"""Local OpenAI-compatible model gateway for deterministic stubbing, recording, and replaying.

Provides an endpoint on http://127.0.0.1:18080/v1 that supports:
  - POST /v1/embeddings
  - POST /v1/chat/completions
  - GET  /v1/models (health check)

Modes:
  --mode stub:
      Deterministic fake responses (free, offline, $0).
  --mode record:
      Forwards requests to upstream provider, logs spend, and saves response bodies
      to runs/cache/<sha256>.json. Stops with HTTP 402 if budget is exceeded.
  --mode replay:
      Serves strictly from runs/cache/<sha256>.json. Returns HTTP 500 on cache misses;
      makes zero outbound network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DIMS = 1536
WORD_RE = re.compile(r"[a-z0-9]+")
CAPITALISED_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
STOP_WORDS = {
    "The", "A", "An", "In", "On", "It", "He", "She", "They", "This",
    "That", "What", "Which", "Who", "When", "Where", "Answer", "Return",
    "Question", "Graph", "If", "Every", "Then", "There",
}


def compute_stub_embedding(text: str) -> list[float]:
    """Generate a deterministic 1536-dimensional L2-normalized vector from text."""
    vec = [0.0] * DIMS
    for token in WORD_RE.findall(text.lower()):
        token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        sign = 1.0 if (token_hash >> 20) % 2 else -1.0
        vec[token_hash % DIMS] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def last_user_prompt(body: dict[str, Any]) -> str:
    """Extract the last user message text from an OpenAI chat payload."""
    for message in reversed(body.get("messages", [])):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content or "")
    return ""


def extract_input_text(prompt: str) -> str:
    """Extract paragraph text from a neo4j-graphrag extraction prompt."""
    match = re.search(r"Input text:\s*(.*)", prompt, flags=re.S)
    return (match.group(1) if match else prompt).strip()


def stub_schema_response() -> dict[str, Any]:
    """Deterministic schema response for GraphSchemaExtractionOutput."""
    return {
        "node_types": [
            {"label": "Company", "description": "A business organization", "properties": [{"name": "name", "type": "STRING", "description": ""}]},
            {"label": "Entity", "description": "A general entity", "properties": [{"name": "name", "type": "STRING", "description": ""}]},
        ],
        "relationship_types": [
            {"label": "ACQUIRED", "description": "Commercial acquisition", "properties": [{"name": "since", "type": "STRING", "description": ""}]},
            {"label": "RELATED_TO", "description": "Generic relation", "properties": []},
        ],
        "patterns": [
            {"source": "Company", "relationship": "ACQUIRED", "target": "Company"},
            {"source": "Entity", "relationship": "RELATED_TO", "target": "Entity"},
        ],
        "constraints": [],
    }


def stub_graph_response(text: str) -> dict[str, Any]:
    """Deterministic Neo4jGraph extraction response based on input text."""
    acquired = re.search(
        r"([A-Z][\w]+(?:\s+[A-Z][\w]+)*) acquired ([A-Z][\w]+(?:\s+[A-Z][\w]+)*)(?: in (\d{4}))?",
        text,
    )
    if acquired:
        props = {"since": acquired.group(3)} if acquired.group(3) else {}
        return {
            "nodes": [
                {"id": "0", "label": "Company", "properties": {"name": acquired.group(1)}},
                {"id": "1", "label": "Company", "properties": {"name": acquired.group(2)}},
            ],
            "relationships": [
                {"start_node_id": "0", "end_node_id": "1", "type": "ACQUIRED", "properties": props}
            ],
        }

    names: list[str] = []
    for name in CAPITALISED_RE.findall(text):
        if name not in STOP_WORDS and name not in names:
            names.append(name)
    names = names[:4]
    nodes = [{"id": str(i), "label": "Entity", "properties": {"name": n}} for i, n in enumerate(names)]
    rels = [
        {"start_node_id": "0", "end_node_id": str(i), "type": "RELATED_TO", "properties": {}}
        for i in range(1, len(names))
    ]
    return {"nodes": nodes, "relationships": rels}


def stub_answer_response(prompt: str) -> str:
    """Deterministic grounded answer response citing evidence chunk ids."""
    ids = re.findall(r"\[([^\]]+)\]", prompt)
    payload = {"answer": "Northwind Labs", "citations": ids[:1] or ["none"]}
    return json.dumps(payload)


def canonical_request_hash(body: dict[str, Any]) -> str:
    """Generate SHA-256 hash of canonical JSON request body."""
    normalized = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


class ModelGatewayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for stub, record, and replay modes."""

    server_mode: str = "stub"
    cache_dir: Path = Path("runs/cache")
    budget_usd: float = 5.0
    spent_usd: float = 0.0
    upstream_base_url: str = "https://openrouter.ai/api/v1"
    request_history: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # Silence default HTTP server console logging

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/v1/models"):
            self._send_json(
                200,
                {
                    "status": "healthy",
                    "mode": self.server_mode,
                    "spent_usd": self.spent_usd,
                    "budget_usd": self.budget_usd,
                },
            )
        elif self.path in ("/requests", "/v1/requests"):
            self._send_json(200, {"requests": self.request_history})
        elif self.path in ("/requests/clear", "/v1/requests/clear"):
            self.request_history.clear()
            self._send_json(200, {"status": "cleared"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        self.request_history.append(
            {
                "path": self.path,
                "body": body,
                "headers": dict(self.headers),
                "timestamp": time.time(),
            }
        )

        req_hash = canonical_request_hash(body)

        # 1. REPLAY MODE
        if self.server_mode == "replay":
            cache_file = self.cache_dir / f"{req_hash}.json"
            if not cache_file.exists():
                self._send_json(
                    500,
                    {
                        "error": "Cache miss in replay mode. Zero external calls permitted.",
                        "request_hash": req_hash,
                        "path": self.path,
                    },
                )
                return
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            self._send_json(200, cached.get("response", {}))
            return

        # 2. RECORD MODE
        if self.server_mode == "record":
            cache_file = self.cache_dir / f"{req_hash}.json"
            if cache_file.exists():
                with open(cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                self._send_json(200, cached.get("response", {}))
                return

            if self.spent_usd >= self.budget_usd:
                self._send_json(
                    402,
                    {
                        "error": "Model gateway budget exceeded.",
                        "budget_usd": self.budget_usd,
                        "spent_usd": self.spent_usd,
                    },
                )
                return

            # Forward to upstream
            upstream_url = self.upstream_base_url.rstrip("/") + ("/" + self.path.lstrip("/").replace("v1/", "", 1) if self.upstream_base_url.endswith("/v1") else "/" + self.path.lstrip("/"))
            req = urllib.request.Request(
                upstream_url,
                data=raw_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self.headers.get("Authorization", ""),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_data = resp.read()
                    status = resp.status
            except urllib.error.HTTPError as e:
                self._send_json(e.code, {"error": "Upstream error", "details": e.read().decode("utf-8")})
                return
            except Exception as e:
                self._send_json(502, {"error": f"Gateway connection error: {e}"})
                return

            resp_json = json.loads(resp_data.decode("utf-8"))
            # Track cost if returned by OpenRouter
            call_cost = float(resp_json.get("usage", {}).get("cost", 0.0))
            self.spent_usd += call_cost

            # Write cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "request_hash": req_hash,
                        "timestamp": time.time(),
                        "path": self.path,
                        "request": body,
                        "response": resp_json,
                        "cost": call_cost,
                    },
                    f,
                    indent=2,
                )

            self._send_json(status, resp_json)
            return

        # 3. STUB MODE (Default)
        if self.path.endswith("/embeddings"):
            inputs = body.get("input", [])
            inputs = inputs if isinstance(inputs, list) else [inputs]
            data = [
                {"object": "embedding", "index": i, "embedding": compute_stub_embedding(str(x))}
                for i, x in enumerate(inputs)
            ]
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": body.get("model", "openai/text-embedding-3-small"),
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
            return

        prompt = last_user_prompt(body)
        fmt = body.get("response_format") or {}
        fmt_name = ""
        if isinstance(fmt, dict):
            fmt_name = (fmt.get("json_schema") or {}).get("name", "")

        if self.headers.get("X-Test-Malformed") == "true" or "[FORCE_MALFORMED_JSON]" in prompt:
            content = "This is malformed, not valid JSON output {answer: missing_quotes}"
        elif fmt_name == "GraphSchemaExtractionOutput":
            content = json.dumps(stub_schema_response())
        elif fmt_name == "Neo4jGraph":
            content = json.dumps(stub_graph_response(extract_input_text(prompt)))
        else:
            content = stub_answer_response(prompt)

        self._send_json(
            200,
            {
                "id": "chatcmpl-gateway-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "openai/gpt-4o-mini"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(prompt) + len(content)) // 4,
                },
            },
        )


def run_gateway(
    host: str = "127.0.0.1",
    port: int = 18080,
    mode: str = "stub",
    cache_dir: str = "runs/cache",
    budget_usd: float = 5.0,
    upstream_base_url: str = "https://openrouter.ai/api/v1",
) -> None:
    """Start the model gateway HTTP server."""
    ModelGatewayHandler.server_mode = mode
    ModelGatewayHandler.cache_dir = Path(cache_dir)
    ModelGatewayHandler.budget_usd = budget_usd
    ModelGatewayHandler.upstream_base_url = upstream_base_url

    server = ThreadingHTTPServer((host, port), ModelGatewayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Gateway (Stub, Record, Replay)")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18080, help="Binding port (default: 18080)")
    parser.add_argument("--mode", choices=["stub", "record", "replay"], default="stub")
    parser.add_argument("--cache-dir", default="runs/cache", help="Cache directory")
    parser.add_argument("--budget-usd", type=float, default=5.0, help="Budget ceiling in USD")
    parser.add_argument("--upstream-base-url", default="https://openrouter.ai/api/v1")
    args = parser.parse_args()

    run_gateway(
        host=args.host,
        port=args.port,
        mode=args.mode,
        cache_dir=args.cache_dir,
        budget_usd=args.budget_usd,
        upstream_base_url=args.upstream_base_url,
    )


if __name__ == "__main__":
    main()
