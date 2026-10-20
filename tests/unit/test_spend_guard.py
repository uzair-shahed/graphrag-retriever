"""Unit tests for spend cap guard and cost forecast (Step M4.4)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.forecast import calculate_forecast, main as forecast_main, measure_unit_cost_from_cache
from tools.model_gateway import ModelGatewayHandler


def test_calculate_forecast_computes_expected_total() -> None:
    """Forecast equals planned calls multiplied by unit cost."""
    forecast = calculate_forecast(planned_calls=15, unit_cost=0.002)
    assert forecast["planned_calls"] == 15
    assert forecast["unit_cost_usd"] == 0.002
    assert forecast["forecast_usd"] == 0.03


def test_measure_unit_cost_from_cache(tmp_path: Path) -> None:
    """Average unit cost is calculated from cached run responses."""
    file1 = tmp_path / "req1.json"
    file1.write_text(json.dumps({"cost": 0.002}), encoding="utf-8")
    file2 = tmp_path / "req2.json"
    file2.write_text(json.dumps({"response": {"usage": {"cost": 0.004}}}), encoding="utf-8")

    measured = measure_unit_cost_from_cache(tmp_path)
    assert abs(measured - 0.003) < 1e-6


def test_forecast_cli_runs_with_json_output(capsys) -> None:
    """CLI script outputs valid JSON with planned calls and projected forecast."""
    code = forecast_main(["--planned-calls", "50", "--unit-cost", "0.001", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["planned_calls"] == 50
    assert data["forecast_usd"] == 0.05


def test_gateway_refuses_at_budget_cap_and_matches_forecast() -> None:
    """Gateway stops new calls when budget cap is reached, matching the forecast total."""
    planned_calls = 3
    fake_unit_cost = 0.001
    budget_usd = planned_calls * fake_unit_cost

    forecast = calculate_forecast(planned_calls, fake_unit_cost)
    assert forecast["forecast_usd"] == budget_usd

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelGatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        # Configure gateway limits
        req_cfg = urllib.request.Request(
            f"{base_url}/config",
            data=json.dumps(
                {
                    "budget_usd": budget_usd,
                    "fake_unit_cost": fake_unit_cost,
                    "reset_spend": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_cfg) as resp:
            cfg = json.loads(resp.read().decode())
            assert cfg["budget_usd"] == budget_usd

        # Run planned calls within budget
        for i in range(planned_calls):
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": f"question {i}"}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode())
                assert data["usage"]["cost"] == fake_unit_cost

        # Health check verifies stub's actual total equals the forecast
        with urllib.request.urlopen(f"{base_url}/health") as resp:
            health = json.loads(resp.read().decode())
            assert abs(health["spent_usd"] - forecast["forecast_usd"]) < 1e-6

        # Call beyond the cap must be refused with HTTP 402 and clear message
        req_overflow = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "overflow question"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req_overflow)
            assert False, "Expected HTTPError 402"
        except urllib.error.HTTPError as error:
            assert error.code == 402
            body = json.loads(error.read().decode("utf-8"))
            assert "Model gateway budget exceeded" in body["error"]
            assert f"${budget_usd:.4f}" in body["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_gateway_refuses_at_max_new_calls_limit() -> None:
    """Gateway stops calls when max_new_calls limit is reached."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelGatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        # Set max_new_calls = 2
        req_cfg = urllib.request.Request(
            f"{base_url}/config",
            data=json.dumps(
                {
                    "budget_usd": 10.0,
                    "max_new_calls": 2,
                    "reset_spend": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_cfg) as resp:
            cfg = json.loads(resp.read().decode())
            assert cfg["max_new_calls"] == 2

        # 2 allowed calls
        for i in range(2):
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": f"call {i}"}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200

        # 3rd call must be refused with HTTP 402
        req_3 = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "call 3"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req_3)
            assert False, "Expected HTTPError 402"
        except urllib.error.HTTPError as error:
            assert error.code == 402
            body = json.loads(error.read().decode("utf-8"))
            assert "Model gateway call limit exceeded" in body["error"]
    finally:
        server.shutdown()
        server.server_close()
