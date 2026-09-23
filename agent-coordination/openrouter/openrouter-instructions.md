# OpenRouter Configuration And Model Recommendation

**Last researched:** 23 September 2026

**Purpose:**
Provide the local setup and model-selection guidance for using OpenRouter with
Neo4j GraphRAG `SimpleKGPipeline` and the later Neo4j vector-retrieval path.

**Important:**
OpenRouter's catalog, prices, free availability, provider endpoints, rate
limits, and model support change. Treat this document as a dated baseline, not
as a permanent guarantee. Before a real run, query the OpenRouter Models API
and run the smoke tests below.

## Recommendation In Brief

Use this initial configuration:

| Role | Initial model | Why |
|---|---|---|
| MuSiQue extraction LLM | `qwen/qwen3.8-27b:free` | Free catalog variant; listed with `structured_outputs`, tools, and a 262,144-token context window. Large enough to be a reasonable first structured extraction candidate. |
| Embedding model | `liquid/lfm-2.5-embedding-350m:free` | Free catalog variant; OpenRouter documents 1,024-dimensional text embeddings. |
| Vector similarity | cosine | Matches the project's Neo4j retrieval notes and is appropriate for normalized semantic vectors. |
| First pipeline input | `tests/fixtures/musique/extraction_chunk.json` | Small, source-controlled, provenance-focused fixture. |

The extraction model and embedding model are separate. The LLM produces the
structured graph extraction; the embedding model produces vectors for lexical
chunks and later query-time retrieval.

## Free Router Clarification

OpenRouter has several related concepts that should not be conflated:

- `openrouter/auto` is the Auto Router. It chooses models using task type,
  market usage, model capability, and a cost tier. It is not a free-only
  router. It charges the normal price of whichever model it selects.
- A model's `:free` catalog variant, such as
  `qwen/qwen3.8-27b:free`, explicitly selects a free model entry when that
  variant exists.
- OpenRouter's Free Models Router is primarily described for the Chat
  Playground. For reproducible application runs, prefer a pinned `:free`
  model ID and record the exact model returned by the API.

Do not use `openrouter/auto` for the acceptance run unless the agent restricts
its allowed models to known structured-output-capable free variants and records
the selected model from the response. Even then, automatic routing reduces
reproducibility.

Free endpoints can have lower rate limits, temporary unavailability, provider
retention differences, and changing availability. A failed free request should
be reported clearly rather than silently switching to a paid model.

## Why Qwen3.8-27B Free Is The Baseline

The OpenRouter Models API was queried on 23 September 2026 with the
`structured_outputs` filter. The catalog listed `qwen/qwen3.8-27b:free` with:

- prompt price: `$0`
- completion price: `$0`
- context length: 262,144 tokens
- `structured_outputs` support
- `response_format` support
- tool-choice support
- a non-expired catalog entry at research time

This is a catalog capability check, not a guarantee that every upstream
provider endpoint will enforce JSON Schema correctly. OpenRouter states that
structured-output support is endpoint-specific. The request must therefore set
`provider.require_parameters=true` and use a schema request with
`response_format.type=json_schema`.

Recommended extraction request settings:

```text
model: qwen/qwen3.8-27b:free
temperature: 0 where accepted
stream: false
provider.require_parameters: true
provider.allow_fallbacks: false for the first reproducibility test
on_error: RAISE
perform_entity_resolution: false for raw-output inspection
```

If the free endpoint is unavailable or fails the schema probe, test these
low-cost structured-output alternatives, in this order:

1. `qwen/qwen3.8-27b` - paid version of the same family.
2. `deepseek/deepseek-v4-flash-0731` - low-cost, structured-output-capable
   model listed by the catalog.
3. `google/gemini-2.5-flash-lite` - low-cost, structured-output-capable model.
4. `openai/gpt-4.1-mini` - more expensive, but a conservative compatibility
   fallback.

Prices and exact availability must be rechecked through the Models API before
using any paid fallback. Never let a fallback silently change the model for a
benchmark or acceptance artifact.

## Embedding Recommendation

The OpenRouter embeddings catalog listed these relevant free options on the
research date:

- `liquid/lfm-2.5-embedding-350m:free`: 1,024 dimensions, 512-token context.
- `nvidia/nemotron-3-embed-1b:free`: free text embedding model with a 32,768
  token context; the catalog did not expose its output dimension in the
  returned metadata, so the application must probe the vector length before
  creating an index.
- `nvidia/llama-nemotron-embed-vl-1b-v2:free`: free multimodal model; not
  needed for text-only MuSiQue paragraphs.

Use Liquid as the first reproducible baseline only if the project chunks text
so each embedding input is at most 512 tokens. MuSiQue paragraphs should be
checked with the selected tokenizer before embedding. If a paragraph exceeds
that limit, prefer the NVIDIA free text model after measuring its vector
length, or use a low-cost paid model with a larger context.

The vector index dimension must be measured or verified from the actual API
response. Do not infer an embedding dimension from the model name and do not
create an index until the probe succeeds.

For a low-cost paid fallback, OpenRouter lists `openai/text-embedding-3-small`
at a very low price. Its exact output dimension should still be measured from
the response used by this project, and the same model must be used for both
indexing and query-time embeddings.

## OpenRouter Environment Configuration

Copy the checked-in template [config.example.env](config.example.env) to the
ignored repository `.env` and fill in the key. The template contains no secret.

Required values:

```env
OPENROUTER_API_KEY=replace-with-your-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_LLM_MODEL=qwen/qwen3.8-27b:free
OPENROUTER_EMBEDDING_MODEL=liquid/lfm-2.5-embedding-350m:free
OPENROUTER_EMBEDDING_DIMENSIONS=1024
OPENROUTER_REQUIRE_PARAMETERS=true
OPENROUTER_ALLOW_FALLBACKS=false
OPENROUTER_DATA_COLLECTION=deny
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=musique-graphrag
```

`OPENROUTER_DATA_COLLECTION=deny` is a project default for the first run,
subject to the endpoints available for the selected model. OpenRouter's
provider routing policy may reject a request if no eligible endpoint satisfies
all restrictions. If that happens, record the failure and decide explicitly
whether to allow collection; do not remove the restriction silently.

The API key must never be committed, printed, placed in a run artifact, or
sent through chat. `.env` is already ignored by the repository.

## Python/OpenAI-Compatible Configuration

OpenRouter exposes an OpenAI-compatible API. A GraphRAG OpenAI-compatible LLM
adapter should use:

```text
base_url=https://openrouter.ai/api/v1
api_key=OPENROUTER_API_KEY
model=OPENROUTER_LLM_MODEL
```

The installed `neo4j-graphrag==1.21.0` API must be inspected before choosing the
exact constructor and provider extra. Do not assume the package reads
`OPENROUTER_BASE_URL` automatically. The agent should pass `base_url` directly
when the client supports it or add only a thin compatibility adapter.

For the first structured-output request, the underlying HTTP body should be
equivalent to:

```json
{
  "model": "qwen/qwen3.8-27b:free",
  "messages": [{"role": "user", "content": "Return a test object."}],
  "temperature": 0,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "smoke_test",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": false
      }
    }
  },
  "provider": {
    "require_parameters": true,
    "allow_fallbacks": false
  }
}
```

The agent must validate the response, record the returned model ID and
OpenRouter generation ID, and redact the response content from logs unless it
is explicitly part of a controlled test artifact.

## Embedding Probe

Before creating a Neo4j vector index, call the embeddings endpoint with one
short string and measure the returned list length:

```bash
curl -sS "$OPENROUTER_BASE_URL/embeddings" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$OPENROUTER_EMBEDDING_MODEL"'","input":"MuSiQue Neo4j smoke test"}'
```

The implementation should parse `data[0].embedding`, verify it is numeric, and
set the Neo4j index dimension to `len(embedding)`. The value in
`OPENROUTER_EMBEDDING_DIMENSIONS` is an expectation, not authority; fail if the
measured value differs.

Use the same model, dimension, similarity function, and preprocessing for
both document indexing and query embedding. Never mix vectors produced by two
embedding models in one Neo4j index.

## Neo4j Architecture Choice

There are two supported embedding paths:

### OpenRouter Python embedding path

```text
MuSiQue text -> OpenRouter /embeddings -> Python/GraphRAG embedder -> Neo4j vector property/index
```

This guide configures that path because it is explicit and easy to probe.

### Neo4j server-side embedding path

```text
MuSiQue text -> Neo4j genai.vector.encodeBatch -> Neo4j vector property/index
```

The local server exposes `genai.vector.encodeBatch`, so the agent may later
switch to this path. If it does, document the Neo4j provider configuration,
model, and measured dimension, and remove the Python embedding dependency from
the active path. Do not mix server-generated and OpenRouter-generated vectors.

## Required Setup Steps

1. Create an OpenRouter account and API key.
2. Copy `config.example.env` into the repository `.env` file.
3. Add the OpenRouter key and verify the selected model IDs are still listed.
4. Start the local Neo4j Enterprise `2026.08.1` database.
5. Run the OpenRouter structured-output smoke test.
6. Run the embedding probe and verify its vector dimension.
7. Install only the GraphRAG/OpenAI-compatible provider extra required by the
   installed package version.
8. Run the focused `SimpleKGPipeline` fixture with `on_error=RAISE`.
9. Inspect graph labels, relationships, Document/Chunk metadata, and
   provenance.
10. Create the vector index only after the measured embedding dimension is
    known.

## Verification Commands

```bash
source .venv/bin/activate
python -m pip check
python -m pytest tests -q
python -m pytest tests/integration/test_neo4j_connection_smoke.py -v -rs
```

The agent should also query the current model catalog without exposing the API
key:

```bash
curl -sS "$OPENROUTER_BASE_URL/models?output_modalities=text&supported_parameters=structured_outputs" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

## Acceptance Criteria

The OpenRouter configuration is ready for the plugin run when:

- the LLM model is currently available;
- a strict JSON Schema smoke test succeeds;
- the response reports the expected model family;
- the embedding request succeeds;
- the measured embedding dimension matches configuration;
- provider routing does not silently select an unsupported paid endpoint;
- credentials remain outside source control and logs;
- the same embedding model is used for indexing and querying; and
- model IDs, dimensions, routing settings, package versions, and timestamps
  are recorded in the redacted run artifact.

## Sources Consulted

- OpenRouter Models API: https://openrouter.ai/api/v1/models
- OpenRouter embeddings API: https://openrouter.ai/docs/api-reference/embeddings
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter provider routing: https://openrouter.ai/docs/guides/routing/provider-selection
- OpenRouter Auto Router: https://openrouter.ai/docs/guides/routing/routers/auto-router
- OpenRouter `:free` variants: https://openrouter.ai/docs/guides/routing/model-variants/free
- Neo4j GraphRAG Knowledge Graph Builder:
  https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html

## Caveats

- Free model availability and rate limits can change without notice.
- OpenRouter structured-output support is endpoint-specific, not only model-
  specific.
- The OpenRouter API catalog did not expose the output dimension for every
  embedding model; measure it directly.
- A free model may be unsuitable for production-quality MuSiQue extraction.
  Compare its graph output against the fixture and retain a paid fallback
  configuration only if quality is inadequate.
- This file does not store credentials.
