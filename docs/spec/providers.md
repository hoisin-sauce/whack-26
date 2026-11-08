# Provider seams: LLM, embeddings, and mock mode

Status: build detail derived from the provider research and the live probes in `api-notes.md`.

## Interfaces

```python
class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict: ...

class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...
```

Both return validated Python objects, not raw provider responses. The caller never sees provider-specific JSON.

## Gemini implementation

- Base: `https://generativelanguage.googleapis.com/v1beta`
- Auth: `x-goog-api-key: $GEMINI_API_KEY`
- LLM: `POST /models/{model}:generateContent`
- Embeddings: `POST /models/{model}:batchEmbedContents` (fall back to `:embedContent` one at a time)
- Defaults: `GEMINI_MODEL=gemini-3.6-flash`, `GEMINI_EMBED_MODEL=gemini-embedding-2`, `EMBED_DIMS=768`
- Every deterministic call sets `temperature=0` and `thinkingConfig={"thinkingBudget": 0}`.
- Structured output: `responseMimeType="application/json"` + `responseSchema`; validate with Pydantic; retry once on schema failure with the validation error.
- Timeout: 60 s per LLM call, 30 s per embedding batch. Retries: 2 on 429/5xx/timeout with 1 s and 4 s backoff.
- Embeddings: L2-normalize on receipt. Batch up to 50 texts; if a batch fails, split it and retry; if a single item fails, retry once, then raise.
- The free tier is not confidential; never send PII, secrets, or proprietary schemas. The caller must be able to switch providers for private data.

## Cloudflare Workers AI (fallback)

- OpenAI-compatible base: `https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/v1`
- Auth: `Authorization: Bearer $CLOUDFLARE_API_TOKEN`
- LLM: `@cf/openai/gpt-oss-120b`; embeddings: `@cf/baai/bge-m3`.
- JSON mode is not strict; validate with Pydantic and retry.
- Untested in this environment (no credentials yet). Treat as a fallback, not the default.

## Local embeddings (privacy fallback)

`model2vec` with `potion-base-8M` (256 dims, no key). If selected, set `WHACK26_EMBED_PROVIDER=local` and `EMBED_DIMS=256`. The field store is dimension-sensitive: a mismatch must fail fast at startup with a clear error and a `--reset-fields` instruction. Do not mix embedding providers within one field store.

## Mock mode

`--mock` or `WHACK26_LLM_PROVIDER=mock` / `WHACK26_EMBED_PROVIDER=mock` replaces network calls with deterministic fixtures so the pipeline, artifact, and tests run before keys exist.

- **Embeddings:** hash the text (SHA-256) into a 768-d unit vector. Similar texts still have some signal only if fixtures are crafted; mock mode tests plumbing, not classification quality.
- **LLM:** a `fixtures/` directory keyed by `purpose`:
  - `classification`: `create` a root, then `assign` to the best existing field;
  - `queries`: canned query list built from the field path;
  - `factors`: canned factors with evidence IDs from the mock records;
  - `hypotheses`: one runnable template per family using the profile's numeric columns;
  - `critic`: one `pass` and two seeded issues, chosen by a fixture flag.
- Mock hypotheses must pass the same AST checks and execute in the same subprocess runner. No special-casing in the artifact path.
- Mock mode is visibly labelled in `run.mode = "mock"` and in the preview.

## Selection and configuration

| Env var | Values | Default |
|---|---|---|
| `WHACK26_LLM_PROVIDER` | `gemini`, `cloudflare`, `mock` | `gemini` |
| `WHACK26_EMBED_PROVIDER` | `gemini`, `local`, `mock` | `gemini` |
| `GEMINI_MODEL` | model id | `gemini-3.6-flash` |
| `GEMINI_EMBED_MODEL` | model id | `gemini-embedding-2` |
| `EMBED_DIMS` | int | `768` |
| `WHACK26_OFFLINE` | `0`, `1` | `0` |
| `WHACK26_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` | `INFO` |

`doctor` checks: Python version, installed packages, corpus presence, key presence, Gemini model latency (one tiny call), embedding dimensions, OpenAlex auth, and cache writability. It exits non-zero on a blocking problem and prints a remediation hint.

## Logging and privacy

- Log latency, token counts, provider, model, and status. Do not log prompts or responses by default (`WHACK26_LOG_PROMPTS=0`).
- Never write API keys to the artifact, cache metadata, logs, or error messages.
- Redact `Authorization`, `x-goog-api-key`, and `api_key` query values from exception text.
- The artifact stores model ids and provider names, not credentials.

## Failure behavior

- LLM unavailable: retry; if still unavailable, the pipeline fails with a typed error and a hint to use `--mock` or Cloudflare. Do not silently produce a partial artifact.
- Embedding unavailable: same, except the classifier can fall back to the local provider if configured.
- A single schema failure: retry once; a second failure is fatal for that step.
- All errors include the purpose (`classification`, `queries`, `factors`, `hypothesis:<family>`, `critic:<hyp_id>`) so the run log identifies the stage.
