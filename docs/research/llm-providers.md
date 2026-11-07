# LLM and embedding providers for the research → hypothesis slice

**Ticket:** [#6](https://github.com/hoisin-sauce/whack-26/issues/6) · **Date:** 2026-11-07 · **Status:** recommendation, ready to implement

---

## TL;DR

- **Primary: Google Gemini API (AI Studio).** A single `GEMINI_API_KEY` covers both jobs:
  - LLM: `gemini-3.8-flash` — free tier, native JSON-Schema structured outputs, strong code generation.
  - Embeddings: `gemini-embedding-2` — free for text input, 128–3072 dimensions (768 recommended here), 8K-token input.
- **Fallback: Cloudflare Workers AI.** One account, two credentials (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`), covers both jobs:
  - LLM: `@cf/openai/gpt-oss-120b`.
  - Embeddings: `@cf/baai/bge-m3`.
  - 10,000 Neurons/day free; zero data retention and no training. JSON mode is not a strict schema, so validate outputs client-side.
- **If strict schema is non-negotiable:** Groq (`openai/gpt-oss-120b` or `qwen/qwen3.8-27b`, strict `json_schema`) plus Jina (`jina-embeddings-v5-text-small`, free 100 RPM / 100K TPM) or local `sentence-transformers`. More setup, stronger schema guarantee.
- **Eliminated for free use:** GitHub Models (retired 2026-07-30), Hugging Face Inference Providers (no free monthly credits), Mistral API (free mode has no API access), OpenRouter free tier (50 requests/day without a $10 lifetime top-up).
- **Legal caveat that matters most:** Gemini's unpaid tier is not private. Google uses free-tier prompts/responses to improve its products and human reviewers may read them. Do **not** send sensitive, confidential, or personal data through the free tier. In the EEA, Switzerland, and the UK, paid-tier data-use terms apply even to unpaid quota, but you may only use Paid Services when making an API client available to users in those regions (relevant only if you distribute the tool).

---

## 1. What the slice actually needs

The 6–8 hour slice (map issue #1) takes any pandas-readable dataset plus an optional description and:

1. classifies columns into a field hierarchy;
2. fetches and caches free-API research;
3. generates 3 runnable Python hypotheses (functions with applicability predicates);
4. runs an agentic critic over the derivation method;
5. writes a JSON artifact plus a Markdown preview.

That maps to exactly **two provider capabilities**:

| Job | Capability | Why |
|---|---|---|
| Field classification | Embeddings + LLM | Embeddings give cheap similarity between column names/values and candidate hierarchy nodes; the LLM disambiguates and emits the final structure. |
| Research retrieval | Embeddings | Embed cached research summaries, retrieve top-k evidence per hypothesis. |
| Research summarisation | LLM | Condense fetched pages into short, citable notes. |
| Hypothesis generation | LLM + code generation | Emit runnable pandas/numpy functions plus predicates. |
| Critic | LLM + structured output | Return a machine-readable verdict and scores. |

**Not needed:** a reranker, a vector database, fine-tuning, a hosted RAG service, or separate vendors for LLM and embeddings. A few hundred vectors fit in the JSON artifact; cosine similarity in NumPy is enough.

**Constraints:** local-first, free APIs preferred, legal use, Python 3.12 (3.15 also installed), no Docker/Postgres, no keys yet, rate limits explicitly not a concern for this slice.

---

## 2. Evaluation criteria

1. **Structured output** — native JSON Schema or strict mode, so hypotheses and critic verdicts parse reliably.
2. **Python code generation** — models that reliably emit runnable pandas/numpy code.
3. **Embedding quality and control** — retrieval quality, dimensions, task instructions, multilingual support.
4. **Python 3.12 support** — installable, maintained client. (3.15 is too new to assume every transitive dependency supports it; pin 3.12 for the slice.)
5. **Free tier** — enough for iterative local runs, ideally without a credit card.
6. **Terms and data use** — legal in the user's jurisdiction, no training on submitted data, acceptable retention.
7. **Operational simplicity** — number of vendors and keys; OpenAI-compatible APIs are a bonus.

---

## 3. Comparison

### 3.1 LLM candidates

| Provider / model | Structured output | Python code gen | Free tier | Data use | Python 3.12 client | Verdict |
|---|---|---|---|---|---|---|
| **Google Gemini** `gemini-3.8-flash` | Native `responseSchema` / JSON Schema [G5] | Strong; Flash class is positioned for software engineering and agents [G1] | Free of charge; per-model RPM/RPD shown in AI Studio [G1][G4] | Unpaid tier used to improve products; human reviewers may read input/output; EEA/CH/UK paid terms apply [G2] | `google-genai` 3.0.0, `>=3.11` [G7] | **Primary** |
| **Cloudflare Workers AI** `@cf/openai/gpt-oss-120b` | JSON mode, no strict schema documented for this model [CF4] | Capable open-weight 120B model [CF2] | 10,000 Neurons/day [CF1] | Zero data retention, no training [CF6] | `cloudflare` 5.10.0 `>=3.9`; also OpenAI-compatible REST [CF7][CF5] | **Fallback** |
| **Groq** `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` | Strict `json_schema` (`strict: true`) [GR2] | Good; code-capable open models [GR3] | 30 RPM / 1,000 RPD / 8K TPM / 200K TPD [GR1] | No retention by default, no training, ZDR available [GR4] | `groq` 1.7.0, `>=3.10` [GR5] | Alternative if strict schema |
| **Cohere** Command A+/A | Structured outputs [C5] | Good | Trial key free, 1,000 calls/month; chat 20 RPM [C1][C4] | Trial keys not permitted for production or commercial use; data-use terms not confirmed [C1][C2] | `cohere` 7.3.1, `>=3.10,<4.0` [C6] | Backup only |
| **OpenRouter** (12 `:free` models as of 2026-11-07) | Per-endpoint support [OR2] | Varies by model | 20 RPM / 50 RPD while lifetime credits < $10; 1,000 RPD after [OR1] | Per-provider; free endpoints may log or train [OR4] | OpenAI-compatible | Not recommended |
| **Hugging Face Inference Providers** | Varies | Varies | No monthly credits for free users; $2/month only on PRO/Team/Enterprise [HF1] | Per-provider | `huggingface_hub` | Not viable free |
| **GitHub Models** | — | — | Retired 2026-07-30 [GH1] | — | — | Eliminated |
| **Mistral** | — | — | Free mode includes no API access; API requires billing activation [M1] | — | `mistralai` | Eliminated |

### 3.2 Embedding candidates

| Provider / model | Dimensions / context | Quality signal | Free tier | Data use | Client | Verdict |
|---|---|---|---|---|---|---|
| **Gemini** `gemini-embedding-2` | 128–3072 (recommended 768 / 1536 / 3072); 8,192-token input [G3] | MTEB 67.99 at 768 dims, 68.16 at 2048 [G3] | Text input free of charge [G1] | Same terms as the LLM [G2] | `google-genai` [G7] | **Primary** |
| **Cloudflare** `@cf/baai/bge-m3` | 1,024 dims; 8,192-token input | Multilingual retrieval model [CF2] | 1,075 Neurons per 1M tokens [CF1] | Zero retention, no training [CF6] | OpenAI-compatible `/v1/embeddings` [CF5] | **Fallback** |
| **Jina** `jina-embeddings-v5-text-small` | MRL down to 32 dims; 32K-token input [J1] | MMTEB 67.0, English MTEB 71.7 [J1] | Free key: 100 RPM / 100K TPM; no batch limit [J1] | API data never used for training [J1] | REST, OpenAI-compatible schema [J1] | Best free embedding-only option |
| **Cohere** `embed-v5.0-pro` / `embed-v5.0-fast` | 256–2048 dims; 128K-token input [C3] | 100+ languages; no public benchmark cited [C3] | Trial key: 1,000 calls/month [C1] | Trial restrictions apply; data-use not confirmed [C1][C2] | `cohere` [C6] | Backup |
| **Local** `BAAI/bge-m3` via `sentence-transformers` | 1,024 dims; 8,192-token input [S2] | Strong multilingual retrieval model [S2] | Free (local compute) | Data never leaves the machine | `sentence-transformers` 6.2.0, `>=3.10` [S1] | Privacy fallback |

---

## 4. Findings by provider

### 4.1 Google Gemini API (AI Studio) — primary

**Why it wins:** it is the only candidate that provides a strong code-generation LLM and a high-quality embedding model behind one free key with terms suitable for this use, plus native structured outputs.

**Models and pricing (2026-11-07):**

- `gemini-3.8-flash`: free tier input, output, and context caching are **free of charge**. Paid price is $0.75/M input and $3.75/M output through 2026-12-31 [G1].
- `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite`: also free of charge; useful for cheap classification/summarisation calls [G1].
- `gemini-4-argon`: **not available on the free tier** [G1].
- `gemini-3.1-pro-preview`: **not available on the free tier** [G1].
- `gemini-embedding-2`: text input free of charge; 8,192-token input; output dimensions 128–3072; MTEB 67.99 at 768 dims vs 68.16 at 2048 — 768 loses ~0.17 points while shrinking vectors 2.7× [G1][G3].

**Structured output:** `responseSchema` / `response_json_schema` accepts a JSON Schema subset and is the native way to constrain output [G5].

**Embedding details that affect implementation:**

- `gemini-embedding-2` does **not** accept `task_type`. For text-only use, put the task in the prompt: `task: classification | query: {content}` for classification, and `task: search result | query: {content}` for queries with `title: {title} | text: {content}` for documents [G3].
- Passing a plain list of strings to `contents` returns **one aggregated embedding**, not one per string. Wrap each chunk in its own `types.Content` object (or use the Batch API) to get separate vectors [G3].
- `gemini-embedding-2` auto-renormalises truncated dimensions; `gemini-embedding-001` requires manual normalisation below 3072 [G3].
- Embedding spaces between `gemini-embedding-001` and `gemini-embedding-2` are incompatible; do not mix them in one index [G3].

**Free-tier limits:** the public rate-limit page does not list per-model RPM/RPD; it points to AI Studio, where the limits are shown per project/model [G4]. Do not assume a specific daily number — check the AI Studio dashboard before committing to a free-only workflow.

**Terms — the important part:**

- Unpaid Services: Google uses submitted content and generated responses to provide, improve, and develop its products and ML technologies; **human reviewers may read, annotate, and process API input and output**. Do not submit sensitive, confidential, or personal information [G2].
- Paid Services: prompts/responses are **not** used to improve products; they are logged for a limited period for abuse prevention [G2].
- EEA / Switzerland / UK: the paid-tier data-use terms apply to all Services, including unpaid quota; however, you may use **only Paid Services when making API Clients available to users** in those regions [G2]. A local-only tool used by the developer is not "making an API Client available to users", but distributing it there would require paid service.

**Python client:** `google-genai` 3.0.0 declares `requires_python >=3.11` and classifiers for 3.12 and 3.15 [G7]. Pin Python 3.12 for the slice.

### 4.2 Cloudflare Workers AI — fallback

**Why it is the fallback:** one vendor covers both modalities, the free quota is predictable, and the data terms are the strongest of the free options.

- **Free allocation:** 10,000 Neurons/day [CF1].
- **LLM:** `@cf/openai/gpt-oss-120b` costs ~31,818 Neurons per 1M input tokens and ~68,182 per 1M output tokens [CF1].
- **Embeddings:** `@cf/baai/bge-m3` costs ~1,075 Neurons per 1M input tokens [CF1].
- **APIs:** Workers AI exposes OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoints, so the `openai` Python client works with a custom `base_url`; the `cloudflare` SDK also works [CF5][CF7].
- **Structured output:** JSON mode is documented for a subset of models; no strict JSON-Schema guarantee was found for `@cf/openai/gpt-oss-120b` [CF4]. Treat the schema as a prompt contract and validate with Pydantic, retrying once on failure.
- **Data use:** zero data retention and no training on Workers AI inputs/outputs [CF6].

**Trade-off:** weaker schema enforcement than Gemini or Groq, and the neuron budget supports roughly 1–2 full slice runs per day (see §7). That is acceptable for a fallback.

### 4.3 Groq + Jina — strict-schema alternative

Use this if the primary and fallback both fail and strict schema matters more than having one vendor.

- **Groq free plan:** 30 RPM, 1,000 RPD, 8K TPM, 200K TPD for `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` [GR1].
- **Structured output:** strict `json_schema` (`strict: true`) is supported on the GPT-OSS 20B/120B and Qwen3.8 27B models [GR2].
- **No embeddings:** Groq does not serve embedding models, so pair it with Jina or local `sentence-transformers`.
- **Data use:** inference data is not retained by default; no training; zero data retention is available [GR4].
- **Jina free key:** 100 RPM / 100K TPM, no batch-size limit, no training on API data, one key across Reader/Embeddings/Reranker [J1].
- **Jina model:** `jina-embeddings-v5-text-small` (677M parameters, 32K context, MRL down to 32 dims, MMTEB 67.0 / English MTEB 71.7) [J1].
- **Catch:** 8K TPM means an 80K-token run takes at least ~10 minutes of wall-clock time; 200K TPD allows only 1–2 runs/day. Use it as a correctness fallback, not a high-throughput primary.

### 4.4 Cohere — single-key backup

- **Trial key:** free, 1,000 API calls/month; chat 20 RPM; Embed 2,000 inputs/min [C1][C4].
- **Models:** Command A+/A for chat, `embed-v5.0-pro` / `embed-v5.0-fast` for embeddings (256–2048 dims, 128K context, 100+ languages) [C3].
- **Structured outputs:** supported on Command A+/A/R+/R [C5].
- **Terms:** trial keys are free but **not permitted for production or commercial purposes** [C1]. The public pricing page does not state whether trial inputs are used for training; the SaaS agreement/DPA would need review before sending anything sensitive [C2].
- **Verdict:** viable as a one-key backup for non-sensitive experimentation, but the data-use ambiguity disqualifies it as the recommended fallback for a "legal use" requirement.

### 4.5 Rejected options

- **OpenRouter:** only 12 `:free` models on 2026-11-07, and free usage is capped at 20 RPM / 50 RPD until lifetime credits exceed $10 [OR1][OR3]. Structured outputs are per-endpoint [OR2], and free endpoints may have provider-specific logging/training policies [OR4]. Too restrictive and too variable for a primary or fallback.
- **Hugging Face Inference Providers:** free users receive no monthly credits; the $2/month included credit is only for PRO, Team, and Enterprise plans [HF1]. Not a free option.
- **GitHub Models:** retired on 2026-07-30 [GH1]. Do not design around it.
- **Mistral:** "Free mode" includes a monthly budget for Vibe and Studio but **does not include API access**; creating API keys and using the API require billing activation and pay-as-you-go [M1]. Not a free API option.
- **OpenAI / Anthropic / Voyage direct:** capable but paid; out of scope for a free-API-first slice.

---

## 5. Recommendation

### 5.1 Primary — Google Gemini API

| Decision | Value |
|---|---|
| Key | `GEMINI_API_KEY` |
| LLM | `gemini-3.8-flash` |
| Cheap helper model | `gemini-3.1-flash-lite` (free tier) |
| Embedding model | `gemini-embedding-2` |
| Embedding dimensions | `768` (MTEB 67.99; keeps the JSON artifact small) |
| LLM structured output | `response_schema` / `response_json_schema` |
| Embedding task format | `task: classification \| query: ...` and `task: search result \| query: ...` / `title: ... \| text: ...` |
| Cost | $0 on the free tier |

**Why:** one key, both modalities, native schema enforcement, strong code generation, and embeddings that retain quality at 768 dims. The slice's expected volume (3 hypotheses, a few hundred embeddings) is small relative to typical free-tier quotas, but confirm the exact RPM/RPD in AI Studio.

**Condition:** do not send sensitive/confidential/personal data through the free tier. If the dataset is sensitive, use Cloudflare or local embeddings instead (see §8).

### 5.2 Fallback — Cloudflare Workers AI

| Decision | Value |
|---|---|
| Credentials | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| LLM | `@cf/openai/gpt-oss-120b` |
| Embedding model | `@cf/baai/bge-m3` |
| Structured output | JSON mode + Pydantic validation + one repair retry |
| Cost | $0 within 10,000 Neurons/day |

**Why:** one vendor, both modalities, predictable free quota, zero data retention, and OpenAI-compatible endpoints that keep the client code almost identical to Groq/OpenRouter. It is the right fallback when the primary is unavailable, or when the data must go to a no-retention provider.

### 5.3 If strict schema is non-negotiable — Groq + Jina

Use `GROQ_API_KEY` with `openai/gpt-oss-120b` (or `qwen/qwen3.8-27b`) and `JINA_API_KEY` with `jina-embeddings-v5-text-small`. This is the strongest schema guarantee among the free options, at the cost of two keys and a tighter token budget. If you want to avoid the second key entirely, run `BAAI/bge-m3` locally with `sentence-transformers` — no API key, no data egress, but a ~2 GB model download and slower first run.

### 5.4 What not to do

- Do not build against GitHub Models or Mistral's free API; neither exists in a usable form as of 2026-11-07.
- Do not use OpenRouter's free tier as the primary; 50 requests/day is below one full slice run.
- Do not mix embedding models in one index; vector spaces are incompatible.
- Do not rely on the Gemini free tier for sensitive data.

---

## 6. Keys to supply

Only the primary is needed to start. The fallback is optional but recommended before the first run, so a provider outage does not block the slice.

| Variable | Required | Where to get it | Used for |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes (primary)** | https://aistudio.google.com/api-keys | `gemini-3.8-flash`, `gemini-embedding-2` |
| `CLOUDFLARE_ACCOUNT_ID` | Fallback | Cloudflare dashboard → Workers & Pages → Account ID | Workers AI REST base URL |
| `CLOUDFLARE_API_TOKEN` | Fallback | Cloudflare dashboard → My Profile → API Tokens (Workers AI permission) | Workers AI auth |
| `JINA_API_KEY` | Optional | https://jina.ai/api-dashboard | Embeddings if using the Groq+Jina alternative |
| `GROQ_API_KEY` | Optional | https://console.groq.com/keys | Strict-schema LLM alternative |

**Storage:** put these in a repo-root `.env` and add `.env` to `.gitignore`, or set them as OS-level environment variables. Never commit them. The application code should read them via `os.environ` (or `python-dotenv` if the project already uses it).

**Do not add any key for:** GitHub Models, Mistral, or Hugging Face — they are not viable free options for this slice.

---

## 7. Free-tier budget for one slice run

**Assumptions (deliberately conservative):**

- LLM: 80K–150K input tokens and 20K–30K output tokens per full run (classification + summarisation + 3 hypotheses + 3 critic passes).
- Embeddings: 300–800 chunks × 200–400 tokens = 150K–300K tokens per run.
- One "run" = one dataset → 3 hypotheses → JSON artifact.

| Provider | LLM usage per run | Embedding usage per run | Runs/day within free quota |
|---|---|---|---|
| Gemini 3.8 Flash + embedding-2 | 80K–150K in / 20K–30K out | 150K–300K tokens | Limited by per-model RPM/RPD shown in AI Studio [G4]; cost is $0 [G1] |
| Cloudflare gpt-oss-120b + bge-m3 | 3.9K–6.8K Neurons | 161–323 Neurons | ~1–2 full runs/day at the high estimate; ~2–3 at the low estimate (10K/day) [CF1] |
| Groq gpt-oss-120b + Jina v5 | 100K–180K tokens (of 200K TPD) | 150K–300K tokens (of 100K TPM) | ~1–2 runs/day on Groq; embeddings take ~2–3 minutes of TPM [GR1][J1] |
| Cohere trial | ~20–40 chat calls | ~5–10 embed calls | ~20–40 runs/month (1,000 calls) [C1] |

**Reading the numbers:**

- Gemini is the only option where the slice's volume is comfortably free without thinking about daily quotas; the exact RPM/RPD still needs a look in AI Studio.
- Cloudflare's 10K Neurons/day is enough for development and a demo, but not for repeated full runs. Embeddings are negligible (a few hundred Neurons).
- Groq's 8K TPM is the binding constraint, not the daily token count; long prompts are slow.
- Cohere's 1,000 calls/month is plenty for this slice if the trial restriction is acceptable.

---

## 8. Legal and privacy notes

1. **Gemini free tier is not confidential.** Prompts and responses are used to improve Google products; human reviewers may read them. Do not send PII, credentials, proprietary schemas, or confidential research. Use synthetic or public datasets for development. [G2]
2. **Gemini paid tier is the privacy upgrade.** Paid quota is not used for product improvement and is logged only for abuse prevention. If the project later uses real data, budget for paid Gemini or switch to Cloudflare/local embeddings. [G2]
3. **EEA / Switzerland / UK.** Paid-tier data-use terms apply to unpaid quota, but making an API client available to users in those regions requires Paid Services. Local, single-user use is a different case; confirm before distributing. [G2]
4. **Cloudflare** states zero data retention and no training for Workers AI. This is the best free data posture among the candidates. [CF6]
5. **Groq** does not retain inference data by default and does not train on it; ZDR is available. [GR4]
6. **Jina** does not use API requests, inputs, or outputs to train its models. [J1]
7. **Cohere trial keys** are free but not permitted for production or commercial use; data-use terms are not stated on the public pricing page. Treat as non-sensitive experimentation only. [C1][C2]
8. **OpenRouter free endpoints** may have provider-specific logging/training policies; check each model before use. [OR4]
9. **Dataset hygiene.** Even on a no-retention provider, minimise what you send: hash or aggregate high-cardinality column values, drop direct identifiers, and send only column names, dtypes, summary statistics, and public research text.
10. **Local embeddings are the ultimate privacy fallback.** `BAAI/bge-m3` via `sentence-transformers` keeps all data on the machine and needs no key. [S1][S2]

---

## 9. Implementation notes

**Provider abstraction.** Keep the slice provider-agnostic behind two small functions:

```python
def generate_structured(prompt: str, schema: dict, model: str) -> dict: ...
def embed(texts: list[str], task: str, model: str) -> list[list[float]]: ...
```

Configuration selects `gemini` or `cloudflare`; the rest of the slice never imports a vendor SDK directly. This makes the fallback a config change, not a rewrite.

**Structured output handling.**

- Gemini: pass the JSON Schema via `response_schema` / `response_json_schema`; parse the response text and validate with Pydantic. [G5]
- Cloudflare: send `response_format={"type": "json_object"}` and include the schema in the prompt; validate with Pydantic and retry once with the validation error appended. [CF4]
- Groq: use `response_format={"type": "json_schema", "json_schema": {"name": ..., "strict": True, "schema": ...}}`. [GR2]
- Always keep a repair retry: LLMs occasionally emit invalid JSON even with a schema.

**Embedding handling.**

- Pin `output_dimensionality=768` for Gemini; it is within 0.17 MTEB points of 2048 and keeps a 500-vector artifact to a few megabytes of JSON. [G3]
- For `gemini-embedding-2`, wrap each chunk in `types.Content` (or use the Batch API) so you get one vector per chunk, not one aggregated vector. [G3]
- Apply the task prefix consistently: classification uses `task: classification | query: ...`; retrieval uses `task: search result | query: ...` for queries and `title: ... | text: ...` for documents. [G3]
- Cache embeddings by SHA-256 of the chunk text so re-runs do not re-embed unchanged research.
- Store vectors as plain JSON floats for the slice; base64 float32 or int8 are easy later optimisations.

**Rate limits and retries.** Use exponential backoff on HTTP 429 and 5xx, batch embedding calls, and cap concurrency at 2–4 requests. The slice explicitly does not care about throughput, so correctness and quota preservation win.

**Model pinning.** Put model IDs in one config module. Providers rename and retire models frequently; a pinned ID plus a clear error message is easier to debug than a floating alias.

---

## 10. Residual risks and things to re-check at implementation time

| Risk | Mitigation |
|---|---|
| Gemini free-tier RPM/RPD are not published in the docs and may be lower than expected | Check AI Studio before the first run; keep `gemini-3.1-flash-lite` as a cheaper fallback model |
| Cloudflare JSON mode is not a strict schema for `gpt-oss-120b` | Pydantic validation + one repair retry; do not trust the model to honour the schema |
| Cohere trial data-use terms are unconfirmed | Do not use for sensitive data; read the SaaS agreement/DPA before relying on it |
| Model names and prices change frequently (Gemini 4 Argon is paid-only; several models are free today) | Re-check the pricing and models pages before implementation; pin IDs |
| Python 3.15 support is not guaranteed by every client | Use Python 3.12 for the slice; 3.15 is available for later experiments |
| Embedding spaces differ across providers/models | Never mix vectors from different models in one index; re-embed when switching |
| Free-tier data may be used for training (Gemini) | Use public/synthetic data or switch to Cloudflare/local embeddings |

---

## 11. Sources

All sources accessed 2026-11-07.

**Google Gemini**

- [G1] Gemini Developer API pricing — https://ai.google.dev/gemini-api/docs/pricing
- [G2] Gemini API Additional Terms of Service — https://ai.google.dev/gemini-api/terms
- [G3] Gemini API embeddings — https://ai.google.dev/gemini-api/docs/embeddings
- [G4] Gemini API rate limits — https://ai.google.dev/gemini-api/docs/rate-limits
- [G5] Gemini API structured output — https://ai.google.dev/gemini-api/docs/structured-output
- [G6] Gemini API models — https://ai.google.dev/gemini-api/docs/models
- [G7] `google-genai` on PyPI — https://pypi.org/project/google-genai/

**Cloudflare Workers AI**

- [CF1] Workers AI pricing — https://developers.cloudflare.com/workers-ai/platform/pricing/
- [CF2] Workers AI models — https://developers.cloudflare.com/workers-ai/models/
- [CF3] Workers AI limits — https://developers.cloudflare.com/workers-ai/platform/limits/
- [CF4] Workers AI JSON mode — https://developers.cloudflare.com/workers-ai/features/json-mode/
- [CF5] Workers AI OpenAI compatibility — https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/
- [CF6] Workers AI data usage — https://developers.cloudflare.com/workers-ai/platform/data-usage/
- [CF7] `cloudflare` on PyPI — https://pypi.org/project/cloudflare/

**Groq**

- [GR1] Groq rate limits — https://console.groq.com/docs/rate-limits
- [GR2] Groq structured outputs — https://console.groq.com/docs/structured-outputs
- [GR3] Groq models — https://console.groq.com/docs/models
- [GR4] Groq data use — https://console.groq.com/docs/your-data
- [GR5] `groq` on PyPI — https://pypi.org/project/groq/

**Jina**

- [J1] Jina Embeddings API, models, rate limits, and data-use FAQ — https://jina.ai/embeddings/
- [J2] Jina legal — https://jina.ai/legal/

**Cohere**

- [C1] Cohere pricing — https://cohere.com/pricing
- [C2] Cohere "Going Live" (trial vs production keys) — https://docs.cohere.com/docs/going-live
- [C3] Cohere Embed models — https://docs.cohere.com/docs/cohere-embed
- [C4] Cohere rate limits — https://docs.cohere.com/docs/rate-limits
- [C5] Cohere structured outputs — https://docs.cohere.com/docs/structured-outputs
- [C6] `cohere` on PyPI — https://pypi.org/project/cohere/

**OpenRouter**

- [OR1] OpenRouter limits — https://openrouter.ai/docs/api-reference/limits
- [OR2] OpenRouter structured outputs — https://openrouter.ai/docs/features/structured-outputs
- [OR3] OpenRouter models API — https://openrouter.ai/api/v1/models
- [OR4] OpenRouter FAQ — https://openrouter.ai/docs/faq

**Other**

- [HF1] Hugging Face Inference Providers pricing — https://huggingface.co/docs/inference-providers/pricing
- [GH1] GitHub Models documentation (retirement notice) — https://docs.github.com/en/github-models
- [M1] Mistral usage and limits — https://docs.mistral.ai/admin/billing-usage/usage-limits
- [M2] Mistral pricing — https://mistral.ai/pricing/
- [S1] `sentence-transformers` on PyPI — https://pypi.org/project/sentence-transformers/
- [S2] BGE-M3 model card — https://huggingface.co/BAAI/bge-m3
- [S3] `openai` on PyPI (OpenAI-compatible clients) — https://pypi.org/project/openai/
