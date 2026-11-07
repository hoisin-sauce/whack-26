# Python and package stack for the first slice

- **Ticket:** #7 — Verify Python and package compatibility
- **Status:** research complete, recommendation below
- **Date:** 2026-11-07
- **Environment tested:** Windows `win_amd64`; Python **3.12.4** and **3.15.0**; pip 26.3 in fresh venvs; no Docker, Postgres, `uv`, Ollama, or API keys.
- **Scope:** runtime version, package manager, minimal dependency set, and the storage seam needed to move from a local vector index to Postgres/pgvector later.

## Recommendation

**Use Python 3.12 (`3.12.4`, already installed) for the slice, with `venv` + `pip` and pinned requirements.** The same core stack also installs and runs on 3.15.0, so the code is not locked to 3.12, but 3.15 currently blocks several optional packages the project may want later (`onnxruntime`-based embedders, `faiss-cpu` on Windows, and `litellm`).

The slice stack:

| Need | Choice | Why |
|---|---|---|
| Runtime | **Python 3.12.4** | All candidates install from wheels; 3.12 is in security support until Oct 2028. |
| Package manager | **stdlib `venv` + `pip`** | `uv`, Poetry, PDM, pipx, and conda are not installed. Pip is sufficient for a hackathon slice. |
| Data input | **pandas 3.1.1** | CSV works with no extra dependency; Parquet/Excel are optional extras. |
| Embeddings | **model2vec 0.10.0** (`minishlab/potion-base-8M`, 256-d) | Local, CPU-only, no API key, ~30 MB model. `openai` embeddings adapter is the remote swap-in. |
| LLM client | **openai 3.46.0** | Python 3.10+, custom `base_url`, chat and embeddings both verified. Works with a fake local server now and any OpenAI-compatible endpoint later. |
| Local vector index | **flat NumPy store** behind a `VectorStore` protocol; **sqlite-vec 0.1.9** optional | NumPy has no native dependency and is enough for demo-scale data. `sqlite-vec` is tested on both Pythons if persistence/SQL filtering is wanted. |
| Generated-code safety | **RestrictedPython 9.0 + subprocess isolation** | RestrictedPython is not a sandbox by itself; the subprocess timeout/allowlist is the real boundary. |
| Postgres port later | **psycopg[binary] 3.3.8 + pgvector 0.5.1 + SQLAlchemy 2.1.8** | Installed, imported, and `VECTOR(256)` DDL compiled on both Pythons without a server. |
| HTTP for free APIs | **httpx 0.28.1** | Sync/async, explicit timeouts; live OpenAlex request returned 200 on both Pythons. |

## Why 3.12 rather than 3.15

Python 3.15.0 was released 2026-10-09 and is still in its bugfix phase. The core slice runs on it, but the surrounding ecosystem is not uniformly ready:

| Package / capability | Python 3.12 | Python 3.15 | Detail |
|---|---|---|---|
| pandas 3.1.1 | ✅ tested | ✅ tested | `cp312` and `cp315` Windows wheels exist; CSV round-trip passed. |
| numpy 2.5.4 | ✅ tested | ✅ tested | `cp312`/`cp315` wheels; cosine search and `.npz` persistence passed. |
| httpx 0.28.1 | ✅ tested | ✅ tested | Pure Python; live OpenAlex GET returned 200. |
| openai 3.46.0 | ✅ tested | ✅ tested | Pure Python + `jiter` wheels; chat and embeddings round-trips against a local fake server passed. |
| model2vec 0.10.0 | ✅ tested | ✅ tested | `py3-none-any`; `tokenizers`/`safetensors` have abi3 wheels; actual embedding passed. |
| RestrictedPython 9.0 | ✅ tested | ✅ tested | Pure Python; requires `>=3.11,<3.16`; compile+exec passed. |
| sqlite-vec 0.1.9 | ✅ tested | ✅ tested | Platform wheel; `vec0` KNN passed. |
| psycopg[binary] 3.3.8 | ✅ tested | ✅ tested | `psycopg-binary` has a `cp315` Windows wheel; import passed. |
| pgvector 0.5.1 | ✅ tested | ✅ tested | Pure Python; SQLAlchemy `VECTOR(256)` DDL compiled. |
| SQLAlchemy 2.1.8 | ✅ tested | ✅ tested | `cp315` wheels; DDL compiled. |
| **faiss-cpu 1.15.1** | ✅ tested | ❌ blocked | Windows wheels stop at `cp314`; `pip` on 3.15 reports no matching distribution. (Linux/macOS have abi3 wheels, but the slice is on Windows.) |
| **fastembed 0.9.1** | not tested | ❌ blocked | Requires `onnxruntime>=1.24.2`; `onnxruntime 1.31.1` has no `cp315` wheel and no sdist. |
| **chromadb 1.5.9** | not tested | ❌ blocked | Its core dependency list includes `onnxruntime>=1.14.1`; same gap. |
| **litellm 1.108.2** | not tested | ❌ blocked | Package metadata says `requires-python <3.15,>=3.10`. |
| sentence-transformers 6.2.0 | not tested | ⚠️ resolvable | `torch 2.15.0` has a `cp315` Windows wheel; a 3.15 dry-run resolves. Heavy install; not needed for the slice. |
| pyarrow 26.0.1 | not tested | ⚠️ wheel exists | `cp315` Windows wheel exists; optional for Parquet only. |
| psycopg2-binary 2.9.13 | not tested | ⚠️ wheel exists | `cp315` wheel exists, but psycopg 3 is the preferred target. |

The core seven packages are not the problem on 3.15; the optional/native add-ons are. Choosing 3.12 preserves those options and costs nothing for this slice. If the team later standardises on 3.15, the pinned core stack is already proven there; just avoid `onnxruntime`-based embedders, `faiss-cpu` on Windows, and `litellm`.

Python support dates: 3.12 first released 2023-10-02, now in security-only phase, EOL October 2028; 3.15 first released 2026-10-09, bugfix phase, EOL October 2031 ([Python devguide status table](https://devguide.python.org/versions/)).

## Minimal dependency set

Direct runtime dependencies, pinned:

```text
# requirements.txt
pandas==3.1.1
numpy==2.5.4
httpx==0.28.1
openai==3.46.0
pydantic==2.14.3
model2vec==0.10.0
RestrictedPython==9.0
```

Optional, add only when the slice needs them:

```text
# requirements-optional.txt
python-dotenv==1.2.4   # .env loading for OPENAI_API_KEY / OPENAI_BASE_URL
tenacity==9.2.1         # retry/backoff for free APIs
sqlite-vec==0.1.9       # persistent local vector index, if NumPy store is not enough
```

Development:

```text
# requirements-dev.txt
pytest==9.2.0
```

Postgres/pgvector port, deliberately **not** installed for the slice:

```text
# requirements-postgres.txt (future)
psycopg[binary]==3.3.8
pgvector==0.5.1
SQLAlchemy==2.1.8
```

Optional input format:

```text
# requirements-parquet.txt (only if Parquet is required)
pyarrow==26.0.1
```

Notes:

- `pandas` pulls `numpy`, `python-dateutil`, and `tzdata`; `openai` pulls `httpx2`, `jiter`, `pydantic`, `anyio`, and `typing-extensions`; `model2vec` pulls `tokenizers`, `safetensors`, `huggingface-hub`, `joblib`, and `jinja2`. All of those installed without a compiler on both Pythons.
- The one minor 3.15 friction was PyYAML: no `cp315` wheel, so pip built a small pure-Python wheel from the sdist. It succeeded in seconds; no Visual Studio toolchain was needed.
- Do **not** add `fastembed`, `chromadb`, or `litellm` while targeting 3.15.
- `pydantic` is already a transitive dependency of `openai`; pinning it directly is cheap because the domain models and structured LLM output use it.

## Component findings

### pandas input

pandas 3.1.1 requires Python `>=3.11` and NumPy `>=2.0.2` ([pandas installation docs](https://pandas.pydata.org/docs/getting_started/install.html)). Both `cp312` and `cp315` Windows wheels exist ([PyPI metadata](https://pypi.org/pypi/pandas/json)). A CSV write/read round-trip passed on 3.12.4 and 3.15.0 with pandas 3.1.1 and numpy 2.5.4.

For the slice, accept CSV first. Add `openpyxl` only if Excel input is required, or `pyarrow` only if Parquet is required; neither is needed to read a normal CSV.

### Embedding client

Use a small protocol, not a concrete library at call sites:

```python
class EmbeddingClient(Protocol):
    model_id: str
    dimension: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

Two adapters:

1. **Local default — model2vec `StaticModel`.** `minishlab/potion-base-8M` produced 256-dimensional `float32` embeddings in 7.6 s on first load on 3.12 (including download) and 1.7 s warm on 3.15. The causal pair (“causal inference from observational data” vs “statistical methods for causal effect estimation”) scored 0.658 cosine, while shipping vs cooking scored 0.057, so the similarity signal is usable for field matching. The base package is lightweight and uses NumPy as its only major dependency; the 8M model is ~30 MB ([model2vec README](https://github.com/MinishLab/model2vec)).
2. **Remote — OpenAI embeddings through the same SDK.** The SDK’s `client.embeddings.create(...)` path was exercised against a local fake OpenAI-compatible server and parsed correctly. No live OpenAI call was made because no key is available. The real `text-embedding-3-small` model is 1536-d, so store that dimension in the record; the SDK accepts a custom `base_url`, so the same adapter works with OpenAI, Azure OpenAI, Ollama, or LM Studio.

Because 256-d (model2vec), 384-d (MiniLM), and 1536-d (OpenAI) are not interchangeable, every stored record must carry `embedding_model` and `dimension`. Do not mix models in one index or column.

For higher retrieval quality later, `sentence-transformers` with `all-MiniLM-L6-v2` (384-d) is the conventional upgrade. Its 3.15 dry-run resolves with `torch 2.15.0` (`cp315` Windows wheel exists), but the install is much heavier; it was not installed for this research. On 3.12 it is a standard install.

Avoid `fastembed` for this environment: it needs `onnxruntime`, which has no Python 3.15 distribution.

### LLM client

Use the official `openai` SDK, not a provider wrapper:

- `openai 3.46.0` requires Python 3.10+ and uses HTTPX2 under the hood ([openai-python README](https://github.com/openai/openai-python)).
- A local `http.server` fake served both `/v1/chat/completions` and `/v1/embeddings`; `OpenAI(api_key="test-key", base_url="http://127.0.0.1:<port>/v1")` completed both calls on 3.12.4 and 3.15.0.
- With no key today, put a `FakeLLMClient` behind an `LLMClient` protocol and return canned JSON for demo/tests. When a key or local server exists, flip `OPENAI_API_KEY` / `OPENAI_BASE_URL`; no application code changes.
- `litellm` is not viable on 3.15 (`requires-python <3.15`), another reason not to depend on it.

### Vector store

Put storage behind a narrow interface so the Postgres adapter is a drop-in:

```python
class VectorStore(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> None: ...
    def query(
        self,
        embedding: Sequence[float],
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]: ...
    def delete(self, ids: Sequence[str]) -> None: ...
```

`VectorRecord` should include `id`, `embedding`, `text`, and metadata (`embedding_model`, `dimension`, `source`, `created_at`).

**Default local implementation: flat NumPy store.** Normalize vectors on write, keep a `float32[N, d]` matrix and a parallel metadata list, compute cosine similarity with a single matrix-vector product, persist to `.npz` plus JSONL. A 1,000 × 64 test returned the query vector itself as the top hit and survived a save/load round-trip. This is enough for demo-scale data and has zero native dependencies.

**Optional: `sqlite-vec`.** A `vec0` virtual table with a `float[4]` column returned the expected two nearest neighbours on both Pythons (`sqlite_vec.load(db)`, `serialize_float32`, `MATCH ... ORDER BY distance LIMIT k`). Use it if the demo needs a single-file store with SQL-style metadata filters. It is 0.1.9 (pre-1.0), so pin it. Python’s Windows `sqlite3` supports `enable_load_extension`; the test passed on both versions.

**Rejected for the slice:**

- **Chroma** — full install is blocked on 3.15 by `onnxruntime`; also a large dependency tree for a local demo.
- **FAISS** — works on 3.12 and is fast, but has no Windows `cp315` wheel and adds a native library that the NumPy store does not need.
- **LanceDB** — its wheel is abi3 and may work on 3.15, but it brings Arrow/Lance and a larger stack than the slice needs.
- **DuckDB** — not needed; the slice has no SQL analytics requirement.

### HTTP for free APIs

`httpx 0.28.1` is pure Python and worked on both versions. A live request to `https://api.openalex.org/works?search=causal%20inference&per-page=1` returned HTTP 200 and a result count on both Pythons. Use explicit timeouts and a descriptive `User-Agent`; add `tenacity` only if retries are needed. The OpenAI SDK’s own HTTPX2 dependency does not conflict with `httpx`; both installed cleanly and `pip check` reported no broken requirements.

### Safe execution of generated Python

The devplan requires generated Python functions, and the gap audit flags the missing execution contract. For a demo, “safe enough” means two layers:

1. **Syntax/global restriction.** Compile with `RestrictedPython.compile_restricted`, execute with `safe_globals`, provide only the whitelisted names the function needs (`pd`, `np`, approved model helpers), and deliberately do **not** provide an `__import__`. The RestrictedPython README is explicit: it is “not a sandbox system or a secured environment” and ships no `__import__`; if you add one, the security boundary moves into your own import policy ([RestrictedPython README](https://github.com/zopefoundation/RestrictedPython)).
2. **Process isolation.** Run the compiled function in a subprocess started with `sys.executable -I`, a temporary working directory, a scrubbed environment, and a hard `timeout`; reject filesystem/network/process modules in an AST allowlist. Optionally cap memory with a Windows Job Object. The application process should never `exec` model-generated code.

This is adequate for a local hackathon demo, not for exposing generated code to untrusted users. Stronger isolation (separate container/VM) can come later; Docker is not available now.

## Postgres/pgvector portability

The port target is viable on both Python versions, even though there is no Postgres server or Docker here:

- `psycopg[binary] 3.3.8` installs from a `cp315` Windows wheel; `pgvector 0.5.1` and `SQLAlchemy 2.1.8` install from wheels on both.
- Importing `psycopg`, `pgvector.psycopg.register_vector`, `pgvector.sqlalchemy.Vector`, and compiling `CreateTable(... Vector(256) ...)` against the PostgreSQL dialect produced `embedding VECTOR(256)` on both Pythons.
- pgvector-python explicitly supports Psycopg 3, Psycopg 2, SQLAlchemy, SQLModel, asyncpg, and others; for SQLAlchemy it exposes `VECTOR`, `HALFVEC`, `BIT`, `SPARSEVEC`, and distance methods such as `cosine_distance` ([pgvector-python README](https://github.com/pgvector/pgvector-python)).
- Server-side, enable the extension once with `CREATE EXTENSION vector;`. For cosine similarity use the `<=>` operator and `vector_cosine_ops`; an HNSW index is `CREATE INDEX ON chunk USING hnsw (embedding vector_cosine_ops);` ([pgvector README](https://github.com/pgvector/pgvector)).
- pgvector’s `vector` type stores up to 16,000 dimensions, but HNSW/IVFFlat indexes support only up to **2,000** dimensions; `halfvec` raises the index limit to 4,000. Both 256-d (model2vec) and 1536-d (OpenAI small) fit.
- Windows installation of the pgvector server extension needs Visual Studio C++ build tools and `nmake`, or a hosted Postgres with pgvector preinstalled. Since there is no Docker/Postgres locally, defer this; the Python client/DDL path is already verified.

Porting plan: implement `PgVectorStore` against the same `VectorStore` protocol, keep `embedding_model`/`dimension` in a metadata table, and use a migration when the embedding model changes dimension. This directly addresses the audit’s S1/S2/S6 gaps without forcing Postgres into the first demo.

## Setup commands

```powershell
# From the repo root. Note: plain `python` is 3.15.0 on this machine.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Why these choices:

- `python --version` returns **3.15.0**; `py -3.12` returns **3.12.4**. Always create the venv with `py -3.12` and invoke `.venv\Scripts\python.exe` explicitly.
- The 3.12 system pip is 24.0; the venv’s pip upgrades to 26.3. Upgrade it before installing.
- PowerShell execution policy blocked `npm.ps1` on this machine, so `.venv\Scripts\Activate.ps1` may also be blocked. Calling `.venv\Scripts\python.exe` directly avoids the issue; alternatively run `Set-ExecutionPolicy -Scope Process Bypass` for the session.
- Pin exact versions in `requirements.txt`, then run `.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt` to capture transitives. `pip-tools` is an option if constraint solving becomes painful, but it is not required for this stack.
- If a `pyproject.toml` is added, set `requires-python = ">=3.12,<3.16"` and keep 3.12-compatible syntax so the same code runs on 3.15.

## Verification log

What was actually run on 2026-11-07 (Windows `win_amd64`):

1. Created fresh venvs with `py -3.12 -m venv` and `py -3.15 -m venv`; both reported pip 26.3.
2. Installed `pandas==3.1.1`, `numpy==2.5.4`, `httpx==0.28.1`, `openai==3.46.0`, `pydantic==2.14.3`, `model2vec==0.10.0`, `RestrictedPython==9.0`, and `pytest==9.2.0` on both. `pip check` on both: “No broken requirements found.”
3. `smoke_stack.py` on both: pandas CSV round-trip; NumPy cosine search + `.npz` persistence; RestrictedPython compile/exec; model2vec `potion-base-8M` embedding of three texts (256-d, `float32`).
4. Installed `sqlite-vec==0.1.9`, `psycopg[binary]==3.3.8`, `pgvector==0.5.1`, and `SQLAlchemy==2.1.8` on both. `smoke_store_pg.py` on both: `vec_version()` = `v0.1.9`; `vec0` KNN returned the expected ids; `VECTOR(256)` DDL compiled.
5. `smoke_llm.py` on both: local fake OpenAI-compatible server served chat completion and embedding responses; `openai` client round-trips passed; live OpenAlex GET returned 200 with count 745,875.
6. Failure probes on 3.15: `pip install --only-binary=:all: onnxruntime==1.31.1` → no matching distribution; `faiss-cpu==1.15.1` → no matching distribution; `litellm==1.108.2` → no matching distribution (all versions filtered by `requires-python`). `faiss-cpu==1.15.1` installed and `IndexFlatIP` returned the query’s own vector as top-1 on 3.12.

Not tested: a live OpenAI/Anthropic call (no key); a live Ollama/LM Studio server (not installed); a live Postgres connection (no server/Docker); a full `sentence-transformers` install; full `fastembed`/`chromadb` installs; Excel/Parquet input.

## Risks and follow-ups

- **3.12.4 is an old patch.** 3.12 is security-only, so new Windows installers are not expected. Fine for a local demo; for a deployed service, use a maintained base image or 3.13/3.15.
- **Model quality.** Model2Vec static embeddings are fast but weaker than transformer embeddings. Measure retrieval quality on the actual demo dataset; swapping to `all-MiniLM-L6-v2` or OpenAI embeddings is an adapter change, but the vector dimension changes, so plan the index rebuild.
- **Offline demo risk.** `model2vec` downloads its model from Hugging Face on first use. Pre-download it and load from a local path (or cache it in the image) before judging; also pin the model revision.
- **Brand-new pins.** pandas 3.1.1 (2026-11-04) and openai 3.46.0 (2026-11-05) are days old. They passed the smoke tests; if the team prefers more soak time, pin one minor version back and re-run the same tests.
- **No live provider test.** The fake-server test proves the SDK/HTTP path, not OpenAI or an alternative provider’s behaviour. Re-test with a key before relying on structured outputs/tool calling.
- **Sandbox limits.** RestrictedPython is syntax-level only. A subprocess timeout does not cap memory or block all OS-level abuse; add a Job Object/worker isolation before accepting untrusted code.
- **`sqlite-vec` is pre-1.0.** If chosen, pin `0.1.9` and keep the store behind the interface so it can be replaced.

## Sources

Retrieved 2026-11-07 unless noted.

- Python version status and support dates: <https://devguide.python.org/versions/>
- pandas installation, Python/NumPy requirements, venv recommendation: <https://pandas.pydata.org/docs/getting_started/install.html>
- PyPI release metadata and wheel tags (pandas, numpy, httpx, openai, pydantic, model2vec, RestrictedPython, sqlite-vec, psycopg, pgvector, SQLAlchemy, faiss-cpu, onnxruntime, fastembed, chromadb, litellm, sentence-transformers, pyarrow): <https://pypi.org/pypi/pandas/json>, <https://pypi.org/pypi/numpy/json>, <https://pypi.org/pypi/openai/json>, <https://pypi.org/pypi/model2vec/json>, <https://pypi.org/pypi/RestrictedPython/json>, <https://pypi.org/pypi/sqlite-vec/json>, <https://pypi.org/pypi/psycopg/json>, <https://pypi.org/pypi/pgvector/json>, <https://pypi.org/pypi/SQLAlchemy/json>, <https://pypi.org/pypi/faiss-cpu/json>, <https://pypi.org/pypi/onnxruntime/json>, <https://pypi.org/pypi/litellm/json>
- OpenAI Python SDK requirements, HTTPX2, custom `base_url`, retries: <https://github.com/openai/openai-python>
- Model2Vec static embeddings, lightweight base package, model sizes: <https://github.com/MinishLab/model2vec>
- RestrictedPython security caveat and no `__import__`: <https://github.com/zopefoundation/RestrictedPython>
- pgvector SQL operators, HNSW/IVFFlat, dimension limits: <https://github.com/pgvector/pgvector>
- pgvector-python Psycopg 3/SQLAlchemy support: <https://github.com/pgvector/pgvector-python>
- sqlite-vec Python loading and KNN API: <https://alexgarcia.xyz/sqlite-vec/python.html>
- Python `venv` docs: <https://docs.python.org/3/library/venv.html>
