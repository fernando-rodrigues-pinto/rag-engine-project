# AGENTS.md — Directives & Architectural Governance

## System Overview & Architecture Scope
This repository houses a production-grade, multi-stage Hybrid Retrieval-Augmented Generation (RAG) engine.
Agents interacting with this codebase must adhere strictly to modularity, deterministic pipelines, strict typing, and automated evaluation benchmarking.

### Core Stack & Tooling
- **Runtime & Management:** Python 3.11+, managed strictly via `uv`.
- **API & Schemas:** FastAPI, Pydantic v2 (`BaseModel` with explicit validation), `pydantic-settings`.
- **Vector Engine:** Qdrant (accessed via gRPC for high-throughput operations).
- **ML / Inference Runways:** `fastembed` (ONNX Runtime engine) for dense embeddings; `rank-bm25` for lexical search; `bge-reranker-base` (Cross-Encoder) for stage-2 reranking.
- **Quality & Evals:** `ruff`, `mypy` (strict mode), `pytest`, `ragas` / `deepeval`.

---

## Architectural Rules & Module Boundaries

Code modifications must strictly respect layer boundaries. Circular dependencies or cross-layer leaks are prohibited:

1. `src/core/`:
   - Contains domain models, abstract protocols/interfaces, and custom exceptions.
   - Zero dependencies on external infrastructure (no Qdrant clients, no FastAPI imports).
2. `src/ingestion/`:
   - Responsible for raw document AST parsing (Markdown/PDF) and hierarchical chunking.
   - Must preserve semantic header breadcrumbs within chunk metadata.
3. `src/retrieval/`:
   - Hosts dense retrieval (`dense.py`), sparse BM25 indexing (`sparse.py`), Reciprocal Rank Fusion (`fusion.py`), and Cross-Encoder reranking (`reranker.py`).
   - Fusion algorithms must be deterministic and accept uniform candidate schemas.
4. `src/evals/`:
   - Standalone evaluation runners (`metrics.py`, `runner.py`) computing Faithfulness, Context Recall, and Context Precision over the golden dataset.
5. `src/api/`:
   - FastAPI application layer exposing REST endpoints. Depends on business logic solely through dependency injection (`dependencies.py`).

---

## Coding Standards & Invariants

- **Strict Type Annotations:** Every function signature, method, and return value must be typed explicitly. Avoid `typing.Any`; prefer `TypeVar`, generics, or Pydantic models.
- **Error Handling:** Avoid generic `try/except Exception: pass`. Catch explicit exceptions and wrap infrastructure failures in custom exceptions declared in `src.core.exceptions`.
- **Immutability & Schemas:** Core data structures (e.g., `Document`, `Chunk`, `SearchResult`) must be defined using Pydantic models with frozen configurations or explicit validation.
- **Path Handling:** Always use `pathlib.Path` instead of string manipulations with `os.path`.
- **Asynchronous Execution:** FastAPI route handlers and I/O-bound integrations (network, gRPC) must use `async/await` patterns correctly without blocking the event loop.

---

## Negative Constraints (What NOT to Do)

1. **NO PyTorch Bloat:** Do not introduce heavy PyTorch dependencies (`torch`, `torchvision`) for inference where ONNX Runtime (`fastembed`) or lightweight quantized execution is specified.
2. **NO Hardcoded Values:** Never hardcode paths, collection names, vector dimensions, or thresholds in implementation files. All configurations must be sourced from `src/config.py` via `pydantic-settings`.
3. **NO Ad-hoc Directory Modifications:** Do not introduce root-level directories, utility dumping grounds (`utils/`), or change existing naming conventions without explicit specification.
4. **NO Code Dumps or Stubbed Functions:** Never leave incomplete placeholders (`# TODO: implement later`) in committed modules. Implement the complete contract with unit test coverage.
5. **NO Direct Modification of Golden Datasets:** Never mutate files in `data/golden_dataset/` programmatically during testing or eval runs. Treat them as immutable baselines.

---

## Verification & Command Execution Protocols

Before confirming any implementation or changes, run and verify the following commands locally via `uv`:

```bash
# 1. Format and lint checks
uv run ruff check .
uv run ruff format --check .

# 2. Strict static type analysis
uv run mypy src

# 3. Unit and integration testing
uv run pytest tests/unit
uv run pytest tests/integration
```