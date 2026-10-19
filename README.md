# Production-Grade Hybrid RAG Engine with Automated Evals

This repository is documented in both English and Portuguese. Use the navigation links below to jump to your preferred language.

Esta documentação está disponível em inglês e português. Utilize os links abaixo para navegar para o idioma de sua preferência.

[🇺🇸 English](#english) | [🇧🇷 Português](#português)

---

<a id="english"></a>
## English

### Overview
A production-ready Hybrid Retrieval-Augmented Generation (RAG) engine designed to overcome the failure modes of naive vector search. This project implements a deterministic multi-stage retrieval pipeline combining lexical search (BM25), dense semantic representations (Qdrant), Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking, coupled with a continuous automated evaluation pipeline in CI/CD.

Developed as an applied engineering and academic research initiative, this project benchmarks trade-offs across chunking strategies, retrieval latency, and answer faithfulness under strict software engineering principles.

### Architecture Highlights
- **Document Chunking:** Structural/AST-based hierarchical chunking preserving Markdown headers and context.
- **Dual Retrieval Pipeline:**
  - Sparse/Lexical: BM25 indexation for deterministic keyword, acronym, and identifier matching.
  - Dense/Semantic: Vector embeddings via `bge-small-en-v1.5` optimized through ONNX Runtime.
- **Fusion Layer:** Reciprocal Rank Fusion (RRF) to combine disparate score distributions without manual weighting.
- **Reranking:** Cross-Encoder (`bge-reranker-base`) for top-k contextual scoring and false-positive elimination.
- **Continuous Evaluation (CI/CD):** Automated regressions on a golden dataset calculating Faithfulness and Context Precision/Recall via Ragas/DeepEval on GitHub Actions.

### Tech Stack
- **Language & Runtime:** Python 3.11+, uv package manager
- **API Framework:** FastAPI, Pydantic v2, Pydantic-Settings
- **Vector Database:** Qdrant (via Docker with gRPC communication)
- **ML / Inference Runtime:** FastEmbed (ONNX Runtime), sentence-transformers
- **Evaluation & Testing:** Ragas / DeepEval, Pytest, Ruff, Mypy
- **Infrastructure:** Docker, Docker Compose, GitHub Actions

### Prerequisites & Requirements
- **Operating System:** Linux (Fedora/Debian/Ubuntu recommended) or macOS
- **Runtimes:** Python 3.11+ and Docker Engine with Docker Compose
- **Tooling:** `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Hardware:** Minimum 4 GB RAM recommended for local ONNX inference and Qdrant storage.

### Quick Start

#### 1. Clone the repository and configure environment

```bash
git clone https://github.com/fernando-rodrigues-pinto/rag-engine-project.git
cd hybrid-rag-engine
cp .env.example .env
```

#### 2. Start the local Vector Database
```bash
docker compose -f docker/docker-compose.yml up -d
```
Verify that Qdrant is running at `http://localhost:6334/dashboard`.

#### 3. Install dependencies
```bash
uv sync --all-extras
```

#### 4. Run database migrations / collection setup
```bash
uv run python -m src.retrieval.vector_store
```

#### 5. Launch the API
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

### Running Tests and Quality Checks
```bash
# Type check and linting
uv run ruff check .
uv run mypy src

# Unit and integration tests
uv run pytest tests/unit
uv run pytest tests/integration
```

### Project Roadmap & Status
- [ ] **Phase 1: Structured Ingestion and Vectorization** [In Progress]
  - [x] Directory scaffolding and environment configuration
  - [ ] AST Markdown parser and hierarchical chunker
  - [ ] FastEmbed ONNX dense runner integration
  - [ ] Qdrant collection orchestration via gRPC
- [ ] **Phase 2: Hybrid Retrieval and Cross-Encoder Reranking** [To be completed]
  - [ ] In-memory/persisted BM25 sparse indexer
  - [ ] Reciprocal Rank Fusion (RRF) algorithm
  - [ ] BGE-Reranker cross-encoder integration
  - [ ] FastAPI search endpoints (`/search/hybrid`)
- [ ] **Phase 3: Automated Evals and CI/CD Benchmarking** [To be completed]
  - [ ] Golden dataset curation and synthetic generation
  - [ ] Ragas/DeepEval metric runner (Faithfulness, Context Precision)
  - [ ] GitHub Actions workflow for evaluation regressions on PRs

### Benchmark Results
*(To be filled upon completion of Phase 3 evaluation pipeline)*

| Metric | Naive Dense RAG | BM25 Only | Hybrid (RRF) | Hybrid + Reranker |
| :--- | :--- | :--- | :--- | :--- |
| Context Precision | TBD | TBD | TBD | TBD |
| Context Recall | TBD | TBD | TBD | TBD |
| Faithfulness | TBD | TBD | TBD | TBD |
| Latency p95 (ms) | TBD | TBD | TBD | TBD |

---

<a id="português"></a>
## Português

### Visão Geral
Motor de Geração Aumentada por Recuperação (RAG) Híbrido, voltado para cenários de produção e desenhado para superar as limitações do RAG vetorial ingênuo (*naive RAG*). Este projeto implementa um pipeline determinístico de múltiplos estágios que combina busca esparsa (BM25), representações semânticas densas (Qdrant), Reciprocal Rank Fusion (RRF) e reordenação com Cross-Encoder, integrado a uma esteira de avaliação automatizada contínua em CI/CD.

Desenvolvido com caráter acadêmico e de engenharia aplicada, este projeto estabelece benchmarks formais entre estratégias de chunking, latência de recuperação e fidelidade de respostas sob práticas rigorosas de engenharia de software.

### Destaques da Arquitetura
- **Chunking Estruturado:** Divisão hierárquica baseada na árvore sintática (AST) de arquivos Markdown, preservando cabeçalhos e relações de contexto.
- **Pipeline Duplo de Recuperação:**
  - Esparsa/Léxica: Indexação BM25 para correspondência determinística de palavras-chave, siglas e termos técnicos exatos.
  - Densa/Semântica: Embeddings vetoriais com `bge-small-en-v1.5` executados via ONNX Runtime.
- **Camada de Fusão:** Reciprocal Rank Fusion (RRF) para combinar distribuições heterogêneas de ranqueamento sem necessidade de pesos manuais.
- **Reranking:** Cross-Encoder (`bge-reranker-base`) para pontuação contextual profunda dos top-k candidatos e eliminação de falsos positivos.
- **Avaliação Contínua (CI/CD):** Regressões automatizadas sobre um dataset de referência (*Golden Dataset*), aferindo métricas de *Faithfulness* e *Context Precision/Recall* via Ragas/DeepEval no GitHub Actions.

### Tecnologias Utilizadas
- **Linguagem e Runtime:** Python 3.11+, gerenciador de pacotes uv
- **Framework de API:** FastAPI, Pydantic v2, Pydantic-Settings
- **Banco de Dados Vetorial:** Qdrant (via Docker com comunicação gRPC)
- **Runtime de Inferência e ML:** FastEmbed (ONNX Runtime), sentence-transformers
- **Testes e Avaliação:** Ragas / DeepEval, Pytest, Ruff, Mypy
- **Infraestrutura:** Docker, Docker Compose, GitHub Actions

### Pré-requisitos e Requisitos de Sistema
- **Sistema Operacional:** Linux (Fedora/Debian/Ubuntu recomendados) ou macOS
- **Runtimes:** Python 3.11+ e Docker Engine com Docker Compose
- **Ferramentas:** `uv` instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Hardware:** Mínimo de 4 GB de RAM recomendado para execução local do Qdrant e inferência ONNX.

### Inicialização Rápida

#### 1. Clonar o repositório e configurar variáveis
```bash
git clone https://github.com/fernando-rodrigues-pinto/rag-engine-project.git
cd hybrid-rag-engine
cp .env.example .env
```

#### 2. Iniciar o Banco de Dados Vetorial local
```bash
docker compose -f docker/docker-compose.yml up -d
```
Verifique o status do Qdrant acessando `http://localhost:6334/dashboard`.

#### 3. Instalar dependências
```bash
uv sync --all-extras
```

#### 4. Inicializar coleções no banco
```bash
uv run python -m src.retrieval.vector_store
```

#### 5. Executar a API
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
A documentação interativa estará disponível em `http://localhost:8000/docs`.

### Executando Testes e Validações de Qualidade
```bash
# Verificação de tipos e padronização de código
uv run ruff check .
uv run mypy src

# Testes unitários e de integração
uv run pytest tests/unit
uv run pytest tests/integration
```

### Roadmap e Status de Implementação
- [ ] **Fase 1: Ingestão Estruturada e Vetorização** [Em Andamento]
  - [x] Estruturação do repositório e variáveis de ambiente
  - [ ] Parser de AST para Markdown e chunker hierárquico
  - [ ] Integração do gerador denso via FastEmbed e ONNX
  - [ ] Orquestração de coleções no Qdrant via gRPC
- [ ] **Fase 2: Recuperação Híbrida e Reranking com Cross-Encoder** [A preencher]
  - [ ] Indexador esparso BM25 persistido/em memória
  - [ ] Algoritmo determinístico de Reciprocal Rank Fusion (RRF)
  - [ ] Integração do Cross-Encoder com BGE-Reranker
  - [ ] Endpoints de busca da API FastAPI (`/search/hybrid`)
- [ ] **Fase 3: Avaliação Automatizada (Evals) e Benchmarking em CI/CD** [A preencher]
  - [ ] Curadoria de Golden Dataset e geração sintética
  - [ ] Runner de métricas com Ragas/DeepEval (Faithfulness, Context Precision)
  - [ ] Workflow no GitHub Actions para regressão de métricas em Pull Requests

### Tabela de Benchmarks
*(A ser preenchida após a finalização da Fase 3 do projeto)*

| Métrica | Naive Dense RAG | Apenas BM25 | RAG Híbrido (RRF) | Híbrido + Reranker |
| :--- | :--- | :--- | :--- | :--- |
| Precisão de Contexto | A preencher | A preencher | A preencher | A preencher |
| Cobertura de Contexto | A preencher | A preencher | A preencher | A preencher |
| Fidelidade (Faithfulness) | A preencher | A preencher | A preencher | A preencher |
| Latência p95 (ms) | A preencher | A preencher | A preencher | A preencher |