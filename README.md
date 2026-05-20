# DocuMind

> A production RAG system for PDF question answering. Upload documents, ask questions, get answers with cited sources — backed by FastAPI, LangChain, ChromaDB, and OpenAI, deployed on AWS with HTTPS.

**🔗 Live demo:** [https://documind.mroshan454.dev/docs](https://documind.mroshan454.dev/docs)

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1c3c3c)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

DocuMind is an end-to-end document question-answering backend. Upload a PDF and it:

1. **Ingests** the document — extracts text, splits into semantic chunks, embeds each chunk into a vector store
2. **Retrieves** relevant chunks when you ask a question (semantic similarity search)
3. **Generates** a grounded answer using GPT, returning the answer along with the source chunks it cited

Idempotent ingestion (re-uploading the same PDF doesn't duplicate chunks). Async query handling. Graceful error responses for edge cases like image-only PDFs.

---

## Architecture

### Ingestion pipeline

```
User PDF upload
       │
       ▼
LangChain PyPDFLoader  →  extracts text per page
       │
       ▼
RecursiveCharacterTextSplitter  →  ~800-char chunks, 150-char overlap
       │
       ▼
OpenAI text-embedding-3-small  →  1536-d vector per chunk
       │
       ▼
ChromaDB  →  stores vector + text + metadata, keyed by SHA-256 chunk ID
```

### Query pipeline

```
User question
       │
       ▼
Embed question (same model)
       │
       ▼
ChromaDB similarity search  →  top-k relevant chunks
       │
       ▼
GPT-4o-mini  →  generates answer grounded in retrieved chunks
       │
       ▼
Response  →  { answer, sources[], metadata }
```

### Deployment topology

```
Browser
   │ HTTPS (Let's Encrypt cert, auto-renewing)
   ▼
Cloudflare DNS  →  AWS Elastic IP  →  AWS EC2 (Ubuntu 26.04)
                                          │
                                          ▼
                                   Nginx reverse proxy (:443)
                                          │
                                          ▼
                                   Docker container → FastAPI :8000
                                          │
                                          ▼
                                   LangChain + ChromaDB + OpenAI API
```

---

## Tech stack

| Layer | Tools |
|---|---|
| **API** | FastAPI, Pydantic, async/await, Uvicorn |
| **RAG** | LangChain, ChromaDB (persistent vector store), OpenAI API |
| **LLM** | `gpt-4o-mini` (responses), `text-embedding-3-small` (embeddings) |
| **Deployment** | Docker, AWS EC2, Nginx (reverse proxy), Let's Encrypt (SSL), Cloudflare DNS |
| **Observability** | Structured JSON logging (structlog), custom exception middleware |
| **Testing** | pytest, pytest-asyncio, FastAPI TestClient, mocked LLM/vectorstore |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API metadata + entry points |
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Upload a PDF; returns `{ filename, file_hash, pages, chunks }` |
| `POST` | `/query` | Body: `{ question, k }`; returns answer with cited source chunks |
| `GET` | `/docs` | Interactive Swagger UI |

Explore live at [documind.mroshan454.dev/docs](https://documind.mroshan454.dev/docs).

---

## Running locally

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- An OpenAI API key

### Quick start (Docker — recommended)

```bash
# 1. Clone
git clone https://github.com/mroshan454/documind.git
cd documind

# 2. Create .env with your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Build + run
docker compose up -d --build

# 4. Open Swagger UI
open http://localhost:8000/docs
```

### Without Docker

```bash
# 1. Clone + set up venv
git clone https://github.com/mroshan454/documind.git
cd documind
python -m venv venv
source venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Export OPENAI_API_KEY (or use .env + python-dotenv)
export OPENAI_API_KEY=sk-...

# 4. Run
uvicorn app.main:app --reload
```

### Running tests

```bash
pytest                    # full suite
pytest -v --tb=short      # verbose
pytest tests/integration  # integration tests only
```

---

## Production engineering highlights

A few decisions worth calling out — these were the hard parts.

### Reduced Docker image from 8.9 GB → under 1 GB

The first build shipped 8.9 GB of dependencies. The culprit: PyTorch was pulling in 7 GB of CUDA libraries to support HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` embedding model — none of which runs on a CPU-only EC2 instance.

**Tradeoff analysis:**

| Option | Image size | Inference cost | Latency |
|---|---|---|---|
| Local HF embeddings (CPU) | 8.9 GB | $0 | slow on CPU |
| Local HF embeddings (GPU) | 8.9 GB | + GPU instance cost | fast |
| OpenAI embeddings API | < 1 GB | ~$0.02 per 1M tokens | low (network) |

Chose **OpenAI embeddings** — for portfolio-scale traffic the per-document cost is ~$0.001, and shrinking the image meaningfully improves cold-start time and reduces EC2 disk usage. If this scaled to enterprise volumes (millions of documents), the math would flip and local GPU embeddings would win.

### Idempotent ingestion with SHA-256 chunk IDs

Re-uploading the same PDF used to create duplicate chunks. Fixed by hashing the file contents and using deterministic IDs (`<sha256>-chunk-<i>`) for ChromaDB upserts. Same file → same IDs → upsert overwrites cleanly.

### Custom exception hierarchy

A user uploaded an image-only PDF (scanned/rasterized text). PyPDFLoader returned zero characters. The system crashed with a 500 stack trace. Replaced with:

```python
class EmptyDocumentError(IngestionException):
    status_code = 400
    error_code = "no_extractable_text"
    message = "No extractable text found. The PDF may be a scanned or image-only document."
```

Caught in the endpoint, returns a clean HTTP 400 with `{ error_code, message }`. Useful both for UX and for debugging.

### Async LangChain integration

Query path uses `ainvoke()` so multiple concurrent queries don't block on each other's OpenAI round-trips. Tested with `pytest-asyncio`.

### Structured logging

Every request emits JSON logs with `event`, `level`, `timestamp`, plus context (`filename`, `pages`, `chunks`, `file_hash`). Greppable, ready to ship to a log aggregator.

---

## Project structure

```
documind/
├── app/
│   ├── main.py                # FastAPI app + endpoints + exception handlers
│   ├── core/
│   │   ├── exceptions.py      # Custom exception hierarchy
│   │   └── logging.py         # structlog configuration
│   └── services/
│       ├── ingestion_service.py   # PDF load → split → embed → store
│       └── query_service.py       # Retrieve → prompt → LLM → answer
├── tests/                     # pytest unit + integration tests
├── uploads/                   # Mounted volume for incoming PDFs
├── chroma_db/                 # Persistent vector store (mounted volume)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] **OCR support** — Tesseract integration for scanned / image-only PDFs (currently returns HTTP 400)
- [ ] **Gradio frontend** — clean UI on top of the API for non-technical users
- [ ] **Agentic retrieval** — multi-step retrieval with query refinement
- [ ] **Rate limiting** — per-IP request limits at the Nginx layer
- [ ] **Streaming responses** — Server-Sent Events for token-by-token output
- [ ] **Conversational memory** — multi-turn QA over the same document

---

## License

MIT. See `LICENSE`.

---

## About

Built by [Roshan Mohammed](https://linkedin.com/in/roshan-mohammed-068008279), MSc AI graduate based in Melbourne. Open to Junior AI Engineer / ML Engineer / Backend Engineer roles — full Australian work rights via MATES Visa (Subclass 403), no sponsorship required.

- 🌐 Live demo: [documind.mroshan454.dev/docs](https://documind.mroshan454.dev/docs)
- 🐙 GitHub: [github.com/mroshan454](https://github.com/mroshan454)
- 🤗 Hugging Face: [huggingface.co/roshan454](https://huggingface.co/roshan454)
- 💼 LinkedIn: [in/roshan-mohammed-068008279](https://linkedin.com/in/roshan-mohammed-068008279)
