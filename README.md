# Zania Document QA

A FastAPI service that answers a list of questions against a document (PDF or JSON) using retrieval + `gpt-4o-mini`.

You give it two files, a JSON list of questions and the document, and get back question/answer pairs. If the document doesn't contain the answer, it returns `Data Not Available` instead of making something up.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # put your OPENAI_API_KEY in here

# answer the 5 sample questions, no server needed
python scripts/run_sample.py tests/fixtures/sample_questions.json \
  tests/fixtures/sample_document.json -o docs/sample_output.json
```

Python 3.11+.

## How it works

```
questions (.json) ──> list[str]
document (.pdf/.json) ──> pages / records ──> chunks (1000 chars, 150 overlap)
                                          ──> embeddings (text-embedding-3-small) ──> FAISS
for each question: top-5 chunks ──> gpt-4o-mini (temperature 0) ──> answer + sources
```

Design choices:

- **FAISS in memory.** One document per request, so a hosted vector DB would be overkill. The index is cached by the document's SHA-256, so sending the same document again doesn't re-embed it (the key has a $5 limit).
- **Grounded prompt.** The model answers only from the retrieved context. It can make direct conclusions (hosted in GCP → cloud provider is GCP) and gives a partial answer when only part of a question is covered. If nothing relevant is retrieved it says `Data Not Available`. Temperature is 0 so answers are repeatable.
- **JSON documents.** Nested JSON is flattened to `path.to.key: value` lines. If the top level is a list (e.g. a spreadsheet export of Q/A rows), each record becomes its own document so a row never gets split across chunks.
- **Parallel answering.** Questions are answered with `chain.batch`, max 5 at a time.
- **Sources.** Each answer lists the PDF page numbers (from 1) or JSON record indexes (from 0) of the chunks retrieved for that question. They're retrieval provenance, not exact citations, so some may be only loosely related. They're left empty when the answer is `Data Not Available`.

## Running the API

```bash
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t zania-qa .
docker run -p 8000:8000 --env-file .env zania-qa
```

Swagger UI is at http://localhost:8000/docs. You can upload files and try it there. `GET /health` returns `{"status": "ok"}`.

### `POST /qa`

Multipart form with two files:

| field | type |
|---|---|
| `questions_file` | `.json`: `["q1", "q2"]`, `[{"question": "q1"}]` or `{"questions": [...]}` |
| `document_file` | `.pdf` or `.json` |

```bash
curl -X POST http://localhost:8000/qa \
  -F "questions_file=@tests/fixtures/sample_questions.json" \
  -F "document_file=@tests/fixtures/sample_document.json"
```

Response:

```json
{
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "answer": "We rely on Google Cloud Platform (GCP) for our data hosting and cryptographic key management.",
      "sources": [0, 4, 3, 18, 8]
    }
  ]
}
```

### Errors

| status | when |
|---|---|
| 400 | empty upload |
| 413 | file bigger than `MAX_UPLOAD_MB` (default 20) |
| 415 | document isn't .pdf/.json, or questions file isn't .json |
| 422 | malformed JSON, no questions, too many questions, PDF with no text |
| 429 | OpenAI rate limit or the key is out of credit |
| 502 | OpenAI rejected the key or the call failed |
| 500 | `OPENAI_API_KEY` not set |

## Command line (no server)

`scripts/run_sample.py` runs the same pipeline as `POST /qa` and reads the key from `.env` the same way.

```bash
python scripts/run_sample.py QUESTIONS.json DOCUMENT.(pdf|json) [-o OUT.json]
```

Without `-o` it prints the result to stdout. Input or OpenAI errors exit with a one-line message.

## Sample data and output

- `tests/fixtures/sample_questions.json`: the 5 questions from the challenge appendix.
- `tests/fixtures/sample_document.json`: the challenge's sample spreadsheet (`Sample_JSON.xlsx`, 19 Q/A records), converted with:
  ```bash
  python scripts/xlsx_to_json.py Sample_JSON.xlsx tests/fixtures/sample_document.json
  ```
- The sample SOC 2 PDF link in the challenge (getnave.com) now returns 404, so it isn't included. Any PDF works the same way.

[`docs/sample_output.json`](docs/sample_output.json) has real `gpt-4o-mini` output for the 5 questions against the sample document:

| question | result |
|---|---|
| Incident notification criteria / SLAs | `Data Not Available`: the document mentions an incident response plan but no client notification SLA |
| Personal data shared with third parties | `Data Not Available`: not covered |
| Cloud providers | GCP (hosting and key management) |
| Primary / backup data center region | US Central on GCP; says backup locations aren't covered |
| APM / EUM / DEM monitoring | `Data Not Available`: not covered |

## Config

All settings come from env vars or `.env` (see `app/config.py`):

| variable | default |
|---|---|
| `OPENAI_API_KEY` | required |
| `LLM_MODEL` | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` |
| `TOP_K` | `5` |
| `MAX_UPLOAD_MB` | `20` |
| `MAX_QUESTIONS` | `50` |

`.env` is in `.gitignore`. Don't commit the key.

## Tests

```bash
pytest
```

The tests swap the LLM and embeddings for LangChain's fake models (through FastAPI dependency overrides), so they run offline and don't use the API key. They cover the loaders, the API happy path for PDF and JSON, the main input errors (400/415/422), OpenAI auth errors (502), the index cache, and the command-line script.

## Project layout

```
app/
  main.py        routes, upload validation, error mapping, dependency wiring
  loaders.py     PDF / JSON parsing into LangChain Documents
  rag.py         chunking, FAISS index, prompt, batch answering, cache
  config.py      settings
  schemas.py     response models
docs/
  sample_output.json   real output for the 5 sample questions
scripts/
  run_sample.py        full pipeline from the command line
  xlsx_to_json.py      sample spreadsheet -> JSON document
tests/
  fixtures/            sample questions + sample document
  test_api.py  test_loaders.py  test_run_sample.py
```

## Limitations / what I'd do next

- **Citations.** `sources` is what was retrieved, not what the answer used. Next step is having the model return the records it actually cited (structured output).
- **Scanned PDFs** have no text layer; they'd need OCR.
- **Cache is per process.** With multiple workers I'd move to a persistent store (pgvector or Qdrant) keyed by document hash.
- **No auth or rate limiting** on the endpoint.
- **Retrieval quality** on long compliance docs could improve with a reranker or hybrid (BM25 + vector) search, and answers could be streamed.
- `langchain-community` is being sunset; FAISS would move to its standalone package once that's stable.
