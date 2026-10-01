import logging

import openai
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings
from app.loaders import LoaderError, load_document, parse_questions
from app.rag import IndexCache, answer_questions, build_index, doc_hash
from app.schemas import QAResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("zania-qa")

app = FastAPI(title="Zania Document QA", version="0.1.0")
_cache = IndexCache()


# these two are separate deps so tests can override them with fakes
def get_llm(settings: Settings = Depends(get_settings)) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)


def get_embeddings(settings: Settings = Depends(get_settings)) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


async def _read_upload(f: UploadFile, max_bytes: int) -> bytes:
    raw = await f.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{f.filename} is too large")
    if not raw:
        raise HTTPException(status_code=400, detail=f"{f.filename} is empty")
    return raw


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/qa", response_model=QAResponse)
async def qa(
    questions_file: UploadFile = File(..., description="JSON list of questions"),
    document_file: UploadFile = File(..., description="PDF or JSON document"),
    settings: Settings = Depends(get_settings),
    llm: BaseChatModel = Depends(get_llm),
    embeddings: Embeddings = Depends(get_embeddings),
):
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if not (questions_file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=415, detail="questions_file must be a .json file")

    q_raw = await _read_upload(questions_file, max_bytes)
    d_raw = await _read_upload(document_file, max_bytes)

    try:
        questions = parse_questions(q_raw)
    except LoaderError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if len(questions) > settings.max_questions:
        raise HTTPException(status_code=422, detail=f"too many questions (max {settings.max_questions})")

    key = doc_hash(d_raw)
    index = _cache.get(key)
    try:
        if index is None:
            try:
                docs = load_document(document_file.filename, d_raw)
            except LoaderError as e:
                code = 415 if "must be a .pdf or .json" in str(e) else 422
                raise HTTPException(status_code=code, detail=str(e))
            index = build_index(docs, embeddings, settings.chunk_size, settings.chunk_overlap)
            _cache.put(key, index)
        else:
            log.info("reusing cached index for %s", key[:12])

        results = answer_questions(index, questions, llm, top_k=settings.top_k)
    except openai.AuthenticationError as e:
        log.error("openai auth failed: %s", e)
        raise HTTPException(status_code=502, detail="OpenAI rejected the API key (check OPENAI_API_KEY in .env)")
    except openai.RateLimitError as e:
        # also what you get when the key is out of credit
        log.error("openai rate limit / quota: %s", e)
        raise HTTPException(status_code=429, detail=f"OpenAI rate limit or quota exceeded: {e}")
    except openai.APIError as e:
        log.exception("openai call failed")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    return {"results": results}
