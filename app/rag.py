import hashlib
import logging
from collections import OrderedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

NOT_FOUND = "Data Not Available"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a document using only the context given below.\n"
            "- The context can be earlier answered questionnaire entries, use them even when "
            "they're worded differently from the question.\n"
            "- Direct conclusions are fine (e.g. data hosted in GCP means the cloud provider is GCP).\n"
            "- If only part of the question is covered, answer that part and say which part isn't.\n"
            f"- Only if nothing in the context is relevant, reply with exactly: {NOT_FOUND}\n"
            "Keep it short and factual, don't make things up.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class IndexCache:
    """Tiny LRU so uploading the same doc twice doesn't re-embed it (we have a $5 budget)."""

    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._store: OrderedDict[str, FAISS] = OrderedDict()

    def get(self, key):
        idx = self._store.get(key)
        if idx is not None:
            self._store.move_to_end(key)
        return idx

    def put(self, key, index):
        self._store[key] = index
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


def doc_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_index(docs: list[Document], embeddings: Embeddings, chunk_size: int, chunk_overlap: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    log.info("indexing %d chunks from %d docs", len(chunks), len(docs))
    return FAISS.from_documents(chunks, embeddings)


def _source_of(d: Document):
    md = d.metadata or {}
    return md.get("page", md.get("record"))


def answer_questions(index: FAISS, questions: list[str], llm: BaseChatModel, top_k: int = 5) -> list[dict]:
    retriever = index.as_retriever(search_kwargs={"k": top_k})
    chain = PROMPT | llm | StrOutputParser()

    inputs, sources = [], []
    for q in questions:
        hits = retriever.invoke(q)
        ctx = "\n\n---\n\n".join(h.page_content for h in hits)
        inputs.append({"context": ctx, "question": q})
        # dedupe but keep order
        srcs = []
        for h in hits:
            s = _source_of(h)
            if s is not None and s not in srcs:
                srcs.append(s)
        sources.append(srcs)

    # batch runs them in parallel, 5 at a time is gentle on rate limits
    answers = chain.batch(inputs, config={"max_concurrency": 5})

    out = []
    for q, a, srcs in zip(questions, answers, sources):
        a = a.strip()
        out.append({"question": q, "answer": a, "sources": [] if a == NOT_FOUND else srcs})
    return out
