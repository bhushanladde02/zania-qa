import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

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
        self._store: OrderedDict = OrderedDict()

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


# split on paragraphs, then lines, then sentences. pdf text has no newlines after
# cleanup, so without ". " the splitter would cut mid sentence (it cut a list of
# regions in half on the boardingware pdf)
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# tiny stopword list, just enough so bm25 doesn't match on "do you have the"
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "has", "have",
    "if", "in", "is", "it", "of", "on", "or", "our", "the", "to", "we", "what", "which", "with",
    "you", "your", "any", "following", "part", "please", "describe", "yes",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class HybridIndex:
    """Vector index (meaning) + BM25 (exact words) over the same chunks."""

    chunks: list[Document]
    vectors: FAISS
    bm25: BM25Okapi


def build_index(docs: list[Document], embeddings: Embeddings, chunk_size: int, chunk_overlap: int) -> HybridIndex:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS, keep_separator="end"
    )
    # json records (one answered question each) are kept whole so a question never gets
    # separated from its answer. anything really big gets split anyway
    whole = [d for d in docs if d.metadata.get("whole") and len(d.page_content) <= chunk_size * 4]
    rest = [d for d in docs if not (d.metadata.get("whole") and len(d.page_content) <= chunk_size * 4)]
    chunks = whole + splitter.split_documents(rest)
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i  # so results from both searches can be matched up
    log.info("indexing %d chunks from %d docs", len(chunks), len(docs))
    return HybridIndex(
        chunks=chunks,
        vectors=FAISS.from_documents(chunks, embeddings),
        bm25=BM25Okapi([_tokens(c.page_content) or ["_"] for c in chunks]),
    )


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score = sum of 1/(k + rank) over each ranking."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: -scores[c])


def keyword_search(index: HybridIndex, question: str, k: int) -> list[int]:
    scores = index.bm25.get_scores(_tokens(question))
    ranked = sorted(range(len(index.chunks)), key=lambda i: -scores[i])
    return [i for i in ranked[:k] if scores[i] > 0]


def retrieve(index: HybridIndex, question: str, k: int) -> list[Document]:
    # meaning search finds paraphrases, keyword search finds exact terms
    # (e.g. "monitoring" -> "CloudWatch ... monitoring"). merge both with rrf
    by_meaning = [d.metadata["chunk"] for d in index.vectors.similarity_search(question, k=k)]
    by_keyword = keyword_search(index, question, k)
    return [index.chunks[i] for i in rrf([by_meaning, by_keyword])[:k]]


def _source_of(d: Document):
    md = d.metadata or {}
    return md.get("page", md.get("record"))


def answer_questions(index: HybridIndex, questions: list[str], llm: BaseChatModel, top_k: int = 8) -> list[dict]:
    chain = PROMPT | llm | StrOutputParser()

    inputs, sources = [], []
    for q in questions:
        hits = retrieve(index, q, top_k)
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
