from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.rag import build_index, keyword_search, retrieve, rrf

DOCS = [
    Document(page_content="All data in transit is secured by TLS over HTTPS. Servers use NTP for clock sync.", metadata={"page": 1}),
    Document(page_content="Cloud monitoring: we use CloudWatch to monitor resources and alarm on latency and errors.", metadata={"page": 2}),
    Document(page_content="Passwords must be at least 10 characters. Reset links expire after 1 day.", metadata={"page": 3}),
    Document(page_content="Backups are encrypted and stored in the same region across availability zones.", metadata={"page": 4}),
]


def _index(**kw):
    return build_index(DOCS, DeterministicFakeEmbedding(size=32), kw.get("size", 500), kw.get("overlap", 50))


def test_rrf_rewards_agreement():
    # chunk 2 is 2nd in both lists, it should beat chunks that are 1st in only one
    assert rrf([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_keyword_search_finds_exact_terms():
    idx = _index()
    hits = keyword_search(idx, "Which of the following are part of your monitoring process?", k=2)
    assert idx.chunks[hits[0]].metadata["page"] == 2


def test_keyword_search_ignores_stopword_only_questions():
    assert keyword_search(_index(), "do you have the", k=3) == []


def test_retrieve_includes_keyword_hit_and_no_duplicates():
    idx = _index()
    got = retrieve(idx, "monitoring process for the service", k=3)
    pages = [d.metadata["page"] for d in got]
    assert 2 in pages
    assert len(pages) == len(set(pages)) <= 3


def test_splitter_keeps_sentences_whole():
    # pdf text has no newlines after cleanup, chunks should still end on a sentence
    text = ("Intro sentence about the company. " * 6) + (
        "The available regions are: Virginia, Ireland and Sydney. Schools choose one region. "
    ) + ("Closing sentence about support. " * 6)
    idx = build_index([Document(page_content=text, metadata={"page": 7})], DeterministicFakeEmbedding(size=8), 200, 40)
    assert any("Virginia, Ireland and Sydney." in c.page_content for c in idx.chunks)
    assert all(c.page_content.rstrip().endswith(".") for c in idx.chunks)


def test_json_records_are_not_split():
    from app.loaders import load_json_doc
    import json

    rec = {"question": "Where are your data centres located?", "answer": "US Central on GCP. " * 40}
    docs = load_json_doc(json.dumps([rec, {"question": "q2", "answer": "a2"}]).encode())
    idx = build_index(docs, DeterministicFakeEmbedding(size=8), 300, 50)
    assert len(idx.chunks) == 2  # ~800 char record stayed one chunk (cap is 4 x chunk size)
    assert "Where are your data centres" in idx.chunks[0].page_content and "US Central" in idx.chunks[0].page_content
