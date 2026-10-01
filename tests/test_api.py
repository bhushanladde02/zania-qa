import json


def _post(client, questions, doc_bytes, doc_name="doc.pdf", q_name="questions.json"):
    return client.post(
        "/qa",
        files={
            "questions_file": (q_name, json.dumps(questions).encode(), "application/json"),
            "document_file": (doc_name, doc_bytes, "application/octet-stream"),
        },
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_qa_pdf_happy_path(client, sample_pdf):
    r = _post(client, ["Which cloud and region?", "What is the CEO's name?"], sample_pdf)
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [x["question"] for x in res] == ["Which cloud and region?", "What is the CEO's name?"]
    assert res[0]["answer"] == "AWS us-east-1"
    assert res[0]["sources"]  # got page numbers back
    assert res[1]["answer"] == "Data Not Available"
    assert res[1]["sources"] == []


def test_qa_json_document(client):
    doc = json.dumps([{"question": "Cloud provider?", "answer": "AWS"}]).encode()
    r = _post(client, [{"question": "Which cloud?"}], doc, doc_name="kb.json")
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 1


def test_bad_doc_type(client):
    r = _post(client, ["q?"], b"hello", doc_name="notes.txt")
    assert r.status_code == 415


def test_questions_must_be_json_file(client, sample_pdf):
    r = _post(client, ["q?"], sample_pdf, q_name="questions.txt")
    assert r.status_code == 415


def test_malformed_questions(client, sample_pdf):
    r = client.post(
        "/qa",
        files={
            "questions_file": ("q.json", b"{not json", "application/json"),
            "document_file": ("doc.pdf", sample_pdf, "application/pdf"),
        },
    )
    assert r.status_code == 422


def test_empty_document(client):
    r = _post(client, ["q?"], b"", doc_name="doc.pdf")
    assert r.status_code == 400


def test_index_is_cached(client, sample_pdf, monkeypatch):
    from app import main

    calls = {"n": 0}
    real = main.build_index

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(main, "build_index", counting)
    _post(client, ["one?"], sample_pdf)
    _post(client, ["two?"], sample_pdf)
    assert calls["n"] == 1


def test_qa_with_zania_sample_doc(client):
    # the real sample file from the challenge, converted from the xlsx
    from pathlib import Path

    fx = Path(__file__).parent / "fixtures"
    r = client.post(
        "/qa",
        files={
            "questions_file": ("q.json", (fx / "sample_questions.json").read_bytes(), "application/json"),
            "document_file": ("sample_document.json", (fx / "sample_document.json").read_bytes(), "application/json"),
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 5


def test_openai_auth_error_is_readable(client, sample_pdf):
    import httpx
    import openai
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from app import main

    class BadKeyEmbeddings(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            raise openai.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)

    main.app.dependency_overrides[main.get_embeddings] = lambda: BadKeyEmbeddings(size=8)
    r = _post(client, ["q?"], sample_pdf)
    assert r.status_code == 502
    assert "API key" in r.json()["detail"]
