import json

import pytest

from app.loaders import LoaderError, load_document, load_json_doc, parse_questions


def test_questions_plain_list():
    assert parse_questions(b'["a?", " b? ", ""]') == ["a?", "b?"]


def test_questions_objects_and_wrapper():
    raw = json.dumps({"questions": [{"question": "x?"}, "y?"]}).encode()
    assert parse_questions(raw) == ["x?", "y?"]


@pytest.mark.parametrize("raw", [b"not json", b'{"foo": 1}', b"[]", b"[1, 2]"])
def test_questions_bad_input(raw):
    with pytest.raises(LoaderError):
        parse_questions(raw)


def test_json_doc_one_doc_per_record():
    raw = json.dumps([{"question": "Cloud?", "answer": "AWS"}, {"question": "SSO?", "answer": "Yes"}]).encode()
    docs = load_json_doc(raw)
    assert len(docs) == 2
    assert "answer: AWS" in docs[0].page_content
    assert docs[1].metadata["record"] == 1


def test_json_doc_nested_dict():
    docs = load_json_doc(b'{"infra": {"cloud": "GCP", "regions": ["us", "eu"]}}')
    txt = docs[0].page_content
    assert "infra.cloud: GCP" in txt
    assert "infra.regions[1]: eu" in txt


def test_pdf_pages(sample_pdf):
    docs = load_document("report.PDF", sample_pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "us-east-1" in docs[0].page_content


def test_broken_pdf():
    with pytest.raises(LoaderError):
        load_document("x.pdf", b"%PDF-garbage")


def test_unsupported_ext():
    with pytest.raises(LoaderError):
        load_document("notes.txt", b"hello")


def test_pdf_text_cleanup():
    from app.loaders import _clean_pdf_text

    messy = "Boardingware  has  partnered\n \nwith\n \nAWS  for  conﬁguration"
    assert _clean_pdf_text(messy) == "Boardingware has partnered with AWS for configuration"
