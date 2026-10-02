import io
import json
import re
import unicodedata
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class LoaderError(ValueError):
    """Raised when an uploaded file can't be parsed. API turns this into a 4xx."""


def parse_questions(raw: bytes) -> list[str]:
    # accept either ["q1", "q2"] or [{"question": "q1"}, ...]
    # also {"questions": [...]} since people send that a lot
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LoaderError(f"questions file is not valid JSON: {e}") from e

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]

    if not isinstance(data, list):
        raise LoaderError("questions file should be a JSON list")

    questions = []
    for item in data:
        if isinstance(item, str):
            q = item
        elif isinstance(item, dict) and isinstance(item.get("question"), str):
            q = item["question"]
        else:
            raise LoaderError(f"don't know how to read this question entry: {item!r}")
        q = q.strip()
        if q:
            questions.append(q)

    if not questions:
        raise LoaderError("no questions found in the file")
    return questions


def _clean_pdf_text(txt: str) -> str:
    # real pdfs (google docs exports especially) come out with one word per line and
    # ligatures like "\ufb01" instead of "fi". NFKC fixes the ligatures, then collapse
    # all the whitespace noise so chunks hold actual words and embeddings aren't diluted
    txt = unicodedata.normalize("NFKC", txt)
    return re.sub(r"\s+", " ", txt).strip()


def load_pdf(raw: bytes) -> list[Document]:
    try:
        reader = PdfReader(io.BytesIO(raw))
    except (PdfReadError, ValueError) as e:
        raise LoaderError(f"couldn't read PDF: {e}") from e

    docs = []
    for i, page in enumerate(reader.pages, start=1):
        txt = _clean_pdf_text(page.extract_text() or "")
        if txt:  # skip blank / image only pages
            docs.append(Document(page_content=txt, metadata={"page": i}))

    if not docs:
        # most likely a scanned pdf, we dont do OCR (yet)
        raise LoaderError("no extractable text in PDF (scanned document?)")
    return docs


def _flatten(obj: Any, prefix: str = "") -> list[str]:
    # turns nested json into "a.b: value" lines so the llm can actually read it
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lines.extend(_flatten(v, key))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            lines.extend(_flatten(v, f"{prefix}[{idx}]" if prefix else f"[{idx}]"))
    elif obj is not None and str(obj).strip():
        lines.append(f"{prefix}: {obj}" if prefix else str(obj))
    return lines


def load_json_doc(raw: bytes) -> list[Document]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LoaderError(f"document is not valid JSON: {e}") from e

    # if its a list of records (like the sample spreadsheet export) keep one
    # record per Document so a Q/A pair doesnt get cut in half by the splitter
    if isinstance(data, list):
        docs = []
        for i, rec in enumerate(data):
            text = "\n".join(_flatten(rec))
            if text:
                docs.append(Document(page_content=text, metadata={"record": i, "whole": True}))
    else:
        text = "\n".join(_flatten(data))
        docs = [Document(page_content=text, metadata={"record": 0})] if text else []

    if not docs:
        raise LoaderError("JSON document is empty")
    return docs


def load_document(filename: str, raw: bytes) -> list[Document]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return load_pdf(raw)
    if name.endswith(".json"):
        return load_json_doc(raw)
    raise LoaderError("document must be a .pdf or .json file")
