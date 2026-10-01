import io

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from reportlab.pdfgen import canvas

from app import main


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for txt in pages:
        y = 800
        for line in txt.split("\n"):
            c.drawString(50, y, line)
            y -= 15
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def client():
    # fake llm just returns canned answers in order, no network calls / no $$
    fake_llm = FakeListChatModel(responses=["AWS us-east-1", "Data Not Available"] * 3)
    main.app.dependency_overrides[main.get_llm] = lambda: fake_llm
    main.app.dependency_overrides[main.get_embeddings] = lambda: DeterministicFakeEmbedding(size=64)
    main._cache._store.clear()
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf():
    return make_pdf(
        [
            "Nave hosts the service on Amazon Web Services.\nPrimary region is us-east-1.",
            "Backups are stored in us-west-2.\nIncidents are notified within 72 hours.",
        ]
    )
