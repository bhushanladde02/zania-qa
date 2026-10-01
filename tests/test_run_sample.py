import json
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_sample  # noqa: E402

from app.config import Settings  # noqa: E402

FX = Path(__file__).parent / "fixtures"


def test_run_end_to_end_with_fakes():
    llm = FakeListChatModel(responses=["GCP"] * 5)
    res = run_sample.run(
        FX / "sample_questions.json",
        FX / "sample_document.json",
        llm,
        DeterministicFakeEmbedding(size=64),
        Settings(openai_api_key="x"),
    )
    assert len(res["results"]) == 5
    assert all(r["answer"] == "GCP" for r in res["results"])
    json.dumps(res)  # has to be serialisable for the -o file


def test_main_without_key_exits(monkeypatch):
    monkeypatch.setattr(run_sample, "get_settings", lambda: Settings(openai_api_key="", _env_file=None))
    with pytest.raises(SystemExit) as e:
        run_sample.main([str(FX / "sample_questions.json"), str(FX / "sample_document.json")])
    assert "OPENAI_API_KEY" in str(e.value)
