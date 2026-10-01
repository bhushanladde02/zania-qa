from pydantic import BaseModel


class QAPair(BaseModel):
    question: str
    answer: str
    sources: list[int | str] = []


class QAResponse(BaseModel):
    results: list[QAPair]
