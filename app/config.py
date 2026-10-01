from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"  # brief says stick to this one
    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5

    # 20MB felt like plenty for a SOC2 report, bump it if needed
    max_upload_mb: int = 20
    max_questions: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
