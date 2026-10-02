"""
Central application configuration, loaded from environment variables
(or a local .env file via python-dotenv). Every other module should pull
its settings from here rather than reading os.environ directly — that
keeps all the "what's configurable" knowledge in one place.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider — "ollama" (fully local, $0 forever) or "gemini" (free
    # tier via Google AI Studio, no local hardware needed). Every agent
    # calls chat_completion() in services/llm_provider.py regardless of
    # which one is active — nothing else in the codebase needs to change.
    llm_provider: str = "ollama"

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # LLM (Gemini) — free tier via https://aistudio.google.com, no card
    # required. Defaults to the "latest" alias (auto-updates, ~2 weeks
    # notice before it points at a new model); pin to a dated model
    # (e.g. "gemini-3.6-flash") instead if you want it to never change
    # under you.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Vector store
    chroma_persist_dir: str = "./.chroma_data"
    chroma_collection: str = "company_policies"

    # Guardrails
    guardrails_enabled: bool = True

    # App
    app_env: str = "development"
    log_level: str = "INFO"


settings = Settings()