"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str = ""

    # LLM for entity extraction. Opus 4.7 chosen 2026-04-23 for consistency with
    # the rest of the SDI Engage stack (reduce-phase synthesis, SOP generation,
    # insight reconciliation all use Opus 4.7). Opus yields richer entities,
    # more accurate relationships, and better contradiction/ambiguity handling
    # at the cost of ~5x higher per-token spend vs. Sonnet and somewhat slower
    # per-episode wall-clock. Override via GRAPHITI_LLM_MODEL env var.
    anthropic_api_key: str = ""
    graphiti_llm_model: str = "claude-opus-4-7"

    # OpenAI for embeddings (Graphiti default)
    openai_api_key: str = ""

    # Service settings
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
