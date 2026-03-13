"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str = ""

    # LLM for entity extraction (Sonnet — better reasoning for contradictions/ambiguity)
    anthropic_api_key: str = ""
    graphiti_llm_model: str = "claude-sonnet-4-6"

    # OpenAI for embeddings (Graphiti default)
    openai_api_key: str = ""

    # Service settings
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
