"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str = ""

    # LLM for entity extraction. The model string MUST be one graphiti-core's
    # token map recognizes, or it silently falls back to a 16384-token output
    # cap. graphiti-core 0.28.2 recognizes the Claude 4.5 generation
    # (claude-sonnet-4-5-latest, claude-haiku-4-5-latest -> 65536) but NOT any
    # Opus 4.x — the prior "claude-opus-4-7" was unrecognized and capped.
    # Sonnet 4.5 is the quality choice; set GRAPHITI_LLM_MODEL=claude-haiku-4-5-latest
    # for the cheaper routine-extraction tier.
    anthropic_api_key: str = ""
    graphiti_llm_model: str = "claude-sonnet-4-5-latest"

    # Embeddings. With no Voyage key set, Graphiti uses its implicit OpenAI
    # default embedder. Setting VOYAGE_API_KEY switches to Voyage (Anthropic's
    # recommended embeddings provider), removing OpenAI as the single embedding
    # point of failure. NOTE: switching providers makes the existing OpenAI
    # vectors incompatible, so existing graphs must be re-embedded at cutover.
    openai_api_key: str = ""
    voyage_api_key: str = ""
    embedding_model: str = "voyage-4-large"
    embedding_dim: int = 1024

    # Service settings
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
