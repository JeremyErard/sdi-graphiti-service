"""Application configuration from environment variables."""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FalkorDB connection
    # How many episode extractions may run at once.
    #
    # One, because this service runs on a SINGLE CPU and extraction is a long
    # chain of model calls with real CPU work between them (embedding handling,
    # JSON parsing, rank fusion). Two concurrent extractions starved the asyncio
    # event loop badly enough that the HTTP server stopped answering: on
    # 2026-08-28 both POST /ingest/episode/async and POST /ingest/jobs/status
    # timed out at the caller while the service was alive and had not restarted.
    #
    # Serialising is not a throughput loss worth minding — the work is
    # CPU-bound on one core either way. What it buys is a service that keeps
    # answering, so a queued job is reported as queued instead of looking dead.
    max_concurrent_ingests: int = 1

    # Socket timeout for the SYNCHRONOUS FalkorDB handle, in seconds.
    #
    # Load-bearing. That handle is used from inside async request paths, so a
    # call with no timeout blocks the whole event loop for as long as FalkorDB
    # takes — which, if FalkorDB is wedged, is forever. On 2026-08-28 this
    # service stopped answering even /health (a 3s redis ping) while an
    # extraction held the loop; nothing else could be scheduled at all.
    #
    # Generous rather than tight, because admin export/import legitimately run
    # long. The point is that it is FINITE.
    falkordb_socket_timeout_seconds: int = 120

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

    # P1 provenance rollout is deliberately non-enforcing by default. ``legacy``
    # preserves the established search response. ``shadow`` adds a separately
    # versioned preview without changing prompt-visible facts. Only ``enforce``
    # returns the fail-closed v3 response.
    graphiti_provenance_mode: Literal["legacy", "shadow", "enforce"] = "legacy"

    # The source-anchored structured-v2 writer remains dormant until an operator
    # explicitly selects the staged lifecycle. There is intentionally no unsafe
    # direct-enable mode.
    graphiti_structured_v2_write_mode: Literal["off", "staged"] = "off"

    # Dedicated acceptance-probe processes expose only the signed, enforced
    # search contract and use FalkorDB's read-only query command. The default is
    # deliberately false so ordinary service processes retain their established
    # retrieval/fallback behavior.
    graphiti_acceptance_probe_mode: bool = False

    # Declared with the flags it governs rather than at the end of the class so
    # this block does not sit on the same insertion point every other settings
    # change uses. Field order is irrelevant to pydantic; the validator runs
    # after every field below is populated, including the auth mode.
    @model_validator(mode="after")
    def validate_mode_combinations(self):
        if (
            self.graphiti_structured_v2_write_mode == "staged"
            and self.graphiti_provenance_mode != "enforce"
        ):
            raise ValueError(
                "GRAPHITI_STRUCTURED_V2_WRITE_MODE=staged requires "
                "GRAPHITI_PROVENANCE_MODE=enforce"
            )
        if self.graphiti_acceptance_probe_mode and (
            self.graphiti_provenance_mode != "enforce"
            or self.graphiti_auth_mode != "required"
        ):
            raise ValueError(
                "GRAPHITI_ACCEPTANCE_PROBE_MODE=true requires "
                "GRAPHITI_PROVENANCE_MODE=enforce and "
                "GRAPHITI_AUTH_MODE=required"
            )
        if self.graphiti_acceptance_probe_mode and not self.voyage_api_key.strip():
            raise ValueError(
                "GRAPHITI_ACCEPTANCE_PROBE_MODE=true requires VOYAGE_API_KEY "
                "for the exact fast-path embedder"
            )
        return self

    # Engage -> Graphiti service authentication. ``off`` preserves the current
    # production contract during a coordinated rollout; ``optional`` accepts
    # unsigned legacy traffic but verifies any signed request; ``required``
    # rejects every non-health request that is not scope- and tenant-bound.
    graphiti_auth_mode: Literal["off", "optional", "required"] = "off"
    graphiti_search_secret: str = ""
    graphiti_ingest_secret: str = ""
    graphiti_admin_secret: str = ""
    graphiti_auth_max_clock_skew_seconds: int = 300

    # Governed exact-ID projection (/ingest/projection/v2). The envelope carries
    # both origins so the direct import path and the later Outcome projector emit
    # one shape, but only the direct import lane is authorized in this phase.
    # Default false keeps the Outcome projector lane refused until it is ratified.
    projection_v2_allow_outcome_event: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
