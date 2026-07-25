"""Versioned Graphiti-owned provenance contract constants.

These values describe graph structure only. They deliberately carry no claim
that a source identifier resolves in a tenant database or grounds fact content.
"""

STRUCTURED_PROVENANCE_CONTRACT_VERSION = "structured_provenance_v2"
EPISODE_PROVENANCE_CONTRACT_VERSION = "engage_episode_v2"
LEGACY_STRUCTURED_CONTRACT_VERSION = "legacy_structured_v1"
LEGACY_EPISODE_CONTRACT_VERSION = "legacy_episode_v0"
SEARCH_CONTEXT_CONTRACT_VERSION = "graphiti_search_context_v3"
PROVENANCE_SUMMARY_CONTRACT_VERSION = "graphiti_provenance_summary_v1"
PROVENANCE_SHADOW_CONTRACT_VERSION = "graphiti_provenance_shadow_v1"

PROVENANCE_WRITE_STATE_STAGING = "staging"
PROVENANCE_WRITE_STATE_COMPLETE = "complete"

V2_ANCHOR_MODES = frozenset({"typed_source", "engagement"})
V2_PRODUCER_CONTRACT_VERSIONS = frozenset(
    {
        STRUCTURED_PROVENANCE_CONTRACT_VERSION,
        EPISODE_PROVENANCE_CONTRACT_VERSION,
    }
)
STRUCTURALLY_ANCHORED_MODES = frozenset(
    {*V2_ANCHOR_MODES, LEGACY_EPISODE_CONTRACT_VERSION}
)
