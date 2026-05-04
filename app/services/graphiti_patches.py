"""
Runtime patches for graphiti-core (third-party library) compatibility.

Bug being patched
-----------------
graphiti-core's EpisodicNode model defines:

    entity_edges: list[str] = Field(..., default_factory=list)

The default_factory is only applied when the field is OMITTED. When the
field is explicitly set to None — which happens when an Episodic node in
FalkorDB has never had its entity_edges property written (e.g., legacy
episodes from before that field was tracked, or rows where the property
was deleted) — Pydantic raises:

    1 validation error for EpisodicNode
    entity_edges
      Input should be a valid list [type=list_type, input_value=None,
      input_type=NoneType]

This blocks every subsequent ingestion in our pokagon graph because
add_episode() reads existing episodes during dedup/lookup; a single bad
record poisons the whole pipeline. Observed in production: 41 jobs in
the kg-ingest-dead queue, 5 failed, 0 ingested since 2026-04-22.

Both library parser functions hit this:

  graphiti_core/nodes.py:998                  get_episodic_node_from_record
  graphiti_core/driver/record_parsers.py:86   episodic_node_from_record

Both call sites read record['entity_edges'] directly. We wrap each to
coerce None -> [] before delegating.

Why monkey-patch instead of editing the venv
--------------------------------------------
- venv edits get overwritten on the next `pip install`.
- Forking graphiti-core is heavy for a one-line bug.
- The PR upstream is the right long-term fix; this patch is the bridge.

Other module attributes (entity_node_from_record, community_node_from_record)
are left alone — only the EpisodicNode parsers exhibit this pattern in
production.
"""

import logging

logger = logging.getLogger("graphiti_service")


def install() -> None:
    """Install runtime patches. Idempotent — safe to call once at startup."""
    _patched = 0

    try:
        from graphiti_core import nodes as _nodes_mod

        _orig_get = _nodes_mod.get_episodic_node_from_record

        def _safe_get_episodic_node_from_record(record):
            if record is not None and record.get("entity_edges") is None:
                # Don't mutate the caller's record; copy and patch.
                record = dict(record)
                record["entity_edges"] = []
            return _orig_get(record)

        _nodes_mod.get_episodic_node_from_record = (
            _safe_get_episodic_node_from_record
        )
        _patched += 1
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[graphiti-patch] could not patch nodes.get_episodic_node_from_record: %s",
            e,
        )

    try:
        from graphiti_core.driver import record_parsers as _rp_mod

        _orig_parse = _rp_mod.episodic_node_from_record

        def _safe_episodic_node_from_record(record):
            if record is not None and record.get("entity_edges") is None:
                record = dict(record)
                record["entity_edges"] = []
            return _orig_parse(record)

        _rp_mod.episodic_node_from_record = _safe_episodic_node_from_record
        _patched += 1
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[graphiti-patch] could not patch record_parsers.episodic_node_from_record: %s",
            e,
        )

    logger.info(
        "[graphiti-patch] installed entity_edges-null guard on %d parser(s)",
        _patched,
    )
