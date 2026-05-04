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
    """Install runtime patches. Idempotent — safe to call once at startup.

    The first attempt at this fix patched the parser functions
    (record_parsers.episodic_node_from_record and
    nodes.get_episodic_node_from_record). That didn't work because the
    falkor and neo4j operation modules each do
        from graphiti_core.driver.record_parsers import episodic_node_from_record
    at module load time, capturing a *local reference* to the original
    function. Replacing the source-module attribute later does not affect
    callers that already imported the function by name.

    Instead we patch EpisodicNode.__init__ itself — which catches every
    construction path, regardless of how the caller obtained the model
    class. Coerce record.get('entity_edges') from None to [] BEFORE
    Pydantic v2 validation runs.
    """
    import ast

    def _coerce_list(value):
        """Coerce a value that should be a list into one.

        Handles three real-world failure modes from FalkorDB:
        - None  -> []                      (property never set / cleared)
        - "['a','b']" -> ['a','b']         (Falkor sometimes returns the str
                                            repr of a list rather than a real
                                            array; happens for some properties)
        - "[]"  -> []                      (str repr of empty list)
        Anything that's already a list passes through. Anything else we
        leave alone and let Pydantic validate.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return value

    try:
        from graphiti_core.nodes import EpisodicNode

        if not getattr(EpisodicNode.__init__, "_sdi_patched", False):
            _orig_episodic_init = EpisodicNode.__init__

            def _patched_episodic_init(self, **data):  # type: ignore[no-untyped-def]
                # entity_edges: list[str] — FalkorDB returns None for legacy
                # episodes; Pydantic Field(default_factory=list) only applies
                # when the key is omitted, not present-with-None.
                data["entity_edges"] = _coerce_list(data.get("entity_edges"))
                _orig_episodic_init(self, **data)

            _patched_episodic_init._sdi_patched = True  # type: ignore[attr-defined]
            EpisodicNode.__init__ = _patched_episodic_init  # type: ignore[method-assign]
            logger.info("[graphiti-patch] EpisodicNode.__init__ patched: entity_edges None|str -> list")
    except Exception as e:  # noqa: BLE001
        logger.warning("[graphiti-patch] could not patch EpisodicNode: %s", e)

    try:
        from graphiti_core.edges import EntityEdge

        if not getattr(EntityEdge.__init__, "_sdi_patched", False):
            _orig_entity_edge_init = EntityEdge.__init__

            def _patched_entity_edge_init(self, **data):  # type: ignore[no-untyped-def]
                # episodes: list[str] — FalkorDB returns this property as the
                # string representation of a list ("['uuid-1','uuid-2']")
                # rather than a real array, which Pydantic rejects.
                data["episodes"] = _coerce_list(data.get("episodes"))
                _orig_entity_edge_init(self, **data)

            _patched_entity_edge_init._sdi_patched = True  # type: ignore[attr-defined]
            EntityEdge.__init__ = _patched_entity_edge_init  # type: ignore[method-assign]
            logger.info("[graphiti-patch] EntityEdge.__init__ patched: episodes None|str -> list")
    except Exception as e:  # noqa: BLE001
        logger.warning("[graphiti-patch] could not patch EntityEdge: %s", e)
