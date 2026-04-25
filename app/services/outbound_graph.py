"""Outbound GTM Knowledge Graph — direct FalkorDB access for the hyperloop.

This module owns the `gtm_outbound` graph: a per-recipient relationship graph
capturing every send as a hypothesis-bound experiment with delayed-reward edges.

Distinct from the per-client engagement graphs (which use Graphiti's episode
abstraction) — this graph is structured Cypher with explicit Recipient/Send/
Hypothesis/Outcome/CoachingEvent/Principle nodes designed for compose-time
retrieval and outcome-driven learning.

Schema designed 2026-04-25 — see project_thrive_llm_hyperloop_architecture.md.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import redis

from app.config import settings

logger = logging.getLogger("graphiti_service")

GRAPH_NAME = "gtm_outbound"

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    """Lazy-init the redis client to FalkorDB."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            decode_responses=True,
        )
        _redis_client.ping()  # fail fast if unreachable
    return _redis_client


def close():
    """Close the redis connection."""
    global _redis_client
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception:
            pass
        _redis_client = None


# ---------------------------------------------------------------------------
# Cypher execution helpers
# ---------------------------------------------------------------------------

def _query(cypher: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a Cypher query against the gtm_outbound graph.

    FalkorDB's GRAPH.QUERY does not natively support parameter binding the way
    Neo4j does — we inline values into the query string with safe escaping.
    Returns a dict with header (column names) and rows (list of lists).
    """
    client = _get_redis()
    final_cypher = _inline_params(cypher, params or {})
    try:
        # GRAPH.QUERY <key> <cypher> --compact
        raw = client.execute_command("GRAPH.QUERY", GRAPH_NAME, final_cypher)
    except redis.exceptions.ResponseError as e:
        logger.error(f"[outbound] GRAPH.QUERY error: {e}\nCypher: {final_cypher[:500]}")
        raise

    # FalkorDB response shape: [header, rows, statistics] or [statistics] for write-only
    if not isinstance(raw, list):
        return {"header": [], "rows": [], "stats": str(raw)}
    if len(raw) >= 3:
        header_raw, rows_raw, stats_raw = raw[0], raw[1], raw[2]
        header = [h[1] if isinstance(h, list) and len(h) >= 2 else str(h) for h in header_raw]
        rows = [list(r) for r in rows_raw]
        stats = stats_raw
    elif len(raw) == 1:
        header, rows, stats = [], [], raw[0]
    else:
        header, rows, stats = [], [], raw
    return {"header": header, "rows": rows, "stats": stats}


def _escape(value: Any) -> str:
    """Escape a value for safe inlining into Cypher."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_escape(v) for v in value) + "]"
    if isinstance(value, dict):
        # Convert dict to a Cypher map literal
        parts = [f"{k}: {_escape(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    # String — JSON-encode then strip wrapping quotes and re-wrap with single quotes
    s = str(value)
    # Use double-quote-aware escaping — FalkorDB accepts both single and double quoted strings
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _inline_params(cypher: str, params: dict[str, Any]) -> str:
    """Replace $name placeholders with safely-escaped literal values."""
    if not params:
        return cypher
    # Sort keys longest-first so $foo_bar replaces before $foo
    for key in sorted(params.keys(), key=len, reverse=True):
        placeholder = f"${key}"
        cypher = cypher.replace(placeholder, _escape(params[key]))
    return cypher


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

# Order matters: indexes after first node creation. We create dummy nodes
# (one per label) to ensure FalkorDB recognizes the labels for index creation,
# then drop them. FalkorDB requires at least one node of a label before
# CREATE INDEX. Newer versions auto-create — we cover both paths.
SCHEMA_STATEMENTS = [
    # Create one of each node type so labels exist
    "MERGE (:Recipient {id: '__schema_seed__'})",
    "MERGE (:Send {id: '__schema_seed__'})",
    "MERGE (:Hypothesis {id: '__schema_seed__'})",
    "MERGE (:CoachingEvent {id: '__schema_seed__'})",
    "MERGE (:Outcome {id: '__schema_seed__'})",
    "MERGE (:Principle {id: '__schema_seed__'})",
    "MERGE (:SubSegment {id: '__schema_seed__'})",
    "MERGE (:Company {id: '__schema_seed__'})",
    # Property indexes for hot-path lookups
    "CREATE INDEX FOR (r:Recipient) ON (r.id)",
    "CREATE INDEX FOR (r:Recipient) ON (r.normalized_name)",
    "CREATE INDEX FOR (r:Recipient) ON (r.company_id)",
    "CREATE INDEX FOR (s:Send) ON (s.id)",
    "CREATE INDEX FOR (s:Send) ON (s.sent_at)",
    "CREATE INDEX FOR (s:Send) ON (s.body_hash)",
    "CREATE INDEX FOR (s:Send) ON (s.sub_segment)",
    "CREATE INDEX FOR (h:Hypothesis) ON (h.id)",
    "CREATE INDEX FOR (h:Hypothesis) ON (h.model_version)",
    "CREATE INDEX FOR (c:CoachingEvent) ON (c.id)",
    "CREATE INDEX FOR (c:CoachingEvent) ON (c.extracted_rule)",
    "CREATE INDEX FOR (o:Outcome) ON (o.id)",
    "CREATE INDEX FOR (o:Outcome) ON (o.kind)",
    "CREATE INDEX FOR (p:Principle) ON (p.rule)",
    "CREATE INDEX FOR (p:Principle) ON (p.scope)",
    "CREATE INDEX FOR (g:SubSegment) ON (g.id)",
    "CREATE INDEX FOR (co:Company) ON (co.id)",
    # Drop the seed nodes
    "MATCH (n) WHERE n.id = '__schema_seed__' DELETE n",
]


def init_schema() -> dict[str, Any]:
    """Initialize the gtm_outbound graph schema. Idempotent — re-runnable safely."""
    results = {"executed": 0, "skipped": 0, "errors": []}
    for stmt in SCHEMA_STATEMENTS:
        try:
            _query(stmt)
            results["executed"] += 1
        except redis.exceptions.ResponseError as e:
            msg = str(e)
            # Index already exists is fine
            if "already" in msg.lower() or "exists" in msg.lower() or "equivalent" in msg.lower():
                results["skipped"] += 1
            else:
                results["errors"].append({"stmt": stmt, "error": msg})
    return results


def health() -> dict[str, Any]:
    """Confirm the graph is reachable + return basic counts."""
    try:
        r = _query("MATCH (n) RETURN count(n) AS total, count(DISTINCT labels(n)[0]) AS labels")
        total = r["rows"][0][0] if r["rows"] else 0
        labels = r["rows"][0][1] if r["rows"] else 0
        return {"reachable": True, "graph": GRAPH_NAME, "node_count": total, "distinct_labels": labels}
    except Exception as e:
        return {"reachable": False, "graph": GRAPH_NAME, "error": str(e)}


# ---------------------------------------------------------------------------
# Recipient management
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def upsert_recipient(payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update a Recipient. Identity is `id` (provided) or
    derived from normalized_name+company_id if id is missing.

    Required: full_name OR linkedin_urn OR id (one of three).
    Optional: company, title, seniority, function, sector, tier, geo,
              dossier_path, primary_email, response_profile (list[float]).
    """
    full_name = payload.get("full_name") or ""
    company = payload.get("company") or ""
    rcpt_id = payload.get("id") or ""

    if not rcpt_id:
        if full_name:
            normalized = _normalize_name(full_name)
            company_slug = re.sub(r"[^a-z0-9]", "", (company or "_").lower())
            rcpt_id = f"rcpt_{company_slug}_{re.sub(r'[^a-z0-9]', '', normalized)}"[:80]
        elif payload.get("linkedin_urn"):
            rcpt_id = f"rcpt_{payload['linkedin_urn']}"
        else:
            raise ValueError("upsert_recipient requires id, full_name, or linkedin_urn")

    now = datetime.now(timezone.utc).isoformat()

    # Build properties dict — only include non-None values
    props: dict[str, Any] = {
        "id": rcpt_id,
        "updated_at": now,
    }
    for k in ("full_name", "linkedin_urn", "primary_email", "company", "company_id",
              "title", "seniority", "function", "sector", "tier", "geo", "dossier_path"):
        v = payload.get(k)
        if v is not None and v != "":
            props[k] = v
    if full_name:
        props["normalized_name"] = _normalize_name(full_name)

    # Properties to set only on insert (not overwritten on update)
    on_create_props = {
        "created_at": now,
        "touch_count": 0,
        "reply_count": 0,
        "reply_rate_smoothed": 0.0,
    }

    # MERGE on id, SET on existing, ON CREATE SET on new
    set_clauses = ", ".join(f"r.{k} = ${'p_' + k}" for k in props.keys())
    create_clauses = ", ".join(f"r.{k} = ${'c_' + k}" for k in on_create_props.keys())

    cypher = f"""
        MERGE (r:Recipient {{id: $rcpt_id}})
        ON CREATE SET {create_clauses}
        SET {set_clauses}
        RETURN r.id, r.touch_count, r.reply_count
    """
    params = {"rcpt_id": rcpt_id}
    for k, v in props.items():
        params[f"p_{k}"] = v
    for k, v in on_create_props.items():
        params[f"c_{k}"] = v

    result = _query(cypher, params)
    if result["rows"]:
        row = result["rows"][0]
        return {"id": row[0], "touch_count": row[1], "reply_count": row[2]}
    return {"id": rcpt_id}


def record_send(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a Send node + RECEIVED edge from Recipient + optional Hypothesis.

    Required: recipient_id, channel, sent_at OR drafted_at.
    Optional: framing, hook_type, length_tokens, sub_segment, body_hash,
              body_preview, drafted_by, signal_basis (list), variant_id,
              client_context, hypothesis (dict with predicted_reply_rate, rationale).
    """
    recipient_id = payload.get("recipient_id")
    if not recipient_id:
        raise ValueError("record_send requires recipient_id")

    send_id = payload.get("id") or f"snd_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()

    send_props: dict[str, Any] = {
        "id": send_id,
        "channel": payload.get("channel", "email"),
        "drafted_at": payload.get("drafted_at") or now,
        "status": payload.get("status", "drafted"),
    }
    if payload.get("sent_at"):
        send_props["sent_at"] = payload["sent_at"]
    for k in ("variant_id", "framing", "hook_type", "length_tokens", "sub_segment",
              "body_hash", "body_preview", "drafted_by", "promoted_via",
              "client_context", "signal_basis"):
        v = payload.get(k)
        if v is not None:
            send_props[k] = v

    # Build SET clauses for Send
    set_parts = ", ".join(f"s.{k} = ${'sp_' + k}" for k in send_props.keys())
    cypher = f"""
        MATCH (r:Recipient {{id: $recipient_id}})
        CREATE (s:Send)
        SET {set_parts}
        CREATE (r)-[:RECEIVED {{at: '{now}'}}]->(s)
        SET r.touch_count = coalesce(r.touch_count, 0) + 1
    """
    params = {"recipient_id": recipient_id}
    for k, v in send_props.items():
        params[f"sp_{k}"] = v

    hypothesis = payload.get("hypothesis")
    hyp_id = None
    if hypothesis and hypothesis.get("predicted_reply_rate") is not None:
        hyp_id = hypothesis.get("id") or f"hyp_{uuid.uuid4().hex[:16]}"
        hyp_props = {
            "id": hyp_id,
            "predicted_reply_rate": float(hypothesis["predicted_reply_rate"]),
            "rationale": hypothesis.get("rationale", ""),
            "model_version": hypothesis.get("model_version", "thrive_llm_unknown"),
            "registered_at": now,
        }
        hyp_set = ", ".join(f"h.{k} = ${'hp_' + k}" for k in hyp_props.keys())
        cypher += f"""
            CREATE (h:Hypothesis)
            SET {hyp_set}
            CREATE (s)-[:TESTED]->(h)
        """
        for k, v in hyp_props.items():
            params[f"hp_{k}"] = v

    cypher += " RETURN s.id"
    _query(cypher, params)
    return {"send_id": send_id, "hypothesis_id": hyp_id}


def record_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    """Record an Outcome on a Send. Updates Recipient.reply_count when kind=replied.

    Required: send_id, kind (one of: delivered|opened|replied|meeting_booked|
              opportunity|closed_won|closed_lost|bounced|unsubscribed).
    Optional: occurred_at, latency_seconds, reply_sentiment, reply_intent,
              meeting_id, deal_id, attribution_confidence, observed_by.
    """
    send_id = payload.get("send_id")
    kind = payload.get("kind")
    if not send_id or not kind:
        raise ValueError("record_outcome requires send_id and kind")

    out_id = payload.get("id") or f"out_{uuid.uuid4().hex[:16]}"
    occurred_at = payload.get("occurred_at") or datetime.now(timezone.utc).isoformat()

    out_props: dict[str, Any] = {
        "id": out_id,
        "kind": kind,
        "occurred_at": occurred_at,
    }
    for k in ("latency_seconds", "reply_sentiment", "reply_intent",
              "reply_text_hash", "meeting_id", "deal_id",
              "attribution_confidence", "observed_by"):
        v = payload.get(k)
        if v is not None:
            out_props[k] = v

    set_parts = ", ".join(f"o.{k} = ${'op_' + k}" for k in out_props.keys())
    cypher = f"""
        MATCH (s:Send {{id: $send_id}})
        CREATE (o:Outcome)
        SET {set_parts}
        CREATE (s)-[:PRODUCED]->(o)
        WITH s
        OPTIONAL MATCH (r:Recipient)-[:RECEIVED]->(s)
    """
    params = {"send_id": send_id}
    for k, v in out_props.items():
        params[f"op_{k}"] = v

    if kind == "replied":
        cypher += """
            SET r.reply_count = coalesce(r.reply_count, 0) + 1
            WITH r
            SET r.reply_rate_smoothed =
                (2.0 + coalesce(r.reply_count, 0)) /
                (2.0 + 20.0 + coalesce(r.touch_count, 0))
        """
    cypher += " RETURN $send_id"
    _query(cypher, params)
    return {"outcome_id": out_id, "send_id": send_id, "kind": kind}


def record_coaching_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a CoachingEvent linked to a Send + Principle.

    Required: action (KEEP|REWRITE|REJECT). Optional: send_id (the Send being coached),
              extracted_rule, raw_feedback, reviewer, scope.
    """
    action = payload.get("action", "REWRITE").upper()
    coach_id = payload.get("id") or f"coach_{uuid.uuid4().hex[:16]}"
    occurred_at = payload.get("occurred_at") or datetime.now(timezone.utc).isoformat()

    coach_props = {
        "id": coach_id,
        "action": action,
        "reviewer": payload.get("reviewer", "jeremy"),
        "raw_feedback": payload.get("raw_feedback", ""),
        "extracted_rule": payload.get("extracted_rule", ""),
        "rule_scope": payload.get("scope", "general"),
        "occurred_at": occurred_at,
    }
    set_parts = ", ".join(f"c.{k} = ${'cp_' + k}" for k in coach_props.keys())

    send_id = payload.get("send_id")
    if send_id:
        cypher = f"""
            MATCH (s:Send {{id: $send_id}})
            CREATE (c:CoachingEvent)
            SET {set_parts}
            CREATE (s)-[:COACHED_BY]->(c)
        """
        params = {"send_id": send_id}
    else:
        cypher = f"""
            CREATE (c:CoachingEvent)
            SET {set_parts}
        """
        params = {}
    for k, v in coach_props.items():
        params[f"cp_{k}"] = v

    # Upsert Principle if rule given — increments reinforcement_count
    rule = payload.get("extracted_rule")
    principle_id = None
    if rule:
        principle_id = f"prin_{re.sub(r'[^a-z0-9]+', '_', rule.lower())[:50]}"
        cypher += """
            WITH c
            MERGE (p:Principle {id: $principle_id})
            ON CREATE SET p.rule = $principle_rule,
                          p.scope = $principle_scope,
                          p.first_observed_at = $now,
                          p.reinforcement_count = 1
            ON MATCH SET p.last_observed_at = $now,
                         p.reinforcement_count = coalesce(p.reinforcement_count, 0) + 1
            CREATE (c)-[:ARTICULATES]->(p)
        """
        params["principle_id"] = principle_id
        params["principle_rule"] = rule
        params["principle_scope"] = payload.get("scope", "general")
        params["now"] = occurred_at

    cypher += " RETURN $coach_id"
    params["coach_id"] = coach_id
    _query(cypher, params)
    return {"coaching_event_id": coach_id, "principle_id": principle_id}


# ---------------------------------------------------------------------------
# Retrieval helpers (compose-time)
# ---------------------------------------------------------------------------

def get_recipient_history(recipient_id: str, limit: int = 10) -> dict[str, Any]:
    """Return the recipient's profile + recent touch history with outcomes.

    Used at compose time to inject prior-touch context into the prompt.
    """
    cypher = """
        MATCH (r:Recipient {id: $recipient_id})
        OPTIONAL MATCH (r)-[:RECEIVED]->(s:Send)
        OPTIONAL MATCH (s)-[:PRODUCED]->(o:Outcome)
        OPTIONAL MATCH (s)-[:COACHED_BY]->(c:CoachingEvent)
        WITH r, s, o, c
        ORDER BY s.sent_at DESC
        LIMIT $lim
        RETURN
            r.id, r.full_name, r.title, r.company, r.tier, r.function, r.sector,
            r.touch_count, r.reply_count, r.reply_rate_smoothed,
            s.id, s.channel, s.framing, s.hook_type, s.sub_segment, s.sent_at, s.body_preview,
            o.kind, o.reply_sentiment, o.occurred_at,
            c.action, c.extracted_rule
    """
    result = _query(cypher, {"recipient_id": recipient_id, "lim": limit})
    if not result["rows"]:
        return {"recipient_id": recipient_id, "found": False, "history": []}

    first_row = result["rows"][0]
    profile = {
        "id": first_row[0],
        "full_name": first_row[1],
        "title": first_row[2],
        "company": first_row[3],
        "tier": first_row[4],
        "function": first_row[5],
        "sector": first_row[6],
        "touch_count": first_row[7] or 0,
        "reply_count": first_row[8] or 0,
        "reply_rate_smoothed": first_row[9] or 0.0,
    }
    history = []
    for row in result["rows"]:
        if row[10]:  # has a Send
            history.append({
                "send_id": row[10],
                "channel": row[11],
                "framing": row[12],
                "hook_type": row[13],
                "sub_segment": row[14],
                "sent_at": row[15],
                "body_preview": row[16],
                "outcome_kind": row[17],
                "reply_sentiment": row[18],
                "outcome_at": row[19],
                "coaching_action": row[20],
                "coaching_rule": row[21],
            })
    return {"recipient_id": recipient_id, "found": True, "profile": profile, "history": history}


def cohort_stats(filters: dict[str, Any] | None = None, limit: int = 50) -> dict[str, Any]:
    """Return cohort-level reply statistics. Filters: tier, function, sector, sub_segment."""
    filters = filters or {}
    where_clauses = []
    params: dict[str, Any] = {}
    for k in ("tier", "function", "sector"):
        if filters.get(k) is not None:
            where_clauses.append(f"r.{k} = ${k}")
            params[k] = filters[k]
    if filters.get("sub_segment"):
        where_clauses.append("s.sub_segment = $sub_segment")
        params["sub_segment"] = filters["sub_segment"]
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    cypher = f"""
        MATCH (r:Recipient)-[:RECEIVED]->(s:Send)
        {where_sql}
        OPTIONAL MATCH (s)-[:PRODUCED]->(o:Outcome {{kind: 'replied'}})
        WITH s.framing AS framing, s.hook_type AS hook,
             count(DISTINCT s) AS sends, count(DISTINCT o) AS replies
        WHERE sends >= 3
        RETURN framing, hook, sends, replies,
               toFloat(replies) / sends AS reply_rate
        ORDER BY reply_rate DESC
        LIMIT $lim
    """
    params["lim"] = limit
    result = _query(cypher, params)
    rows = []
    for row in result["rows"]:
        rows.append({
            "framing": row[0],
            "hook_type": row[1],
            "sends": row[2],
            "replies": row[3],
            "reply_rate": row[4],
        })
    return {"filters": filters, "cohorts": rows}


def graph_stats() -> dict[str, Any]:
    """Aggregate counts across the gtm_outbound graph."""
    counts = {}
    for label in ("Recipient", "Send", "Hypothesis", "CoachingEvent", "Outcome", "Principle", "SubSegment", "Company"):
        try:
            r = _query(f"MATCH (n:{label}) RETURN count(n)")
            counts[label.lower()] = r["rows"][0][0] if r["rows"] else 0
        except Exception:
            counts[label.lower()] = -1
    return {"graph": GRAPH_NAME, "counts": counts}
