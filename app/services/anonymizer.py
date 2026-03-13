"""PII anonymization for segment knowledge promotion.

Implements Draft → Review → Promote workflow (HITL).
This service generates draft anonymized insights — it does NOT auto-promote.
A consultant must review and approve in the SDI dashboard before promotion.
"""

import logging
import re

logger = logging.getLogger("graphiti_service")


def anonymize_content(
    content: str,
    client_name: str | None = None,
    known_names: list[str] | None = None,
) -> str:
    """Strip PII from content to create a draft segment insight.

    This is a best-effort automated pass. Human review is REQUIRED before
    the insight enters the segment knowledge graph.

    Args:
        content: Raw insight text
        client_name: Client organization name to replace
        known_names: List of known person names to replace with role labels
    """
    result = content

    # Replace client name with industry label
    if client_name:
        result = re.sub(
            re.escape(client_name),
            "the gaming commission",
            result,
            flags=re.IGNORECASE,
        )

    # Replace known person names with role placeholder
    if known_names:
        for name in known_names:
            result = re.sub(
                re.escape(name),
                "[stakeholder]",
                result,
                flags=re.IGNORECASE,
            )

    # Strip email addresses
    result = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[email]",
        result,
    )

    # Strip phone numbers (US format)
    result = re.sub(
        r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[phone]",
        result,
    )

    # Replace specific dollar amounts with ranges
    result = re.sub(
        r"\$[\d,]+(?:\.\d{2})?",
        "[amount]",
        result,
    )

    # Replace specific dates with relative references
    result = re.sub(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "[date]",
        result,
    )

    return result


def generate_draft_insights(
    findings: list[dict],
    client_name: str | None = None,
    known_names: list[str] | None = None,
) -> list[dict]:
    """Generate draft anonymized insights from engagement findings.

    Returns drafts for human review — NOT for direct promotion.
    """
    drafts = []

    for finding in findings:
        title = finding.get("title", "")
        content = finding.get("content", "")
        finding_type = finding.get("type", "")

        anonymized_title = anonymize_content(title, client_name, known_names)
        anonymized_content = anonymize_content(content, client_name, known_names)

        drafts.append(
            {
                "original_title": title,
                "original_content": content,
                "draft_title": anonymized_title,
                "draft_content": anonymized_content,
                "finding_type": finding_type,
            }
        )

    logger.info(f"[graphiti] Generated {len(drafts)} draft segment insights")
    return drafts
