"""Collision-free names for tenant and segment graphs."""

import re

_CLIENT_SLUG = re.compile(r"^[a-z0-9-]+$")

# These names are an immutable compatibility boundary discovered in the July 9
# read-only live inventory. Both tenants already contain graph data under the
# pre-v2 mapper, which dropped hyphens. Never silently point either tenant at a
# new empty graph. A future slug equal to one of the collapsed spellings is sent
# to the uppercase V2 namespace so the mapping remains injective.
LEGACY_TENANT_GRAPH_NAMES = {
    "michigan-restaurant-lodging-association": (
        "client_michiganrestaurantlodgingassociation"
    ),
    "test-provision": "client_testprovision",
}
_RESERVED_LEGACY_GRAPH_NAMES = frozenset(LEGACY_TENANT_GRAPH_NAMES.values())


def graph_name_for_client(client_slug: str) -> str:
    """Losslessly encode the platform's ``[a-z0-9-]+`` tenant slug grammar.

    The two inventoried live hyphen tenants retain their exact populated legacy
    graph names. Underscores are not valid platform slug characters, so new
    hyphenated slugs use an injective underscore encoding. A no-hyphen slug that
    would collide with a reserved legacy name is isolated in an uppercase
    ``V2`` namespace that no valid lowercase slug can otherwise produce.
    """
    if not isinstance(client_slug, str):
        raise ValueError("client_slug must be a string")
    if not _CLIENT_SLUG.fullmatch(client_slug):
        raise ValueError("client_slug must match [a-z0-9-]+")
    legacy = LEGACY_TENANT_GRAPH_NAMES.get(client_slug)
    if legacy:
        return legacy

    legacy_shape = f"client_{client_slug}"
    if legacy_shape in _RESERVED_LEGACY_GRAPH_NAMES:
        return f"client_V2_{client_slug}"
    return f"client_{client_slug.replace('-', '_')}"


def segment_graph_name(industry: str) -> str:
    """Map an internal industry identifier to a safe segment graph name."""
    safe_industry = "".join(c for c in industry if c.isalnum() or c == "_").lower()
    if not safe_industry:
        raise ValueError("industry must contain at least one alphanumeric character")
    return f"segment_{safe_industry}"
