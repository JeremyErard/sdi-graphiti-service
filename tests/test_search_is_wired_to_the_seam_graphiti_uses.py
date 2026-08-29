"""The overrides must sit on the seam graphiti actually calls.

Both the Entity vector index and the bounded fulltext query were implemented on
driver.search_ops. graphiti's search_utils never consults search_ops -- the
string does not appear in that module once. It checks driver.search_interface
and otherwise runs its own inline query. So both shipped, changed nothing, and
FalkorDB's slowlog kept reporting the same unbounded 588s fulltext query
afterwards.

Patching the module functions is not an alternative: callers do
`from ...search_utils import node_similarity_search` and call it bare, so a
module-attribute patch never reaches their bound reference.
"""

import inspect

from graphiti_core.driver.search_interface.search_interface import SearchInterface

from app.services.indexed_falkor import SearchOpsInterface


def _interface_methods():
    return [
        n for n, v in vars(SearchInterface).items()
        if not n.startswith("_") and inspect.isfunction(v)
    ]


def test_every_search_method_is_implemented_except_the_one_with_a_fallback():
    """Setting search_interface is all-or-nothing: any method left inherited
    raises NotImplementedError the moment graphiti calls it.

    get_embeddings_for_communities is the sole exception -- graphiti wraps that
    call in try/except NotImplementedError and falls back to its own query.
    """
    inherited = [
        n for n in _interface_methods()
        if getattr(SearchOpsInterface, n) is getattr(SearchInterface, n)
    ]
    assert inherited == ["get_embeddings_for_communities"], (
        f"these would raise in production: {inherited}"
    )


def test_the_search_calls_route_to_search_ops():
    """Delegation must actually reach the driver's ops, not just exist."""
    seen = {}

    class _Ops:
        async def edge_fulltext_search(self, executor, *a, **kw):
            seen["edge"] = (executor, a)
            return ["edge-result"]

        async def node_similarity_search(self, executor, *a, **kw):
            seen["node"] = (executor, a)
            return ["node-result"]

    class _Driver:
        search_ops = _Ops()

    import asyncio

    d = _Driver()
    iface = SearchOpsInterface()
    assert asyncio.run(iface.edge_fulltext_search(d, "q", None, ["g"], 10)) == ["edge-result"]
    assert asyncio.run(iface.node_similarity_search(d, [0.1], None, ["g"], 10)) == ["node-result"]
    # the driver is passed through as the executor -- search_ops takes one
    assert seen["edge"][0] is d
    assert seen["node"][0] is d


def test_search_utils_consults_search_interface_and_never_search_ops():
    """Locks the reason this shim exists. If a future graphiti starts using
    search_ops, this fails and the shim can go."""
    from graphiti_core.search import search_utils

    src = inspect.getsource(search_utils)
    assert "search_ops" not in src, "graphiti now uses search_ops; re-check this wiring"
    assert "driver.search_interface" in src
