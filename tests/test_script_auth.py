"""Operator scripts must use the same signature contract as the service."""

from scripts.graphiti_http import signed_headers


def test_script_signature_matches_cross_language_vector(monkeypatch):
    monkeypatch.setenv("GRAPHITI_AUTH_MODE", "required")
    monkeypatch.setenv(
        "GRAPHITI_SEARCH_SECRET",
        "search-secret-that-is-at-least-32-characters",
    )
    headers = signed_headers(
        "/search/context",
        b'{"client_slug":"pokagon","value":"preserved"}',
        "pokagon",
        timestamp=1750000000,
        nonce="00112233445566778899aabbccddeeff",
    )
    assert headers["X-SDI-KG-Scope"] == "search"
    assert headers["X-SDI-KG-Client"] == "pokagon"
    assert headers["X-SDI-KG-Nonce"] == "00112233445566778899aabbccddeeff"
    assert headers["X-SDI-KG-Signature"] == (
        "bf3424f70a5827bacb4e3ba526541317dfbd549540628c98f70e46f4f523016d"
    )


def test_required_script_auth_fails_closed_without_scope_secret(monkeypatch):
    monkeypatch.setenv("GRAPHITI_AUTH_MODE", "required")
    monkeypatch.delenv("GRAPHITI_ADMIN_SECRET", raising=False)
    try:
        signed_headers("/admin/init-graph", b'{}', "*")
    except SystemExit as exc:
        assert "GRAPHITI_ADMIN_SECRET is required" in str(exc)
    else:
        raise AssertionError("required mode accepted an unsigned admin request")
