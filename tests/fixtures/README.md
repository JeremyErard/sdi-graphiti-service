# Graphiti provenance contract fixtures

`graphiti_search_context_v3.json` is the canonical cross-repository response
fixture. `tests/test_provenance_search_contract.py` constructs it through the
actual Pydantic response model and compares deterministic, sorted JSON bytes;
normal test imports never rewrite the fixture.

- Response contract: `graphiti_search_context_v3`
- Summary contract: `graphiti_provenance_summary_v1`
- Structured source producer contract: `structured_provenance_v2`
- SHA-256: `41bb8316a1dec2fa3b11eacda5378c42eab1bf97589790cda7f48167f54c6414`

`graphiti_search_context_shadow_v1.json` freezes the non-enforcing rolling
shape. Its top-level `facts` remain the legacy contract; only the additive
`provenance_shadow` member carries v3-safe facts and algebra.

- Shadow contract: `graphiti_provenance_shadow_v1`
- `enforcement_applied`: `false`
- SHA-256: `8936dae2746e057b35c17953f7fdca205e9e94ce9f14de86252bd44ba18290cc`
