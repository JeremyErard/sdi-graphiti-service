# Graphiti provenance contract fixtures

`graphiti_search_context_v3.json` is the canonical cross-repository response
fixture. `tests/test_provenance_search_contract.py` constructs it through the
actual Pydantic response model and compares deterministic, sorted JSON bytes;
normal test imports never rewrite the fixture.

- Response contract: `graphiti_search_context_v3`
- Summary contract: `graphiti_provenance_summary_v1`
- Structured source producer contract: `structured_provenance_v2`
- SHA-256: `41bb8316a1dec2fa3b11eacda5378c42eab1bf97589790cda7f48167f54c6414`
