"""config.yaml's 16 entity types must actually reach the extractor.

The file's own header has always said "These define what Graphiti extracts from
episodes." They did not: `_load_entity_types` had ZERO callers and
`add_episode` was never passed an `entity_types` argument, so every episode
since the beginning was extracted untyped while the platform carried a
16-type domain taxonomy that looked configured.
"""

from datetime import datetime, timezone

import pydantic
import pytest

from app.services import graphiti_client


@pytest.fixture(autouse=True)
def _fresh():
    graphiti_client.reset_entity_type_models()
    yield
    graphiti_client.reset_entity_type_models()


def test_the_taxonomy_becomes_model_classes():
    models = graphiti_client.entity_type_models()
    assert len(models) == 16, "all sixteen configured types must be present"
    assert all(issubclass(m, pydantic.BaseModel) for m in models.values())


def test_the_yaml_description_becomes_the_docstring():
    """graphiti shows the extractor each class's docstring, not its fields, so
    a description stored anywhere else would never reach the model."""
    assert "friction" in (graphiti_client.entity_type_models()["PainPoint"].__doc__ or "")


def test_it_is_built_once_not_per_episode():
    """Rebuilding 16 classes and re-reading YAML on every ingest is waste on the
    hottest path there is."""
    assert graphiti_client.entity_type_models() is graphiti_client.entity_type_models()


def test_a_broken_taxonomy_degrades_to_untyped_rather_than_failing(monkeypatch):
    """A malformed config must not stop ingestion. Untyped extraction is
    exactly the behaviour that shipped for months; a failed ingest is not."""
    monkeypatch.setattr(graphiti_client, "_load_entity_types", lambda: [])
    assert graphiti_client.entity_type_models() == {}


def test_entries_without_a_name_are_skipped(monkeypatch):
    monkeypatch.setattr(
        graphiti_client,
        "_load_entity_types",
        lambda: [{"description": "no name"}, {"name": "Real", "description": "d"}],
    )
    assert list(graphiti_client.entity_type_models()) == ["Real"]


def test_add_episode_passes_the_taxonomy_to_the_extractor(monkeypatch):
    """The whole point. Without this argument the types are decoration."""
    seen: dict = {}

    class _Client:
        async def add_episode(self, **kwargs):
            seen.update(kwargs)
            class _R:
                nodes: list = []
                edges: list = []
            return _R()

    async def _get_client(_slug):
        return _Client()

    monkeypatch.setattr(graphiti_client, "get_client", _get_client)

    import asyncio

    asyncio.run(
        graphiti_client.add_episode(
            client_slug="pokagon",
            engagement_id="eng-1",
            name="n",
            content="c",
            source_description="s",
            reference_time=datetime.now(timezone.utc),
        )
    )
    assert "entity_types" in seen, "extraction ran untyped"
    assert len(seen["entity_types"]) == 16
