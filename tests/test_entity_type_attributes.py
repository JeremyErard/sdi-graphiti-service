"""People entity types carry the platform user id and title as extractable attributes.

Ruled 2026-09-11: a person is recorded as name, title on record and platform
user id, the id being the durable key. The rehearsal on a scratch graph showed
the extractor keeps names and titles in summaries but drops the id, because the
taxonomy had no field for it. A field makes it a stored attribute.
"""

from pydantic import BaseModel

from app.services import graphiti_client


def _fresh_models(monkeypatch, entries):
    monkeypatch.setattr(graphiti_client, "_entity_type_models", None)
    monkeypatch.setattr(graphiti_client, "_load_entity_types", lambda: entries)
    try:
        return graphiti_client.entity_type_models()
    finally:
        monkeypatch.setattr(graphiti_client, "_entity_type_models", None)


def test_attributes_become_optional_string_fields_with_their_descriptions(monkeypatch):
    models = _fresh_models(monkeypatch, [
        {"name": "Stakeholder", "description": "Client staff", "attributes": [
            {"name": "platform_user_id", "description": "the durable key"},
            {"name": "title", "description": "job title"},
        ]},
        {"name": "Department", "description": "A unit"},
    ])
    stakeholder = models["Stakeholder"]
    assert issubclass(stakeholder, BaseModel)
    assert stakeholder.__doc__ == "Client staff"
    assert set(stakeholder.model_fields) == {"platform_user_id", "title"}
    assert stakeholder.model_fields["platform_user_id"].description == "the durable key"
    assert stakeholder(platform_user_id="cmlx1", title=None).platform_user_id == "cmlx1"
    assert stakeholder().platform_user_id is None
    assert models["Department"].model_fields == {}


def test_malformed_attribute_entries_are_skipped_not_fatal(monkeypatch):
    models = _fresh_models(monkeypatch, [
        {"name": "Stakeholder", "description": "Client staff", "attributes": [
            {"name": "not an identifier"}, {"description": "no name"}, None, {"name": "ok"},
        ]},
    ])
    assert set(models["Stakeholder"].model_fields) == {"ok"}


def test_the_shipped_taxonomy_gives_people_the_platform_user_id(monkeypatch):
    monkeypatch.setattr(graphiti_client, "_entity_type_models", None)
    try:
        models = graphiti_client.entity_type_models()
    finally:
        monkeypatch.setattr(graphiti_client, "_entity_type_models", None)
    assert "platform_user_id" in models["Stakeholder"].model_fields
    assert "title" in models["Stakeholder"].model_fields
    assert "platform_user_id" in models["Consultant"].model_fields
    assert len(models) >= 16
