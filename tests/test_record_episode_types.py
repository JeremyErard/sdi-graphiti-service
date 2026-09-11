"""The service accepts the client record's own history as episodes.

Ruled 2026-09-11: the Phase 1 approval and feedback record lives in Engage
and must reach the graph as attributed episodes. Two types, so client-facing
generators can exclude commentary while keeping approval history.
"""

import pydantic
import pytest

from app.models.episode import EpisodeAnchorMode, EpisodeType, IngestEpisodeRequest
from app.provenance_contract import EPISODE_PROVENANCE_CONTRACT_VERSION


def _request(episode_type: str) -> dict:
    return {
        "client_slug": "pokagon",
        "engagement_id": "eng-phase-1",
        "episode_type": episode_type,
        "content": "Approval record: Table Fill Inspection. Map version 2 approved 2026-05-02 by Beverly Robb (Director of Compliance).",
        "source_id": "wf-1",
        "source_type": "process",
        "anchor_mode": "typed_source",
        "producer_contract_version": EPISODE_PROVENANCE_CONTRACT_VERSION,
        "metadata": {"workflowId": "wf-1", "part": 1, "partCount": 1},
    }


@pytest.mark.parametrize(
    ("value", "member"),
    [("approval_history", EpisodeType.APPROVAL_HISTORY), ("review_commentary", EpisodeType.REVIEW_COMMENTARY)],
)
def test_record_types_are_typed_source_episodes_anchored_on_their_workflow(value, member):
    req = IngestEpisodeRequest(**_request(value))
    assert req.episode_type is member
    assert req.episode_type.value == value
    assert req.anchor_mode is EpisodeAnchorMode.TYPED_SOURCE
    assert req.source_type == "process"


def test_the_enum_stays_closed_around_the_new_members():
    with pytest.raises(pydantic.ValidationError) as caught:
        IngestEpisodeRequest(**_request("approval_record"))
    assert caught.value.errors()[0]["loc"] == ("episode_type",)
