"""Tenant graph names must be injective over every valid platform slug."""

import itertools

import pytest

from app.graph_names import LEGACY_TENANT_GRAPH_NAMES, graph_name_for_client


def test_live_tenants_keep_their_exact_populated_graph_names():
    assert graph_name_for_client("pokagon") == "client_pokagon"
    assert graph_name_for_client("michigan-restaurant-lodging-association") == (
        "client_michiganrestaurantlodgingassociation"
    )
    assert graph_name_for_client("test-provision") == "client_testprovision"


def test_collapsed_spellings_cannot_alias_live_hyphen_tenants():
    assert graph_name_for_client("michiganrestaurantlodgingassociation") == (
        "client_V2_michiganrestaurantlodgingassociation"
    )
    assert graph_name_for_client("testprovision") == "client_V2_testprovision"
    assert graph_name_for_client("testprovision") != graph_name_for_client("test-provision")


def test_hyphen_is_encoded_not_dropped():
    assert graph_name_for_client("acme-inc") == "client_acme_inc"
    assert graph_name_for_client("acme-inc") != graph_name_for_client("acmeinc")


def test_mapping_is_injective_for_a_bounded_exhaustive_valid_slug_set():
    alphabet = "ab01-"
    slugs = [
        "".join(chars)
        for length in range(1, 5)
        for chars in itertools.product(alphabet, repeat=length)
    ]
    slugs.extend(LEGACY_TENANT_GRAPH_NAMES)
    slugs.extend(name.removeprefix("client_") for name in LEGACY_TENANT_GRAPH_NAMES.values())
    names = [graph_name_for_client(slug) for slug in set(slugs)]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("slug", ["", "ACME", "acme_inc", "acme.inc", "../acme", "acme/inc"])
def test_invalid_platform_slug_is_rejected(slug):
    with pytest.raises(ValueError):
        graph_name_for_client(slug)
