"""Tests for the isotope envelope API, including a brute-force cross-check."""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def brute_force_peaks(payload: dict) -> dict[int, Fraction]:
    """Enumerate every isotope-choice path explicitly (no merging shortcuts)."""
    dist: dict[int, Fraction] = {}
    per_group_outcomes = []
    for group in payload["atom_groups"]:
        outcomes = []
        for combo in itertools.product(group["isotopes"], repeat=group["count"]):
            shift = sum(iso["shift_mda"] for iso in combo)
            prob = Fraction(1)
            for iso in combo:
                n, d = iso["abundance"].split("/")
                prob *= Fraction(int(n), int(d))
            outcomes.append((shift, prob))
        per_group_outcomes.append(outcomes)
    for path in itertools.product(*per_group_outcomes):
        shift = sum(item[0] for item in path)
        prob = Fraction(1)
        for item in path:
            prob *= item[1]
        dist[shift] = dist.get(shift, Fraction(0)) + prob
    return {s: p for s, p in dist.items() if p != 0}


def parse_prob(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def test_basic_single_group_exact_fractions():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "3/4"},
                    {"shift_mda": 1, "abundance": "1/4"},
                ],
            }
        ],
        "k": 2,
    }
    resp = client.post("/v1/envelope", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # (3/4 + x/4)^2 = 9/16 + 6/16 x + 1/16 x^2, probabilities reduced.
    assert body["peaks"] == [
        {"shift_mda": 0, "probability": "9/16"},
        {"shift_mda": 1, "probability": "3/8"},
        {"shift_mda": 2, "probability": "1/16"},
    ]
    assert body["top_peaks"] == [
        {"shift_mda": 0, "probability": "9/16"},
        {"shift_mda": 1, "probability": "3/8"},
    ]


def test_multi_group_matches_brute_force():
    payload = {
        "atom_groups": [
            {
                "count": 3,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "7/10"},
                    {"shift_mda": 1, "abundance": "1/5"},
                    {"shift_mda": 3, "abundance": "1/10"},
                ],
            },
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "2/3"},
                    {"shift_mda": 2, "abundance": "1/3"},
                ],
            },
        ],
        "k": 5,
    }
    resp = client.post("/v1/envelope", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    expected = brute_force_peaks(payload)
    got = {p["shift_mda"]: parse_prob(p["probability"]) for p in body["peaks"]}
    assert got == expected

    shifts = [p["shift_mda"] for p in body["peaks"]]
    assert shifts == sorted(shifts)

    # Probabilities are fully reduced.
    for peak in body["peaks"]:
        n, d = peak["probability"].split("/")
        assert Fraction(int(n), int(d)).denominator == int(d)

    # Top-k ordering: probability desc, ties by shift asc.
    ranked = sorted(expected.items(), key=lambda item: (-item[1], item[0]))[:5]
    assert body["top_peaks"] == [
        {"shift_mda": s, "probability": f"{p.numerator}/{p.denominator}"}
        for s, p in ranked
    ]


def test_top_peaks_tie_break_and_k_limit():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/2"},
                ],
            }
        ],
        "k": 20,
    }
    resp = client.post("/v1/envelope", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # Probabilities 1/4, 1/2, 1/4 -> tie between shifts 0 and 2, shift asc.
    assert [p["shift_mda"] for p in body["top_peaks"]] == [1, 0, 2]
    # k is capped by the number of existing peaks.
    assert len(body["top_peaks"]) == 3


def test_zero_abundance_isotope_is_allowed():
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/1"},
                    {"shift_mda": 5, "abundance": "0/1"},
                ],
            }
        ],
        "k": 1,
    }
    resp = client.post("/v1/envelope", json=payload)
    assert resp.status_code == 200
    assert resp.json()["peaks"] == [{"shift_mda": 0, "probability": "1/1"}]


@pytest.mark.parametrize(
    "mutation",
    [
        # abundance format / range violations
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="1/2/3"),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="a/b"),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="1/0"),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="1/1001"),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="-1/2"),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance=0.5),
        # abundances do not sum to exactly 1
        lambda p: p["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        # duplicate shift within a group
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        # negative / non-integer shift
        lambda p: p["atom_groups"][0]["isotopes"][0].update(shift_mda=-1),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(shift_mda=1.5),
        # count out of range
        lambda p: p["atom_groups"][0].update(count=0),
        lambda p: p["atom_groups"][0].update(count=41),
        # isotope list size out of range
        lambda p: p["atom_groups"][0].update(
            isotopes=[{"shift_mda": 0, "abundance": "1/1"}]
        ),
        lambda p: p["atom_groups"][0]["isotopes"].append(
            {"shift_mda": 9, "abundance": "0/1"}
        )
        or p["atom_groups"][0]["isotopes"].append(
            {"shift_mda": 10, "abundance": "0/1"}
        ),
        # k out of range
        lambda p: p.update(k=0),
        lambda p: p.update(k=21),
        # too many groups
        lambda p: p.update(atom_groups=p["atom_groups"] * 5),
        # total count > 80
        lambda p: p.update(
            atom_groups=[dict(p["atom_groups"][0], count=40)] * 3
        ),
        # max possible total shift > 2000
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=2001),
        # unknown field
        lambda p: p.update(extra=1),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/2"},
                ],
            }
        ],
        "k": 3,
    }
    mutation(payload)
    resp = client.post("/v1/envelope", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/envelope",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
