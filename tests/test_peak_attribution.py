"""Tests for the peak-attribution endpoint, including a brute-force cross-check."""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def brute_force_attribution(payload: dict, target: int):
    """Enumerate every isotope-choice path, conditioned on the target shift.

    Returns (probability, expectations) where expectations is group-major and
    isotope-input-order, exactly like the API contract.
    """
    groups = payload["atom_groups"]
    per_group_atoms = []
    for group in groups:
        atoms = [
            [
                (j, iso["shift_mda"], parse_fraction(iso["abundance"]))
                for j, iso in enumerate(group["isotopes"])
            ]
            for _ in range(group["count"])
        ]
        per_group_atoms.append(atoms)

    total_prob = Fraction(0)
    # moments[g][j]: summed probability-weighted count of isotope j in group g.
    moments = [
        [Fraction(0)] * len(group["isotopes"]) for group in groups
    ]

    # Build a flat list of (group_index, isotope_index, shift, prob) choices.
    flat = [
        [(g, j, shift, prob) for j, shift, prob in atom]
        for g, atoms in enumerate(per_group_atoms)
        for atom in atoms
    ]
    for choices in itertools.product(*flat):
        shift = sum(choice[2] for choice in choices)
        if shift != target:
            continue
        prob = Fraction(1)
        for *_, choice_prob in choices:
            prob *= choice_prob
        total_prob += prob
        for g, j, _, _ in choices:
            moments[g][j] += prob

    if total_prob == 0:
        return Fraction(0), None
    expectations = [
        [moment / total_prob for moment in group_moments]
        for group_moments in moments
    ]
    return total_prob, expectations


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def basic_payload():
    return {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "3/4"},
                    {"shift_mda": 1, "abundance": "1/4"},
                ],
            }
        ],
        "target_shift_mda": 1,
    }


def test_basic_single_group():
    resp = client.post("/v1/peak-attribution", json=basic_payload())
    assert resp.status_code == 200
    body = resp.json()
    # Shift 1: one light + one heavy, probability 2 * 3/4 * 1/4 = 3/8;
    # conditioned on that, exactly one atom of each isotope.
    assert body == {
        "reachable": True,
        "probability": "3/8",
        "expected_counts": [["1/1", "1/1"]],
    }


def test_extreme_peaks_pin_all_atoms():
    for target, expected in [(0, ["2/1", "0/1"]), (2, ["0/1", "2/1"])]:
        payload = basic_payload()
        payload["target_shift_mda"] = target
        resp = client.post("/v1/peak-attribution", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {
            "reachable": True,
            "probability": {"0": "9/16", "2": "1/16"}[str(target)],
            "expected_counts": [expected],
        }


def test_unreachable_shift():
    payload = basic_payload()
    payload["target_shift_mda"] = 3
    resp = client.post("/v1/peak-attribution", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"reachable": False, "probability": "0/1"}
    assert "expected_counts" not in resp.json()


def test_zero_abundance_isotope_expectation_is_zero():
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
        "target_shift_mda": 0,
    }
    resp = client.post("/v1/peak-attribution", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "reachable": True,
        "probability": "1/1",
        "expected_counts": [["1/1", "0/1"]],
    }


def test_multi_group_ternary_matches_brute_force():
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
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 7, "abundance": "1/2"},
                ],
            },
        ]
    }
    max_shift = 3 * 3 + 2 * 2 + 7
    for target in range(0, max_shift + 1):
        resp = client.post(
            "/v1/peak-attribution",
            json={**payload, "target_shift_mda": target},
        )
        assert resp.status_code == 200
        body = resp.json()
        prob, expectations = brute_force_attribution(payload, target)

        if prob == 0:
            assert body == {"reachable": False, "probability": "0/1"}
            continue

        assert body["reachable"] is True
        assert parse_fraction(body["probability"]) == prob
        assert body["probability"] == fmt(prob)
        got = [
            [parse_fraction(value) for value in row]
            for row in body["expected_counts"]
        ]
        assert got == expectations

        # Every expectation is a reduced fraction...
        for row in body["expected_counts"]:
            for value in row:
                n, d = value.split("/")
                assert Fraction(int(n), int(d)).denominator == int(d)
        # ...each group's expectations sum to that group's atom count,
        # and expectations lie in [0, count].
        for g, group in enumerate(payload["atom_groups"]):
            assert sum(got[g]) == group["count"]
            for value in got[g]:
                assert 0 <= value <= group["count"]


def test_reachable_probability_matches_envelope_peak():
    payload = {
        "atom_groups": [
            {
                "count": 4,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "3/5"},
                    {"shift_mda": 2, "abundance": "2/5"},
                ],
            },
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "5/8"},
                    {"shift_mda": 1, "abundance": "3/8"},
                ],
            },
        ]
    }
    envelope = client.post("/v1/envelope", json={**payload, "k": 20}).json()
    probs = {peak["shift_mda"]: peak["probability"] for peak in envelope["peaks"]}
    for target in range(0, 4 * 2 + 2 * 1 + 1):
        body = client.post(
            "/v1/peak-attribution",
            json={**payload, "target_shift_mda": target},
        ).json()
        if target in probs:
            assert body["reachable"] is True
            assert body["probability"] == probs[target]
        else:
            assert body == {"reachable": False, "probability": "0/1"}


@pytest.mark.parametrize(
    "mutation",
    [
        # target_shift_mda out of range / wrong type
        lambda p: p.update(target_shift_mda=-1),
        lambda p: p.update(target_shift_mda=2001),
        lambda p: p.update(target_shift_mda=1.5),
        lambda p: p.update(target_shift_mda="1"),
        lambda p: p.pop("target_shift_mda"),
        # unknown field
        lambda p: p.update(extra=1),
        # atom_groups violations reused verbatim from /v1/envelope
        lambda p: p["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        lambda p: p["atom_groups"][0].update(count=0),
        lambda p: p.update(atom_groups=[]),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=2001),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/peak-attribution", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/peak-attribution",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
