"""Tests for the fragment-spectrum endpoint, including a brute-force cross-check."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction
from math import comb

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def brute_force_all_targets(payload: dict):
    """Enumerate every (isotope assignment x fragment subset) path explicitly.

    Returns (parent_probs, joints): parent_probs maps the parent shift to its
    exact probability, and joints maps each parent shift to the joint
    (parent, fragment) distribution over fragment shifts. The conditional
    fragment law for a target is joints[target] normalized by
    parent_probs[target].
    """
    groups = payload["atom_groups"]
    frag_counts = payload["fragment_counts"]

    # Per group, all (parent_shift, frag_shift, probability) outcomes.
    per_group = []
    for group, frag_count in zip(groups, frag_counts):
        count = group["count"]
        entries = []
        for combo in itertools.product(group["isotopes"], repeat=count):
            shifts = [iso["shift_mda"] for iso in combo]
            prob = Fraction(1)
            for iso in combo:
                prob *= parse_fraction(iso["abundance"])
            # Every subset of frag_count atoms is equally likely.
            for subset in itertools.combinations(range(count), frag_count):
                frag = sum(shifts[i] for i in subset)
                entries.append((sum(shifts), frag, prob / comb(count, frag_count)))
        per_group.append(entries)

    parent_probs: dict[int, Fraction] = {}
    joints: dict[int, dict[int, Fraction]] = {}
    for path in itertools.product(*per_group):
        parent = sum(entry[0] for entry in path)
        frag = sum(entry[1] for entry in path)
        weight = Fraction(1)
        for entry in path:
            weight *= entry[2]
        parent_probs[parent] = parent_probs.get(parent, Fraction(0)) + weight
        frag_dist = joints.setdefault(parent, {})
        frag_dist[frag] = frag_dist.get(frag, Fraction(0)) + weight
    return parent_probs, joints


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
        "fragment_counts": [1],
        "target_parent_shift_mda": 1,
    }


def test_basic_single_group_hand_computed():
    resp = client.post("/v1/fragment-spectrum", json=basic_payload())
    assert resp.status_code == 200
    # Parent shift 1: one light + one heavy atom, probability 2*3/4*1/4 = 3/8.
    # Conditioned on that, the single drawn atom is light or heavy with
    # probability 1/2 each, so the fragment shift is 0 or 1 equally.
    assert resp.json() == {
        "reachable": True,
        "probability": "3/8",
        "peaks": [
            {"shift_mda": 0, "probability": "1/2"},
            {"shift_mda": 1, "probability": "1/2"},
        ],
    }


def test_draw_none_pins_fragment_to_zero():
    payload = basic_payload()
    payload["fragment_counts"] = [0]
    for target, probability in [(0, "9/16"), (1, "3/8"), (2, "1/16")]:
        payload["target_parent_shift_mda"] = target
        resp = client.post("/v1/fragment-spectrum", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {
            "reachable": True,
            "probability": probability,
            "peaks": [{"shift_mda": 0, "probability": "1/1"}],
        }


def test_draw_all_pins_fragment_to_parent():
    payload = basic_payload()
    payload["fragment_counts"] = [2]
    for target, probability in [(0, "9/16"), (1, "3/8"), (2, "1/16")]:
        payload["target_parent_shift_mda"] = target
        resp = client.post("/v1/fragment-spectrum", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {
            "reachable": True,
            "probability": probability,
            "peaks": [{"shift_mda": target, "probability": "1/1"}],
        }


def test_unreachable_parent_shift():
    payload = basic_payload()
    payload["target_parent_shift_mda"] = 3
    resp = client.post("/v1/fragment-spectrum", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"reachable": False, "probability": "0/1"}
    assert "peaks" not in resp.json()


def test_zero_abundance_isotope_never_drawn():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/1"},
                    {"shift_mda": 5, "abundance": "0/1"},
                ],
            }
        ],
        "fragment_counts": [1],
        "target_parent_shift_mda": 0,
    }
    resp = client.post("/v1/fragment-spectrum", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "reachable": True,
        "probability": "1/1",
        "peaks": [{"shift_mda": 0, "probability": "1/1"}],
    }


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
        "fragment_counts": [2, 1],
    }
    parent_probs, joints = brute_force_all_targets(payload)
    max_shift = 3 * 3 + 2 * 2
    for target in range(0, max_shift + 2):
        resp = client.post(
            "/v1/fragment-spectrum",
            json={**payload, "target_parent_shift_mda": target},
        )
        assert resp.status_code == 200
        body = resp.json()
        parent_prob = parent_probs.get(target, Fraction(0))
        if parent_prob == 0:
            assert body == {"reachable": False, "probability": "0/1"}
            continue
        assert body["reachable"] is True
        assert body["probability"] == fmt(parent_prob)
        expected = {
            frag: weight / parent_prob
            for frag, weight in joints[target].items()
        }
        got = {
            peak["shift_mda"]: parse_fraction(peak["probability"])
            for peak in body["peaks"]
        }
        assert got == expected
        # Non-zero peaks only, ascending by shift, reduced fractions.
        shifts = [peak["shift_mda"] for peak in body["peaks"]]
        assert shifts == sorted(shifts)
        for peak in body["peaks"]:
            assert peak["probability"] == fmt(parse_fraction(peak["probability"]))
            assert parse_fraction(peak["probability"]) > 0
        # The conditional distribution is normalized.
        assert sum(got.values(), Fraction(0)) == 1


def test_randomized_matches_brute_force():
    rng = random.Random(20260926)
    for _ in range(40):
        group_count = rng.randint(1, 2)
        groups = []
        frag_counts = []
        for _ in range(group_count):
            count = rng.randint(1, 3)
            kind = rng.randint(2, 3)
            shifts = rng.sample(range(0, 5), kind)
            cuts = sorted(rng.sample(range(1, 12), kind - 1))
            abundances = [
                Fraction(b - a, 12) for a, b in zip([0] + cuts, cuts + [12])
            ]
            groups.append(
                {
                    "count": count,
                    "isotopes": [
                        {"shift_mda": shift, "abundance": fmt(abundance)}
                        for shift, abundance in zip(shifts, abundances)
                    ],
                }
            )
            frag_counts.append(rng.randint(0, count))
        payload = {"atom_groups": groups, "fragment_counts": frag_counts}
        parent_probs, joints = brute_force_all_targets(payload)
        max_shift = sum(
            group["count"] * max(iso["shift_mda"] for iso in group["isotopes"])
            for group in groups
        )
        for target in range(0, max_shift + 2):
            resp = client.post(
                "/v1/fragment-spectrum",
                json={**payload, "target_parent_shift_mda": target},
            )
            assert resp.status_code == 200
            body = resp.json()
            parent_prob = parent_probs.get(target, Fraction(0))
            if parent_prob == 0:
                assert body == {"reachable": False, "probability": "0/1"}
                continue
            assert body["reachable"] is True
            assert body["probability"] == fmt(parent_prob)
            expected = {
                frag: weight / parent_prob
                for frag, weight in joints[target].items()
            }
            got = {
                peak["shift_mda"]: parse_fraction(peak["probability"])
                for peak in body["peaks"]
            }
            assert got == expected
            assert sum(got.values(), Fraction(0)) == 1


def test_parent_probability_matches_envelope_peaks():
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
        ],
        "fragment_counts": [3, 1],
    }
    envelope = client.post(
        "/v1/envelope", json={"atom_groups": payload["atom_groups"], "k": 20}
    ).json()
    probs = {peak["shift_mda"]: peak["probability"] for peak in envelope["peaks"]}
    for target in range(0, 4 * 2 + 2 * 1 + 1):
        body = client.post(
            "/v1/fragment-spectrum",
            json={**payload, "target_parent_shift_mda": target},
        ).json()
        if target in probs:
            assert body["reachable"] is True
            assert body["probability"] == probs[target]
        else:
            assert body == {"reachable": False, "probability": "0/1"}


def test_supports_max_scale_atom_groups():
    # Largest admissible input: total count 80, max total shift 2000. The
    # parent shift is 25 * (heavy atoms out of 80); conditioning on parent
    # shift 1000 fixes exactly 40 heavy atoms, and drawing 20 + 20 atoms
    # without replacement gives a hypergeometric heavy count in the fragment:
    # P(frag = 25k) = C(40, k) * C(40, 40 - k) / C(80, 40) = C(40, k)^2 / C(80, 40).
    payload = {
        "atom_groups": [
            {
                "count": 40,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 25, "abundance": "1/2"},
                ],
            },
            {
                "count": 40,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 25, "abundance": "1/2"},
                ],
            },
        ],
        "fragment_counts": [20, 20],
        "target_parent_shift_mda": 1000,
    }
    resp = client.post("/v1/fragment-spectrum", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["probability"] == fmt(Fraction(comb(80, 40), 2**80))
    expected = {
        25 * k: Fraction(comb(40, k) ** 2, comb(80, 40)) for k in range(41)
    }
    got = {
        peak["shift_mda"]: parse_fraction(peak["probability"])
        for peak in body["peaks"]
    }
    assert got == expected
    assert [peak["shift_mda"] for peak in body["peaks"]] == list(range(0, 1001, 25))


@pytest.mark.parametrize(
    "mutation",
    [
        # fragment_counts missing / wrong length / out of range / wrong type
        lambda p: p.pop("fragment_counts"),
        lambda p: p.update(fragment_counts=[]),
        lambda p: p.update(fragment_counts=[1, 1]),
        lambda p: p.update(fragment_counts=[3]),
        lambda p: p.update(fragment_counts=[-1]),
        lambda p: p.update(fragment_counts=[1.0]),
        lambda p: p.update(fragment_counts=[1.5]),
        lambda p: p.update(fragment_counts=["1"]),
        lambda p: p.update(fragment_counts=[True]),
        lambda p: p.update(fragment_counts=[None]),
        lambda p: p.update(fragment_counts=1),
        # target_parent_shift_mda out of range / wrong type / missing
        lambda p: p.update(target_parent_shift_mda=-1),
        lambda p: p.update(target_parent_shift_mda=2001),
        lambda p: p.update(target_parent_shift_mda=1.5),
        lambda p: p.update(target_parent_shift_mda="1"),
        lambda p: p.pop("target_parent_shift_mda"),
        # unknown field
        lambda p: p.update(extra=1),
        # atom_groups violations reused verbatim from /v1/envelope
        lambda p: p["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        lambda p: p["atom_groups"][0].update(count=0),
        lambda p: p.update(atom_groups=[]),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/fragment-spectrum", json=payload)
    assert resp.status_code == 422


def test_mismatched_fragment_counts_length_returns_422():
    payload = basic_payload()
    payload["atom_groups"].append(
        {
            "count": 1,
            "isotopes": [
                {"shift_mda": 0, "abundance": "1/2"},
                {"shift_mda": 1, "abundance": "1/2"},
            ],
        }
    )
    # Two groups but still one fragment count.
    resp = client.post("/v1/fragment-spectrum", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/fragment-spectrum",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
