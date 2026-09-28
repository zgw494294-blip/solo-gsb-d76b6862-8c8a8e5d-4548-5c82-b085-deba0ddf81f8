"""Tests for the detection-probability endpoint, with a brute-force cross-check."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


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
                prob *= parse_fraction(iso["abundance"])
            outcomes.append((shift, prob))
        per_group_outcomes.append(outcomes)
    for path in itertools.product(*per_group_outcomes):
        shift = sum(item[0] for item in path)
        prob = Fraction(1)
        for item in path:
            prob *= item[1]
        dist[shift] = dist.get(shift, Fraction(0)) + prob
    return {s: p for s, p in dist.items() if p != 0}


def brute_force_detection(payload: dict):
    """Enumerate every ion sampling sequence over the full envelope.

    Returns (joint, marginals): the probability that every window reaches
    its min_hits simultaneously, and each window's own success probability
    in input order.
    """
    dist = brute_force_peaks(payload)
    shifts = sorted(dist)
    windows = payload["windows"]
    joint = Fraction(0)
    marginals = [Fraction(0)] * len(windows)
    for draws in itertools.product(shifts, repeat=payload["ion_count"]):
        prob = Fraction(1)
        for shift in draws:
            prob *= dist[shift]
        hits = [
            sum(1 for shift in draws if window["l"] <= shift <= window["r"])
            for window in windows
        ]
        ok = [
            hit >= window["min_hits"]
            for hit, window in zip(hits, windows)
        ]
        for j, flag in enumerate(ok):
            if flag:
                marginals[j] += prob
        if all(ok):
            joint += prob
    return joint, marginals


def basic_payload():
    # Envelope: 0 -> 9/16, 1 -> 3/8, 2 -> 1/16.
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
        "ion_count": 2,
        "windows": [
            {"l": 0, "r": 1, "min_hits": 1},
            {"l": 1, "r": 2, "min_hits": 1},
        ],
    }


def test_overlapping_windows_hand_computed():
    # Shift 1 lies in both windows, shift 0 only in the first, shift 2 only
    # in the second. Joint fails only on sequences {0,0} (81/256) and
    # {2,2} (1/256): 1 - 82/256 = 87/128.
    resp = client.post("/v1/detection-probability", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "87/128",
        "marginal_probabilities": ["255/256", "175/256"],
    }


def test_identical_windows():
    # Duplicate windows succeed or fail together: joint equals the marginal.
    payload = basic_payload()
    payload["windows"] = [
        {"l": 0, "r": 1, "min_hits": 1},
        {"l": 0, "r": 1, "min_hits": 1},
    ]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "255/256",
        "marginal_probabilities": ["255/256", "255/256"],
    }


def test_disjoint_windows_keep_outside_mass():
    # Shift 1 hits neither window but still absorbs probability 3/8 per ion.
    # Joint needs one ion at 0 and one at 2: 2 * 9/16 * 1/16 = 9/128.
    payload = basic_payload()
    payload["windows"] = [
        {"l": 0, "r": 0, "min_hits": 1},
        {"l": 2, "r": 2, "min_hits": 1},
    ]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "9/128",
        "marginal_probabilities": ["207/256", "31/256"],
    }


def test_min_hits_two_single_window():
    # Both ions must land in [0, 1]: (15/16)^2 = 225/256.
    payload = basic_payload()
    payload["windows"] = [{"l": 0, "r": 1, "min_hits": 2}]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "225/256",
        "marginal_probabilities": ["225/256"],
    }


def test_min_hits_up_to_four():
    # Three ions, all four required in [0, 1]: impossible -> 0/1; with
    # ion_count 4 it becomes (15/16)^4 = 50625/65536.
    payload = basic_payload()
    payload["ion_count"] = 3
    payload["windows"] = [{"l": 0, "r": 1, "min_hits": 4}]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "0/1",
        "marginal_probabilities": ["0/1"],
    }
    payload["ion_count"] = 4
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "50625/65536",
        "marginal_probabilities": ["50625/65536"],
    }


def test_impossible_window_zeroes_joint_only():
    # min_hits 3 with 2 ions can never succeed: that window and the joint
    # are exactly 0/1, the other window keeps its own marginal.
    payload = basic_payload()
    payload["windows"] = [
        {"l": 0, "r": 1, "min_hits": 1},
        {"l": 0, "r": 2, "min_hits": 3},
    ]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "0/1",
        "marginal_probabilities": ["255/256", "0/1"],
    }


def test_window_without_mass_is_impossible():
    # No peak inside [5, 9]: the window can never be hit.
    payload = basic_payload()
    payload["windows"] = [
        {"l": 0, "r": 1, "min_hits": 1},
        {"l": 5, "r": 9, "min_hits": 1},
    ]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "0/1",
        "marginal_probabilities": ["255/256", "0/1"],
    }


def test_full_cover_window_is_certain():
    # Every shift lies in [0, 2000], so one hit is guaranteed.
    payload = basic_payload()
    payload["windows"] = [{"l": 0, "r": 2000, "min_hits": 1}]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "1/1",
        "marginal_probabilities": ["1/1"],
    }


def test_single_ion():
    # One ion: window [1, 2] hit once has probability 3/8 + 1/16 = 7/16.
    payload = basic_payload()
    payload["ion_count"] = 1
    payload["windows"] = [{"l": 1, "r": 2, "min_hits": 1}]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "7/16",
        "marginal_probabilities": ["7/16"],
    }


def test_marginals_follow_input_order_not_window_order():
    # Same windows, two input orders: marginals must follow the input.
    payload = basic_payload()
    payload["windows"] = [
        {"l": 2, "r": 2, "min_hits": 1},
        {"l": 0, "r": 0, "min_hits": 1},
    ]
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "joint_probability": "9/128",
        "marginal_probabilities": ["31/256", "207/256"],
    }


def test_multi_group_matches_brute_force():
    payload = {
        "atom_groups": [
            {
                "count": 2,
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
        "ion_count": 3,
        "windows": [
            {"l": 0, "r": 2, "min_hits": 2},
            {"l": 1, "r": 4, "min_hits": 1},
            {"l": 2, "r": 2, "min_hits": 1},
            {"l": 3, "r": 8, "min_hits": 3},
        ],
    }
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    joint, marginals = brute_force_detection(payload)
    assert body["joint_probability"] == fmt(joint)
    assert body["marginal_probabilities"] == [fmt(m) for m in marginals]


def test_randomized_matches_brute_force():
    rng = random.Random(20260926)
    for _ in range(30):
        groups = []
        for _ in range(rng.randint(1, 2)):
            count = rng.randint(1, 2)
            shifts = rng.sample(range(0, 6), rng.randint(2, 3))
            cuts = sorted(rng.sample(range(1, 24), len(shifts) - 1))
            parts = [b - a for a, b in zip([0] + cuts, cuts + [24])]
            groups.append(
                {
                    "count": count,
                    "isotopes": [
                        {"shift_mda": shift, "abundance": f"{part}/24"}
                        for shift, part in zip(shifts, parts)
                    ],
                }
            )
        ion_count = rng.randint(1, 4)
        windows = []
        for _ in range(rng.randint(1, 4)):
            left = rng.randint(0, 8)
            right = rng.randint(left, 10)
            windows.append(
                {
                    "l": left,
                    "r": right,
                    "min_hits": rng.randint(1, 4),
                }
            )
        payload = {
            "atom_groups": groups,
            "ion_count": ion_count,
            "windows": windows,
        }
        resp = client.post("/v1/detection-probability", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        joint, marginals = brute_force_detection(payload)
        assert body["joint_probability"] == fmt(joint)
        assert body["marginal_probabilities"] == [fmt(m) for m in marginals]
        # Invariants: reduced fractions, joint <= every marginal.
        joint_value = parse_fraction(body["joint_probability"])
        for text in body["marginal_probabilities"]:
            n, d = text.split("/")
            assert Fraction(int(n), int(d)).denominator == int(d)
            assert joint_value <= parse_fraction(text)


def test_supports_max_scale_atom_groups():
    # Largest admissible input: total count 80, max total shift 2000, with
    # the largest ion_count and four overlapping windows.
    payload = {
        "atom_groups": [
            {
                "count": 40,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/3"},
                    {"shift_mda": 1, "abundance": "1/3"},
                    {"shift_mda": 25, "abundance": "1/3"},
                ],
            },
            {
                "count": 40,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/3"},
                    {"shift_mda": 1, "abundance": "1/3"},
                    {"shift_mda": 25, "abundance": "1/3"},
                ],
            },
        ],
        "ion_count": 80,
        "windows": [
            {"l": 0, "r": 100, "min_hits": 4},
            {"l": 50, "r": 150, "min_hits": 3},
            {"l": 100, "r": 200, "min_hits": 2},
            {"l": 0, "r": 2000, "min_hits": 1},
        ],
    }
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # The full-cover window is certain; the joint event is exactly the other
    # three windows succeeding, so it cannot exceed any marginal.
    assert body["marginal_probabilities"][3] == "1/1"
    joint = parse_fraction(body["joint_probability"])
    marginals = [parse_fraction(m) for m in body["marginal_probabilities"]]
    assert all(joint <= m for m in marginals)
    # All fractions are reduced and printed in full (thousands of digits).
    for text in [body["joint_probability"], *body["marginal_probabilities"]]:
        n, d = text.split("/")
        assert Fraction(int(n), int(d)).denominator == int(d)
    assert len(body["joint_probability"]) > 1000
    # Marginal of a single window equals the binomial tail computed from the
    # envelope peaks directly.
    envelope = client.post(
        "/v1/envelope",
        json={"atom_groups": payload["atom_groups"], "k": 1},
    ).json()
    probs = {
        peak["shift_mda"]: parse_fraction(peak["probability"])
        for peak in envelope["peaks"]
    }
    for window, text in zip(payload["windows"], body["marginal_probabilities"]):
        q = sum(
            (p for s, p in probs.items() if window["l"] <= s <= window["r"]),
            Fraction(0),
        )
        tail = sum(
            Fraction(1)
            * _binom(payload["ion_count"], k)
            * q**k
            * (1 - q) ** (payload["ion_count"] - k)
            for k in range(window["min_hits"], payload["ion_count"] + 1)
        )
        assert parse_fraction(text) == tail


def _binom(n: int, k: int) -> int:
    out = 1
    for i in range(k):
        out = out * (n - i) // (i + 1)
    return out


@pytest.mark.parametrize(
    "mutation",
    [
        # ion_count out of range / wrong type / missing
        lambda p: p.update(ion_count=0),
        lambda p: p.update(ion_count=81),
        lambda p: p.update(ion_count=1.5),
        lambda p: p.update(ion_count="2"),
        lambda p: p.pop("ion_count"),
        # windows list size / missing
        lambda p: p.update(windows=[]),
        lambda p: p.update(windows=p["windows"] * 3),
        lambda p: p.pop("windows"),
        # window field violations
        lambda p: p["windows"][0].update(l=-1),
        lambda p: p["windows"][0].update(l=2001),
        lambda p: p["windows"][0].update(r=2001),
        lambda p: p["windows"][0].update(r=-1),
        lambda p: p["windows"][0].update(l=2, r=1),
        lambda p: p["windows"][0].update(l=1.5),
        lambda p: p["windows"][0].update(r="2"),
        lambda p: p["windows"][0].update(min_hits=0),
        lambda p: p["windows"][0].update(min_hits=5),
        lambda p: p["windows"][0].update(min_hits=1.5),
        lambda p: p["windows"][0].update(min_hits="1"),
        lambda p: p["windows"][0].pop("min_hits"),
        lambda p: p["windows"][0].pop("l"),
        lambda p: p["windows"][0].pop("r"),
        lambda p: p["windows"][0].update(extra=1),
        # unknown top-level field
        lambda p: p.update(extra=1),
        # atom_groups violations reused verbatim from /v1/envelope
        lambda p: p["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        lambda p: p["atom_groups"][0].update(count=0),
        lambda p: p.update(atom_groups=[]),
        lambda p: p.update(atom_groups=p["atom_groups"] * 5),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=2001),
        lambda p: p.update(
            atom_groups=[dict(p["atom_groups"][0], count=40)] * 3
        ),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/detection-probability", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/detection-probability",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
