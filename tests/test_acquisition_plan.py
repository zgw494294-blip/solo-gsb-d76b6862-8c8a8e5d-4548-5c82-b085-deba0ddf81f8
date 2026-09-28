"""Tests for the acquisition-plan endpoint, including a brute-force cross-check."""

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


def brute_force_plan(payload: dict):
    """Exhaustively try every disjoint window set on the shift axis.

    Returns (windows, total_bins, captured) with the API tie-break order:
    captured desc, bins asc, window count asc, flattened endpoints asc.
    Windows outside [0, max_shift] can only waste bins, so the optimum is
    found with l, r inside that range.
    """
    dist = brute_force_peaks(payload)
    max_windows = payload["max_windows"]
    budget = payload["budget"]
    max_shift = max(dist)
    weight = [dist.get(s, Fraction(0)) for s in range(max_shift + 1)]

    intervals = []
    for left in range(max_shift + 1):
        captured = Fraction(0)
        for right in range(left, max_shift + 1):
            captured += weight[right]
            intervals.append((left, right, right - left + 1, captured))
    intervals.sort()

    best_key = None
    best_windows: list[tuple[int, int]] = []

    def search(start: int, chosen: list[tuple[int, int]], bins: int, captured: Fraction):
        nonlocal best_key, best_windows
        flat = tuple(endpoint for window in chosen for endpoint in window)
        key = (-captured, bins, len(chosen), flat)
        if best_key is None or key < best_key:
            best_key = key
            best_windows = list(chosen)
        if len(chosen) == max_windows:
            return
        for i in range(start, len(intervals)):
            left, right, cost, value = intervals[i]
            if bins + cost > budget:
                continue
            chosen.append((left, right))
            # Intervals are sorted by left; only those starting past right
            # stay disjoint from the one just taken.
            nxt = i + 1
            while nxt < len(intervals) and intervals[nxt][0] <= right:
                nxt += 1
            search(nxt, chosen, bins + cost, captured + value)
            chosen.pop()

    search(0, [], 0, Fraction(0))
    total_bins = sum(right - left + 1 for left, right in best_windows)
    captured = sum(
        (dist.get(s, Fraction(0)) for left, right in best_windows for s in range(left, right + 1)),
        Fraction(0),
    )
    return best_windows, total_bins, captured


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
        "max_windows": 2,
        "budget": 5,
    }


def test_basic_hand_computed():
    # Peaks: 0 -> 9/16, 1 -> 3/8, 2 -> 1/16. [0,2] (3 bins, one window)
    # captures everything; it ties [0,1]+[2,2] on bins and wins on fewer
    # windows.
    resp = client.post("/v1/acquisition-plan", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 2]],
        "total_bins": 3,
        "captured_probability": "1/1",
    }


def test_budget_binds_and_empty_bins_are_charged():
    # Same envelope, but only 2 bins: [0,1] (15/16) beats [0,0]+[1,1]
    # (also 15/16 but 2 windows and same bins) and beats [0,2] (3 bins).
    payload = basic_payload()
    payload["budget"] = 2
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 1]],
        "total_bins": 2,
        "captured_probability": "15/16",
    }
    # A single bin takes the heaviest peak.
    payload["budget"] = 1
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "captured_probability": "9/16",
    }


def test_window_cap_binds():
    # One window of 3 bins captures everything.
    payload = basic_payload()
    payload["max_windows"] = 1
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 2]],
        "total_bins": 3,
        "captured_probability": "1/1",
    }


def test_gap_spanning_bills_empty_bins():
    # Peaks at 0 and 10, each 1/2. Spanning [0,10] costs 11 bins.
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 10, "abundance": "1/2"},
                ],
            }
        ],
        "max_windows": 1,
        "budget": 11,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 10]],
        "total_bins": 11,
        "captured_probability": "1/1",
    }
    # With a second window allowed, [0,0]+[10,10] captures everything in
    # just 2 bins and beats the 11-bin span on the bin tie-break.
    payload["max_windows"] = 2
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0], [10, 10]],
        "total_bins": 2,
        "captured_probability": "1/1",
    }
    # One bin short of the span with a single window: only a single peak
    # fits, tie broken to the lexicographically smallest endpoints [0, 0].
    payload["max_windows"] = 1
    payload["budget"] = 10
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "captured_probability": "1/2",
    }


def test_tie_break_prefers_fewer_windows():
    # Peaks at 0 and 1, each 1/2. [0,0]+[1,1] and [0,1] both capture 1 in
    # 2 bins; the single window wins.
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/2"},
                ],
            }
        ],
        "max_windows": 2,
        "budget": 10,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 1]],
        "total_bins": 2,
        "captured_probability": "1/1",
    }


def test_tie_break_prefers_fewer_bins():
    # Single peak at 3 (the other isotope has zero abundance): the optimal
    # window hugs the peak instead of spending the generous budget.
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "0/1"},
                    {"shift_mda": 3, "abundance": "1/1"},
                ],
            }
        ],
        "max_windows": 3,
        "budget": 200,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[3, 3]],
        "total_bins": 1,
        "captured_probability": "1/1",
    }


def test_tie_break_lexicographic_endpoints():
    # Peaks at 0 and 2 with equal weight 1/2 and a 1-bin budget: [0,0] and
    # [2,2] tie on probability, bins and window count; [0,0] wins.
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 2, "abundance": "1/2"},
                ],
            }
        ],
        "max_windows": 1,
        "budget": 1,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "captured_probability": "1/2",
    }


def test_tie_break_lexicographic_flattened_two_windows():
    # Two groups of one atom each: shifts 0/5 and 0/10 produce four equal
    # peaks at 0, 5, 10, 15 (each 1/4). With two 1-bin windows four pairs
    # capture 1/2; the lexicographically smallest flattened endpoint
    # sequence is [0,0] + [5,5].
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 5, "abundance": "1/2"},
                ],
            },
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 10, "abundance": "1/2"},
                ],
            },
        ],
        "max_windows": 2,
        "budget": 2,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0], [5, 5]],
        "total_bins": 2,
        "captured_probability": "1/2",
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
        ],
        "max_windows": 3,
        "budget": 8,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    windows, total_bins, captured = brute_force_plan(payload)
    assert body["windows"] == [[left, right] for left, right in windows]
    assert body["total_bins"] == total_bins
    assert body["captured_probability"] == fmt(captured)


def test_randomized_matches_brute_force():
    rng = random.Random(20260926)
    for _ in range(30):
        groups = []
        for _ in range(rng.randint(1, 2)):
            count = rng.randint(1, 3)
            shifts = rng.sample(range(0, 7), rng.randint(2, 3))
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
        payload = {
            "atom_groups": groups,
            "max_windows": rng.randint(1, 3),
            "budget": rng.randint(1, 10),
        }
        resp = client.post("/v1/acquisition-plan", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        windows, total_bins, captured = brute_force_plan(payload)
        assert body["windows"] == [[left, right] for left, right in windows]
        assert body["total_bins"] == total_bins
        assert body["captured_probability"] == fmt(captured)
        # Invariants: ascending disjoint windows within budget, reduced fraction.
        flat = [endpoint for window in body["windows"] for endpoint in window]
        assert flat == sorted(flat)
        assert all(
            left <= right for left, right in zip(flat[::2], flat[1::2])
        )
        assert body["total_bins"] == sum(
            right - left + 1 for left, right in body["windows"]
        )
        assert body["total_bins"] <= payload["budget"]
        assert len(body["windows"]) <= payload["max_windows"]
        n, d = body["captured_probability"].split("/")
        assert Fraction(int(n), int(d)).denominator == int(d)


def test_supports_max_scale_atom_groups():
    # Largest admissible input: total count 80, max total shift 2000.
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
        "max_windows": 6,
        "budget": 200,
    }
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 1 <= len(body["windows"]) <= 6
    assert body["total_bins"] <= 200
    flat = [endpoint for window in body["windows"] for endpoint in window]
    assert flat == sorted(flat)
    # Captured probability is consistent with the envelope peaks.
    envelope = client.post(
        "/v1/envelope",
        json={"atom_groups": payload["atom_groups"], "k": 1},
    ).json()
    probs = {
        peak["shift_mda"]: parse_fraction(peak["probability"])
        for peak in envelope["peaks"]
    }
    captured = sum(
        (
            probs.get(s, Fraction(0))
            for left, right in body["windows"]
            for s in range(left, right + 1)
        ),
        Fraction(0),
    )
    assert body["captured_probability"] == fmt(captured)


@pytest.mark.parametrize(
    "mutation",
    [
        # max_windows out of range / wrong type
        lambda p: p.update(max_windows=0),
        lambda p: p.update(max_windows=7),
        lambda p: p.update(max_windows=1.5),
        lambda p: p.update(max_windows="2"),
        lambda p: p.pop("max_windows"),
        # budget out of range / wrong type
        lambda p: p.update(budget=0),
        lambda p: p.update(budget=201),
        lambda p: p.update(budget=1.5),
        lambda p: p.update(budget="100"),
        lambda p: p.pop("budget"),
        # unknown field
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
    resp = client.post("/v1/acquisition-plan", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/acquisition-plan",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
