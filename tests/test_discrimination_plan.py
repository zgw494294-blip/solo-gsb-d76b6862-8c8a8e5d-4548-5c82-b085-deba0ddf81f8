"""Tests for the discrimination-plan endpoint, with a brute-force cross-check."""

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

    Returns (windows, total_bins, p0, p1, accuracy, hit_choice) with the
    API tie-break order: |p0 - p1| desc, bins asc, window count asc,
    flattened endpoints asc. Windows outside [0, max_shift] can only
    waste bins, so the optimum is found with l, r inside that range; the
    empty plan is included.
    """
    dist0 = brute_force_peaks(payload["candidates"][0])
    dist1 = brute_force_peaks(payload["candidates"][1])
    max_windows = payload["max_windows"]
    budget = payload["budget"]
    max_shift = max(max(dist0), max(dist1))
    w0 = [dist0.get(s, Fraction(0)) for s in range(max_shift + 1)]
    w1 = [dist1.get(s, Fraction(0)) for s in range(max_shift + 1)]

    intervals = []
    for left in range(max_shift + 1):
        c0 = Fraction(0)
        c1 = Fraction(0)
        for right in range(left, max_shift + 1):
            c0 += w0[right]
            c1 += w1[right]
            intervals.append((left, right, right - left + 1, c0, c1))
    intervals.sort()

    best_key = None
    best_windows: list[tuple[int, int]] = []

    def search(start: int, chosen: list[tuple[int, int]], bins: int,
               c0: Fraction, c1: Fraction):
        nonlocal best_key, best_windows
        flat = tuple(endpoint for window in chosen for endpoint in window)
        key = (-abs(c0 - c1), bins, len(chosen), flat)
        if best_key is None or key < best_key:
            best_key = key
            best_windows = list(chosen)
        if len(chosen) == max_windows:
            return
        for i in range(start, len(intervals)):
            left, right, cost, v0, v1 = intervals[i]
            if bins + cost > budget:
                continue
            chosen.append((left, right))
            # Intervals are sorted by left; only those starting past right
            # stay disjoint from the one just taken.
            nxt = i + 1
            while nxt < len(intervals) and intervals[nxt][0] <= right:
                nxt += 1
            search(nxt, chosen, bins + cost, c0 + v0, c1 + v1)
            chosen.pop()

    search(0, [], 0, Fraction(0), Fraction(0))
    total_bins = sum(right - left + 1 for left, right in best_windows)
    p0 = sum(
        (dist0.get(s, Fraction(0)) for left, right in best_windows for s in range(left, right + 1)),
        Fraction(0),
    )
    p1 = sum(
        (dist1.get(s, Fraction(0)) for left, right in best_windows for s in range(left, right + 1)),
        Fraction(0),
    )
    accuracy = (1 + abs(p0 - p1)) / 2
    hit_choice = None if p0 == p1 else (0 if p0 > p1 else 1)
    return best_windows, total_bins, p0, p1, accuracy, hit_choice


def basic_payload():
    return {
        "candidates": [
            {
                "atom_groups": [
                    {
                        "count": 1,
                        "isotopes": [
                            {"shift_mda": 0, "abundance": "3/4"},
                            {"shift_mda": 1, "abundance": "1/4"},
                        ],
                    }
                ]
            },
            {
                "atom_groups": [
                    {
                        "count": 1,
                        "isotopes": [
                            {"shift_mda": 0, "abundance": "1/4"},
                            {"shift_mda": 1, "abundance": "3/4"},
                        ],
                    }
                ]
            },
        ],
        "max_windows": 1,
        "budget": 1,
    }


def test_basic_hand_computed():
    # A: 0 -> 3/4, 1 -> 1/4. B: 0 -> 1/4, 1 -> 3/4. One 1-bin window:
    # [0,0] gives p0-p1 = 1/2, [1,1] gives p1-p0 = 1/2; both accuracy
    # 3/4, the lexicographically smaller [0,0] wins and prefers A.
    resp = client.post("/v1/discrimination-plan", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "hit_probabilities": ["3/4", "1/4"],
        "accuracy": "3/4",
        "hit_choice": 0,
    }


def test_covering_everything_is_not_beneficial():
    # Same envelopes with a 2-bin budget: [0,1] would cover every peak
    # (p0 = p1 = 1, accuracy 1/2), but discrimination — not coverage —
    # is the objective, so the single most separating bin still wins.
    payload = basic_payload()
    payload["budget"] = 2
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "hit_probabilities": ["3/4", "1/4"],
        "accuracy": "3/4",
        "hit_choice": 0,
    }


def test_spanning_window_beats_two_windows_on_window_count():
    # A: 0 -> 1/2, 1 -> 1/2. B: 2 -> 1/2, 3 -> 1/2. [0,1] and [0,0]+[1,1]
    # both separate perfectly in 2 bins; the single window wins. The
    # mirrored plan [2,3] ties on every earlier key and loses the
    # flattened-endpoint comparison.
    payload = {
        "candidates": [
            {"atom_groups": [{"count": 1, "isotopes": [
                {"shift_mda": 0, "abundance": "1/2"},
                {"shift_mda": 1, "abundance": "1/2"},
            ]}]},
            {"atom_groups": [{"count": 1, "isotopes": [
                {"shift_mda": 2, "abundance": "1/2"},
                {"shift_mda": 3, "abundance": "1/2"},
            ]}]},
        ],
        "max_windows": 2,
        "budget": 2,
    }
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 1]],
        "total_bins": 2,
        "hit_probabilities": ["1/1", "0/1"],
        "accuracy": "1/1",
        "hit_choice": 0,
    }


def test_gap_spanning_bills_empty_bins_and_negative_middle():
    # A: 0 -> 1/2, 10 -> 1/2. B: 5 -> 1/8, 20 -> 1/2, 30 -> 3/8. One
    # window, 11 bins: [0,10] swallows the negative dip at 5 for
    # p0-p1 = 7/8; the mirrored orientation's [20,30] also gives 7/8 in
    # 11 bins and loses the flattened-endpoint tie-break.
    payload = {
        "candidates": [
            {"atom_groups": [{"count": 1, "isotopes": [
                {"shift_mda": 0, "abundance": "1/2"},
                {"shift_mda": 10, "abundance": "1/2"},
            ]}]},
            {"atom_groups": [{"count": 1, "isotopes": [
                {"shift_mda": 5, "abundance": "1/8"},
                {"shift_mda": 20, "abundance": "1/2"},
                {"shift_mda": 30, "abundance": "3/8"},
            ]}]},
        ],
        "max_windows": 1,
        "budget": 11,
    }
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 10]],
        "total_bins": 11,
        "hit_probabilities": ["1/1", "1/8"],
        "accuracy": "15/16",
        "hit_choice": 0,
    }
    # One bin short of either span: both orientations fall back to a
    # single 1/2-valued peak; [0,0] wins the lexicographic tie.
    payload["budget"] = 10
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "hit_probabilities": ["1/2", "0/1"],
        "accuracy": "3/4",
        "hit_choice": 0,
    }
    # Two 1-bin windows separate A perfectly.
    payload["max_windows"] = 2
    payload["budget"] = 2
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0], [10, 10]],
        "total_bins": 2,
        "hit_probabilities": ["1/1", "0/1"],
        "accuracy": "1/1",
        "hit_choice": 0,
    }


def test_hit_choice_prefers_second_candidate():
    # A: 0 -> 1/2, 1 -> 1/2. B: 0 -> 1/1. Both [0,0] (p1-p0 = 1/2) and
    # [1,1] (p0-p1 = 1/2) give accuracy 3/4 in one bin; the
    # lexicographically smaller [0,0] wins and prefers B.
    payload = {
        "candidates": [
            {"atom_groups": [{"count": 1, "isotopes": [
                {"shift_mda": 0, "abundance": "1/2"},
                {"shift_mda": 1, "abundance": "1/2"},
            ]}]},
            {"atom_groups": [{"count": 1, "isotopes": [
                {"shift_mda": 0, "abundance": "1/1"},
                {"shift_mda": 1, "abundance": "0/1"},
            ]}]},
        ],
        "max_windows": 1,
        "budget": 1,
    }
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [[0, 0]],
        "total_bins": 1,
        "hit_probabilities": ["1/2", "1/1"],
        "accuracy": "3/4",
        "hit_choice": 1,
    }


def test_identical_candidates_yield_empty_plan():
    # Identical envelopes: every window has p0 == p1, so the empty plan
    # is optimal and no candidate is preferred on a hit.
    payload = basic_payload()
    payload["candidates"][1] = payload["candidates"][0]
    payload["max_windows"] = 3
    payload["budget"] = 200
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "windows": [],
        "total_bins": 0,
        "hit_probabilities": ["0/1", "0/1"],
        "accuracy": "1/2",
        "hit_choice": None,
    }


def test_multi_group_matches_brute_force():
    payload = {
        "candidates": [
            {
                "atom_groups": [
                    {"count": 3, "isotopes": [
                        {"shift_mda": 0, "abundance": "7/10"},
                        {"shift_mda": 1, "abundance": "1/5"},
                        {"shift_mda": 3, "abundance": "1/10"},
                    ]},
                    {"count": 2, "isotopes": [
                        {"shift_mda": 0, "abundance": "2/3"},
                        {"shift_mda": 2, "abundance": "1/3"},
                    ]},
                ]
            },
            {
                "atom_groups": [
                    {"count": 2, "isotopes": [
                        {"shift_mda": 0, "abundance": "1/2"},
                        {"shift_mda": 4, "abundance": "1/2"},
                    ]},
                ]
            },
        ],
        "max_windows": 3,
        "budget": 8,
    }
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    windows, total_bins, p0, p1, accuracy, hit_choice = brute_force_plan(payload)
    assert body["windows"] == [[left, right] for left, right in windows]
    assert body["total_bins"] == total_bins
    assert body["hit_probabilities"] == [fmt(p0), fmt(p1)]
    assert body["accuracy"] == fmt(accuracy)
    assert body["hit_choice"] == hit_choice


def test_randomized_matches_brute_force():
    rng = random.Random(20260926)
    for _ in range(30):
        candidates = []
        for _ in range(2):
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
            candidates.append({"atom_groups": groups})
        payload = {
            "candidates": candidates,
            "max_windows": rng.randint(1, 3),
            "budget": rng.randint(1, 10),
        }
        resp = client.post("/v1/discrimination-plan", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        windows, total_bins, p0, p1, accuracy, hit_choice = brute_force_plan(payload)
        assert body["windows"] == [[left, right] for left, right in windows]
        assert body["total_bins"] == total_bins
        assert body["hit_probabilities"] == [fmt(p0), fmt(p1)]
        assert body["accuracy"] == fmt(accuracy)
        assert body["hit_choice"] == hit_choice
        # Invariants: ascending disjoint windows within budget, reduced
        # fractions, hit_choice consistent with the hit probabilities.
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
        probs = [parse_fraction(text) for text in body["hit_probabilities"]]
        for text in body["hit_probabilities"] + [body["accuracy"]]:
            n, d = text.split("/")
            assert Fraction(int(n), int(d)).denominator == int(d)
        expected_choice = None if probs[0] == probs[1] else (0 if probs[0] > probs[1] else 1)
        assert body["hit_choice"] == expected_choice
        assert parse_fraction(body["accuracy"]) == (1 + abs(probs[0] - probs[1])) / 2


def test_supports_max_scale_atom_groups():
    # Largest admissible input per candidate: total count 80, max total
    # shift 2000.
    groups_a = [
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
    ]
    groups_b = [
        {
            "count": 40,
            "isotopes": [
                {"shift_mda": 0, "abundance": "1/2"},
                {"shift_mda": 1, "abundance": "1/4"},
                {"shift_mda": 25, "abundance": "1/4"},
            ],
        },
        {
            "count": 40,
            "isotopes": [
                {"shift_mda": 0, "abundance": "1/2"},
                {"shift_mda": 1, "abundance": "1/4"},
                {"shift_mda": 25, "abundance": "1/4"},
            ],
        },
    ]
    payload = {
        "candidates": [
            {"atom_groups": groups_a},
            {"atom_groups": groups_b},
        ],
        "max_windows": 6,
        "budget": 200,
    }
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["windows"]) <= 6
    assert body["total_bins"] <= 200
    flat = [endpoint for window in body["windows"] for endpoint in window]
    assert flat == sorted(flat)
    # Hit probabilities are consistent with the full envelope peaks.
    hit_probabilities = []
    for groups in (groups_a, groups_b):
        envelope = client.post(
            "/v1/envelope",
            json={"atom_groups": groups, "k": 1},
        ).json()
        probs = {
            peak["shift_mda"]: parse_fraction(peak["probability"])
            for peak in envelope["peaks"]
        }
        hit_probabilities.append(
            sum(
                (
                    probs.get(s, Fraction(0))
                    for left, right in body["windows"]
                    for s in range(left, right + 1)
                ),
                Fraction(0),
            )
        )
    assert body["hit_probabilities"] == [fmt(p) for p in hit_probabilities]
    p0, p1 = hit_probabilities
    assert body["accuracy"] == fmt((1 + abs(p0 - p1)) / 2)
    expected_choice = None if p0 == p1 else (0 if p0 > p1 else 1)
    assert body["hit_choice"] == expected_choice


@pytest.mark.parametrize(
    "mutation",
    [
        # candidates wrong shape / count
        lambda p: p.update(candidates=p["candidates"][:1]),
        lambda p: p.update(candidates=p["candidates"] + p["candidates"][:1]),
        lambda p: p.update(candidates=p["candidates"] * 2),
        lambda p: p.pop("candidates"),
        lambda p: p["candidates"][0].update(extra=1),
        lambda p: p["candidates"][0].pop("atom_groups"),
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
        lambda p: p["candidates"][0]["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        lambda p: p["candidates"][0]["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        lambda p: p["candidates"][0]["atom_groups"][0].update(count=0),
        lambda p: p["candidates"][0].update(atom_groups=[]),
        lambda p: p["candidates"][0].update(
            atom_groups=p["candidates"][0]["atom_groups"] * 5
        ),
        lambda p: p["candidates"][0]["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
        lambda p: p["candidates"][0]["atom_groups"][0]["isotopes"][1].update(shift_mda=2001),
        lambda p: p["candidates"][0].update(
            atom_groups=[dict(p["candidates"][0]["atom_groups"][0], count=40)] * 3
        ),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/discrimination-plan", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/discrimination-plan",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
