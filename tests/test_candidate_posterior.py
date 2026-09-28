"""Tests for the candidate-posterior endpoint, with a brute-force cross-check."""

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


def brute_force_posterior(payload: dict):
    """Exact Bayes update by enumerating every ion sequence per (candidate, d).

    Returns (joints, candidate_marginals, offset_marginals) or None when the
    total evidence is zero. The multinomial probability of the observed
    count vector is computed by scanning every length-n ion sequence and
    checking each listed shift's frequency; every unlisted measured shift is
    "other" and only the total leftover count has to match, which is
    automatic since listed counts sum to <= ion_count.
    """
    priors = [parse_fraction(c["prior"]) for c in payload["candidates"]]
    envelopes = [brute_force_peaks(c) for c in payload["candidates"]]
    radius = payload["calibration_radius"]
    offsets = list(range(-radius, radius + 1))
    ion_count = payload["ion_count"]
    listed = {o["shift_mda"]: o["count"] for o in payload["observations"]}

    weights: list[list[Fraction]] = []
    for env in envelopes:
        row: list[Fraction] = []
        for delta in offsets:
            likelihood = Fraction(0)
            for seq in itertools.product(sorted(env), repeat=ion_count):
                measured = [shift + delta for shift in seq]
                if all(measured.count(shift) == count for shift, count in listed.items()):
                    prob = Fraction(1)
                    for shift in seq:
                        prob *= env[shift]
                    likelihood += prob
            row.append(likelihood)
        weights.append(row)

    total = Fraction(0)
    for i, prior in enumerate(priors):
        for likelihood in weights[i]:
            total += prior * likelihood / len(offsets)
    if total == 0:
        return None
    joints = [
        [
            priors[i] * weights[i][j] / len(offsets) / total
            for j in range(len(offsets))
        ]
        for i in range(len(envelopes))
    ]
    candidate_marginals = [
        priors[i] * sum(weights[i], Fraction(0)) / len(offsets) / total
        for i in range(len(envelopes))
    ]
    offset_marginals = [
        sum(
            priors[i] * weights[i][j] / len(offsets) for i in range(len(envelopes))
        )
        / total
        for j in range(len(offsets))
    ]
    return joints, candidate_marginals, offset_marginals


def make_candidate(*isotopes, prior="1/2", count=1):
    return {
        "atom_groups": [
            {
                "count": count,
                "isotopes": [
                    {"shift_mda": shift, "abundance": abundance}
                    for shift, abundance in isotopes
                ],
            }
        ],
        "prior": prior,
    }


def two_bernoulli_payload(**overrides):
    # A: Bernoulli on {0, 1} with p = 1/2; B: Bernoulli on {0, 2} with 3/4 at 2.
    payload = {
        "candidates": [
            make_candidate((0, "1/2"), (1, "1/2")),
            make_candidate((0, "1/4"), (2, "3/4")),
        ],
        "ion_count": 2,
        "observations": [
            {"shift_mda": 1, "count": 1},
            {"shift_mda": 2, "count": 1},
        ],
        "calibration_radius": 1,
    }
    payload.update(overrides)
    return payload


def test_single_observed_shift_fractional_posterior():
    # One ion at measured shift 2, the other ion unlisted (other bin), n = 2.
    # A needs delta = 1: 2 * 1/2 * 1/2 = 1/2; B needs delta = 0:
    # 2 * 3/4 * 1/4 = 3/8. With priors 1/2 and uniform delta prior 1/3 the
    # weights are 1/12 = 4/48 and 1/16 = 3/48, so the posterior is 4/7 vs 3/7.
    payload = two_bernoulli_payload(
        observations=[{"shift_mda": 2, "count": 1}]
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "possible": True,
        "joint_posteriors": [
            ["0/1", "0/1", "4/7"],
            ["0/1", "3/7", "0/1"],
        ],
        "candidate_posteriors": ["4/7", "3/7"],
        "offset_posteriors": ["0/1", "3/7", "4/7"],
        "offsets": [-1, 0, 1],
    }


def test_offset_ambiguity_within_and_across_candidates():
    # n = 1, observed single ion at shift 1, r = 1.
    # A: delta 0 from original 1 (1/2), delta 1 from original 0 (1/2).
    # B: delta -1 from original 2 (3/4), delta 1 from original 0 (1/4).
    # Weights (prior 1/2, delta prior 1/3):
    # A d=0: 1/12, A d=1: 1/12, B d=-1: 1/8, B d=1: 1/24; Z = 1/3.
    payload = two_bernoulli_payload(
        ion_count=1, observations=[{"shift_mda": 1, "count": 1}]
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "possible": True,
        "joint_posteriors": [
            ["0/1", "1/4", "1/4"],
            ["3/8", "0/1", "1/8"],
        ],
        "candidate_posteriors": ["1/2", "1/2"],
        "offset_posteriors": ["3/8", "1/4", "3/8"],
        "offsets": [-1, 0, 1],
    }


def test_radius_zero_outputs_single_offset():
    # r = 0: only delta 0. Observed one ion at 0 and one at 1: A gives
    # 2 * 1/2 * 1/2 = 1/2, B has no shift 1 at all, so B is ruled out.
    payload = two_bernoulli_payload(
        calibration_radius=0,
        observations=[
            {"shift_mda": 0, "count": 1},
            {"shift_mda": 1, "count": 1},
        ],
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "possible": True,
        "joint_posteriors": [["1/1"], ["0/1"]],
        "candidate_posteriors": ["1/1", "0/1"],
        "offset_posteriors": ["1/1"],
        "offsets": [0],
    }


def test_calibration_offset_extends_reach():
    # Observed shifts 1 and 2 pin A to delta 1 for certain: B's supported
    # measured pairs never equal {1, 2}.
    payload = two_bernoulli_payload(
        observations=[
            {"shift_mda": 1, "count": 1},
            {"shift_mda": 2, "count": 1},
        ]
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["possible"] is True
    assert body["joint_posteriors"] == [
        ["0/1", "0/1", "1/1"],
        ["0/1", "0/1", "0/1"],
    ]
    assert body["candidate_posteriors"] == ["1/1", "0/1"]
    assert body["offset_posteriors"] == ["0/1", "0/1", "1/1"]


def test_zero_prior_candidate_gets_zero_posterior():
    # A also fits the data (delta 1) but carries prior 0/1: its posterior is
    # exactly 0/1 even though its likelihood is positive.
    payload = two_bernoulli_payload(
        candidates=[
            make_candidate((0, "1/2"), (1, "1/2"), prior="0/1"),
            make_candidate((0, "1/4"), (2, "3/4"), prior="1/1"),
        ],
        observations=[{"shift_mda": 2, "count": 1}],
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["possible"] is True
    assert body["candidate_posteriors"] == ["0/1", "1/1"]
    assert body["joint_posteriors"][0] == ["0/1", "0/1", "0/1"]
    assert body["joint_posteriors"][1] == ["0/1", "1/1", "0/1"]


def test_impossible_data_returns_possible_false():
    # Neither envelope reaches measured shift 5 even after any delta in [0, 0].
    payload = two_bernoulli_payload(
        calibration_radius=0,
        observations=[{"shift_mda": 5, "count": 1}],
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"possible": False}


def test_zero_count_unreachable_bin_does_not_invalidate():
    # Combinatorial 0**0 = 1: an unreachable listed bin with count 0 must not
    # kill the likelihood. Shift 5 gets zero ions, one ion lands at shift 2:
    # only B (delta 0) survives, posterior exactly 1/1.
    payload = two_bernoulli_payload(
        calibration_radius=0,
        observations=[
            {"shift_mda": 5, "count": 0},
            {"shift_mda": 2, "count": 1},
        ],
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "possible": True,
        "joint_posteriors": [["0/1"], ["1/1"]],
        "candidate_posteriors": ["0/1", "1/1"],
        "offset_posteriors": ["1/1"],
        "offsets": [0],
    }


def test_all_ions_into_other_bin():
    # No listed bin collects an ion: every ion must avoid shift 1. A with
    # delta 0 gives (1/2)^2 = 1/4, B with delta 0 gives 1 (no peak at 1).
    payload = two_bernoulli_payload(
        calibration_radius=0,
        observations=[{"shift_mda": 1, "count": 0}],
    )
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # Weights 1/2 * 1/4 = 1/8 vs 1/2 * 1 = 1/2; posterior 1/5 vs 4/5.
    assert body["candidate_posteriors"] == ["1/5", "4/5"]
    assert body["joint_posteriors"] == [["1/5"], ["4/5"]]
    assert body["offset_posteriors"] == ["1/1"]


def test_multi_group_matches_brute_force():
    payload = {
        "candidates": [
            {
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
                        "count": 1,
                        "isotopes": [
                            {"shift_mda": 0, "abundance": "2/3"},
                            {"shift_mda": 2, "abundance": "1/3"},
                        ],
                    },
                ],
                "prior": "1/3",
            },
            {
                "atom_groups": [
                    {
                        "count": 2,
                        "isotopes": [
                            {"shift_mda": 0, "abundance": "1/2"},
                            {"shift_mda": 1, "abundance": "1/2"},
                        ],
                    }
                ],
                "prior": "2/3",
            },
        ],
        "ion_count": 3,
        "observations": [
            {"shift_mda": 1, "count": 1},
            {"shift_mda": 2, "count": 1},
        ],
        "calibration_radius": 1,
    }
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    joints, candidates, offsets = brute_force_posterior(payload)
    assert body["possible"] is True
    assert body["joint_posteriors"] == [[fmt(v) for v in row] for row in joints]
    assert body["candidate_posteriors"] == [fmt(v) for v in candidates]
    assert body["offset_posteriors"] == [fmt(v) for v in offsets]
    assert body["offsets"] == [-1, 0, 1]


def test_randomized_matches_brute_force():
    rng = random.Random(20260927)
    for _ in range(40):
        candidate_count = rng.randint(2, 3)
        candidates = []
        for _ in range(candidate_count):
            shifts = rng.sample(range(0, 5), rng.randint(2, 3))
            cuts = sorted(rng.sample(range(1, 24), len(shifts) - 1))
            parts = [b - a for a, b in zip([0] + cuts, cuts + [24])]
            candidates.append(
                {
                    "atom_groups": [
                        {
                            "count": rng.randint(1, 2),
                            "isotopes": [
                                {"shift_mda": shift, "abundance": f"{part}/24"}
                                for shift, part in zip(shifts, parts)
                            ],
                        }
                    ],
                    "prior": "0/1",
                }
            )
        # Uniform priors over candidates (all share a common denominator).
        unit = Fraction(1, candidate_count)
        for candidate in candidates:
            candidate["prior"] = fmt(unit)

        radius = rng.randint(0, 2)
        ion_count = rng.randint(1, 4)
        observation_count = rng.randint(1, 3)
        observed_shifts = rng.sample(range(0, 7), observation_count)
        # Distribute at most ion_count ions across the listed bins.
        allocated = rng.randint(0, ion_count)
        counts = [0] * observation_count
        for _ in range(allocated):
            counts[rng.randrange(observation_count)] += 1
        observations = [
            {"shift_mda": shift, "count": count}
            for shift, count in zip(observed_shifts, counts)
        ]
        payload = {
            "candidates": candidates,
            "ion_count": ion_count,
            "observations": observations,
            "calibration_radius": radius,
        }
        resp = client.post("/v1/candidate-posterior", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        expected = brute_force_posterior(payload)
        if expected is None:
            assert body == {"possible": False}
            continue
        joints, candidate_marginals, offset_marginals = expected
        assert body["possible"] is True
        assert body["offsets"] == list(range(-radius, radius + 1))
        assert body["joint_posteriors"] == [
            [fmt(v) for v in row] for row in joints
        ]
        assert body["candidate_posteriors"] == [
            fmt(v) for v in candidate_marginals
        ]
        assert body["offset_posteriors"] == [fmt(v) for v in offset_marginals]

        # Invariants: reduced fractions, table/marginals consistency and
        # normalization over Q.
        def reduced(text: str) -> Fraction:
            n, d = text.split("/")
            value = Fraction(int(n), int(d))
            assert value.denominator == int(d)
            return value

        joint_values = [
            reduced(text)
            for row in body["joint_posteriors"]
            for text in row
        ]
        assert sum(joint_values, Fraction(0)) == 1
        for i, row in enumerate(body["joint_posteriors"]):
            assert sum((reduced(text) for text in row), Fraction(0)) == reduced(
                body["candidate_posteriors"][i]
            )
        for j, text in enumerate(body["offset_posteriors"]):
            column = sum(
                (reduced(body["joint_posteriors"][i][j]) for i in range(candidate_count)),
                Fraction(0),
            )
            assert column == reduced(text)
        assert sum(map(reduced, body["candidate_posteriors"]), Fraction(0)) == 1
        assert sum(map(reduced, body["offset_posteriors"]), Fraction(0)) == 1


def basic_valid_payload():
    return two_bernoulli_payload()


@pytest.mark.parametrize(
    "mutation",
    [
        # candidate list size
        lambda p: p.update(candidates=[p["candidates"][0]]),
        lambda p: p.update(candidates=[p["candidates"][0]] * 5),
        lambda p: p.pop("candidates"),
        # priors must be valid n/d, non-negative and sum to 1
        lambda p: p["candidates"][0].update(prior="2/2"),
        lambda p: p["candidates"][0].update(prior="1/3"),
        lambda p: p["candidates"][0].update(prior="0/0"),
        lambda p: p["candidates"][0].update(prior="1"),
        lambda p: p["candidates"][0].update(prior="a/b"),
        lambda p: p["candidates"][0].update(prior="1/1001"),
        lambda p: p["candidates"][0].pop("prior"),
        # candidate atom_groups reuse the envelope validation
        lambda p: p["candidates"][1]["atom_groups"][0]["isotopes"][1].update(
            abundance="1/2"
        ),
        lambda p: p["candidates"][0]["atom_groups"][0].update(count=0),
        lambda p: p["candidates"][0]["atom_groups"][0]["isotopes"][0].update(
            shift_mda=2001
        ),
        lambda p: p["candidates"][0].update(extra=1),
        # ion_count
        lambda p: p.update(ion_count=0),
        lambda p: p.update(ion_count=81),
        lambda p: p.update(ion_count=1.5),
        lambda p: p.update(ion_count="2"),
        lambda p: p.pop("ion_count"),
        # observations
        lambda p: p.update(observations=[]),
        lambda p: p.update(observations=p["observations"] * 5),
        lambda p: p.pop("observations"),
        lambda p: p["observations"][0].update(shift_mda=-1),
        lambda p: p["observations"][0].update(shift_mda=2001),
        lambda p: p["observations"][0].update(shift_mda=1.5),
        lambda p: p["observations"][0].update(count=-1),
        lambda p: p["observations"][0].update(count=81),
        lambda p: p["observations"][0].update(count=1.5),
        lambda p: p["observations"][0].update(count="1"),
        lambda p: p["observations"][0].pop("shift_mda"),
        lambda p: p["observations"][0].pop("count"),
        lambda p: p["observations"][0].update(extra=1),
        lambda p: p["observations"][1].update(shift_mda=1),  # duplicate shift
        lambda p: p.update(
            observations=[
                {"shift_mda": 1, "count": 1},
                {"shift_mda": 2, "count": 2},
            ]
        ),  # counts 3 > ion_count 2
        # calibration radius
        lambda p: p.update(calibration_radius=-1),
        lambda p: p.update(calibration_radius=4),
        lambda p: p.update(calibration_radius=1.5),
        lambda p: p.update(calibration_radius="1"),
        lambda p: p.pop("calibration_radius"),
        # unknown top-level field
        lambda p: p.update(extra=1),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_valid_payload()
    mutation(payload)
    resp = client.post("/v1/candidate-posterior", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/candidate-posterior",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
