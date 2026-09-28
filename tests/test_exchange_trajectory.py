"""Tests for the exchange-trajectory endpoint, including a brute-force cross-check."""

from __future__ import annotations

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


def brute_force_trajectory(payload: dict):
    """Exact trajectory inference by explicit exchange-sequence enumeration.

    Every step picks one of the n*m (molecule site, bath site) pairs, so each
    of the (n*m)**steps pick sequences is equally likely; sequences are
    enumerated recursively with the actual heavy-site sets tracked, and every
    sequence passing all observation filters adds one to the joint count and
    to the (step, heavy_count) bin of every prefix. Returns the joint
    probability of the observations and, per step, the conditional
    heavy-count distribution given all of them.
    """
    n, m, h, b, steps = (payload[k] for k in ("n", "m", "h", "b", "steps"))
    filters = {
        obs["step"]: (obs["lower"], obs["upper"])
        for obs in payload.get("observations", [])
    }
    z = 0
    hits = [dict() for _ in range(steps + 1)]

    def walk(t, mol_heavy, bath_heavy, xs):
        nonlocal z
        xs = xs + (len(mol_heavy),)
        if t in filters:
            lower, upper = filters[t]
            if not lower <= xs[t] <= upper:
                return
        if t == steps:
            z += 1
            for s, x in enumerate(xs):
                hits[s][x] = hits[s].get(x, 0) + 1
            return
        for i in range(n):
            for j in range(m):
                i_heavy = i in mol_heavy
                j_heavy = j < bath_heavy  # bath sites 0..bath_heavy-1 heavy
                if i_heavy == j_heavy:
                    walk(t + 1, mol_heavy, bath_heavy, xs)
                elif i_heavy:  # heavy isotope leaves the molecule
                    walk(t + 1, mol_heavy - {i}, bath_heavy + 1, xs)
                else:  # heavy isotope enters the molecule
                    walk(t + 1, mol_heavy | {i}, bath_heavy - 1, xs)

    walk(0, frozenset(range(h)), b, ())
    probability = Fraction(z, (n * m) ** steps)
    if z == 0:
        return probability, None
    distributions = [
        {x: Fraction(count, z) for x, count in sorted(step_hits.items())}
        for step_hits in hits
    ]
    return probability, distributions


def basic_payload():
    return {
        "n": 2,
        "m": 2,
        "h": 1,
        "b": 1,
        "steps": 2,
        "observations": [{"step": 2, "lower": 2, "upper": 2}],
    }


def test_deterministic_oscillation_without_observations():
    # n = m = 1 with the heavy isotope on the molecule: every step swaps the
    # only molecule site with the only bath site, so the heavy count
    # alternates deterministically 1, 0, 1, 0.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={"n": 1, "m": 1, "h": 1, "b": 0, "steps": 3, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 0, "probability": "1/1"}],
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 0, "probability": "1/1"}],
        ],
    }


def test_unconditional_single_step_hand_computed():
    # From x = 1 with n = m = 2 and total 2: up/down numerators are both 1
    # out of n*m = 4 pairs, so X_1 is 0, 1, 2 with weights 1/4, 1/2, 1/4.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={"n": 2, "m": 2, "h": 1, "b": 1, "steps": 1, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            [{"heavy_count": 1, "probability": "1/1"}],
            [
                {"heavy_count": 0, "probability": "1/4"},
                {"heavy_count": 1, "probability": "1/2"},
                {"heavy_count": 2, "probability": "1/4"},
            ],
        ],
    }


def test_conditioned_trajectory_hand_computed():
    # X_2 = 2 requires the path 1 -> 1 -> 2 (probability 1/2 * 1/4 = 1/8):
    # from x = 2 the chain can only move down, so every earlier state is
    # pinned to 1.
    resp = client.post("/v1/exchange-trajectory", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/8",
        "distributions": [
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 2, "probability": "1/1"}],
        ],
    }


def test_zero_probability_omits_distributions():
    # From x = 2 with total 2 the chain must move down, so X_1 = 2 is
    # impossible.
    payload = basic_payload()
    payload.update(h=2, b=0, steps=1)
    payload["observations"] = [{"step": 1, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_step_zero_observation_filters_initial_state():
    # Step 0 is the initial state X_0 = h; observing a range that excludes h
    # makes the observations impossible.
    payload = basic_payload()
    payload["observations"] = [{"step": 0, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}

    # A range containing h is certain and leaves the trajectory unconditional.
    payload["observations"] = [{"step": 0, "lower": 1, "upper": 1}]
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            [{"heavy_count": 1, "probability": "1/1"}],
            [
                {"heavy_count": 0, "probability": "1/4"},
                {"heavy_count": 1, "probability": "1/2"},
                {"heavy_count": 2, "probability": "1/4"},
            ],
            [
                {"heavy_count": 0, "probability": "1/8"},
                {"heavy_count": 1, "probability": "3/4"},
                {"heavy_count": 2, "probability": "1/8"},
            ],
        ],
    }


def test_steps_after_last_observation_are_smoothed_forward():
    # n = 1, m = 2, total 1: from x = 1 the heavy isotope always leaves
    # (down numerator 2/2), so X_1 = 0 surely; from x = 0 it returns with
    # probability 1/2. Observing X_1 = 0 is certain, and X_2 splits 1/2-1/2.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={
            "n": 1,
            "m": 2,
            "h": 1,
            "b": 0,
            "steps": 2,
            "observations": [{"step": 1, "lower": 0, "upper": 0}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 0, "probability": "1/1"}],
            [
                {"heavy_count": 0, "probability": "1/2"},
                {"heavy_count": 1, "probability": "1/2"},
            ],
        ],
    }


def test_multiple_observations_joint():
    # Same chain as the basic payload plus the redundant filter X_1 >= 1:
    # X_2 = 2 already forces X_1 = 1, so the joint probability stays 1/8.
    payload = basic_payload()
    payload["observations"] = [
        {"step": 1, "lower": 1, "upper": 2},
        {"step": 2, "lower": 2, "upper": 2},
    ]
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/8",
        "distributions": [
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 1, "probability": "1/1"}],
            [{"heavy_count": 2, "probability": "1/1"}],
        ],
    }


def test_same_type_swaps_count_as_steps():
    # n = m = 2 with total 2 at x = 1: of the 4 site pairs, 2 swap equal
    # isotopes and leave the count unchanged, so P(X_1 = 1) = 1/2 — a model
    # skipping same-type pairs would report 0.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={
            "n": 2,
            "m": 2,
            "h": 1,
            "b": 1,
            "steps": 1,
            "observations": [{"step": 1, "lower": 1, "upper": 1}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["probability"] == "1/2"


def test_steps_zero_returns_initial_state_only():
    resp = client.post(
        "/v1/exchange-trajectory",
        json={"n": 3, "m": 2, "h": 2, "b": 1, "steps": 0, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [[{"heavy_count": 2, "probability": "1/1"}]],
    }


def test_randomized_matches_brute_force():
    rng = random.Random(20260927)
    for _ in range(40):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 4)
        obs_count = rng.randint(0, 3)
        observations = []
        for step in rng.sample(range(steps + 1), min(obs_count, steps + 1)):
            lower = rng.randint(0, n)
            upper = rng.randint(lower, n)
            observations.append(
                {"step": step, "lower": lower, "upper": upper}
            )
        payload = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "observations": observations,
        }
        resp = client.post("/v1/exchange-trajectory", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        probability, distributions = brute_force_trajectory(payload)
        assert parse_fraction(body["probability"]) == probability
        if probability == 0:
            assert "distributions" not in body
            continue
        assert len(body["distributions"]) == steps + 1
        for got_peaks, expected in zip(body["distributions"], distributions):
            got = {
                peak["heavy_count"]: parse_fraction(peak["probability"])
                for peak in got_peaks
            }
            assert got == expected
            # Ascending by heavy_count, reduced fractions, none zero.
            counts = [peak["heavy_count"] for peak in got_peaks]
            assert counts == sorted(counts)
            for peak in got_peaks:
                value = parse_fraction(peak["probability"])
                assert peak["probability"] == fmt(value)
                assert value > 0
            # Each step's conditional distribution is normalized.
            assert sum(got.values(), Fraction(0)) == 1


def test_large_request_stays_exact_and_fast():
    payload = {
        "n": 40,
        "m": 40,
        "h": 20,
        "b": 20,
        "steps": 80,
        "observations": [
            {"step": 20, "lower": 15, "upper": 25},
            {"step": 40, "lower": 10, "upper": 30},
            {"step": 60, "lower": 0, "upper": 40},
            {"step": 80, "lower": 18, "upper": 22},
        ],
    }
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    probability = parse_fraction(body["probability"])
    assert 0 < probability < 1
    assert body["probability"] == fmt(probability)
    assert len(body["distributions"]) == 81
    for peaks in body["distributions"]:
        total = sum(
            (parse_fraction(peak["probability"]) for peak in peaks),
            Fraction(0),
        )
        assert total == 1
    # The observation windows constrain the support at their steps.
    for step, lower, upper in ((20, 15, 25), (40, 10, 30), (80, 18, 22)):
        counts = [peak["heavy_count"] for peak in body["distributions"][step]]
        assert counts and min(counts) >= lower and max(counts) <= upper


@pytest.mark.parametrize(
    "mutation",
    [
        # n out of range / wrong type / missing
        lambda p: p.update(n=0),
        lambda p: p.update(n=41),
        lambda p: p.update(n=2.0),
        lambda p: p.update(n="2"),
        lambda p: p.update(n=True),
        lambda p: p.pop("n"),
        # m out of range / wrong type / missing
        lambda p: p.update(m=0),
        lambda p: p.update(m=41),
        lambda p: p.update(m="2"),
        lambda p: p.pop("m"),
        # h out of range / wrong type / missing
        lambda p: p.update(h=-1),
        lambda p: p.update(h=3),  # h > n
        lambda p: p.update(h=1.5),
        lambda p: p.update(h="1"),
        lambda p: p.pop("h"),
        # b out of range / wrong type / missing
        lambda p: p.update(b=-1),
        lambda p: p.update(b=3),  # b > m
        lambda p: p.update(b=False),
        lambda p: p.pop("b"),
        # steps out of range / wrong type / missing
        lambda p: p.update(steps=-1),
        lambda p: p.update(steps=81),
        lambda p: p.update(steps=2.5),
        lambda p: p.update(steps="2"),
        lambda p: p.pop("steps"),
        # observations missing / too many / wrong type
        lambda p: p.pop("observations"),
        lambda p: p.update(
            observations=[{"step": 0, "lower": 0, "upper": 0}] * 9
        ),
        lambda p: p.update(observations={"step": 0}),
        lambda p: p.update(observations=[None]),
        # duplicate observation steps
        lambda p: p.update(
            observations=[
                {"step": 1, "lower": 0, "upper": 1},
                {"step": 1, "lower": 1, "upper": 2},
            ]
        ),
        # observation step / bounds out of range or wrong type
        lambda p: p["observations"][0].update(step=3),  # step > steps
        lambda p: p["observations"][0].update(step=-1),
        lambda p: p["observations"][0].update(step=1.0),
        lambda p: p["observations"][0].update(step="2"),
        lambda p: p["observations"][0].update(upper=3),  # upper > n
        lambda p: p["observations"][0].update(lower=-1),
        lambda p: p["observations"][0].update(lower=2, upper=1),  # lower > upper
        lambda p: p["observations"][0].update(upper="2"),
        lambda p: p["observations"][0].update(lower=None),
        # missing / unknown observation fields
        lambda p: p["observations"][0].pop("lower"),
        lambda p: p["observations"][0].pop("step"),
        lambda p: p["observations"][0].update(extra=1),
        # unknown request field
        lambda p: p.update(extra=1),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/exchange-trajectory",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
