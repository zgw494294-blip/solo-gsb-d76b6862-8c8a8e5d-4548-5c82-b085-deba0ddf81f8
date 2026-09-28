"""Tests for the exchange-first-passage endpoint, including a brute-force cross-check."""

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


def brute_force_first_passage(payload: dict):
    """Exact first-passage inference by explicit exchange-sequence enumeration.

    Every step picks one of the n*m (molecule site, bath site) pairs, so each
    of the (n*m)**steps pick sequences is equally likely; sequences are
    enumerated recursively with the actual heavy-site sets tracked, and the
    first step at which the molecule heavy count reaches the threshold is
    recorded (0 when the initial count already qualifies; falling back below
    the threshold afterwards does not undo the hit). Every sequence passing
    all observation filters adds one to the joint count and to the bin of
    its first-hit step (or to the never bin). Returns the joint probability
    of the observations, the conditional first-hit distribution and the
    conditional never probability given all of them.
    """
    n, m, h, b, steps = (payload[k] for k in ("n", "m", "h", "b", "steps"))
    threshold = payload["threshold"]
    filters = {
        obs["step"]: (obs["lower"], obs["upper"])
        for obs in payload.get("observations", [])
    }
    z = 0
    hits: dict = {}
    never = 0

    def walk(t, mol_heavy, bath_heavy, tau):
        nonlocal z, never
        x = len(mol_heavy)
        if tau < 0 and x >= threshold:
            tau = t
        if t in filters:
            lower, upper = filters[t]
            if not lower <= x <= upper:
                return
        if t == steps:
            z += 1
            if tau < 0:
                never += 1
            else:
                hits[tau] = hits.get(tau, 0) + 1
            return
        for i in range(n):
            for j in range(m):
                i_heavy = i in mol_heavy
                j_heavy = j < bath_heavy  # bath sites 0..bath_heavy-1 heavy
                if i_heavy == j_heavy:
                    walk(t + 1, mol_heavy, bath_heavy, tau)
                elif i_heavy:  # heavy isotope leaves the molecule
                    walk(t + 1, mol_heavy - {i}, bath_heavy + 1, tau)
                else:  # heavy isotope enters the molecule
                    walk(t + 1, mol_heavy | {i}, bath_heavy - 1, tau)

    walk(0, frozenset(range(h)), b, -1)
    probability = Fraction(z, (n * m) ** steps)
    if z == 0:
        return probability, None, None
    distribution = {
        tau: Fraction(count, z) for tau, count in sorted(hits.items())
    }
    return probability, distribution, Fraction(never, z)


def basic_payload():
    return {
        "n": 2,
        "m": 2,
        "h": 1,
        "b": 1,
        "steps": 2,
        "threshold": 2,
        "observations": [],
    }


def test_initial_state_already_at_threshold_hits_at_zero():
    # h = 1 >= threshold = 1: the first-passage time is 0 regardless of the
    # later trajectory (here the count alternates deterministically 1, 0, 1).
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 1,
            "m": 1,
            "h": 1,
            "b": 0,
            "steps": 3,
            "threshold": 1,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 0, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_deterministic_hit_after_one_step():
    # n = m = 1 with the heavy isotope in the bath: the count alternates
    # 0, 1, 0, 1 deterministically, so the threshold is first reached at 1.
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 1,
            "m": 1,
            "h": 0,
            "b": 1,
            "steps": 3,
            "threshold": 1,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 1, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_threshold_zero_always_hits_at_zero():
    payload = basic_payload()
    payload["threshold"] = 0
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 0, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_single_step_hand_computed():
    # From x = 1 with n = m = 2 and total 2: X_1 is 0, 1, 2 with weights
    # 1/4, 1/2, 1/4, so the threshold 2 is first reached at step 1 with
    # probability 1/4 and never within one step with probability 3/4.
    payload = basic_payload()
    payload["steps"] = 1
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [{"step": 1, "probability": "1/4"}],
        "never_probability": "3/4",
    }


def test_fall_back_after_hit_still_counts_first_reaching():
    # n = m = 2, total 2, threshold 2 over two steps (integer numerators
    # up = [4, 1, 0], stay = [0, 2, 0], down = [0, 1, 4] over n*m = 4):
    # the path 1 -> 2 -> 1 (weight 1*4 = 4) reaches the threshold at step 1
    # and falls back below it at step 2, yet its first-passage time stays 1;
    # 1 -> 1 -> 2 (weight 2*1 = 2) first reaches at step 2; the remaining
    # weight 4 + 2 + 4 = 10 never reaches 2.
    resp = client.post("/v1/exchange-first-passage", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [
            {"step": 1, "probability": "1/4"},
            {"step": 2, "probability": "1/8"},
        ],
        "never_probability": "5/8",
    }


def test_observation_reweights_first_hit_times():
    # Same chain observing X_2 = 1: the surviving paths are 1 -> 0 -> 1 and
    # 1 -> 1 -> 1 (never hit, weight 4 + 4 = 8) and 1 -> 2 -> 1 (hit at 1,
    # weight 4), so the joint probability is 12/16 = 3/4 and the hit at
    # step 1 carries conditional probability 4/12 = 1/3.
    payload = basic_payload()
    payload["observations"] = [{"step": 2, "lower": 1, "upper": 1}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "3/4",
        "first_hit_distribution": [{"step": 1, "probability": "1/3"}],
        "never_probability": "2/3",
    }


def test_observation_can_force_late_first_hit():
    # Observing X_2 = 2 keeps only 1 -> 1 -> 2 (weight 2) and 1 -> 2 -> 2 is
    # impossible (from x = 2 the chain must move down), so the threshold is
    # first reached at step 2 with certainty given the observation.
    payload = basic_payload()
    payload["observations"] = [{"step": 2, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/8",
        "first_hit_distribution": [{"step": 2, "probability": "1/1"}],
        "never_probability": "0/1",
    }


def test_zero_probability_omits_distribution():
    # Step 0 is the initial state X_0 = h = 1; observing X_0 = 2 is
    # impossible.
    payload = basic_payload()
    payload["observations"] = [{"step": 0, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_steps_zero_below_threshold_never_hits():
    resp = client.post(
        "/v1/exchange-first-passage",
        json={
            "n": 3,
            "m": 2,
            "h": 1,
            "b": 1,
            "steps": 0,
            "threshold": 2,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "first_hit_distribution": [],
        "never_probability": "1/1",
    }


def test_randomized_matches_brute_force():
    rng = random.Random(20260928)
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
            "threshold": rng.randint(0, n),
            "observations": observations,
        }
        resp = client.post("/v1/exchange-first-passage", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        probability, distribution, never = brute_force_first_passage(payload)
        assert parse_fraction(body["probability"]) == probability
        if probability == 0:
            assert "first_hit_distribution" not in body
            assert "never_probability" not in body
            continue
        got = {
            peak["step"]: parse_fraction(peak["probability"])
            for peak in body["first_hit_distribution"]
        }
        assert got == distribution
        assert parse_fraction(body["never_probability"]) == never
        # Ascending by step, reduced fractions, none zero, and the
        # conditional first-hit probabilities plus the never probability
        # are normalized.
        steps_seen = [peak["step"] for peak in body["first_hit_distribution"]]
        assert steps_seen == sorted(steps_seen)
        for peak in body["first_hit_distribution"]:
            value = parse_fraction(peak["probability"])
            assert peak["probability"] == fmt(value)
            assert value > 0
        assert sum(got.values(), never) == 1


def test_large_request_stays_exact_and_fast():
    payload = {
        "n": 40,
        "m": 40,
        "h": 20,
        "b": 20,
        "steps": 80,
        "threshold": 30,
        "observations": [
            {"step": 20, "lower": 15, "upper": 25},
            {"step": 40, "lower": 10, "upper": 30},
            {"step": 60, "lower": 0, "upper": 40},
            {"step": 80, "lower": 18, "upper": 22},
        ],
    }
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    probability = parse_fraction(body["probability"])
    assert 0 < probability < 1
    assert body["probability"] == fmt(probability)
    steps_seen = [peak["step"] for peak in body["first_hit_distribution"]]
    assert steps_seen == sorted(steps_seen)
    total = parse_fraction(body["never_probability"])
    for peak in body["first_hit_distribution"]:
        value = parse_fraction(peak["probability"])
        assert peak["probability"] == fmt(value)
        assert value > 0
        total += value
    # The conditional first-hit distribution plus the never probability
    # sums to exactly 1.
    assert total == 1
    assert body["never_probability"] == fmt(parse_fraction(body["never_probability"]))


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
        # threshold out of range / wrong type / missing
        lambda p: p.update(threshold=-1),
        lambda p: p.update(threshold=3),  # threshold > n
        lambda p: p.update(threshold=1.5),
        lambda p: p.update(threshold="2"),
        lambda p: p.update(threshold=True),
        lambda p: p.update(threshold=None),
        lambda p: p.pop("threshold"),
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
        lambda p: p.update(
            observations=[{"step": 3, "lower": 0, "upper": 1}]  # step > steps
        ),
        lambda p: p.update(observations=[{"step": -1, "lower": 0, "upper": 1}]),
        lambda p: p.update(
            observations=[{"step": 1.0, "lower": 0, "upper": 1}]
        ),
        lambda p: p.update(
            observations=[{"step": 1, "lower": 0, "upper": 3}]  # upper > n
        ),
        lambda p: p.update(observations=[{"step": 1, "lower": -1, "upper": 1}]),
        lambda p: p.update(
            observations=[{"step": 1, "lower": 2, "upper": 1}]  # lower > upper
        ),
        lambda p: p.update(observations=[{"step": 1, "lower": 0, "upper": "2"}]),
        # missing / unknown observation fields
        lambda p: p.update(observations=[{"step": 1, "lower": 0}]),
        lambda p: p.update(
            observations=[{"step": 1, "lower": 0, "upper": 1, "extra": 1}]
        ),
        # unknown request field
        lambda p: p.update(extra=1),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/exchange-first-passage", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/exchange-first-passage",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
