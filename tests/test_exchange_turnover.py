"""Tests for the exchange-turnover endpoint, including a brute-force cross-check."""

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


def brute_force_turnover(payload: dict):
    """Exact loss-count inference by explicit exchange-sequence enumeration.

    Every step picks one of the n*m (molecule site, bath site) pairs, so each
    of the (n*m)**steps pick sequences is equally likely; sequences are
    enumerated recursively with the actual heavy-site sets tracked, and every
    down swap (a heavy isotope leaves the molecule) adds one to the running
    loss count. Every sequence passing all observation filters adds one to
    the joint count and to the bin of its total loss count. Returns the joint
    probability of the observations and the conditional distribution of the
    total loss count given all of them.
    """
    n, m, h, b, steps = (payload[k] for k in ("n", "m", "h", "b", "steps"))
    filters = {
        obs["step"]: (obs["lower"], obs["upper"])
        for obs in payload.get("observations", [])
    }
    z = 0
    losses: dict = {}

    def walk(t, mol_heavy, bath_heavy, lost):
        nonlocal z
        if t in filters:
            lower, upper = filters[t]
            if not lower <= len(mol_heavy) <= upper:
                return
        if t == steps:
            z += 1
            losses[lost] = losses.get(lost, 0) + 1
            return
        for i in range(n):
            for j in range(m):
                i_heavy = i in mol_heavy
                j_heavy = j < bath_heavy  # bath sites 0..bath_heavy-1 heavy
                if i_heavy == j_heavy:
                    walk(t + 1, mol_heavy, bath_heavy, lost)
                elif i_heavy:  # heavy isotope leaves the molecule
                    walk(t + 1, mol_heavy - {i}, bath_heavy + 1, lost + 1)
                else:  # heavy isotope enters the molecule
                    walk(t + 1, mol_heavy | {i}, bath_heavy - 1, lost)

    walk(0, frozenset(range(h)), b, 0)
    probability = Fraction(z, (n * m) ** steps)
    if z == 0:
        return probability, None
    distribution = {c: Fraction(count, z) for c, count in sorted(losses.items())}
    return probability, distribution


def basic_payload():
    return {
        "n": 2,
        "m": 2,
        "h": 1,
        "b": 1,
        "steps": 2,
        "observations": [{"step": 2, "lower": 1, "upper": 1}],
    }


def test_deterministic_repeated_losses_without_observations():
    # n = m = 1 with the heavy isotope on the molecule: every step swaps the
    # only molecule site with the only bath site, so the heavy count
    # alternates 1, 0, 1, 0, 1 and the isotope leaves the molecule at steps
    # 1 and 3 — repeated outflows are counted one by one.
    resp = client.post(
        "/v1/exchange-turnover",
        json={"n": 1, "m": 1, "h": 1, "b": 0, "steps": 4, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [{"count": 2, "probability": "1/1"}],
    }


def test_unconditional_single_step_hand_computed():
    # From x = 1 with n = m = 2 and total 2: the down numerator is 1 out of
    # n*m = 4 pairs (loss), stay is 2 and up is 1 (no loss), so the loss
    # count is 0 with probability 3/4 and 1 with probability 1/4.
    resp = client.post(
        "/v1/exchange-turnover",
        json={"n": 2, "m": 2, "h": 1, "b": 1, "steps": 1, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [
            {"count": 0, "probability": "3/4"},
            {"count": 1, "probability": "1/4"},
        ],
    }


def test_round_trip_paths_count_losses_separately():
    # Observing X_2 = 1 (same as X_0) keeps three paths: 1 -> 1 -> 1 (two
    # same-type swaps, weight 2*2 = 4, no loss), the round trip 1 -> 0 -> 1
    # (weight 1*4 = 4, one loss) and the round trip 1 -> 2 -> 1 (weight
    # 1*4 = 4, one loss). The net heavy-count change is zero on every path,
    # yet 2/3 of the conditional mass carries exactly one loss.
    resp = client.post("/v1/exchange-turnover", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "3/4",
        "distribution": [
            {"count": 0, "probability": "1/3"},
            {"count": 1, "probability": "2/3"},
        ],
    }


def test_losses_after_last_observation_keep_accumulating():
    # n = m = 1, heavy isotope on the molecule: the count alternates
    # 1, 0, 1, 0 deterministically. The only observation is the (certain)
    # initial state, but the outflows at steps 1 and 3 still count.
    resp = client.post(
        "/v1/exchange-turnover",
        json={
            "n": 1,
            "m": 1,
            "h": 1,
            "b": 0,
            "steps": 3,
            "observations": [{"step": 0, "lower": 1, "upper": 1}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [{"count": 2, "probability": "1/1"}],
    }


def test_zero_probability_omits_distribution():
    # From x = 2 with total 2 the chain must move down, so X_1 = 2 is
    # impossible.
    payload = basic_payload()
    payload.update(h=2, b=0, steps=1)
    payload["observations"] = [{"step": 1, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_step_zero_observation_filters_initial_state():
    # Step 0 is the initial state X_0 = h; observing a range that excludes h
    # makes the observations impossible.
    payload = basic_payload()
    payload["observations"] = [{"step": 0, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_steps_zero_returns_zero_count():
    resp = client.post(
        "/v1/exchange-turnover",
        json={"n": 3, "m": 2, "h": 2, "b": 1, "steps": 0, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [{"count": 0, "probability": "1/1"}],
    }


def test_same_type_swaps_do_not_count():
    # n = m = 2 with total 2 at x = 1: of the 4 site pairs, 2 swap equal
    # isotopes (no loss) and 1 moves a heavy isotope in (no loss); only the
    # single heavy-out pair counts, so observing X_1 = 1 leaves the loss
    # count at 0 with certainty.
    resp = client.post(
        "/v1/exchange-turnover",
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
    assert resp.json() == {
        "probability": "1/2",
        "distribution": [{"count": 0, "probability": "1/1"}],
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
        resp = client.post("/v1/exchange-turnover", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        probability, distribution = brute_force_turnover(payload)
        assert parse_fraction(body["probability"]) == probability
        if probability == 0:
            assert "distribution" not in body
            continue
        got = {
            peak["count"]: parse_fraction(peak["probability"])
            for peak in body["distribution"]
        }
        assert got == distribution
        # Ascending by count, reduced fractions, none zero, normalized.
        counts = [peak["count"] for peak in body["distribution"]]
        assert counts == sorted(counts)
        for peak in body["distribution"]:
            value = parse_fraction(peak["probability"])
            assert peak["probability"] == fmt(value)
            assert value > 0
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
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    probability = parse_fraction(body["probability"])
    assert 0 < probability < 1
    assert body["probability"] == fmt(probability)
    counts = [peak["count"] for peak in body["distribution"]]
    assert counts == sorted(counts)
    assert 0 <= counts[0] and counts[-1] <= 80
    total = Fraction(0)
    for peak in body["distribution"]:
        value = parse_fraction(peak["probability"])
        assert peak["probability"] == fmt(value)
        assert value > 0
        total += value
    assert total == 1


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
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/exchange-turnover",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
