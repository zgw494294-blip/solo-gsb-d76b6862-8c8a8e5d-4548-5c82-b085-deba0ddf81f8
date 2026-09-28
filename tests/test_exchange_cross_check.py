"""Cross-interface checks for the three finite-bath exchange endpoints.

The three statistics share one integer-weight transition recursion, so on
cross scenarios they must agree with each other (not only with the
brute-force solvers in their own test files):

* the joint probability of identical observations is the same for
  ``/v1/exchange-trajectory``, ``/v1/exchange-turnover`` and
  ``/v1/exchange-first-passage`` (and all three omit their distribution
  fields exactly when it is zero);
* the first-passage mass at step t equals the trajectory joint
  probability of the pinning event "below the threshold at steps
  0, ..., t-1 and at or above it at step t", both unconditionally and
  with additional observations strictly after t (the extra statistic
  must factor out of the shared recursion);
* the turnover mean (expected number of heavy-isotope outflows) equals
  the down-step probability summed over the trajectory marginals.
"""

from __future__ import annotations

import random
from fractions import Fraction

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINTS = (
    "/v1/exchange-trajectory",
    "/v1/exchange-turnover",
    "/v1/exchange-first-passage",
)
DISTRIBUTION_FIELDS = (
    "distributions",
    "distribution",
    "first_hit_distribution",
)


def parse_fraction(text: str) -> Fraction:
    numerator, denominator = text.split("/")
    return Fraction(int(numerator), int(denominator))


def post(endpoint: str, payload: dict) -> dict:
    body = dict(payload)
    if endpoint == ENDPOINTS[2]:
        body["threshold"] = payload.get("threshold", payload["n"])
    else:
        body.pop("threshold", None)  # extra fields are rejected (422)
    response = client.post(endpoint, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_zero_step_views_agree_across_the_three_interfaces():
    payload = {"n": 3, "m": 2, "h": 2, "b": 1, "steps": 0, "observations": []}

    trajectory = post(ENDPOINTS[0], payload)
    turnover = post(ENDPOINTS[1], payload)
    first = post(ENDPOINTS[2], {**payload, "threshold": 3})

    assert trajectory["probability"] == "1/1"
    assert trajectory["distributions"] == [
        [{"heavy_count": 2, "probability": "1/1"}]
    ]
    assert turnover["distribution"] == [{"count": 0, "probability": "1/1"}]
    assert first["first_hit_distribution"] == []
    assert first["never_probability"] == "1/1"

    first_hit = post(ENDPOINTS[2], {**payload, "threshold": 2})
    assert first_hit["first_hit_distribution"] == [
        {"step": 0, "probability": "1/1"}
    ]
    assert first_hit["never_probability"] == "0/1"

    # An impossible step-0 observation is zero probability everywhere and
    # every endpoint omits its distribution field.
    impossible = dict(
        payload,
        threshold=3,
        observations=[{"step": 0, "lower": 3, "upper": 3}],
    )
    for endpoint, field in zip(ENDPOINTS, DISTRIBUTION_FIELDS):
        body = post(endpoint, impossible)
        assert body == {"probability": "0/1"}
        assert field not in body
    assert "never_probability" not in post(ENDPOINTS[2], impossible)


def test_randomized_identical_observations_give_identical_joint_probability():
    rng = random.Random(20260928)
    for _ in range(60):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 5)
        observations = []
        for step in rng.sample(range(steps + 1), rng.randint(0, min(3, steps + 1))):
            lower = rng.randint(0, n)
            observations.append(
                {
                    "step": step,
                    "lower": lower,
                    "upper": rng.randint(lower, n),
                }
            )
        payload = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "observations": observations,
        }
        bodies = [post(endpoint, payload) for endpoint in ENDPOINTS]
        probabilities = {body["probability"] for body in bodies}
        assert len(probabilities) == 1
        probability = parse_fraction(next(iter(probabilities)))

        if probability == 0:
            for body, field in zip(bodies, DISTRIBUTION_FIELDS):
                assert field not in body
            assert "never_probability" not in bodies[2]
        else:
            for body, field in zip(bodies, DISTRIBUTION_FIELDS):
                assert field in body
            assert "never_probability" in bodies[2]


def test_first_passage_masses_match_trajectory_pinning_without_observations():
    # P(tau = t) under no observations equals the joint probability of the
    # trajectory pins X_0, ..., X_{t-1} < K and X_t >= K, and the never
    # probability equals the joint probability of X_s < K for every s.
    rng = random.Random(20260928)
    for _ in range(40):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 5)
        threshold = rng.randint(1, n)
        base = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "observations": [],
        }
        first = post(
            ENDPOINTS[2], {**base, "threshold": threshold}
        )
        hits = {
            peak["step"]: parse_fraction(peak["probability"])
            for peak in first["first_hit_distribution"]
        }
        never = parse_fraction(first["never_probability"])

        for t in range(steps + 1):
            pins = [
                {"step": s, "lower": 0, "upper": threshold - 1}
                for s in range(t)
            ]
            pins.append({"step": t, "lower": threshold, "upper": n})
            pinned = post(ENDPOINTS[0], {**base, "observations": pins})
            assert parse_fraction(pinned["probability"]) == hits.get(t, Fraction(0))

        never_pins = [
            {"step": s, "lower": 0, "upper": threshold - 1}
            for s in range(steps + 1)
        ]
        pinned_never = post(
            ENDPOINTS[0], {**base, "observations": never_pins}
        )
        assert parse_fraction(pinned_never["probability"]) == never


def test_first_passage_masses_match_trajectory_pinning_with_later_observations():
    # With observations only at steps strictly after t, P(tau = t, obs)
    # reported by first passage must equal the trajectory joint probability
    # of the same observations plus the t-pinning event: conditioning the
    # extra first-hit dimension must not change the shared transition weights.
    rng = random.Random(20260928)
    for _ in range(40):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(1, 4)
        threshold = rng.randint(1, n)
        t = rng.randint(1, steps)
        later_steps = rng.sample(
            range(t + 1, steps + 1),
            rng.randint(0, min(2, steps - t)),
        )
        later = []
        for step in later_steps:
            lower = rng.randint(0, n)
            later.append(
                {"step": step, "lower": lower, "upper": rng.randint(lower, n)}
            )
        base = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "observations": later,
        }
        first = post(ENDPOINTS[2], {**base, "threshold": threshold})
        joint = parse_fraction(first["probability"])
        if joint == 0:
            # Impossible later observations: the pinned event is impossible
            # under the shared chain as well.
            assert "first_hit_distribution" not in first
            pins = [
                {"step": s, "lower": 0, "upper": threshold - 1}
                for s in range(t)
            ]
            pins.append({"step": t, "lower": threshold, "upper": n})
            pinned = post(
                ENDPOINTS[0], {**base, "observations": later + pins}
            )
            assert pinned["probability"] == "0/1"
            return
        hit_mass = Fraction(0)
        for peak in first["first_hit_distribution"]:
            if peak["step"] == t:
                hit_mass = joint * parse_fraction(peak["probability"])

        pins = [
            {"step": s, "lower": 0, "upper": threshold - 1}
            for s in range(t)
        ]
        pins.append({"step": t, "lower": threshold, "upper": n})
        pinned = post(
            ENDPOINTS[0], {**base, "observations": later + pins}
        )
        assert parse_fraction(pinned["probability"]) == hit_mass


def test_turnover_mean_matches_trajectory_down_step_probabilities():
    # Without observations each down transition is one loss, so the expected
    # total loss count from turnover equals sum_t sum_x P(X_{t-1} = x) *
    # down(x)/(n*m), read entirely off the trajectory marginals.
    rng = random.Random(20260928)
    for _ in range(40):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 6)
        h = rng.randint(0, n)
        b = rng.randint(0, m)
        total = h + b
        base = {
            "n": n,
            "m": m,
            "h": h,
            "b": b,
            "steps": steps,
            "observations": [],
        }
        trajectory = post(ENDPOINTS[0], base)
        turnover = post(ENDPOINTS[1], base)
        marginals = [
            {
                peak["heavy_count"]: parse_fraction(peak["probability"])
                for peak in step_peaks
            }
            for step_peaks in trajectory["distributions"]
        ]

        expected_losses = Fraction(0)
        denom = n * m
        for t in range(1, steps + 1):
            for x, probability in marginals[t - 1].items():
                down_numerator = x * (m - total + x)
                expected_losses += probability * Fraction(down_numerator, denom)

        mean = sum(
            (
                peak["count"] * parse_fraction(peak["probability"])
                for peak in turnover["distribution"]
            ),
            Fraction(0),
        )
        assert mean == expected_losses
        # The turnover law is a proper distribution.
        assert sum(
            (parse_fraction(peak["probability"]) for peak in turnover["distribution"]),
            Fraction(0),
        ) == 1
