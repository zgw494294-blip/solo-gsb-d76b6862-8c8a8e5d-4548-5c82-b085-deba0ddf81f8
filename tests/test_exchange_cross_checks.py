"""Cross-scenario checks for the three finite-bath exchange endpoints.

The trajectory, turnover and first-passage statistics are all produced by
the same integer-weight recurrence (one shared transition table, one
shared forward pass with per-step observation zeroing, the backward pass
reusing the same table); each endpoint only contributes its state
extension and result summarization. These tests do not re-enumerate any
exchange sequences: they check the three HTTP interfaces against each
other on shared scenarios, and check the shared core once directly.
"""

from __future__ import annotations

import random
from fractions import Fraction

from fastapi.testclient import TestClient

from app.main import (
    app,
    exchange_backward_recurrence,
    exchange_forward_recurrence,
    exchange_transition_table,
)

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def base_payload(rng: random.Random, steps: int) -> dict:
    n = rng.randint(1, 4)
    m = rng.randint(1, 4)
    obs_count = rng.randint(0, min(3, steps + 1))
    observations = []
    for step in rng.sample(range(steps + 1), obs_count):
        lower = rng.randint(0, n)
        upper = rng.randint(lower, n)
        observations.append(
            {"step": step, "lower": lower, "upper": upper}
        )
    return {
        "n": n,
        "m": m,
        "h": rng.randint(0, n),
        "b": rng.randint(0, m),
        "steps": steps,
        "observations": observations,
    }


def post_all(payload: dict, threshold: int):
    trajectory = client.post("/v1/exchange-trajectory", json=payload).json()
    turnover = client.post("/v1/exchange-turnover", json=payload).json()
    first_passage = dict(payload, threshold=threshold)
    passage = client.post(
        "/v1/exchange-first-passage", json=first_passage
    ).json()
    return trajectory, turnover, passage


def test_joint_probabilities_agree_across_three_statistics():
    # The state extension neither creates nor destroys mass: with the same
    # model parameters and the same observations all three endpoints must
    # report the exact same joint probability, randomly and with zero cases.
    rng = random.Random(20260928)
    for _ in range(60):
        steps = rng.randint(0, 5)
        payload = base_payload(rng, steps)
        threshold = rng.randint(0, payload["n"])
        trajectory, turnover, passage = post_all(payload, threshold)
        assert trajectory["probability"] == turnover["probability"]
        assert trajectory["probability"] == passage["probability"]


def test_zero_probability_omits_distributions_in_all_three():
    # A step-0 observation excluding the deterministic initial state is
    # impossible: all three endpoints answer "0/1" and every conditional
    # distribution field is omitted.
    rng = random.Random(11)
    for _ in range(10):
        payload = base_payload(rng, rng.randint(1, 4))
        payload["observations"] = [
            {"step": 0, "lower": payload["n"] + 0, "upper": payload["n"]}
        ]
        if payload["h"] == payload["n"]:
            payload["h"] -= 1
        threshold = rng.randint(0, payload["n"])
        trajectory, turnover, passage = post_all(payload, threshold)
        assert trajectory == {"probability": "0/1"}
        assert turnover == {"probability": "0/1"}
        assert passage == {"probability": "0/1"}


def test_readme_round_trip_scenario_three_ways():
    # The README's round-trip scenario (n = m = 2, h = b = 1, X_2 = 1):
    # joint probability 3/4 in every endpoint; the trajectory's conditional
    # endpoint law is 0 -> 1/4, 1 -> 1/2, 2 -> 1/4, turnover splits losses
    # 0 -> 1/3, 1 -> 2/3, and first passage at threshold 2 puts the hit at
    # step 1 on 1/3 of the conditional mass (the 1 -> 2 -> 1 round trip),
    # with 2/3 never reaching the threshold within two steps.
    payload = {
        "n": 2,
        "m": 2,
        "h": 1,
        "b": 1,
        "steps": 2,
        "observations": [{"step": 2, "lower": 1, "upper": 1}],
    }
    trajectory = client.post(
        "/v1/exchange-trajectory", json=payload
    ).json()
    assert trajectory["probability"] == "3/4"
    # The final-step observation pins X_2 = 1; one step earlier each of
    # 0, 1, 2 carries one surviving round-trip path of weight 4, so the
    # conditional law at step 1 is uniform over the three counts.
    assert trajectory["distributions"][0] == [
        {"heavy_count": 1, "probability": "1/1"}
    ]
    assert trajectory["distributions"][1] == [
        {"heavy_count": 0, "probability": "1/3"},
        {"heavy_count": 1, "probability": "1/3"},
        {"heavy_count": 2, "probability": "1/3"},
    ]
    assert trajectory["distributions"][2] == [
        {"heavy_count": 1, "probability": "1/1"}
    ]
    turnover = client.post("/v1/exchange-turnover", json=payload).json()
    assert turnover == {
        "probability": "3/4",
        "distribution": [
            {"count": 0, "probability": "1/3"},
            {"count": 1, "probability": "2/3"},
        ],
    }
    passage = client.post(
        "/v1/exchange-first-passage", json=dict(payload, threshold=2)
    ).json()
    assert passage == {
        "probability": "3/4",
        "first_hit_distribution": [
            {"step": 1, "probability": "1/3"}
        ],
        "never_probability": "2/3",
    }

    # The complementary README observation X_2 = 2 has joint probability
    # 1/8 in all three: the trajectory is pinned to 1, 1, 2, turnover sees
    # no down move (count 0 surely), and first passage at threshold 2 is
    # forced to step 2.
    payload["observations"] = [{"step": 2, "lower": 2, "upper": 2}]
    trajectory = client.post(
        "/v1/exchange-trajectory", json=payload
    ).json()
    turnover = client.post("/v1/exchange-turnover", json=payload).json()
    passage = client.post(
        "/v1/exchange-first-passage", json=dict(payload, threshold=2)
    ).json()
    assert trajectory["probability"] == turnover["probability"] == "1/8"
    assert passage["probability"] == "1/8"
    assert all(
        peaks[0]["probability"] == "1/1"
        for peaks in trajectory["distributions"]
    )
    assert turnover["distribution"] == [
        {"count": 0, "probability": "1/1"}
    ]
    assert passage["first_hit_distribution"] == [
        {"step": 2, "probability": "1/1"}
    ]
    assert passage["never_probability"] == "0/1"


def test_turnover_mean_losses_equal_down_probabilities_from_trajectory():
    # A pathwise identity between two endpoints, computed without any
    # sequence enumeration: the loss count increments exactly on down
    # transitions, so with no observations
    #   E[C_steps] = sum_{t=1..steps} sum_x P(X_{t-1} = x) * down(x)/(n*m)
    # where the one-step marginal laws come from the trajectory endpoint
    # and down(x) = x * (m - T + x) is the shared transition numerator.
    rng = random.Random(4242)
    for _ in range(30):
        payload = {
            "n": rng.randint(1, 4),
            "m": rng.randint(1, 4),
            "h": 0,
            "b": 0,
            "steps": rng.randint(0, 6),
            "observations": [],
        }
        payload["h"] = rng.randint(0, payload["n"])
        payload["b"] = rng.randint(0, payload["m"])
        n, m, total = payload["n"], payload["m"], payload["h"] + payload["b"]
        denom = n * m

        trajectory = client.post(
            "/v1/exchange-trajectory", json=payload
        ).json()
        expected = Fraction(0)
        for t in range(1, payload["steps"] + 1):
            law = {
                peak["heavy_count"]: parse_fraction(peak["probability"])
                for peak in trajectory["distributions"][t - 1]
            }
            for x, prob in law.items():
                down = x * (m - total + x)
                if down > 0:
                    expected += prob * down / denom

        turnover = client.post(
            "/v1/exchange-turnover", json=payload
        ).json()
        mean = sum(
            peak["count"] * parse_fraction(peak["probability"])
            for peak in turnover["distribution"]
        )
        assert mean == expected


def test_first_passage_agreement_with_trajectory_laws():
    # Identity linking the first-passage endpoint to the trajectory laws:
    # first reaching at step t requires X_t >= threshold, and never
    # reaching by `steps` requires X_steps < threshold, so under the same
    # observations (and hence the same conditional measure)
    #   P(tau = t | obs) <= P(X_t >= threshold | obs)   for every t,
    #   P(never | obs)   <= P(X_steps < threshold | obs).
    rng = random.Random(99)
    for _ in range(40):
        steps = rng.randint(0, 5)
        payload = base_payload(rng, steps)
        threshold = rng.randint(0, payload["n"])
        trajectory, _turnover, passage = post_all(payload, threshold)
        if trajectory["probability"] == "0/1":
            continue
        laws = [
            {
                peak["heavy_count"]: parse_fraction(peak["probability"])
                for peak in peaks
            }
            for peaks in trajectory["distributions"]
        ]
        hits = {
            peak["step"]: parse_fraction(peak["probability"])
            for peak in passage["first_hit_distribution"]
        }
        for t, prob in hits.items():
            tail = sum(
                (p for x, p in laws[t].items() if x >= threshold),
                Fraction(0),
            )
            assert prob <= tail
        never = parse_fraction(passage["never_probability"])
        below = sum(
            (p for x, p in laws[steps].items() if x < threshold),
            Fraction(0),
        )
        assert never <= below
        # Conditional first-hit probabilities plus the never probability
        # normalize to exactly 1.
        assert sum(hits.values(), never) == 1


def test_unreachable_threshold_never_hits():
    # The molecule can never hold more than the conserved total heavy
    # count: a threshold above that total is unreachable at every step, so
    # the first-passage endpoint answers an empty distribution and
    # never_probability 1/1 (whatever the observations), while its joint
    # probability still agrees with the other two endpoints.
    rng = random.Random(7)
    for _ in range(15):
        payload = base_payload(rng, rng.randint(1, 5))
        if payload["h"] + payload["b"] >= payload["n"]:
            continue
        threshold = payload["h"] + payload["b"] + 1
        trajectory, turnover, passage = post_all(payload, threshold)
        assert passage["probability"] == trajectory["probability"]
        assert passage["probability"] == turnover["probability"]
        if passage["probability"] != "0/1":
            assert passage["first_hit_distribution"] == []
            assert passage["never_probability"] == "1/1"


def test_threshold_zero_or_initial_hit_is_step_zero():
    # threshold = 0 (or any threshold <= h) is reached at step 0 with
    # conditional probability 1; the joint probability still equals the
    # one the trajectory and turnover endpoints report.
    rng = random.Random(123)
    for _ in range(20):
        payload = base_payload(rng, rng.randint(0, 5))
        threshold = rng.choice([0, payload["h"]])
        trajectory, _turnover, passage = post_all(payload, threshold)
        assert passage["probability"] == trajectory["probability"]
        if passage["probability"] != "0/1":
            assert passage["first_hit_distribution"] == [
                {"step": 0, "probability": "1/1"}
            ]
            assert passage["never_probability"] == "0/1"


def test_steps_zero_semantics_agree():
    # With steps = 0 there are no transitions: the trajectory is delta at
    # h, turnover is delta at 0 losses, first passage is delta at tau = 0
    # when h >= threshold and empty hits with never 1/1 otherwise — and
    # the joint probability (1/1 when the step-0 interval contains h,
    # else 0/1 with all distributions omitted) is identical everywhere.
    rng = random.Random(314)
    for _ in range(30):
        n = rng.randint(1, 4)
        m = rng.randint(1, 4)
        h = rng.randint(0, n)
        with_obs = rng.random() < 0.5
        observations = []
        if with_obs:
            lower = rng.randint(0, n)
            observations = [
                {"step": 0, "lower": lower, "upper": rng.randint(lower, n)}
            ]
        payload = {
            "n": n,
            "m": m,
            "h": h,
            "b": rng.randint(0, m),
            "steps": 0,
            "observations": observations,
        }
        threshold = rng.randint(0, n)
        trajectory, turnover, passage = post_all(payload, threshold)
        expected_prob = "1/1" if not observations or (
            observations[0]["lower"] <= h <= observations[0]["upper"]
        ) else "0/1"
        assert trajectory["probability"] == expected_prob
        assert turnover["probability"] == expected_prob
        assert passage["probability"] == expected_prob
        if expected_prob == "0/1":
            assert set(trajectory) == set(turnover) == set(passage) == {
                "probability"
            }
            continue
        assert trajectory["distributions"] == [
            [{"heavy_count": h, "probability": "1/1"}]
        ]
        assert turnover["distribution"] == [
            {"count": 0, "probability": "1/1"}
        ]
        if h >= threshold:
            assert passage["first_hit_distribution"] == [
                {"step": 0, "probability": "1/1"}
            ]
            assert passage["never_probability"] == "0/1"
        else:
            assert passage["first_hit_distribution"] == []
            assert passage["never_probability"] == "1/1"


def test_shared_core_table_forward_and_backward_weights():
    # Exercise the unified core directly on the README's worked example
    # (n = m = 2, h = b = 1): integer numerators up = [4, 1, 0],
    # stay = [0, 2, 0], down = [0, 1, 4] indexed by x, so the per-state
    # move table (stay/up/down order, zeros dropped) is:
    table = exchange_transition_table(2, 2, 2)
    assert table[0] == ((1, 4),)  # x = 0: only up (weight 4)
    assert table[1] == ((1, 2), (2, 1), (0, 1))
    assert table[2] == ((1, 4),)  # x = 2: only down (weight 4)

    # Observing X_2 = 2: forward weights are 1 at step 0 and the full
    # unfiltered birth-death spread at step 1, with only {2: 2} surviving
    # the step-2 filter — hence joint probability 2/4**2 = 1/8.
    forward = exchange_forward_recurrence(
        table,
        steps=2,
        observations=[(2, 2, 2)],
        initial={1: 1},
        heavy_of=lambda x: x,
        advance=lambda x, next_x, t: next_x,
    )
    assert forward == [{1: 1}, {0: 1, 1: 2, 2: 1}, {2: 2}]
    backward = exchange_backward_recurrence(table, forward, 2)
    # At step 1 only x = 1 can still reach the filtered state x = 2 in one
    # move (up numerator 1); from x = 2 the last filter already passed.
    assert backward[2] == {2: 1}
    assert backward[1] == {0: 0, 1: 1, 2: 0}
    assert backward[0] == {1: 2}

    # The state-extended recurrences reuse exactly the same table and
    # driver: turnover weights (x, c) and first passage weights (x, tau).
    losses = exchange_forward_recurrence(
        table,
        steps=2,
        observations=[(2, 1, 1)],
        initial={(1, 0): 1},
        heavy_of=lambda state: state[0],
        advance=lambda state, next_x, t: (
            next_x,
            state[1] + (1 if next_x < state[0] else 0),
        ),
    )
    assert losses[2] == {(1, 0): 4, (1, 1): 8}  # total surviving weight 12
    passages = exchange_forward_recurrence(
        table,
        steps=2,
        observations=[],
        initial={(1, -1): 1},
        heavy_of=lambda state: state[0],
        advance=lambda state, next_x, t: (
            (next_x, t) if state[1] < 0 and next_x >= 2 else (next_x, state[1])
        ),
    )
    final = passages[2]
    hit1 = sum(weight for (x, tau), weight in final.items() if tau == 1)
    hit2 = sum(weight for (x, tau), weight in final.items() if tau == 2)
    never = sum(weight for (x, tau), weight in final.items() if tau < 0)
    assert (hit1, hit2, never) == (4, 2, 10)
