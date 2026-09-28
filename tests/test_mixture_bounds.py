"""Tests for the mixture-bounds endpoint, including exact cross-checks."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_component(*isotopes, count=1):
    return {
        "atom_groups": [
            {
                "count": count,
                "isotopes": [
                    {"shift_mda": shift, "abundance": abundance}
                    for shift, abundance in isotopes
                ],
            }
        ]
    }


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def brute_force_envelope(component_payload: dict) -> dict[int, Fraction]:
    """Exact envelope of one component by explicit path enumeration."""
    dist = {0: Fraction(1)}
    for group in component_payload["atom_groups"]:
        outcomes = []
        for combo in itertools.product(group["isotopes"], repeat=group["count"]):
            shift = sum(iso["shift_mda"] for iso in combo)
            prob = Fraction(1)
            for iso in combo:
                prob *= parse_fraction(iso["abundance"])
            outcomes.append((shift, prob))
        merged = {}
        for (shift_a, prob_a), (shift_b, prob_b) in itertools.product(
            dist.items(), outcomes
        ):
            shift = shift_a + shift_b
            merged[shift] = merged.get(shift, Fraction(0)) + prob_a * prob_b
        dist = merged
    return {s: p for s, p in dist.items() if p != 0}


def test_two_components_forced_to_exact_point():
    payload = {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (2, "1/2")),
        ],
        "observations": [
            {"shift_mda": 1, "lower": "1/4", "upper": "1/1"},
            {"shift_mda": 2, "lower": "1/4", "upper": "1/1"},
        ],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    # w_a/2 >= 1/4 and w_b/2 >= 1/4 with w_a + w_b = 1 force w_a = w_b = 1/2.
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "1/2", "max": "1/2"},
            {"min": "1/2", "max": "1/2"},
        ],
        "weights": ["1/2", "1/2"],
    }


def test_two_components_interval_and_missing_peak_counts_as_zero():
    payload = {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (2, "1/2")),
        ],
        # Component 2 has no peak at shift 1: mixture prob there is w_1/2.
        "observations": [{"shift_mda": 1, "lower": "1/4", "upper": "1/2"}],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    # 1/4 <= w_1/2 <= 1/2  =>  w_1 in [1/2, 1], w_2 in [0, 1/2].
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "1/2", "max": "1/1"},
            {"min": "0/1", "max": "1/2"},
        ],
        "weights": ["1/2", "1/2"],
    }


def test_infeasible_returns_200_with_feasible_false():
    payload = {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (2, "1/2")),
        ],
        # Mixture prob at shift 1 is w_1/2 <= 1/2 < 3/4: impossible.
        "observations": [{"shift_mda": 1, "lower": "3/4", "upper": "1/1"}],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"feasible": False}


def test_three_components_lexicographically_smallest_weights():
    payload = {
        "components": [
            make_component((0, "1/1"), (9, "0/1")),
            make_component((1, "1/1"), (9, "0/1")),
            make_component((2, "1/1"), (9, "0/1")),
        ],
        "observations": [
            {"shift_mda": 0, "lower": "1/4", "upper": "1/1"},
            {"shift_mda": 1, "lower": "1/4", "upper": "1/1"},
        ],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    # w_1 >= 1/4, w_2 >= 1/4, w_3 >= 0; lexicographic minimum is (1/4, 1/4, 1/2).
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "1/4", "max": "3/4"},
            {"min": "1/4", "max": "3/4"},
            {"min": "0/1", "max": "1/2"},
        ],
        "weights": ["1/4", "1/4", "1/2"],
    }


def test_four_components_ranges_and_lex_min():
    payload = {
        "components": [
            make_component((0, "1/1"), (9, "0/1")),
            make_component((1, "1/1"), (9, "0/1")),
            make_component((2, "1/1"), (9, "0/1")),
            make_component((3, "1/1"), (9, "0/1")),
        ],
        "observations": [
            {"shift_mda": 0, "lower": "1/8", "upper": "1/1"},
            {"shift_mda": 1, "lower": "1/8", "upper": "1/1"},
            {"shift_mda": 2, "lower": "1/4", "upper": "1/1"},
        ],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "1/8", "max": "5/8"},
            {"min": "1/8", "max": "5/8"},
            {"min": "1/4", "max": "3/4"},
            {"min": "0/1", "max": "1/2"},
        ],
        "weights": ["1/8", "1/8", "1/4", "1/2"],
    }


def test_identical_envelopes_stay_independent_components():
    payload = {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/1"), (9, "0/1")),
        ],
        # Mixture prob at shift 1 is (w_1 + w_2)/2, so the observation only
        # constrains the sum w_1 + w_2 to [1/2, 1]. The two identical
        # components remain independent variables: each still ranges over
        # [0, 1] and the lex-min splits the required mass as 0, 1/2.
        "observations": [{"shift_mda": 1, "lower": "1/4", "upper": "1/2"}],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "0/1", "max": "1/1"},
            {"min": "0/1", "max": "1/1"},
            {"min": "0/1", "max": "1/2"},
        ],
        "weights": ["0/1", "1/2", "1/2"],
    }


def test_unobserved_shifts_add_no_constraint_and_no_renormalization():
    payload = {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (2, "1/2")),
        ],
        # Shift 0 accepts everything; shifts 1 and 2 stay unconstrained.
        "observations": [{"shift_mda": 0, "lower": "0/1", "upper": "1/1"}],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "0/1", "max": "1/1"},
            {"min": "0/1", "max": "1/1"},
        ],
        "weights": ["0/1", "1/1"],
    }


def test_observation_at_shift_absent_from_all_envelopes():
    components = [
        make_component((0, "1/2"), (1, "1/2")),
        make_component((0, "1/2"), (2, "1/2")),
    ]
    # Mixture prob at shift 7 is identically 0, which lies in [0, 1/2].
    resp = client.post(
        "/v1/mixture-bounds",
        json={
            "components": components,
            "observations": [{"shift_mda": 7, "lower": "0/1", "upper": "1/2"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["feasible"] is True

    # ...but not in [1/4, 1/2].
    resp = client.post(
        "/v1/mixture-bounds",
        json={
            "components": components,
            "observations": [{"shift_mda": 7, "lower": "1/4", "upper": "1/2"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"feasible": False}


def test_multi_group_component_uses_full_envelope():
    # Component 1 envelope: (1/2 + x/2)(1/2 + x^2/2) -> shifts 0..3 each 1/4.
    component1 = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/2"},
                ],
            },
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 2, "abundance": "1/2"},
                ],
            },
        ]
    }
    payload = {
        "components": [component1, make_component((0, "1/1"), (9, "0/1"))],
        # w_1/4 >= 1/4 forces w_1 = 1.
        "observations": [{"shift_mda": 3, "lower": "1/4", "upper": "1/1"}],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "1/1", "max": "1/1"},
            {"min": "0/1", "max": "0/1"},
        ],
        "weights": ["1/1", "0/1"],
    }


def test_output_fractions_are_reduced():
    payload = {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (2, "1/2")),
        ],
        # w_1/2 >= 2/6 = 1/3  =>  w_1 >= 2/3 (reported reduced).
        "observations": [{"shift_mda": 1, "lower": "2/6", "upper": "4/4"}],
    }
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "feasible": True,
        "ranges": [
            {"min": "2/3", "max": "1/1"},
            {"min": "0/1", "max": "1/3"},
        ],
        "weights": ["2/3", "1/3"],
    }


def expected_two_component(payload: dict) -> dict | None:
    """Independent exact solution for the 2-component case.

    With w_2 = 1 - w_1 every observation is a pair of linear inequalities
    in the single variable w_1, so the feasible set is an interval that can
    be tightened directly.
    """
    envs = [brute_force_envelope(c) for c in payload["components"]]
    lo, hi = Fraction(0), Fraction(1)
    for obs in payload["observations"]:
        prob_a = envs[0].get(obs["shift_mda"], Fraction(0))
        prob_b = envs[1].get(obs["shift_mda"], Fraction(0))
        coeff = prob_a - prob_b  # mixture prob = coeff * w_1 + prob_b
        for rhs, is_lower in (
            (parse_fraction(obs["lower"]) - prob_b, True),
            (parse_fraction(obs["upper"]) - prob_b, False),
        ):
            # coeff * w_1 >= rhs (lower side) or coeff * w_1 <= rhs (upper).
            if coeff == 0:
                if (is_lower and rhs > 0) or (not is_lower and rhs < 0):
                    return None
                continue
            bound = rhs / coeff
            if (coeff > 0) == is_lower:
                lo = max(lo, bound)
            else:
                hi = min(hi, bound)
    if lo > hi:
        return None
    return {
        "feasible": True,
        "ranges": [
            {"min": fmt(lo), "max": fmt(hi)},
            {"min": fmt(1 - hi), "max": fmt(1 - lo)},
        ],
        "weights": [fmt(lo), fmt(1 - lo)],
    }


ABUNDANCE_POOLS = [
    ["1/2", "1/2"],
    ["1/3", "2/3"],
    ["1/4", "3/4"],
    ["2/5", "3/5"],
    ["5/6", "1/6"],
    ["1/2", "1/4", "1/4"],
    ["1/3", "1/3", "1/3"],
    ["1/6", "1/3", "1/2"],
]
BOUND_POOL = [
    "0/1", "1/6", "1/5", "1/4", "1/3", "2/5",
    "1/2", "3/5", "2/3", "3/4", "4/5", "5/6", "1/1",
]


def random_component(rng: random.Random, shifts, count) -> dict:
    pool = rng.choice(ABUNDANCE_POOLS)
    chosen = rng.sample(list(shifts), len(pool))
    return make_component(*zip(chosen, pool), count=count)


def random_observations(rng: random.Random, n: int, shift_choices) -> list:
    observations = []
    for shift in rng.sample(list(shift_choices), n):
        lower, upper = sorted(
            (rng.choice(BOUND_POOL) for _ in range(2)), key=parse_fraction
        )
        observations.append({"shift_mda": shift, "lower": lower, "upper": upper})
    return observations


def test_two_components_randomized_cross_check():
    rng = random.Random(20260926)
    for _ in range(50):
        payload = {
            "components": [
                random_component(rng, range(0, 6), rng.randint(1, 3)),
                random_component(rng, range(0, 6), rng.randint(1, 3)),
            ],
            "observations": random_observations(
                rng, rng.randint(1, 4), range(0, 10)
            ),
        }
        resp = client.post("/v1/mixture-bounds", json=payload)
        assert resp.status_code == 200
        expected = expected_two_component(payload)
        if expected is None:
            assert resp.json() == {"feasible": False}
        else:
            assert resp.json() == expected


def expected_disjoint_supports(payload: dict) -> dict | None:
    """Independent exact oracle for components with disjoint shift supports.

    Every observation then constrains exactly one weight to an interval, so
    the feasible set is the simplex intersected with a box [L_k, U_k]:
    feasible iff sum(L) <= 1 <= sum(U), with closed-form extrema and a
    greedy lexicographic minimum.
    """
    envs = [brute_force_envelope(c) for c in payload["components"]]
    count = len(envs)
    lows = [Fraction(0)] * count
    highs = [Fraction(1)] * count
    for obs in payload["observations"]:
        owners = [k for k, env in enumerate(envs) if obs["shift_mda"] in env]
        assert len(owners) <= 1, "test construction requires disjoint supports"
        if not owners:
            # Mixture prob is identically 0 at this shift.
            if parse_fraction(obs["lower"]) > 0:
                return None
            continue
        k = owners[0]
        prob = envs[k][obs["shift_mda"]]
        lows[k] = max(lows[k], parse_fraction(obs["lower"]) / prob)
        highs[k] = min(highs[k], parse_fraction(obs["upper"]) / prob)
    if any(lo > hi for lo, hi in zip(lows, highs)):
        return None
    if sum(lows) > 1 or sum(highs) < 1:
        return None
    ranges = [
        {
            "min": fmt(max(lows[i], 1 - sum(highs) + highs[i])),
            "max": fmt(min(highs[i], 1 - sum(lows) + lows[i])),
        }
        for i in range(count)
    ]
    weights = []
    remaining = Fraction(1)
    for i in range(count):
        w = max(lows[i], remaining - sum(highs[i + 1 :]))
        weights.append(fmt(w))
        remaining -= w
    return {"feasible": True, "ranges": ranges, "weights": weights}


def test_disjoint_supports_randomized_cross_check():
    rng = random.Random(7)
    for _ in range(40):
        count = rng.randint(2, 4)
        # Component k draws isotope shifts from [3k, 3k+2]; with count <= 2
        # its envelope support stays within [6k, 6k+4]: disjoint across k.
        components = [
            random_component(rng, range(3 * k, 3 * k + 3), rng.randint(1, 2))
            for k in range(count)
        ]
        observations = random_observations(
            rng, rng.randint(1, 6), range(0, 6 * count + 4)
        )
        payload = {"components": components, "observations": observations}
        resp = client.post("/v1/mixture-bounds", json=payload)
        assert resp.status_code == 200
        expected = expected_disjoint_supports(payload)
        if expected is None:
            assert resp.json() == {"feasible": False}
        else:
            assert resp.json() == expected


def fm_eliminate(constraints, var):
    """Fourier-Motzkin elimination of one variable over Q.

    constraints: (coeffs, const) meaning const + coeffs·x >= 0. Returns the
    exact projection onto the remaining variables.
    """
    pos, neg, rest = [], [], []
    for coeffs, const in constraints:
        coeff = coeffs[var]
        others = coeffs[:var] + coeffs[var + 1 :]
        if coeff > 0:
            pos.append((coeff, others, const))
        elif coeff < 0:
            neg.append((-coeff, others, const))
        else:
            rest.append((others, const))
    out = list(rest)
    for cp, rp, kp in pos:
        for cn, rn, kn in neg:
            merged = tuple(cn * a + cp * b for a, b in zip(rp, rn))
            out.append((merged, cn * kp + cp * kn))
    return list(set(out))  # exact duplicates are redundant


def fm_min_max(constraints, dim, keep):
    """Exact (min, max) of x_keep over the polytope, or None if empty."""
    cons = constraints
    current = list(range(dim))
    for var in (v for v in range(dim) if v != keep):
        cons = fm_eliminate(cons, current.index(var))
        current.remove(var)
    lo = hi = None
    for coeffs, const in cons:
        coeff = coeffs[0]
        if coeff == 0:
            if const < 0:
                return None
        elif coeff > 0:
            bound = -const / coeff
            lo = bound if lo is None else max(lo, bound)
        else:
            bound = -const / coeff
            hi = bound if hi is None else min(hi, bound)
    assert lo is not None and hi is not None, "polytope must be bounded"
    if lo > hi:
        return None
    return lo, hi


def fm_oracle(payload: dict):
    """Independent exact solver via Fourier-Motzkin elimination.

    Uses the same reduced-space modeling as the app (weights w_1..w_{m-1}
    free, w_m = 1 - sum) but a completely different solution algorithm.
    """
    envs = [brute_force_envelope(c) for c in payload["components"]]
    count = len(envs)
    dim = count - 1

    def build_constraints(order):
        cons = []
        for i in range(dim):
            coeffs = [Fraction(0)] * dim
            coeffs[i] = Fraction(1)
            cons.append((tuple(coeffs), Fraction(0)))
        cons.append((tuple([Fraction(-1)] * dim), Fraction(1)))
        for obs in payload["observations"]:
            probs = [envs[idx].get(obs["shift_mda"], Fraction(0)) for idx in order]
            base = probs[-1]
            coeffs = tuple(probs[i] - base for i in range(dim))
            lower = parse_fraction(obs["lower"])
            upper = parse_fraction(obs["upper"])
            cons.append((coeffs, base - lower))
            cons.append((tuple(-c for c in coeffs), upper - base))
        return cons

    base_cons = build_constraints(list(range(count)))
    if fm_min_max(base_cons, dim, 0) is None:
        return None

    ranges = []
    for j in range(count):
        # Rotate component j to the front so its weight is a free variable.
        order = [j] + [k for k in range(count) if k != j]
        ranges.append(fm_min_max(build_constraints(order), dim, 0))

    # Lexicographic minimum: pin each prefix weight to its exact minimum.
    cons = list(base_cons)
    weights = []
    for i in range(dim):
        lo, _ = fm_min_max(cons, dim, i)
        weights.append(lo)
        coeffs = [Fraction(0)] * dim
        coeffs[i] = Fraction(1)
        cons.append((tuple(coeffs), -lo))
        cons.append((tuple(-c for c in coeffs), lo))
    weights.append(1 - sum(weights))
    return ranges, weights


def test_randomized_general_cross_check_fourier_motzkin():
    rng = random.Random(20260927)
    for _ in range(30):
        count = rng.randint(2, 4)
        components = [
            random_component(rng, range(0, 5), rng.randint(1, 2))
            for _ in range(count)
        ]
        observations = random_observations(rng, rng.randint(1, 5), range(0, 10))
        payload = {"components": components, "observations": observations}
        resp = client.post("/v1/mixture-bounds", json=payload)
        assert resp.status_code == 200
        expected = fm_oracle(payload)
        if expected is None:
            assert resp.json() == {"feasible": False}
        else:
            ranges, weights = expected
            assert resp.json() == {
                "feasible": True,
                "ranges": [
                    {"min": fmt(lo), "max": fmt(hi)} for lo, hi in ranges
                ],
                "weights": [fmt(w) for w in weights],
            }


def base_payload() -> dict:
    return {
        "components": [
            make_component((0, "1/2"), (1, "1/2")),
            make_component((0, "1/2"), (2, "1/2")),
        ],
        "observations": [{"shift_mda": 1, "lower": "1/4", "upper": "3/4"}],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        # component count out of range
        lambda p: p.update(components=p["components"][:1]),
        lambda p: p.update(components=p["components"] * 2 + p["components"][:1]),
        # observations count out of range
        lambda p: p.update(observations=[]),
        lambda p: p.update(
            observations=[
                {"shift_mda": i, "lower": "0/1", "upper": "1/1"} for i in range(9)
            ]
        ),
        # duplicate observation shifts
        lambda p: p["observations"].append(
            {"shift_mda": 1, "lower": "0/1", "upper": "1/1"}
        ),
        # observation shift out of range / wrong type
        lambda p: p["observations"][0].update(shift_mda=-1),
        lambda p: p["observations"][0].update(shift_mda=2001),
        lambda p: p["observations"][0].update(shift_mda=1.5),
        lambda p: p["observations"][0].update(shift_mda="1"),
        # bound format / range violations
        lambda p: p["observations"][0].update(lower="1/2/3"),
        lambda p: p["observations"][0].update(lower="a/b"),
        lambda p: p["observations"][0].update(lower="1/0"),
        lambda p: p["observations"][0].update(lower="1/1001"),
        lambda p: p["observations"][0].update(upper="5/4"),
        lambda p: p["observations"][0].update(lower="-1/2"),
        lambda p: p["observations"][0].update(lower=0.5),
        # lower > upper
        lambda p: p["observations"][0].update(lower="3/4", upper="1/4"),
        # component-level atom_groups validation is reused
        lambda p: p["components"][0]["atom_groups"][0]["isotopes"][0].update(
            abundance="1/3"
        ),
        lambda p: p["components"][0]["atom_groups"][0]["isotopes"][1].update(
            shift_mda=0
        ),
        lambda p: p["components"][0]["atom_groups"][0].update(count=0),
        lambda p: p["components"][0].update(
            atom_groups=p["components"][0]["atom_groups"] * 5
        ),
        lambda p: p["components"][0].update(
            atom_groups=[
                dict(p["components"][0]["atom_groups"][0], count=40)
                for _ in range(3)
            ]
        ),
        lambda p: p["components"][0]["atom_groups"][0]["isotopes"][1].update(
            shift_mda=2001
        ),
        lambda p: p["components"][0]["atom_groups"][0].update(
            isotopes=[{"shift_mda": 0, "abundance": "1/1"}]
        ),
        # unknown / missing fields
        lambda p: p.update(extra=1),
        lambda p: p["observations"][0].update(extra=1),
        lambda p: p["components"][0].update(extra=1),
        lambda p: p.pop("components"),
        lambda p: p.pop("observations"),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = base_payload()
    mutation(payload)
    resp = client.post("/v1/mixture-bounds", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/mixture-bounds",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
