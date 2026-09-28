"""Tests for the abundance-sensitivity endpoint, including a brute-force cross-check."""

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


def brute_force_derivative(payload: dict, transfer: dict) -> dict[int, Fraction]:
    """Exact dP(shift)/d(epsilon) at epsilon = 0 by explicit path enumeration.

    Every atom's isotope probability is a linear polynomial c0 + c1*epsilon
    (c1 is -1 for the donor isotope of the transferred group, +1 for the
    receptor, 0 otherwise). A path's probability polynomial is the product of
    its atoms' polynomials; only the coefficients of epsilon^0 and epsilon^1
    are tracked, since the epsilon^1 coefficient of the full envelope at
    epsilon = 0 is exactly the right derivative. Paths are enumerated with
    itertools.product and merged by total shift.
    """
    groups = payload["atom_groups"]
    g = transfer["group_index"]
    per_group_options = []
    for gi, group in enumerate(groups):
        options = []
        for ii, iso in enumerate(group["isotopes"]):
            coeff = Fraction(0)
            if gi == g:
                if ii == transfer["from_index"]:
                    coeff = Fraction(-1)
                elif ii == transfer["to_index"]:
                    coeff = Fraction(1)
            options.append((iso["shift_mda"], parse_fraction(iso["abundance"]), coeff))
        per_group_options.append(options)

    deriv: dict[int, Fraction] = {}
    choices = [
        itertools.product(per_group_options[gi], repeat=group["count"])
        for gi, group in enumerate(groups)
    ]
    for path in itertools.product(*choices):
        shift = 0
        p0, p1 = Fraction(1), Fraction(0)
        for group_path in path:
            for atom_shift, c0, c1 in group_path:
                shift += atom_shift
                p0, p1 = p0 * c0, p0 * c1 + p1 * c0
        deriv[shift] = deriv.get(shift, Fraction(0)) + p1
    return {shift: value for shift, value in deriv.items() if value != 0}


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
        "transfers": [{"group_index": 0, "from_index": 0, "to_index": 1}],
    }


def test_basic_single_group_hand_computed():
    resp = client.post("/v1/abundance-sensitivity", json=basic_payload())
    assert resp.status_code == 200
    # a0(eps) = 3/4 - eps, a1(eps) = 1/4 + eps with two independent atoms:
    # P(0) = a0^2 -> -2*a0 = -3/2; P(1) = 2*a0*a1 -> 2*(a0 - a1) = 1;
    # P(2) = a1^2 -> 2*a1 = 1/2.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-3/2"},
                {"shift_mda": 1, "derivative": "1/1"},
                {"shift_mda": 2, "derivative": "1/2"},
            ]
        ]
    }


def test_reverse_transfer_flips_signs():
    payload = basic_payload()
    payload["transfers"] = [{"group_index": 0, "from_index": 1, "to_index": 0}]
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "3/2"},
                {"shift_mda": 1, "derivative": "-1/1"},
                {"shift_mda": 2, "derivative": "-1/2"},
            ]
        ]
    }


def test_multiple_transfers_in_input_order_and_independent():
    payload = basic_payload()
    payload["transfers"] = [
        {"group_index": 0, "from_index": 0, "to_index": 1},
        {"group_index": 0, "from_index": 1, "to_index": 0},
        {"group_index": 0, "from_index": 0, "to_index": 1},
    ]
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    derivatives = resp.json()["derivatives"]
    assert len(derivatives) == 3
    # Each transfer is applied to the unperturbed envelope on its own.
    assert derivatives[0] == derivatives[2]
    forward = {p["shift_mda"]: p["derivative"] for p in derivatives[0]}
    backward = {p["shift_mda"]: p["derivative"] for p in derivatives[1]}
    assert set(forward) == set(backward)
    for shift in forward:
        assert parse_fraction(forward[shift]) == -parse_fraction(backward[shift])


def test_zero_derivative_entries_are_omitted():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/2"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 0, "to_index": 1}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    # dP(1)/d(eps) = 2*(a0 - a1) = 0 exactly, so shift 1 is left out.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-1/1"},
                {"shift_mda": 2, "derivative": "1/1"},
            ]
        ]
    }


def test_zero_baseline_peak_with_nonzero_derivative_is_kept():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/1"},
                    {"shift_mda": 3, "abundance": "0/1"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 0, "to_index": 1}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    # Baseline envelope is a single peak at 0, but P(3) = 2*(1-eps)*eps has
    # derivative 2, so the zero-baseline peak at shift 3 must be listed.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-2/1"},
                {"shift_mda": 3, "derivative": "2/1"},
            ]
        ]
    }


def test_multi_group_hand_computed():
    payload = {
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
                    {"shift_mda": 0, "abundance": "1/3"},
                    {"shift_mda": 2, "abundance": "2/3"},
                ],
            },
        ],
        "transfers": [{"group_index": 1, "from_index": 0, "to_index": 1}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    # Group 1's derivative is -1 at shift 0 and +1 at shift 2; convolving
    # with group 0's {0: 1/2, 1: 1/2} gives the four entries below.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-1/2"},
                {"shift_mda": 1, "derivative": "-1/2"},
                {"shift_mda": 2, "derivative": "1/2"},
                {"shift_mda": 3, "derivative": "1/2"},
            ]
        ]
    }


def test_derivatives_sum_to_zero():
    # Total probability is 1 for every epsilon, so the derivatives sum to 0.
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
        "transfers": [
            {"group_index": 0, "from_index": 0, "to_index": 2},
            {"group_index": 0, "from_index": 1, "to_index": 0},
            {"group_index": 1, "from_index": 0, "to_index": 1},
        ],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    for peaks in resp.json()["derivatives"]:
        total = sum(
            (parse_fraction(peak["derivative"]) for peak in peaks), Fraction(0)
        )
        assert total == 0


def test_randomized_matches_brute_force():
    rng = random.Random(20260927)
    for _ in range(40):
        group_count = rng.randint(1, 2)
        groups = []
        for _ in range(group_count):
            count = rng.randint(1, 3)
            kind = rng.randint(2, 3)
            shifts = rng.sample(range(0, 5), kind)
            cuts = sorted(rng.sample(range(1, 12), kind - 1))
            abundances = [
                Fraction(b - a, 12) for a, b in zip([0] + cuts, cuts + [12])
            ]
            if rng.random() < 0.3:
                # Force a zero-abundance isotope (valid donor still exists).
                abundances[rng.randrange(kind)] = Fraction(0)
                abundances[0] = 1 - sum(abundances[1:], Fraction(0))
            groups.append(
                {
                    "count": count,
                    "isotopes": [
                        {"shift_mda": shift, "abundance": fmt(abundance)}
                        for shift, abundance in zip(shifts, abundances)
                    ],
                }
            )
        transfers = []
        for _ in range(rng.randint(1, 3)):
            g = rng.randrange(group_count)
            donor_pool = [
                i
                for i, iso in enumerate(groups[g]["isotopes"])
                if parse_fraction(iso["abundance"]) > 0
            ]
            from_index = rng.choice(donor_pool)
            to_index = rng.choice(
                [i for i in range(len(groups[g]["isotopes"])) if i != from_index]
            )
            transfers.append(
                {
                    "group_index": g,
                    "from_index": from_index,
                    "to_index": to_index,
                }
            )
        payload = {"atom_groups": groups, "transfers": transfers}
        resp = client.post("/v1/abundance-sensitivity", json=payload)
        assert resp.status_code == 200
        derivatives = resp.json()["derivatives"]
        assert len(derivatives) == len(transfers)
        for transfer, peaks in zip(transfers, derivatives):
            expected = brute_force_derivative(payload, transfer)
            got = {
                peak["shift_mda"]: parse_fraction(peak["derivative"])
                for peak in peaks
            }
            assert got == expected
            # Ascending by shift, reduced signed fractions, none zero.
            shifts = [peak["shift_mda"] for peak in peaks]
            assert shifts == sorted(shifts)
            for peak in peaks:
                assert peak["derivative"] == fmt(parse_fraction(peak["derivative"]))
                assert parse_fraction(peak["derivative"]) != 0


def test_derivative_consistent_with_envelope_perturbation():
    # Cross-check against the envelope endpoint with an exact small epsilon:
    # the secant slope (P(eps) - P(0)) / eps of each peak must match the
    # reported derivative up to the quadratic remainder (count is 2, so
    # every P(shift) is a quadratic in eps with |second-order coefficient|
    # <= 2, and the gap is exactly that coefficient times eps).
    payload = basic_payload()
    eps = Fraction(1, 1000)
    base = client.post(
        "/v1/envelope", json={"atom_groups": payload["atom_groups"], "k": 20}
    ).json()
    perturbed_groups = [
        {
            "count": 2,
            "isotopes": [
                {"shift_mda": 0, "abundance": fmt(Fraction(3, 4) - eps)},
                {"shift_mda": 1, "abundance": fmt(Fraction(1, 4) + eps)},
            ],
        }
    ]
    moved = client.post(
        "/v1/envelope", json={"atom_groups": perturbed_groups, "k": 20}
    ).json()
    base_probs = {p["shift_mda"]: parse_fraction(p["probability"]) for p in base["peaks"]}
    moved_probs = {p["shift_mda"]: parse_fraction(p["probability"]) for p in moved["peaks"]}
    got = {
        p["shift_mda"]: parse_fraction(p["derivative"])
        for p in client.post("/v1/abundance-sensitivity", json=payload).json()[
            "derivatives"
        ][0]
    }
    for shift in set(base_probs) | set(moved_probs):
        diff = moved_probs.get(shift, Fraction(0)) - base_probs.get(shift, Fraction(0))
        # P(eps) is a quadratic in eps here (count 2), so the secant slope
        # differs from the derivative by at most 2 * eps.
        assert abs(diff / eps - got.get(shift, Fraction(0))) <= 2 * eps


@pytest.mark.parametrize(
    "mutation",
    [
        # transfers missing / empty / too many / wrong type
        lambda p: p.pop("transfers"),
        lambda p: p.update(transfers=[]),
        lambda p: p.update(
            transfers=[{"group_index": 0, "from_index": 0, "to_index": 1}] * 9
        ),
        lambda p: p.update(transfers={"group_index": 0}),
        lambda p: p.update(transfers=[None]),
        # group_index out of range / negative / wrong type
        lambda p: p["transfers"][0].update(group_index=1),
        lambda p: p["transfers"][0].update(group_index=-1),
        lambda p: p["transfers"][0].update(group_index=0.0),
        lambda p: p["transfers"][0].update(group_index="0"),
        lambda p: p["transfers"][0].update(group_index=True),
        # from_index / to_index out of range / negative / wrong type
        lambda p: p["transfers"][0].update(from_index=2),
        lambda p: p["transfers"][0].update(to_index=2),
        lambda p: p["transfers"][0].update(from_index=-1),
        lambda p: p["transfers"][0].update(to_index=1.5),
        lambda p: p["transfers"][0].update(to_index="1"),
        lambda p: p["transfers"][0].update(from_index=None),
        # from_index == to_index
        lambda p: p["transfers"][0].update(to_index=0),
        # missing transfer fields / unknown transfer field
        lambda p: p["transfers"][0].pop("from_index"),
        lambda p: p["transfers"][0].update(extra=1),
        # unknown request field
        lambda p: p.update(extra=1),
        # atom_groups violations reused verbatim from /v1/envelope
        lambda p: p["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        lambda p: p["atom_groups"][0].update(count=0),
        lambda p: p.update(atom_groups=[]),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 422


def test_zero_abundance_donor_returns_422():
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/1"},
                    {"shift_mda": 5, "abundance": "0/1"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 1, "to_index": 0}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/abundance-sensitivity",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
