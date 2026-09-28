"""Isotope peak-envelope calculation API.

Computes the exact isotope fine-structure envelope for independent atom
groups using rational (Fraction) arithmetic. No pruning is applied at any
stage: every path is merged by mass shift and kept with its exact
probability. A second endpoint solves for mixture weights of several
components against observed peak-probability intervals, also exactly over Q.
A third endpoint attributes a single observed peak: its exact probability
and, conditioned on observing that shift, the expected atom count of every
isotope in every group, again exactly over Q. A fourth endpoint plans
acquisition windows for the full envelope: at most K disjoint closed
intervals within a bin budget, maximizing exactly captured probability with
fully specified exact tie-breaking. A fifth endpoint samples a finite number
of ions independently from the full envelope and computes, exactly over Q,
the joint probability that every (possibly overlapping) detection window
collects at least its required number of hits, together with each window's
marginal success probability. A sixth endpoint performs exact Bayesian
candidate/offset inference: each of 2-4 mutually exclusive candidates
carries its own envelope and an ``n/d`` prior, all ions share one
uniform-prior integer calibration offset, and the exact multinomial
likelihood of binned ion counts yields joint, per-candidate and per-offset
posteriors, all reduced over Q. A seventh endpoint plans acquisition
windows that discriminate between two equal-prior candidates: at most K
disjoint closed intervals within a bin budget, exactly maximizing the
accuracy (1 + |p0 - p1|) / 2 of the rule "an ion inside the windows votes
for the candidate with the higher hit probability, an ion outside votes
for the other one", with fully specified exact tie-breaking. An eighth
endpoint conditions on an observed parent-ion shift: the parent atoms take
isotopes independently, then each group draws a fixed number of its atoms
uniformly without replacement to form a fragment (the rest are the
residual); summed over all isotope assignments and draw paths, it returns
the parent peak probability and the exact conditional distribution of the
fragment shift, again exactly over Q. A ninth endpoint computes abundance
sensitivities: each of 1-8 transfers moves an infinitesimal epsilon of
abundance from one isotope (donor) to another (receptor) inside one group,
and the exact right derivative dP(shift)/d(epsilon) at epsilon = 0 of the
full envelope is returned per transfer — symbolically, never by finite
differences, and keeping peaks whose baseline probability is zero but whose
derivative is not. A tenth endpoint infers finite-bath exchange
trajectories: a molecule of n sites and a bath of m sites start with h and
b heavy isotopes, and each step swaps one uniformly and independently
chosen molecule site with one bath site (same-type swaps count as steps,
the bath updates synchronously and the total heavy count is conserved), so
the molecule heavy count is a birth-death chain; given closed-interval
observations of that count at distinct steps it returns their exact joint
probability and, conditioned on all of them, the exact heavy-count
distribution at every step, again exactly over Q and never enumerating
exchange sequences. An eleventh endpoint counts heavy-isotope losses in
the same finite-bath exchange: every swap that moves a heavy isotope out
of the molecule (a down transition) adds one to the loss count while
same-type swaps and up transitions add nothing, and it returns the exact
joint probability of the observations together with the exact conditional
distribution of the total loss count over the whole trajectory, tracked
as an extra dimension of the same integer-weight forward recursion.
A twelfth endpoint computes first-passage times in the same finite-bath
exchange: the first step at which the molecule heavy count reaches a
threshold (0 when the initial count already does, and falling back below
the threshold afterwards does not undo the hit), returning the exact
joint probability of the observations and, conditioned on all of them,
the exact distribution of the first-hit step together with the
probability of never reaching the threshold within the horizon, again
tracked as an extra dimension of the same integer-weight forward
recursion.
"""

from __future__ import annotations

import itertools
import re
import sys
from fractions import Fraction
from math import comb, gcd
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

# Detection probabilities are exact fractions whose reduced denominator can
# reach (lcm of abundance denominators)**(total atoms * ions) — tens of
# thousands of digits — and must be printed in full, so lift Python's
# default 4300-digit int<->str conversion cap well past that bound.
sys.set_int_max_str_digits(1_000_000)

ABUNDANCE_PATTERN = re.compile(r"^(\d{1,4})/(\d{1,4})$")
MAX_DENOMINATOR = 1000

# A probability distribution over total mass shift (mDa), exact rationals.
Distribution = Dict[int, Fraction]


def parse_abundance(value: str) -> Fraction:
    """Parse an ``n/d`` abundance string into a Fraction.

    Enforces 0 <= n <= d and 1 <= d <= 1000.
    """
    match = ABUNDANCE_PATTERN.match(value)
    if match is None:
        raise ValueError("abundance must be a string of the form 'n/d'")
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if not 1 <= denominator <= MAX_DENOMINATOR:
        raise ValueError("abundance denominator d must satisfy 1 <= d <= 1000")
    if numerator > denominator:
        raise ValueError("abundance must satisfy 0 <= n <= d")
    return Fraction(numerator, denominator)


class Isotope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shift_mda: int = Field(strict=True, ge=0)
    abundance: str

    @field_validator("abundance")
    @classmethod
    def _validate_abundance(cls, value: str) -> str:
        parse_abundance(value)
        return value

    @property
    def abundance_fraction(self) -> Fraction:
        return parse_abundance(self.abundance)


class AtomGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(strict=True, ge=1, le=40)
    isotopes: List[Isotope] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def _validate_group(self) -> "AtomGroup":
        shifts = [iso.shift_mda for iso in self.isotopes]
        if len(set(shifts)) != len(shifts):
            raise ValueError("shift_mda values within one group must be distinct")
        total = sum((iso.abundance_fraction for iso in self.isotopes), Fraction(0))
        if total != 1:
            raise ValueError("isotope abundances within one group must sum to exactly 1")
        return self


def validate_atom_groups(groups: List[AtomGroup]) -> None:
    """Cross-group constraints shared by every atom_groups list."""
    if sum(group.count for group in groups) > 80:
        raise ValueError("total count across all groups must be <= 80")
    max_total_shift = sum(
        group.count * max(iso.shift_mda for iso in group.isotopes)
        for group in groups
    )
    if max_total_shift > 2000:
        raise ValueError("maximum possible total shift must be <= 2000 mDa")


class EnvelopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    k: int = Field(strict=True, ge=1, le=20)

    @model_validator(mode="after")
    def _validate_request(self) -> "EnvelopeRequest":
        validate_atom_groups(self.atom_groups)
        return self


class Peak(BaseModel):
    shift_mda: int
    probability: str  # reduced fraction "n/d"


class EnvelopeResponse(BaseModel):
    peaks: List[Peak]
    top_peaks: List[Peak]


class PeakAttributionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    target_shift_mda: int = Field(strict=True, ge=0, le=2000)

    @model_validator(mode="after")
    def _validate_request(self) -> "PeakAttributionRequest":
        validate_atom_groups(self.atom_groups)
        return self


class PeakAttributionResponse(BaseModel):
    reachable: bool
    probability: str  # reduced fraction "n/d"
    # expected_counts[g][i]: expected atoms of isotope i in group g, in the
    # input ordering, conditioned on the target shift; reduced "n/d".
    expected_counts: Optional[List[List[str]]] = None


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shift_mda: int = Field(strict=True, ge=0, le=2000)
    lower: str
    upper: str

    @field_validator("lower", "upper")
    @classmethod
    def _validate_bound(cls, value: str) -> str:
        parse_abundance(value)
        return value

    @model_validator(mode="after")
    def _validate_bounds(self) -> "Observation":
        if self.lower_fraction > self.upper_fraction:
            raise ValueError("lower must be <= upper")
        return self

    @property
    def lower_fraction(self) -> Fraction:
        return parse_abundance(self.lower)

    @property
    def upper_fraction(self) -> Fraction:
        return parse_abundance(self.upper)


class Component(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_component(self) -> "Component":
        validate_atom_groups(self.atom_groups)
        return self


class MixtureBoundsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: List[Component] = Field(min_length=2, max_length=4)
    observations: List[Observation] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_request(self) -> "MixtureBoundsRequest":
        shifts = [obs.shift_mda for obs in self.observations]
        if len(set(shifts)) != len(shifts):
            raise ValueError("observation shift_mda values must be distinct")
        return self


class WeightRange(BaseModel):
    min: str  # reduced fraction "n/d"
    max: str  # reduced fraction "n/d"


class MixtureBoundsResponse(BaseModel):
    feasible: bool
    ranges: Optional[List[WeightRange]] = None
    weights: Optional[List[str]] = None


class AcquisitionPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    max_windows: int = Field(strict=True, ge=1, le=6)
    budget: int = Field(strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def _validate_request(self) -> "AcquisitionPlanRequest":
        validate_atom_groups(self.atom_groups)
        return self


class AcquisitionPlanResponse(BaseModel):
    windows: List[List[int]]  # [l, r] pairs, ascending by l
    total_bins: int
    captured_probability: str  # reduced fraction "n/d"


class DetectionWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l: int = Field(strict=True, ge=0, le=2000)
    r: int = Field(strict=True, ge=0, le=2000)
    min_hits: int = Field(strict=True, ge=1, le=4)

    @model_validator(mode="after")
    def _validate_window(self) -> "DetectionWindow":
        if self.l > self.r:
            raise ValueError("l must be <= r")
        return self


class DetectionProbabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    ion_count: int = Field(strict=True, ge=1, le=80)
    windows: List[DetectionWindow] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_request(self) -> "DetectionProbabilityRequest":
        validate_atom_groups(self.atom_groups)
        return self


class DetectionProbabilityResponse(BaseModel):
    joint_probability: str  # reduced fraction "n/d"
    # marginal_probabilities[j]: P(window j reaches min_hits), input order.
    marginal_probabilities: List[str]  # reduced "n/d"


class PosteriorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shift_mda: int = Field(strict=True, ge=0, le=2000)
    count: int = Field(strict=True, ge=0, le=80)


class PosteriorCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    prior: str

    @field_validator("prior")
    @classmethod
    def _validate_prior(cls, value: str) -> str:
        parse_abundance(value)
        return value

    @model_validator(mode="after")
    def _validate_candidate(self) -> "PosteriorCandidate":
        validate_atom_groups(self.atom_groups)
        return self

    @property
    def prior_fraction(self) -> Fraction:
        return parse_abundance(self.prior)


class CandidatePosteriorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: List[PosteriorCandidate] = Field(min_length=2, max_length=4)
    ion_count: int = Field(strict=True, ge=1, le=80)
    observations: List[PosteriorObservation] = Field(min_length=1, max_length=8)
    calibration_radius: int = Field(strict=True, ge=0, le=3)

    @model_validator(mode="after")
    def _validate_request(self) -> "CandidatePosteriorRequest":
        if sum(
            (candidate.prior_fraction for candidate in self.candidates),
            Fraction(0),
        ) != 1:
            raise ValueError("candidate priors must be non-negative and sum to 1")
        shifts = [obs.shift_mda for obs in self.observations]
        if len(set(shifts)) != len(shifts):
            raise ValueError("observation shift_mda values must be distinct")
        if sum(obs.count for obs in self.observations) > self.ion_count:
            raise ValueError("observation counts must sum to <= ion_count")
        return self


class CandidatePosteriorResponse(BaseModel):
    possible: bool
    # joint_posteriors[i][j]: P(candidate i, offset delta_j | data), candidates
    # in input order and offsets ascending within each candidate.
    joint_posteriors: Optional[List[List[str]]] = None
    # candidate_posteriors[i]: P(candidate i | data), input order.
    candidate_posteriors: Optional[List[str]] = None
    # offset_posteriors[j]: P(offset delta_j | data), offsets ascending.
    offset_posteriors: Optional[List[str]] = None
    offsets: Optional[List[int]] = None


class DiscriminationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_candidate(self) -> "DiscriminationCandidate":
        validate_atom_groups(self.atom_groups)
        return self


class DiscriminationPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: List[DiscriminationCandidate] = Field(min_length=2, max_length=2)
    max_windows: int = Field(strict=True, ge=1, le=6)
    budget: int = Field(strict=True, ge=1, le=200)


class DiscriminationPlanResponse(BaseModel):
    windows: List[List[int]]  # [l, r] pairs, ascending by l
    total_bins: int
    # hit_probabilities[i]: P(ion lands in the windows | candidate i), input order.
    hit_probabilities: List[str]  # reduced "n/d"
    accuracy: str  # reduced fraction "n/d"
    # Candidate preferred when the ion lands in the windows; null on a tie.
    hit_choice: Optional[int]


class FragmentSpectrumRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    # fragment_counts[g]: how many of group g's atoms form the fragment.
    fragment_counts: List[StrictInt] = Field(min_length=1, max_length=4)
    target_parent_shift_mda: int = Field(strict=True, ge=0, le=2000)

    @model_validator(mode="after")
    def _validate_request(self) -> "FragmentSpectrumRequest":
        validate_atom_groups(self.atom_groups)
        if len(self.fragment_counts) != len(self.atom_groups):
            raise ValueError(
                "fragment_counts must have exactly one entry per atom group"
            )
        for frag_count, group in zip(self.fragment_counts, self.atom_groups):
            if not 0 <= frag_count <= group.count:
                raise ValueError(
                    "each fragment count must satisfy 0 <= fragment_count "
                    "<= the corresponding group count"
                )
        return self


class FragmentSpectrumResponse(BaseModel):
    reachable: bool
    probability: str  # reduced fraction "n/d"
    # Conditional fragment-shift distribution given the parent shift: non-zero
    # peaks ascending by shift. Only present when reachable.
    peaks: Optional[List[Peak]] = None


class AbundanceTransfer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # All indices are 0-based; bounds are checked against the actual groups.
    group_index: int = Field(strict=True, ge=0)
    from_index: int = Field(strict=True, ge=0)
    to_index: int = Field(strict=True, ge=0)


class AbundanceSensitivityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_groups: List[AtomGroup] = Field(min_length=1, max_length=4)
    transfers: List[AbundanceTransfer] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_request(self) -> "AbundanceSensitivityRequest":
        validate_atom_groups(self.atom_groups)
        for transfer in self.transfers:
            if transfer.group_index >= len(self.atom_groups):
                raise ValueError(
                    "group_index must be < the number of atom groups"
                )
            group = self.atom_groups[transfer.group_index]
            if max(transfer.from_index, transfer.to_index) >= len(group.isotopes):
                raise ValueError(
                    "from_index/to_index must be < the number of isotopes "
                    "in the transferred group"
                )
            if transfer.from_index == transfer.to_index:
                raise ValueError("from_index and to_index must differ")
            if group.isotopes[transfer.from_index].abundance_fraction <= 0:
                raise ValueError("the donor isotope abundance must be > 0")
        return self


class DerivativePeak(BaseModel):
    shift_mda: int
    derivative: str  # reduced signed fraction "n/d"


class AbundanceSensitivityResponse(BaseModel):
    # derivatives[t]: non-zero dP(shift)/d(epsilon) values of transfer t
    # (input order), ascending by shift.
    derivatives: List[List[DerivativePeak]]


class ExchangeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(strict=True, ge=0)
    lower: int = Field(strict=True, ge=0)
    upper: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def _validate_observation(self) -> "ExchangeObservation":
        if self.lower > self.upper:
            raise ValueError("lower must be <= upper")
        return self


class ExchangeTrajectoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(strict=True, ge=1, le=40)
    m: int = Field(strict=True, ge=1, le=40)
    h: int = Field(strict=True, ge=0)
    b: int = Field(strict=True, ge=0)
    steps: int = Field(strict=True, ge=0, le=80)
    observations: List[ExchangeObservation] = Field(min_length=0, max_length=8)

    @model_validator(mode="after")
    def _validate_request(self) -> "ExchangeTrajectoryRequest":
        if self.h > self.n:
            raise ValueError("h must satisfy 0 <= h <= n")
        if self.b > self.m:
            raise ValueError("b must satisfy 0 <= b <= m")
        steps_seen = [obs.step for obs in self.observations]
        if len(set(steps_seen)) != len(steps_seen):
            raise ValueError("observation steps must be distinct")
        for obs in self.observations:
            if obs.step > self.steps:
                raise ValueError("observation step must be <= steps")
            if obs.upper > self.n:
                raise ValueError("observation upper must be <= n")
        return self


class HeavyCountPeak(BaseModel):
    heavy_count: int
    probability: str  # reduced fraction "n/d"


class ExchangeTrajectoryResponse(BaseModel):
    probability: str  # reduced fraction "n/d"
    # distributions[t]: conditional law of the molecule heavy count at step t
    # given all observations; non-zero entries ascending by heavy_count.
    # Omitted when the observations have joint probability zero.
    distributions: Optional[List[List[HeavyCountPeak]]] = None


class ExchangeTurnoverRequest(ExchangeTrajectoryRequest):
    """Identical fields and strict validation to the trajectory request."""


class TurnoverCountPeak(BaseModel):
    count: int
    probability: str  # reduced fraction "n/d"


class ExchangeTurnoverResponse(BaseModel):
    probability: str  # reduced fraction "n/d"
    # Conditional law of the total loss count over steps 0..steps given all
    # observations; non-zero entries ascending by count. Omitted when the
    # observations have joint probability zero.
    distribution: Optional[List[TurnoverCountPeak]] = None


class ExchangeFirstPassageRequest(ExchangeTrajectoryRequest):
    """Trajectory fields plus the heavy-count threshold to first reach."""

    threshold: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def _validate_threshold(self) -> "ExchangeFirstPassageRequest":
        if self.threshold > self.n:
            raise ValueError("threshold must satisfy 0 <= threshold <= n")
        return self


class FirstHitPeak(BaseModel):
    step: int
    probability: str  # reduced fraction "n/d"


class ExchangeFirstPassageResponse(BaseModel):
    probability: str  # reduced fraction "n/d"
    # Conditional law of the first step at which the molecule heavy count
    # reaches the threshold, given all observations; non-zero entries
    # ascending by step. Omitted when the observations have joint
    # probability zero.
    first_hit_distribution: Optional[List[FirstHitPeak]] = None
    # Conditional probability of never reaching the threshold within
    # steps 0..steps; the distribution entries and this value sum to 1.
    never_probability: Optional[str] = None


def convolve(left: Distribution, right: Distribution) -> Distribution:
    """Exact convolution of two shift distributions over Q; no pruning."""
    out: Distribution = {}
    for shift_a, prob_a in left.items():
        if prob_a == 0:
            continue
        for shift_b, prob_b in right.items():
            if prob_b == 0:
                continue
            shift = shift_a + shift_b
            out[shift] = out.get(shift, Fraction(0)) + prob_a * prob_b
    return out


def group_distribution(group: AtomGroup) -> Distribution:
    """Distribution of the summed shift of ``count`` independent atoms."""
    single: Distribution = {
        iso.shift_mda: iso.abundance_fraction
        for iso in group.isotopes
        if iso.abundance_fraction != 0
    }
    dist: Distribution = {0: Fraction(1)}
    for _ in range(group.count):
        dist = convolve(dist, single)
    return dist


def envelope_distribution(groups: List[AtomGroup]) -> Distribution:
    """Full envelope: convolve every group's distribution, merging by shift."""
    dist: Distribution = {0: Fraction(1)}
    for group in groups:
        dist = convolve(dist, group_distribution(group))
    return dist


# Attribution state for a fixed target shift, using a factored common
# denominator: entry weights and first moments at partial shift s are the
# numerators (plain integers) for the probability * denom and the
# probability-weighted summed atom counts * denom, respectively. Each group
# contributes the factor L_g**count where L_g is the LCM of its abundance
# denominators, so the inner loop stays in Z and a single Fraction reduction
# happens at the end.
IntAttributionState = Dict[int, Tuple[int, List[int]]]


def convolve_int_attribution(
    left: IntAttributionState, right: IntAttributionState
) -> IntAttributionState:
    """Integer convolution: cross-multiply weights and first moments."""
    out: IntAttributionState = {}
    for shift_a, (weight_a, moments_a) in left.items():
        for shift_b, (weight_b, moments_b) in right.items():
            shift = shift_a + shift_b
            entry = out.get(shift)
            if entry is None:
                out[shift] = (
                    weight_a * weight_b,
                    [
                        moment_a * weight_b + moment_b * weight_a
                        for moment_a, moment_b in zip(moments_a, moments_b)
                    ],
                )
            else:
                prev_weight, moments = entry
                for k, (moment_a, moment_b) in enumerate(
                    zip(moments_a, moments_b)
                ):
                    moments[k] += moment_a * weight_b + moment_b * weight_a
                out[shift] = (prev_weight + weight_a * weight_b, moments)
    return out


def group_int_attribution(
    group: AtomGroup, offset: int, width: int, target: int
) -> Tuple[IntAttributionState, int]:
    """Integer attribution state of one group's ``count`` atoms.

    The LCM ``L_g`` of the abundance denominators is factored out: every path
    weight carries ``L_g**count`` in the caller-held common denominator, so
    this state uses plain numerators (isotope j is scaled to
    ``n_j * L_g / d_j``). Moments are likewise probability-weighted counts
    times ``L_g`` per atom. Shifts above ``target`` are discarded.
    """
    isotopes = group.isotopes
    fractions = [iso.abundance_fraction for iso in isotopes]
    scale = 1
    for fraction in fractions:
        scale = scale * fraction.denominator // gcd(scale, fraction.denominator)
    numerators = [
        fraction.numerator * (scale // fraction.denominator)
        for fraction in fractions
    ]
    shifts = [iso.shift_mda for iso in isotopes]

    state: IntAttributionState = {0: (1, [0] * width)}
    for _ in range(group.count):
        single: IntAttributionState = {}
        for j, (shift, numerator) in enumerate(zip(shifts, numerators)):
            if numerator == 0:
                continue
            moments = [0] * width
            moments[offset + j] = numerator
            single[shift] = (numerator, moments)
        state = convolve_int_attribution(state, single)
        state = {shift: entry for shift, entry in state.items() if shift <= target}
    return state, scale ** group.count


def peak_attribution(
    groups: List[AtomGroup], target: int
) -> Tuple[Fraction, List[List[Fraction]]]:
    """Peak probability at ``target`` and per-isotope conditional expectations.

    Returns the exact unconditional probability of the target shift together
    with the expected atom count of each isotope (group-major, isotope input
    order) conditioned on observing that shift. Zero-probability targets give
    probability 0 and all-zero expectations (the caller omits them).
    """
    offsets: List[int] = []
    total = 0
    for group in groups:
        offsets.append(total)
        total += len(group.isotopes)

    state: IntAttributionState = {0: (1, [0] * total)}
    denominator = 1
    for group, offset in zip(groups, offsets):
        group_state, factor = group_int_attribution(
            group, offset, total, target
        )
        denominator *= factor
        state = convolve_int_attribution(state, group_state)
        state = {shift: entry for shift, entry in state.items() if shift <= target}

    if target not in state:
        zero = Fraction(0)
        return zero, [[zero] * len(group.isotopes) for group in groups]
    weight, moments = state[target]
    probability = Fraction(weight, denominator)
    # E[count | target] = moment / weight; the factored denominator cancels.
    expectations = [
        [
            Fraction(moment, weight)
            for moment in moments[offset : offset + len(group.isotopes)]
        ]
        for group, offset in zip(groups, offsets)
    ]
    return probability, expectations


def to_fraction_str(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def to_peak(shift: int, prob: Fraction) -> Peak:
    return Peak(shift_mda=shift, probability=to_fraction_str(prob))


# A linear constraint const + sum(coeffs[i] * x_i) >= 0 over Q.
Constraint = Tuple[List[Fraction], Fraction]


def solve_linear_system(
    matrix: List[List[Fraction]], rhs: List[Fraction]
) -> Optional[List[Fraction]]:
    """Solve a square system over Q by Gauss-Jordan elimination.

    Returns the unique solution, or None when the matrix is singular.
    """
    size = len(matrix)
    aug = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(size):
        pivot = next((r for r in range(col, size) if aug[r][col] != 0), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(size):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for k in range(col, size + 1):
                    aug[row][k] -= factor * aug[col][k]
    return [aug[i][size] / aug[i][i] for i in range(size)]


def enumerate_vertices(
    constraints: List[Constraint], dim: int
) -> List[List[Fraction]]:
    """Vertices of {x : const + coeffs·x >= 0} via exact constraint intersections."""
    vertices: List[List[Fraction]] = []
    for active in itertools.combinations(constraints, dim):
        point = solve_linear_system(
            [list(coeffs) for coeffs, _ in active],
            [-const for _, const in active],
        )
        if point is None:
            continue
        if all(
            const + sum(c * x for c, x in zip(coeffs, point)) >= 0
            for coeffs, const in constraints
        ):
            vertices.append(point)
    return vertices


def mixture_weight_bounds(
    envelopes: List[Distribution], observations: List[Observation]
) -> Optional[Tuple[List[Tuple[Fraction, Fraction]], List[Fraction]]]:
    """Exact feasible set of mixture weights, or None when empty.

    Weights are non-negative and sum to 1; each observation constrains the
    mixture probability at its shift (missing peaks count as 0) to its
    closed interval, while unobserved shifts impose nothing. Eliminating
    w_m = 1 - w_1 - ... - w_{m-1} leaves a bounded polytope, so per-weight
    extrema and the lexicographically smallest feasible vector are read off
    its vertices. Returns per-component (min, max) ranges and that
    lexicographically smallest weight vector, both in input order.
    """
    count = len(envelopes)
    dim = count - 1

    constraints: List[Constraint] = []
    for i in range(dim):  # w_i >= 0
        coeffs = [Fraction(0)] * dim
        coeffs[i] = Fraction(1)
        constraints.append((coeffs, Fraction(0)))
    # w_m = 1 - (w_1 + ... + w_{m-1}) >= 0
    constraints.append(([Fraction(-1)] * dim, Fraction(1)))

    for obs in observations:
        probs = [env.get(obs.shift_mda, Fraction(0)) for env in envelopes]
        base = probs[-1]
        coeffs = [probs[i] - base for i in range(dim)]
        # lower <= base + coeffs·x <= upper
        constraints.append((coeffs, base - obs.lower_fraction))
        constraints.append(([-c for c in coeffs], obs.upper_fraction - base))

    vertices = enumerate_vertices(constraints, dim)
    if not vertices:
        return None

    weights = [point + [Fraction(1) - sum(point, Fraction(0))] for point in vertices]
    ranges = [
        (min(w[i] for w in weights), max(w[i] for w in weights))
        for i in range(count)
    ]
    return ranges, min(weights)


def optimal_weighted_windows(
    weighted: List[Tuple[int, int]], max_windows: int, budget: int
) -> Tuple[List[Tuple[int, int]], int, int]:
    """Optimal disjoint windows over signed integer bin weights, exactly.

    ``weighted`` holds (shift, weight) pairs ascending by shift; every
    grid bin not listed has weight 0. Chooses at most ``max_windows``
    pairwise disjoint closed integer intervals [l, r] whose total bin
    count sum(r - l + 1) (every grid bin of a window is billed, weighted
    or not) stays within ``budget``, maximizing the summed weight inside
    their union; the empty plan is allowed. Ties break by fewer total
    bins, then fewer windows, then the lexicographically smaller
    flattened endpoint list [l1, r1, l2, r2, ...] with windows ordered
    by start.

    Window endpoints can be restricted to listed bins with positive
    weight: sliding an endpoint past a bin of weight <= 0 never changes
    the captured weight and strictly reduces the bin count, so the
    tie-break optimum always has endpoints on positive weights. The DP
    below therefore walks the listed bins left to right (negative
    weights stay in the walk: one may sit inside a window spanning two
    positive bins, but no optimal window starts or ends on one).

    Per processed bin, ``h[k][b]`` holds the best (captured, endpoints)
    using exactly k windows and exactly b billed bins over the bins seen
    so far, and ``g[k][b]`` the same restricted to states whose last bin
    is covered (open window ending there). Transitions at bin x:

    - leave x uncovered: carry h forward unchanged;
    - cover x by extending the open window from the previous bin
      (billed gap extra bins, last endpoint moves to x);
    - cover x by opening a new window [x, x] on top of any h state with
      one fewer window and one fewer bin.

    Within one (k, b) cell the better candidate is the higher captured
    weight, ties broken by the lexicographically smaller endpoint tuple;
    appending a shared suffix or replacing a shared last endpoint
    preserves that order, so keeping only the best prefix per cell is
    exact. Returns the windows (ascending by start), the exact total bin
    count and the captured weight as a plain integer.
    """
    windows_cap, bins_cap = max_windows, budget
    # Cells are (captured_weight, flattened_endpoint_tuple) or None.
    h: List[List[Optional[Tuple[int, Tuple[int, ...]]]]] = [
        [None] * (bins_cap + 1) for _ in range(windows_cap + 1)
    ]
    g: List[List[Optional[Tuple[int, Tuple[int, ...]]]]] = [
        [None] * (bins_cap + 1) for _ in range(windows_cap + 1)
    ]
    h[0][0] = (0, ())

    prev_x: Optional[int] = None
    for x, weight in weighted:
        gap = x - prev_x if prev_x is not None else 0
        prev_x = x
        g_next: List[List[Optional[Tuple[int, Tuple[int, ...]]]]] = [
            [None] * (bins_cap + 1) for _ in range(windows_cap + 1)
        ]
        for k in range(1, windows_cap + 1):
            g_open = g[k]
            h_prev = h[k - 1]
            g_row = g_next[k]
            for b in range(k, bins_cap + 1):
                best: Optional[Tuple[int, Tuple[int, ...]]] = None
                if gap and b >= gap:
                    cand = g_open[b - gap]
                    if cand is not None:
                        best = (cand[0] + weight, cand[1][:-1] + (x,))
                cand = h_prev[b - 1]
                if cand is not None:
                    prob = cand[0] + weight
                    seq = cand[1] + (x, x)
                    if (
                        best is None
                        or prob > best[0]
                        or (prob == best[0] and seq < best[1])
                    ):
                        best = (prob, seq)
                g_row[b] = best
        for k in range(1, windows_cap + 1):
            h_row = h[k]
            g_row = g_next[k]
            for b in range(k, bins_cap + 1):
                cand = g_row[b]
                if cand is None:
                    continue
                cur = h_row[b]
                if (
                    cur is None
                    or cand[0] > cur[0]
                    or (cand[0] == cur[0] and cand[1] < cur[1])
                ):
                    h_row[b] = cand
        g = g_next

    best_key: Optional[Tuple[int, int, int, Tuple[int, ...]]] = None
    best_seq: Tuple[int, ...] = ()
    best_num = 0
    best_bins = 0
    for k in range(windows_cap + 1):
        for b in range(bins_cap + 1):
            state = h[k][b]
            if state is None:
                continue
            key = (-state[0], b, k, state[1])
            if best_key is None or key < best_key:
                best_key = key
                best_num, best_seq = state
                best_bins = b

    windows = [
        (best_seq[i], best_seq[i + 1]) for i in range(0, len(best_seq), 2)
    ]
    return windows, best_bins, best_num


def acquisition_plan(
    dist: Distribution, max_windows: int, budget: int
) -> Tuple[List[Tuple[int, int]], int, Fraction]:
    """Optimal acquisition windows over the full envelope, exactly.

    Every non-zero peak contributes its probability as the bin weight;
    all weights are scaled by the lcm of the peak denominators so the DP
    runs on plain integers and stays exact, then delegated to
    ``optimal_weighted_windows`` (see its docstring for the window
    semantics, the DP and the tie-break chain). Returns the windows
    (ascending by start), the exact total bin count and the captured
    probability as a reduced Fraction.
    """
    peaks = sorted(
        (shift, prob) for shift, prob in dist.items() if prob != 0
    )
    scale = 1
    for _, prob in peaks:
        scale = scale * prob.denominator // gcd(scale, prob.denominator)
    weighted = [
        (shift, prob.numerator * (scale // prob.denominator))
        for shift, prob in peaks
    ]
    windows, total_bins, captured_num = optimal_weighted_windows(
        weighted, max_windows, budget
    )
    return windows, total_bins, Fraction(captured_num, scale)


def discrimination_plan(
    dist0: Distribution,
    dist1: Distribution,
    max_windows: int,
    budget: int,
) -> Tuple[List[Tuple[int, int]], int, Fraction, Fraction, Fraction, Optional[int]]:
    """Optimal windows discriminating two equal-prior candidates, over Q.

    The two candidates carry the full envelope distributions ``dist0``
    and ``dist1`` (no pruning) and share the prior 1/2. A plan is at
    most ``max_windows`` pairwise disjoint closed integer intervals
    [l, r] within the 0..2000 shift axis whose total bin count
    sum(r - l + 1) (every grid bin billed, peaks or not) stays within
    ``budget``; the empty plan is allowed. An ion landing inside the
    union votes for the candidate with the higher hit probability and an
    ion outside votes for the other one, so with
    p_i = P(hit | candidate i) the exact accuracy is
    (1 + |p0 - p1|) / 2 and the plan maximizes |p0 - p1|. Ties break by
    fewer total bins, then fewer windows, then the lexicographically
    smaller flattened endpoint list with windows ordered by start.

    Maximizing |p0 - p1| splits into two orientations: maximize
    p0 - p1 (per-bin weight w0 - w1) or p1 - p0 (weight w1 - w0). Each
    orientation is exactly the signed-weight window DP of
    ``optimal_weighted_windows`` over the union of both envelopes' peak
    positions (bins outside the union carry weight 0 in both envelopes),
    with all masses scaled by one common denominator lcm so the two
    orientations compare exactly. The orientation with the higher value
    wins; equal values fall back to the shared tie-break chain (bins,
    windows, flattened endpoints) with no preference between
    orientations. The hit probabilities, the accuracy and the preferred
    candidate are then read off the chosen windows directly from the
    full envelopes.

    Returns the windows (ascending by start), the exact total bin count,
    the per-candidate hit probabilities (candidate input order), the
    accuracy and the hit-preferred candidate index (0 or 1; None when
    the hit probabilities tie), every probability a reduced Fraction.
    """
    scale = 1
    for dist in (dist0, dist1):
        for prob in dist.values():
            if prob != 0:
                scale = scale * prob.denominator // gcd(scale, prob.denominator)
    masses = [
        {
            shift: prob.numerator * (scale // prob.denominator)
            for shift, prob in dist.items()
            if prob != 0
        }
        for dist in (dist0, dist1)
    ]
    positions = sorted(set(masses[0]) | set(masses[1]))
    forward = [
        (x, masses[0].get(x, 0) - masses[1].get(x, 0)) for x in positions
    ]
    backward = [(x, -weight) for x, weight in forward]

    best_key: Optional[Tuple[int, int, int, Tuple[int, ...]]] = None
    best_windows: List[Tuple[int, int]] = []
    best_bins = 0
    for weighted in (forward, backward):
        windows, bins, value = optimal_weighted_windows(
            weighted, max_windows, budget
        )
        flat = tuple(endpoint for window in windows for endpoint in window)
        key = (-value, bins, len(windows), flat)
        if best_key is None or key < best_key:
            best_key = key
            best_windows, best_bins = windows, bins

    p0 = sum(
        (
            dist0.get(x, Fraction(0))
            for left, right in best_windows
            for x in range(left, right + 1)
        ),
        Fraction(0),
    )
    p1 = sum(
        (
            dist1.get(x, Fraction(0))
            for left, right in best_windows
            for x in range(left, right + 1)
        ),
        Fraction(0),
    )
    accuracy = (1 + abs(p0 - p1)) / 2
    hit_choice = None if p0 == p1 else (0 if p0 > p1 else 1)
    return best_windows, best_bins, p0, p1, accuracy, hit_choice


def detection_probability(
    dist: Distribution, windows: List[Tuple[int, int, int]], ion_count: int
) -> Tuple[Fraction, List[Fraction]]:
    """Joint and marginal window-detection probabilities, exactly over Q.

    Each of ``ion_count`` ions is drawn independently from the full envelope
    ``dist``; an ion landing at shift s scores one hit in every window
    ``(l, r, min_hits)`` with ``l <= s <= r`` (windows may overlap or
    coincide, and probability mass outside all windows is kept as a
    hit-nothing outcome). A window succeeds when its hit count reaches
    ``min_hits``. Returns the probability that every window succeeds
    together with each window's own success probability, in input order.

    A single ion realizes one of at most 2m+1 hit masks (sweeping the shift
    axis, the mask of covered windows changes only at window endpoints), so
    the ions follow a multinomial over those masks. The count vector is
    tracked capped at each ``min_hits`` — at most 5**4 = 625 states — and
    advanced ion by ion, never enumerating sampling sequences. All mask
    probabilities are scaled by the lcm of their denominators, so the DP
    runs on plain integers with denominator ``scale**ion_count`` carried
    symbolically; each output is reduced exactly once at the end. A window
    that can never succeed (e.g. ``min_hits > ion_count``) simply gets
    numerator 0, yielding the exact "0/1" for itself and the joint event.
    """
    # Probability of each hit mask: bit j set iff the shift lies in window j.
    mask_probs: Dict[int, Fraction] = {}
    for shift, prob in dist.items():
        if prob == 0:
            continue
        mask = 0
        for j, (left, right, _) in enumerate(windows):
            if left <= shift <= right:
                mask |= 1 << j
        mask_probs[mask] = mask_probs.get(mask, Fraction(0)) + prob

    scale = 1
    for prob in mask_probs.values():
        scale = scale * prob.denominator // gcd(scale, prob.denominator)
    scaled = [
        (mask, prob.numerator * (scale // prob.denominator))
        for mask, prob in mask_probs.items()
        if prob != 0
    ]

    caps = [min_hits for _, _, min_hits in windows]
    # Capped count vector encoded in mixed radix: idx = sum(counts[j]*mult[j]).
    radices = [cap + 1 for cap in caps]
    multipliers = [1]
    for radix in radices[:-1]:
        multipliers.append(multipliers[-1] * radix)
    state_count = multipliers[-1] * radices[-1]

    # transitions[idx][k]: state after one more ion drawing mask scaled[k][0].
    transitions: List[List[int]] = []
    for idx in range(state_count):
        counts = [
            (idx // mult) % radix for mult, radix in zip(multipliers, radices)
        ]
        row = []
        for mask, _weight in scaled:
            target = 0
            for j, (mult, cap) in enumerate(zip(multipliers, caps)):
                count = counts[j] + ((mask >> j) & 1)
                if count > cap:
                    count = cap
                target += count * mult
            row.append(target)
        transitions.append(row)

    weights = [weight for _, weight in scaled]
    dp = [0] * state_count
    dp[0] = 1
    for _ in range(ion_count):
        nxt = [0] * state_count
        for idx, value in enumerate(dp):
            if value == 0:
                continue
            row = transitions[idx]
            for k, weight in enumerate(weights):
                nxt[row[k]] += value * weight
        dp = nxt

    denominator = scale ** ion_count
    full_idx = sum(
        cap * mult for cap, mult in zip(caps, multipliers)
    )
    joint = Fraction(dp[full_idx], denominator)
    marginals = [
        Fraction(
            sum(
                value
                for idx, value in enumerate(dp)
                if value and (idx // mult) % radix == cap
            ),
            denominator,
        )
        for mult, radix, cap in zip(multipliers, radices, caps)
    ]
    return joint, marginals


def multinomial_binned_likelihood(
    dist: Distribution,
    observations: List[Tuple[int, int]],
    ion_count: int,
) -> Fraction:
    """Exact multinomial probability of binned ion counts, over Q.

    Each ion is drawn independently from ``dist``; ``observations`` lists
    ``(measured shift, count)`` bins and every unlisted shift collapses into
    a single "other" bin holding the remaining ions. Returns the multinomial
    probability of exactly that count vector. Following the standard
    combinatorial convention ``0**0 = 1``, a zero-probability bin with a
    zero count contributes nothing; the likelihood is zero exactly when a
    positive-count bin (including "other") has zero probability.
    """
    remaining = ion_count
    likelihood = Fraction(1)
    covered = Fraction(0)
    for shift, count in observations:
        probability = dist.get(shift, Fraction(0))
        likelihood *= probability**count
        covered += probability
        remaining -= count
    other_probability = Fraction(1) - covered
    likelihood *= other_probability**remaining
    return likelihood


def candidate_posterior(
    envelopes: List[Distribution],
    priors: List[Fraction],
    observations: List[Tuple[int, int]],
    ion_count: int,
    radius: int,
) -> Optional[Tuple[List[List[Fraction]], List[Fraction], List[Fraction]]]:
    """Exact joint/candidate/offset posteriors, or None when data impossible.

    Candidate ``i`` is drawn once with prior ``priors[i]``; conditioned on it,
    all ions share one calibration offset delta drawn uniformly from
    ``[-radius, radius]``, and each ion's measured shift is its independently
    sampled envelope shift plus delta. The likelihood of the observed
    count vector is the exact multinomial probability under
    ``multinomial_binned_likelihood`` against the shifted envelope. Joint
    unnormalized weights are

        prior_i * (1/(2r+1)) * multinomial(counts | candidate i, delta)

    (the multinomial coefficient is common to every weight and omitted).
    Every weight is reduced exactly once; when the total evidence is zero the
    observation has probability zero under every hypothesis and the caller
    reports ``possible=false`` with no posteriors. Returns the joint table
    (candidate input order, offsets ascending within each candidate), the
    per-candidate marginals and the per-offset marginals, all normalized.
    """
    offsets = list(range(-radius, radius + 1))
    weights: List[List[Fraction]] = []
    for dist, prior in zip(envelopes, priors):
        row: List[Fraction] = []
        for delta in offsets:
            shifted = {shift + delta: prob for shift, prob in dist.items()}
            likelihood = multinomial_binned_likelihood(
                shifted, observations, ion_count
            )
            # The uniform 1/(2r+1) offset prior is common to every cell and
            # cancels in normalization, but keep it explicit for exactness.
            row.append(prior * likelihood / len(offsets))
        weights.append(row)

    total = sum((sum(row, Fraction(0)) for row in weights), Fraction(0))
    if total == 0:
        return None
    joints = [[weight / total for weight in row] for row in weights]
    candidate_marginals = [
        sum(row, Fraction(0)) / total for row in weights
    ]
    offset_marginals = [
        sum(weights[i][j] for i in range(len(weights))) / total
        for j in range(len(offsets))
    ]
    return joints, candidate_marginals, offset_marginals


def group_fragment_pairs(
    group: AtomGroup, frag_count: int, target: int
) -> Tuple[Dict[Tuple[int, int], int], int]:
    """Integer joint (parent, fragment) shift-pair weights of one group.

    The group's ``count`` atoms take isotopes independently; then
    ``frag_count`` of them are drawn uniformly without replacement to form
    the fragment (the rest are the residual). Returns the joint distribution
    over (parent shift, fragment shift) as plain integer weights together
    with the common denominator ``L**count * C(count, frag_count)``, where
    ``L`` is the LCM of the group's abundance denominators: each pair's exact
    probability is ``weight / denominator``.

    A composition n (atom counts per isotope) carries the integer weight
    ``multinomial(count; n) * prod(scaled_abundance_j**n_j)`` and a fragment
    draw realizing the sub-composition m carries ``prod C(n_j, m_j)``; both
    are enumerated recursively (compositions, then sub-compositions) and
    merged by shift pair. Compositions whose parent shift already exceeds
    ``target`` are pruned (all shifts are non-negative), and zero-abundance
    isotopes are dropped up front since every composition using them has
    weight zero.
    """
    count = group.count
    scale = 1
    for iso in group.isotopes:
        fraction = iso.abundance_fraction
        scale = scale * fraction.denominator // gcd(scale, fraction.denominator)
    shifts: List[int] = []
    powers: List[List[int]] = []
    for iso in group.isotopes:
        fraction = iso.abundance_fraction
        if fraction == 0:
            continue
        scaled = fraction.numerator * (scale // fraction.denominator)
        shifts.append(iso.shift_mda)
        powers.append([scaled**atoms for atoms in range(count + 1)])
    kind = len(shifts)

    pairs: Dict[Tuple[int, int], int] = {}
    composition = [0] * kind
    suffix = [0] * kind  # suffix[j]: sum of composition[j:], set per leaf

    def walk_fragment(
        j: int,
        frag_left: int,
        frag_shift: int,
        choose: int,
        parent_shift: int,
        weight: int,
    ) -> None:
        if frag_left > suffix[j]:
            return  # the remaining isotopes cannot supply enough atoms
        if j == kind - 1:
            choose *= comb(composition[j], frag_left)
            frag_shift += frag_left * shifts[j]
            key = (parent_shift, frag_shift)
            pairs[key] = pairs.get(key, 0) + weight * choose
            return
        for take in range(min(frag_left, composition[j]) + 1):
            walk_fragment(
                j + 1,
                frag_left - take,
                frag_shift + take * shifts[j],
                choose * comb(composition[j], take),
                parent_shift,
                weight,
            )

    def walk_isotopes(j: int, atoms_left: int, parent_shift: int, weight: int) -> None:
        if j == kind - 1:
            composition[j] = atoms_left
            parent_shift += atoms_left * shifts[j]
            if parent_shift > target:
                return
            total = 0
            for t in range(kind - 1, -1, -1):
                total += composition[t]
                suffix[t] = total
            walk_fragment(
                0, frag_count, 0, 1, parent_shift, weight * powers[j][atoms_left]
            )
            return
        for take in range(atoms_left + 1):
            composition[j] = take
            next_shift = parent_shift + take * shifts[j]
            if next_shift > target:
                break  # shifts are non-negative and `take` only grows
            walk_isotopes(
                j + 1,
                atoms_left - take,
                next_shift,
                weight * comb(atoms_left, take) * powers[j][take],
            )

    walk_isotopes(0, count, 0, 1)
    return pairs, scale**count * comb(count, frag_count)


def fragment_spectrum(
    groups: List[AtomGroup], frag_counts: List[int], target: int
) -> Tuple[Fraction, Optional[Dict[int, Fraction]]]:
    """Parent-peak probability and conditional fragment-shift law, over Q.

    Every atom of every group first takes an isotope independently (the
    parent ion); then, within each group, ``frag_counts[g]`` of the group's
    atoms are drawn uniformly without replacement to form the fragment and
    the rest are the residual. Conditioning on the parent's total shift
    being ``target``, the fragment shift is the summed shift of the drawn
    atoms. Returns the exact (unconditional) probability of the parent
    shift ``target`` and, when that is non-zero, the exact conditional
    distribution of the fragment shift as a shift -> probability mapping;
    a zero-probability parent shift yields ``(0, None)``.

    Each group's joint (parent shift, fragment shift) pair-distribution is
    computed by ``group_fragment_pairs`` as plain integer weights with the
    common denominator ``L_g**count * C(count, frag_count)`` held
    symbolically; the pair-distributions are convolved across groups in
    integer arithmetic, pruning every partial parent shift above ``target``
    (all shifts are non-negative). Each output is reduced exactly once at
    the end: the parent probability is ``numerator / denominator`` and the
    conditional fragment probabilities are ``weight / parent_numerator``
    (the factored denominators cancel).
    """
    state: Dict[Tuple[int, int], int] = {(0, 0): 1}
    denominator = 1
    for group, frag_count in zip(groups, frag_counts):
        pairs, factor = group_fragment_pairs(group, frag_count, target)
        denominator *= factor
        merged: Dict[Tuple[int, int], int] = {}
        for (parent_a, frag_a), weight_a in state.items():
            for (parent_b, frag_b), weight_b in pairs.items():
                parent = parent_a + parent_b
                if parent > target:
                    continue
                key = (parent, frag_a + frag_b)
                merged[key] = merged.get(key, 0) + weight_a * weight_b
        state = merged

    parent_numerator = 0
    fragment_weights: Dict[int, int] = {}
    for (parent, frag), weight in state.items():
        if parent != target:
            continue
        parent_numerator += weight
        fragment_weights[frag] = fragment_weights.get(frag, 0) + weight
    if parent_numerator == 0:
        return Fraction(0), None
    probability = Fraction(parent_numerator, denominator)
    peaks = {
        frag: Fraction(weight, parent_numerator)
        for frag, weight in fragment_weights.items()
    }
    return probability, peaks


def transfer_derivative(
    groups: List[AtomGroup],
    group_dists: List[Distribution],
    transfer: AbundanceTransfer,
) -> Distribution:
    """Exact dP(shift)/d(epsilon) at epsilon = 0 for one abundance transfer.

    The transfer perturbs one group: the donor isotope's abundance becomes
    ``a_from - epsilon`` and the receptor's ``a_to + epsilon``, so the
    group's single-atom distribution is ``single(0) + epsilon * delta``
    with ``delta`` the signed distribution holding -1 at the donor shift
    and +1 at the receptor shift. The group distribution is the
    ``count``-fold convolution power of ``single(epsilon)``, hence by the
    product rule (all ``count`` atom positions are symmetric)

        d/d(eps) single(eps)^{*count} |_{eps=0}
            = count * delta * single(0)^{*(count-1)}   (* = convolution)

    and the full envelope derivative convolves this with every other
    group's unperturbed distribution. Everything is exact rational
    arithmetic — no finite differences, no pruning — so shifts whose
    baseline probability is zero (e.g. paths that must use a zero-abundance
    receptor) still appear whenever their derivative is non-zero. Returns
    the signed shift -> derivative mapping; entries may be negative.
    """
    group = groups[transfer.group_index]
    from_shift = group.isotopes[transfer.from_index].shift_mda
    to_shift = group.isotopes[transfer.to_index].shift_mda
    delta: Distribution = {from_shift: Fraction(-1), to_shift: Fraction(1)}
    single: Distribution = {
        iso.shift_mda: iso.abundance_fraction
        for iso in group.isotopes
        if iso.abundance_fraction != 0
    }
    deriv: Distribution = dict(delta)
    for _ in range(group.count - 1):
        deriv = convolve(deriv, single)
    deriv = {shift: group.count * prob for shift, prob in deriv.items()}

    dist: Distribution = {0: Fraction(1)}
    for index, group_dist in enumerate(group_dists):
        dist = convolve(dist, deriv if index == transfer.group_index else group_dist)
    return dist


def exchange_trajectory(
    n: int,
    m: int,
    h: int,
    b: int,
    steps: int,
    observations: List[Tuple[int, int, int]],
) -> Tuple[Fraction, Optional[List[Dict[int, Fraction]]]]:
    """Exact finite-bath exchange-trajectory inference, over Q.

    A molecule of ``n`` sites starts with ``h`` heavy isotopes and a bath of
    ``m`` sites with ``b``; each step picks one molecule site and one bath
    site uniformly and independently (all ``n*m`` pairs equally likely) and
    swaps their isotopes — same-type swaps count as steps. The bath holds
    the remaining ``T - X_t`` heavy isotopes (``T = h + b`` is conserved),
    so the molecule heavy count X_t is a birth-death chain on 0..n whose
    one-step transitions all share the denominator ``n*m``, with integer
    numerators

        up(x)   = (n - x) * (T - x)        (light site out, heavy site in)
        down(x) = x * (m - T + x)          (heavy site out, light site in)
        stay(x) = n*m - up(x) - down(x)    (same-type swaps)

    which vanish exactly at the feasible boundary
    max(0, T - m) <= x <= min(n, T). ``observations`` lists
    ``(step, lower, upper)`` constraints ``lower <= X_step <= upper`` at
    distinct steps (step 0 is the initial state, X_0 = h). Returns the
    exact joint probability of all observations and, when it is non-zero,
    for every step 0..steps the conditional distribution of X_t given all
    observations as a count -> probability mapping; a zero joint
    probability yields ``(0, None)``.

    The forward pass tracks exact integer weights ``A_t`` with
    ``alpha_t(x) = A_t[x] / (n*m)**t``, zeroing every state that violates
    the observation at its step; the remaining mass at the last observed
    step ``last`` is the joint probability ``z / (n*m)**last``. The
    backward pass tracks ``B_t(x) = P(later observations | X_t = x) *
    (n*m)**(last - t)`` over the same integer weights, so each conditional
    probability is the single Fraction reduction
    ``A_t[x] * B_t[x] / z``; steps after ``last`` carry ``B = 1`` and pick
    up the denominator factor ``(n*m)**(t - last)``. No exchange sequence
    is ever enumerated.
    """
    total = h + b  # conserved heavy-isotope count (molecule + bath)
    denom = n * m  # every one-step transition has this denominator

    # Integer transition numerators at molecule heavy count x.
    ups = [(n - x) * (total - x) for x in range(n + 1)]
    downs = [x * (m - total + x) for x in range(n + 1)]
    stays = [denom - ups[x] - downs[x] for x in range(n + 1)]

    filters = {step: (lower, upper) for step, lower, upper in observations}

    # Forward pass over exact integer weights: alpha_t(x) = A_t[x]/denom**t,
    # with states violating an observation zeroed at its step.
    forward: List[Dict[int, int]] = []
    current: Dict[int, int] = {h: 1}
    if 0 in filters:
        lower, upper = filters[0]
        if not lower <= h <= upper:
            current = {}
    forward.append(current)
    for t in range(1, steps + 1):
        nxt: Dict[int, int] = {}
        for x, weight in current.items():
            if stays[x]:
                nxt[x] = nxt.get(x, 0) + weight * stays[x]
            if ups[x]:
                nxt[x + 1] = nxt.get(x + 1, 0) + weight * ups[x]
            if downs[x]:
                nxt[x - 1] = nxt.get(x - 1, 0) + weight * downs[x]
        if t in filters:
            lower, upper = filters[t]
            nxt = {x: w for x, w in nxt.items() if lower <= x <= upper}
        forward.append(nxt)
        current = nxt

    if not filters:
        # No observations: the joint event is certain and the conditional
        # law at step t is the unconditional one.
        distributions = [
            {x: Fraction(weight, denom**t) for x, weight in forward[t].items()}
            for t in range(steps + 1)
        ]
        return Fraction(1), distributions

    last = max(filters)
    z_num = sum(forward[last].values())
    if z_num == 0:
        return Fraction(0), None
    probability = Fraction(z_num, denom**last)

    # Backward pass over the same integer weights:
    # B_t(x) = P(observations after t | X_t = x) * denom**(last - t); states
    # failing the filter at t simply never enter backward[t].
    backward: Dict[int, Dict[int, int]] = {last: {x: 1 for x in forward[last]}}
    for t in range(last, 0, -1):
        following = backward[t]
        prev: Dict[int, int] = {}
        for x in forward[t - 1]:
            value = 0
            if stays[x]:
                value += stays[x] * following.get(x, 0)
            if ups[x]:
                value += ups[x] * following.get(x + 1, 0)
            if downs[x]:
                value += downs[x] * following.get(x - 1, 0)
            prev[x] = value
        backward[t - 1] = prev

    distributions = []
    for t in range(steps + 1):
        step_dist: Dict[int, Fraction] = {}
        if t <= last:
            for x, weight in forward[t].items():
                joint = weight * backward[t][x]
                if joint:
                    step_dist[x] = Fraction(joint, z_num)
        else:
            scale = z_num * denom ** (t - last)
            for x, weight in forward[t].items():
                step_dist[x] = Fraction(weight, scale)
        distributions.append(step_dist)
    return probability, distributions


def exchange_turnover(
    n: int,
    m: int,
    h: int,
    b: int,
    steps: int,
    observations: List[Tuple[int, int, int]],
) -> Tuple[Fraction, Optional[Dict[int, Fraction]]]:
    """Exact finite-bath exchange loss-count (turnover) inference, over Q.

    The exchange dynamics are exactly those of ``exchange_trajectory``:
    each step picks one of the ``n*m`` (molecule site, bath site) pairs
    uniformly and swaps (same-type swaps count as steps), so the molecule
    heavy count X_t is a birth-death chain with integer transition
    numerators up/down/stay over the common denominator ``n*m``. The loss
    count C_t additionally increments by 1 on every down transition (a
    heavy isotope leaves the molecule); same-type swaps and up
    transitions leave it unchanged, and repeated outflows accumulate one
    by one. ``observations`` lists ``(step, lower, upper)`` constraints
    on X_t at distinct steps (step 0 is the initial state X_0 = h).
    Returns the exact joint probability of all observations and, when it
    is non-zero, the exact conditional distribution of the total loss
    count C_steps given all of them as a count -> probability mapping; a
    zero joint probability yields ``(0, None)``.

    The forward recursion carries exact integer weights keyed by the
    joint state (heavy count, loss count): A_t[(x, c)] with
    alpha_t(x, c) = A_t[(x, c)] / (n*m)**t, zeroing every state that
    violates the observation at its step. Paths that end at the same
    heavy count but accumulated different loss counts (round trips such
    as 1 -> 0 -> 1 or 1 -> 2 -> 1 versus staying at 1) occupy distinct
    cells and are merged only when the loss-count marginal is read off,
    so every outflow is counted separately as required. The remaining
    mass at step ``steps`` gives the joint probability
    ``z / (n*m)**steps`` and each conditional loss probability is the
    single Fraction reduction ``weight_c / z``. No exchange sequence is
    ever enumerated.
    """
    total = h + b  # conserved heavy-isotope count (molecule + bath)
    denom = n * m  # every one-step transition has this denominator

    # Integer transition numerators at molecule heavy count x.
    ups = [(n - x) * (total - x) for x in range(n + 1)]
    downs = [x * (m - total + x) for x in range(n + 1)]
    stays = [denom - ups[x] - downs[x] for x in range(n + 1)]

    filters = {step: (lower, upper) for step, lower, upper in observations}

    # Forward pass over exact integer weights keyed by (heavy count, loss
    # count): alpha_t(x, c) = current[(x, c)] / denom**t, with states
    # violating an observation zeroed at its step.
    current: Dict[Tuple[int, int], int] = {(h, 0): 1}
    if 0 in filters:
        lower, upper = filters[0]
        if not lower <= h <= upper:
            current = {}
    for t in range(1, steps + 1):
        nxt: Dict[Tuple[int, int], int] = {}
        for (x, c), weight in current.items():
            if stays[x]:
                key = (x, c)
                nxt[key] = nxt.get(key, 0) + weight * stays[x]
            if ups[x]:
                key = (x + 1, c)
                nxt[key] = nxt.get(key, 0) + weight * ups[x]
            if downs[x]:
                key = (x - 1, c + 1)
                nxt[key] = nxt.get(key, 0) + weight * downs[x]
        if t in filters:
            lower, upper = filters[t]
            nxt = {
                key: w for key, w in nxt.items() if lower <= key[0] <= upper
            }
        current = nxt

    z_num = sum(current.values())
    if z_num == 0:
        return Fraction(0), None
    probability = Fraction(z_num, denom**steps)

    # Marginalize the heavy count: each loss count's conditional weight is
    # reduced exactly once against the surviving total mass.
    losses: Dict[int, int] = {}
    for (_x, c), weight in current.items():
        losses[c] = losses.get(c, 0) + weight
    distribution = {
        c: Fraction(weight, z_num) for c, weight in losses.items() if weight
    }
    return probability, distribution


def exchange_first_passage(
    n: int,
    m: int,
    h: int,
    b: int,
    steps: int,
    observations: List[Tuple[int, int, int]],
    threshold: int,
) -> Tuple[Fraction, Optional[Dict[int, Fraction]], Optional[Fraction]]:
    """Exact first-passage inference for the finite-bath exchange, over Q.

    The exchange dynamics are exactly those of ``exchange_trajectory``:
    each step picks one of the ``n*m`` (molecule site, bath site) pairs
    uniformly and swaps (same-type swaps count as steps), so the molecule
    heavy count X_t is a birth-death chain with integer transition
    numerators up/down/stay over the common denominator ``n*m``. The
    first-passage time tau is the first step at which X_t >= ``threshold``
    (tau = 0 when the initial count h already qualifies); once the
    threshold has been reached, falling back below it does not undo the
    hit. ``observations`` lists ``(step, lower, upper)`` constraints on
    X_t at distinct steps (step 0 is the initial state X_0 = h). Returns
    the exact joint probability of all observations and, when it is
    non-zero, the exact conditional distribution of tau over 0..steps
    given all of them as a step -> probability mapping together with the
    conditional probability of never reaching the threshold within the
    horizon; a zero joint probability yields ``(0, None, None)``.

    The forward recursion carries exact integer weights keyed by the
    joint state (heavy count, first-hit step): A_t[(x, tau)] with
    alpha_t(x, tau) = A_t[(x, tau)] / (n*m)**t, where tau = -1 marks
    "threshold not reached yet" and zeroing every state that violates
    the observation at its step. A not-yet-hit state that crosses the
    threshold on step t moves to tau = t; states already carrying a
    first-hit step keep it whatever the later counts are, so a path
    that reaches the threshold and falls back is still credited with
    its first reaching time. Paths whose first hit happens at different
    steps occupy distinct cells and are merged only when the tau
    marginal is read off. The remaining mass at step ``steps`` gives
    the joint probability ``z / (n*m)**steps`` and each conditional
    probability is the single Fraction reduction ``weight / z``; the
    never-hit mass is the surviving tau = -1 weight. No exchange
    sequence is ever enumerated.
    """
    total = h + b  # conserved heavy-isotope count (molecule + bath)
    denom = n * m  # every one-step transition has this denominator

    # Integer transition numerators at molecule heavy count x.
    ups = [(n - x) * (total - x) for x in range(n + 1)]
    downs = [x * (m - total + x) for x in range(n + 1)]
    stays = [denom - ups[x] - downs[x] for x in range(n + 1)]

    filters = {step: (lower, upper) for step, lower, upper in observations}

    # Forward pass over exact integer weights keyed by (heavy count x,
    # first-hit step tau): alpha_t(x, tau) = current[(x, tau)] / denom**t,
    # with states violating an observation zeroed at its step. tau = -1
    # marks "threshold not reached yet"; such states always have
    # x < threshold.
    current: Dict[Tuple[int, int], int] = {
        (h, 0 if h >= threshold else -1): 1
    }
    if 0 in filters:
        lower, upper = filters[0]
        if not lower <= h <= upper:
            current = {}
    for t in range(1, steps + 1):
        nxt: Dict[Tuple[int, int], int] = {}
        for (x, tau), weight in current.items():
            for nx, num in ((x, stays[x]), (x + 1, ups[x]), (x - 1, downs[x])):
                if not num:
                    continue
                if tau < 0 and nx >= threshold:
                    ntau = t  # first reaching of the threshold
                else:
                    ntau = tau
                key = (nx, ntau)
                nxt[key] = nxt.get(key, 0) + weight * num
        if t in filters:
            lower, upper = filters[t]
            nxt = {
                key: w for key, w in nxt.items() if lower <= key[0] <= upper
            }
        current = nxt

    z_num = sum(current.values())
    if z_num == 0:
        return Fraction(0), None, None
    probability = Fraction(z_num, denom**steps)

    # Marginalize the heavy count: each first-hit step's conditional weight
    # is reduced exactly once against the surviving total mass; the tau = -1
    # weight is the probability of never reaching the threshold by `steps`.
    hits: Dict[int, int] = {}
    never_weight = 0
    for (_x, tau), weight in current.items():
        if tau < 0:
            never_weight += weight
        else:
            hits[tau] = hits.get(tau, 0) + weight
    distribution = {
        tau: Fraction(weight, z_num) for tau, weight in hits.items() if weight
    }
    never = Fraction(never_weight, z_num)
    return probability, distribution, never


app = FastAPI(title="Isotope Envelope API", version="1.11.0")


@app.post("/v1/envelope", response_model=EnvelopeResponse)
def compute_envelope(request: EnvelopeRequest) -> EnvelopeResponse:
    dist = envelope_distribution(request.atom_groups)

    # All non-zero peaks, ascending by shift.
    peaks = [to_peak(shift, prob) for shift, prob in sorted(dist.items()) if prob != 0]

    # Top-k: probability descending, ties broken by shift ascending.
    ranked = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
    top_peaks = [to_peak(shift, prob) for shift, prob in ranked[: request.k]]

    return EnvelopeResponse(peaks=peaks, top_peaks=top_peaks)


@app.post(
    "/v1/peak-attribution",
    response_model=PeakAttributionResponse,
    response_model_exclude_none=True,
)
def compute_peak_attribution(
    request: PeakAttributionRequest,
) -> PeakAttributionResponse:
    probability, expectations = peak_attribution(
        request.atom_groups, request.target_shift_mda
    )
    if probability == 0:
        return PeakAttributionResponse(
            reachable=False, probability=to_fraction_str(Fraction(0))
        )
    return PeakAttributionResponse(
        reachable=True,
        probability=to_fraction_str(probability),
        expected_counts=[
            [to_fraction_str(value) for value in row] for row in expectations
        ],
    )


@app.post(
    "/v1/mixture-bounds",
    response_model=MixtureBoundsResponse,
    response_model_exclude_none=True,
)
def compute_mixture_bounds(request: MixtureBoundsRequest) -> MixtureBoundsResponse:
    envelopes = [
        envelope_distribution(component.atom_groups)
        for component in request.components
    ]
    result = mixture_weight_bounds(envelopes, request.observations)
    if result is None:
        return MixtureBoundsResponse(feasible=False)
    ranges, weights = result
    return MixtureBoundsResponse(
        feasible=True,
        ranges=[
            WeightRange(min=to_fraction_str(lo), max=to_fraction_str(hi))
            for lo, hi in ranges
        ],
        weights=[to_fraction_str(w) for w in weights],
    )


@app.post(
    "/v1/acquisition-plan",
    response_model=AcquisitionPlanResponse,
)
def compute_acquisition_plan(
    request: AcquisitionPlanRequest,
) -> AcquisitionPlanResponse:
    dist = envelope_distribution(request.atom_groups)
    windows, total_bins, captured = acquisition_plan(
        dist, request.max_windows, request.budget
    )
    return AcquisitionPlanResponse(
        windows=[[left, right] for left, right in windows],
        total_bins=total_bins,
        captured_probability=to_fraction_str(captured),
    )


@app.post(
    "/v1/detection-probability",
    response_model=DetectionProbabilityResponse,
)
def compute_detection_probability(
    request: DetectionProbabilityRequest,
) -> DetectionProbabilityResponse:
    dist = envelope_distribution(request.atom_groups)
    windows = [(window.l, window.r, window.min_hits) for window in request.windows]
    joint, marginals = detection_probability(dist, windows, request.ion_count)
    return DetectionProbabilityResponse(
        joint_probability=to_fraction_str(joint),
        marginal_probabilities=[to_fraction_str(value) for value in marginals],
    )


@app.post(
    "/v1/candidate-posterior",
    response_model=CandidatePosteriorResponse,
    response_model_exclude_none=True,
)
def compute_candidate_posterior(
    request: CandidatePosteriorRequest,
) -> CandidatePosteriorResponse:
    envelopes = [
        envelope_distribution(candidate.atom_groups)
        for candidate in request.candidates
    ]
    priors = [candidate.prior_fraction for candidate in request.candidates]
    observations = [(obs.shift_mda, obs.count) for obs in request.observations]
    result = candidate_posterior(
        envelopes,
        priors,
        observations,
        request.ion_count,
        request.calibration_radius,
    )
    if result is None:
        return CandidatePosteriorResponse(possible=False)
    joints, candidate_marginals, offset_marginals = result
    offsets = list(
        range(-request.calibration_radius, request.calibration_radius + 1)
    )
    return CandidatePosteriorResponse(
        possible=True,
        joint_posteriors=[
            [to_fraction_str(value) for value in row] for row in joints
        ],
        candidate_posteriors=[
            to_fraction_str(value) for value in candidate_marginals
        ],
        offset_posteriors=[
            to_fraction_str(value) for value in offset_marginals
        ],
        offsets=offsets,
    )


@app.post(
    "/v1/discrimination-plan",
    response_model=DiscriminationPlanResponse,
)
def compute_discrimination_plan(
    request: DiscriminationPlanRequest,
) -> DiscriminationPlanResponse:
    envelopes = [
        envelope_distribution(candidate.atom_groups)
        for candidate in request.candidates
    ]
    windows, total_bins, p0, p1, accuracy, hit_choice = discrimination_plan(
        envelopes[0], envelopes[1], request.max_windows, request.budget
    )
    return DiscriminationPlanResponse(
        windows=[[left, right] for left, right in windows],
        total_bins=total_bins,
        hit_probabilities=[to_fraction_str(p0), to_fraction_str(p1)],
        accuracy=to_fraction_str(accuracy),
        hit_choice=hit_choice,
    )


@app.post(
    "/v1/fragment-spectrum",
    response_model=FragmentSpectrumResponse,
    response_model_exclude_none=True,
)
def compute_fragment_spectrum(
    request: FragmentSpectrumRequest,
) -> FragmentSpectrumResponse:
    probability, peaks = fragment_spectrum(
        request.atom_groups,
        request.fragment_counts,
        request.target_parent_shift_mda,
    )
    if probability == 0:
        return FragmentSpectrumResponse(
            reachable=False, probability=to_fraction_str(Fraction(0))
        )
    return FragmentSpectrumResponse(
        reachable=True,
        probability=to_fraction_str(probability),
        peaks=[to_peak(shift, prob) for shift, prob in sorted(peaks.items())],
    )


@app.post(
    "/v1/abundance-sensitivity",
    response_model=AbundanceSensitivityResponse,
)
def compute_abundance_sensitivity(
    request: AbundanceSensitivityRequest,
) -> AbundanceSensitivityResponse:
    group_dists = [group_distribution(group) for group in request.atom_groups]
    derivatives = []
    for transfer in request.transfers:
        dist = transfer_derivative(request.atom_groups, group_dists, transfer)
        derivatives.append(
            [
                DerivativePeak(
                    shift_mda=shift, derivative=to_fraction_str(value)
                )
                for shift, value in sorted(dist.items())
                if value != 0
            ]
        )
    return AbundanceSensitivityResponse(derivatives=derivatives)


@app.post(
    "/v1/exchange-trajectory",
    response_model=ExchangeTrajectoryResponse,
    response_model_exclude_none=True,
)
def compute_exchange_trajectory(
    request: ExchangeTrajectoryRequest,
) -> ExchangeTrajectoryResponse:
    observations = [
        (obs.step, obs.lower, obs.upper) for obs in request.observations
    ]
    probability, distributions = exchange_trajectory(
        request.n,
        request.m,
        request.h,
        request.b,
        request.steps,
        observations,
    )
    if distributions is None:
        return ExchangeTrajectoryResponse(probability=to_fraction_str(probability))
    return ExchangeTrajectoryResponse(
        probability=to_fraction_str(probability),
        distributions=[
            [
                HeavyCountPeak(
                    heavy_count=count, probability=to_fraction_str(prob)
                )
                for count, prob in sorted(step_dist.items())
            ]
            for step_dist in distributions
        ],
    )


@app.post(
    "/v1/exchange-turnover",
    response_model=ExchangeTurnoverResponse,
    response_model_exclude_none=True,
)
def compute_exchange_turnover(
    request: ExchangeTurnoverRequest,
) -> ExchangeTurnoverResponse:
    observations = [
        (obs.step, obs.lower, obs.upper) for obs in request.observations
    ]
    probability, distribution = exchange_turnover(
        request.n,
        request.m,
        request.h,
        request.b,
        request.steps,
        observations,
    )
    if distribution is None:
        return ExchangeTurnoverResponse(probability=to_fraction_str(probability))
    return ExchangeTurnoverResponse(
        probability=to_fraction_str(probability),
        distribution=[
            TurnoverCountPeak(count=count, probability=to_fraction_str(prob))
            for count, prob in sorted(distribution.items())
        ],
    )


@app.post(
    "/v1/exchange-first-passage",
    response_model=ExchangeFirstPassageResponse,
    response_model_exclude_none=True,
)
def compute_exchange_first_passage(
    request: ExchangeFirstPassageRequest,
) -> ExchangeFirstPassageResponse:
    observations = [
        (obs.step, obs.lower, obs.upper) for obs in request.observations
    ]
    probability, distribution, never = exchange_first_passage(
        request.n,
        request.m,
        request.h,
        request.b,
        request.steps,
        observations,
        request.threshold,
    )
    if distribution is None:
        return ExchangeFirstPassageResponse(
            probability=to_fraction_str(probability)
        )
    return ExchangeFirstPassageResponse(
        probability=to_fraction_str(probability),
        first_hit_distribution=[
            FirstHitPeak(step=step, probability=to_fraction_str(prob))
            for step, prob in sorted(distribution.items())
        ],
        never_probability=to_fraction_str(never),
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
