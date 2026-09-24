"""Exact solver for the feeder phase adjudication problem.

The engineer submits binary phase variables and weighted XOR observations.
An observation ``left XOR right == xor_value`` with cost ``c`` is either
satisfied (cost 0) or *polluted* (cost c). The adjudication must, over all
2**n phase assignments, lexicographically minimize

    1. the total cost of polluted observations, then
    2. the number of polluted observations,

and decide whether the minimizer is unique.

With at most 26 variables (2**26 ~= 67 million assignments) the optimum is
found by exact enumeration: the violation indicator of an XOR observation is a
quadratic polynomial of binary variables, so all assignments of a block are
evaluated with a few BLAS matrix products (numpy).

Exactness
---------
Observation costs are arbitrary positive integers (no upper bound). Each cost
is split into base-2**LIMB_BITS limbs (LIMB_BITS = 32), giving one quadratic
energy polynomial per limb whose weights are integers below 2**32, plus one
count energy with weight 1 per observation:

* limb energy, weight ``(c >> 32*k) % 2**32`` per observation, an integer in
  [0, 120 * (2**32 - 1)];
* count energy, weight 1 per observation, an integer in [0, 120].

All are computed in float64. Within one limb the absolute coefficients of the
quadratic form are bounded by 600 * 2**32 (constant), 360 * 2**32 (linear)
and 240 * 2**32 (pairwise), so every partial sum during the block evaluation
is an integer below 2**53 and therefore exact; each block is rounded to the
nearest integer and the residual is verified against ``ROUND_TOLERANCE`` as a
guard. A limb *energy* is a sum of up to 120 limb weights and can exceed the
limb base, so after evaluation the limbs of every assignment are carry-
normalized in exact int64 arithmetic (each normalized limb < 2**LIMB_BITS,
plus one top carry limb). The exact polluted cost of an assignment is

    cost(x) = sum_k limb_cost_k(x) * 2**(LIMB_BITS * k)

and assignments are ordered by the exact integer key
``(limb_K(x), ..., limb_0(x), count(x))`` over normalized limbs, which is
precisely the lexicographic (cost, count) order — near-equal and cancelling
large costs are distinguished exactly. The combined scalar objective

    combined = SCALE * cost + count,   SCALE = MAX_OBSERVATIONS + 1

is then formed in arbitrary-precision integer arithmetic. Since count <=
MAX_OBSERVATIONS < SCALE, minimizing it is exactly equivalent to the
lexicographic pair. The chosen assignment is finally re-evaluated with pure
integer arithmetic as an end-to-end guard.

Determinism / order independence
---------------------------------
Variables are ranked by name (in the validation layer), observations are
processed in id order, and assignments are scanned in lexicographic order, so
the first / second best assignments and the polluted id lists are canonical
and independent of submission order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_VARIABLES = 26
MAX_OBSERVATIONS = 120

#: Weight of one unit of polluted cost in the combined scalar objective.
#: Must be strictly greater than the maximum possible polluted count.
SCALE = MAX_OBSERVATIONS + 1

#: Observation costs are split into base-2**LIMB_BITS limbs so that every
#: per-limb energy coefficient stays far below 2**53 and the float64 block
#: evaluation of each limb is exact (see module docstring). Costs of any
#: magnitude are handled exactly; there is no upper bound on a cost.
LIMB_BITS = 32
LIMB_BASE = 1 << LIMB_BITS

#: Number of "fast" (inner, block-local) free variables in the split scan.
#: A block holds 2**FAST_BITS assignments (~524k here, ~80 MB float64/array).
FAST_BITS = 19

#: Every float64 block energy must be within this distance of an integer.
ROUND_TOLERANCE = 0.25


@dataclass(frozen=True)
class Observation:
    """One submitted XOR observation in index form."""

    id: str
    left_rank: int
    right_rank: int
    xor_value: int
    cost: int


@dataclass(frozen=True)
class Problem:
    """Solver input. Variable names are unique; ranks follow name order."""

    variable_names: tuple[str, ...]
    observations: tuple[Observation, ...]
    fixed: dict[int, int]  # variable rank -> reference bit


@dataclass(frozen=True)
class Solution:
    optimal_cost: int
    optimal_polluted_count: int
    unique: bool
    #: lexicographically smallest optimal assignment, name -> bit
    assignment: dict[str, int]
    #: polluted observation ids under ``assignment``, sorted by id
    violated_ids: tuple[str, ...]
    #: another optimal assignment (name -> bit), or None when unique
    witness: dict[str, int] | None
    witness_polluted_cost: int | None
    witness_polluted_count: int | None
    witness_violated_ids: tuple[str, ...] | None


class SolverError(RuntimeError):
    """Unexpected internal failure of the exact solver."""


@dataclass
class _QuadraticEnergy:
    """E(x) = constant + linear^T x + x^T matrix x over the free variables."""

    constant: float
    linear: np.ndarray
    pair: np.ndarray

    @classmethod
    def zeros(cls, f: int) -> "_QuadraticEnergy":
        return cls(0.0, np.zeros(f, dtype=np.float64), np.zeros((f, f), dtype=np.float64))

    def add_observation(
        self,
        weight: float,
        a: int,
        b: int,
        xor_value: int,
        fixed: dict[int, int],
        free_pos: dict[int, int],
    ) -> None:
        """Add the violation indicator of one XOR observation times weight."""
        if xor_value == 0:
            # violated iff xa != xb:  xa + xb - 2 xa xb
            ca = cb = 1.0
            cab = -2.0
        else:
            # violated iff xa == xb:  1 - xa - xb + 2 xa xb
            ca = cb = -1.0
            cab = 2.0
            self.constant += weight

        a_fixed = a in fixed
        b_fixed = b in fixed
        va = fixed[a] if a_fixed else None
        vb = fixed[b] if b_fixed else None

        if a_fixed:
            self.constant += ca * weight * va
        else:
            self.linear[free_pos[a]] += ca * weight
        if b_fixed:
            self.constant += cb * weight * vb
        else:
            self.linear[free_pos[b]] += cb * weight

        if a_fixed and b_fixed:
            self.constant += cab * weight * va * vb
        elif a_fixed:
            self.linear[free_pos[b]] += cab * weight * va
        elif b_fixed:
            self.linear[free_pos[a]] += cab * weight * vb
        else:
            pa, pb = free_pos[a], free_pos[b]
            if pa == pb:  # self-referencing observation on a free variable
                # xa * xa == xa for a binary variable
                self.linear[pa] += cab * weight
            else:
                self.pair[pa, pb] += cab * weight
                self.pair[pb, pa] += cab * weight

    def matrix(self) -> tuple[float, np.ndarray]:
        """Return (constant, M) with E = constant + x^T M x (linear on diag)."""
        m = self.pair / 2.0
        np.fill_diagonal(m, self.linear)
        return self.constant, m


def solve(problem: Problem) -> Solution:
    names = problem.variable_names
    fixed = dict(problem.fixed)
    # Id ordering makes accumulation and every reported result independent of
    # the submitted observation order.
    observations = sorted(problem.observations, key=lambda o: o.id)

    n = len(names)
    free = [rank for rank in range(n) if rank not in fixed]
    f = len(free)
    free_pos = {rank: pos for pos, rank in enumerate(free)}

    # Fast path: without observations every assignment is optimal (cost and
    # count both zero). Canonical assignment is all zeros respecting the
    # references; the lexicographically second witness flips the last free
    # variable, when one exists.
    if not observations:
        assignment = {name: 0 for name in names}
        for rank, bit in fixed.items():
            assignment[names[rank]] = bit
        witness = None
        if free:
            witness = dict(assignment)
            # lexicographically second all-zero-cost assignment: (0,...,0,1)
            witness[names[free[-1]]] = 1
        return Solution(
            optimal_cost=0,
            optimal_polluted_count=0,
            unique=not free,
            assignment=assignment,
            violated_ids=(),
            witness=witness,
            witness_polluted_cost=0 if free else None,
            witness_polluted_count=0 if free else None,
            witness_violated_ids=() if free else None,
        )

    # Split every cost into base-2**LIMB_BITS limbs (least significant first);
    # each limb gets its own quadratic energy so costs of any magnitude are
    # evaluated exactly.
    limb_count = max(
        1, (max(obs.cost for obs in observations).bit_length() + LIMB_BITS - 1)
        // LIMB_BITS
    )
    limb_energies = [_QuadraticEnergy.zeros(f) for _ in range(limb_count)]
    count_energy = _QuadraticEnergy.zeros(f)
    for obs in observations:
        kwargs = dict(
            a=obs.left_rank,
            b=obs.right_rank,
            xor_value=obs.xor_value,
            fixed=fixed,
            free_pos=free_pos,
        )
        remaining = obs.cost
        for limb_energy in limb_energies:
            limb_energy.add_observation(float(remaining % LIMB_BASE), **kwargs)
            remaining //= LIMB_BASE
        count_energy.add_observation(1.0, **kwargs)

    limb_matrices = [energy.matrix() for energy in limb_energies]
    count_const, count_matrix = count_energy.matrix()

    first_code, second_code, best_combined = _enumerate(
        limb_matrices, count_matrix, count_const, f
    )

    def decode(code: int) -> dict[str, int]:
        assignment: dict[str, int] = {}
        for pos, rank in enumerate(free):
            assignment[names[rank]] = (code >> (f - 1 - pos)) & 1
        for rank, bit in fixed.items():
            assignment[names[rank]] = bit
        return {name: assignment[name] for name in names}

    def evaluate(assignment: dict[str, int]) -> tuple[int, int, list[str]]:
        cost = 0
        count = 0
        ids: list[str] = []
        for obs in observations:
            violated = (
                assignment[names[obs.left_rank]]
                ^ assignment[names[obs.right_rank]]
                != obs.xor_value
            )
            if violated:
                cost += obs.cost
                count += 1
                ids.append(obs.id)
        return cost, count, ids

    assignment = decode(first_code)
    cost, count, violated_ids = evaluate(assignment)
    if SCALE * cost + count != best_combined:
        raise SolverError("internal check failed for canonical assignment")

    witness = None
    witness_cost = witness_count = None
    witness_violated_ids: tuple[str, ...] | None = None
    if second_code >= 0:
        witness = decode(second_code)
        witness_cost, witness_count, wids = evaluate(witness)
        witness_violated_ids = tuple(wids)
        if SCALE * witness_cost + witness_count != best_combined:
            raise SolverError("internal check failed for witness assignment")

    return Solution(
        optimal_cost=cost,
        optimal_polluted_count=count,
        unique=second_code < 0,
        assignment=assignment,
        violated_ids=tuple(violated_ids),
        witness=witness,
        witness_polluted_cost=witness_cost,
        witness_polluted_count=witness_count,
        witness_violated_ids=witness_violated_ids,
    )


def _enumerate(
    limb_matrices: list[tuple[float, np.ndarray]],
    count_matrix: np.ndarray,
    count_constant: float,
    f: int,
) -> tuple[int, int, int]:
    """Scan all 2**f assignments in lexicographic order.

    ``limb_matrices`` holds one ``(constant, matrix)`` quadratic energy per
    cost limb, least significant limb first: the exact polluted cost of an
    assignment is ``sum_k energy_k * LIMB_BASE**k``. Returns (first_code,
    second_code, best_combined); second_code is -1 when the optimum is unique.
    Variable ``pos`` is bit ``f-1-pos`` of the code, so ascending codes are
    lexicographically ordered assignments.

    The scan splits variables into slow vars (positions 0..q-1, the block
    index) and fast vars (positions q..f-1, enumerated inside a block); fast
    bit patterns and the fast/fast quadratic contributions are built once per
    limb. Every limb energy and the count energy is evaluated exactly (see
    LIMB_BITS) and rounded through the ROUND_TOLERANCE guard. A limb energy
    sums up to 120 limb weights and may exceed the limb base, so the per-
    assignment limbs are carry-normalized in exact int64 arithmetic; the
    assignments are then ordered by the exact integer key
    ``(limb_K, ..., limb_0, count)`` over normalized limbs — i.e. by
    (polluted cost, polluted count) — and the combined objective
    ``SCALE * cost + count`` is formed in arbitrary-precision arithmetic.
    """
    p = min(f, FAST_BITS)
    q = f - p

    lo_powers = np.arange(p - 1, -1, -1, dtype=np.int64)
    base_index = np.arange(1 << p, dtype=np.int64)
    fast_bits = ((base_index[:, None] >> lo_powers[None, :]) & 1).astype(np.float64)

    def parts(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_ss = matrix[:q, :q]
        m_sf = matrix[:q, q:]
        m_ff = matrix[q:, q:]
        base = ((fast_bits @ m_ff) * fast_bits).sum(axis=1)
        return m_ss, m_sf, base

    limb_parts = [(constant, *parts(matrix)) for constant, matrix in limb_matrices]
    k_ss, k_sf, count_base = parts(count_matrix)

    best_key: tuple[int, ...] | None = None
    first_code = -1
    second_code = -1

    def round_exact(energies: np.ndarray, label: str) -> np.ndarray:
        rounded = np.rint(energies)
        residual = np.max(np.absolute(energies - rounded)) if energies.size else 0.0
        if residual > ROUND_TOLERANCE:
            raise SolverError(
                f"{label} energy rounding residual {residual:.6f} exceeds tolerance "
                f"{ROUND_TOLERANCE}; enumeration is not exact on this input"
            )
        return rounded.astype(np.int64)

    hi_powers = np.arange(q - 1, -1, -1, dtype=np.int64)
    for block in range(1 << q):
        slow = ((block >> hi_powers) & 1).astype(np.float64)

        # Evaluate every cost limb (least significant first) for the block.
        limbs: list[np.ndarray] = []
        for constant, m_ss, m_sf, base in limb_parts:
            block_const = constant + float(slow @ m_ss @ slow)
            cross = 2.0 * (m_sf.T @ slow)
            limbs.append(round_exact(block_const + base + fast_bits @ cross, "cost"))

        count_block_const = count_constant + float(slow @ k_ss @ slow)
        count_cross = 2.0 * (k_sf.T @ slow)
        count_e = round_exact(
            count_block_const + count_base + fast_bits @ count_cross, "count"
        )

        # Carry-normalize: a limb energy sums up to 120 limb weights and may
        # exceed LIMB_BASE. After normalization every limb is in
        # [0, LIMB_BASE) (plus a top carry limb <= 120), so ordering by the
        # integer key (limb_K, ..., limb_0, count) is exactly ordering by
        # (polluted cost, polluted count). All values are far below 2**63,
        # so the shifts and masks are exact.
        normalized: list[np.ndarray] = []
        carry = np.zeros(1 << p, dtype=np.int64)
        for energy in limbs:
            energy = energy + carry
            normalized.append(energy & (LIMB_BASE - 1))
            carry = energy >> LIMB_BITS
        normalized.append(carry)

        # Exact integer key of the block optimum, refined from the most
        # significant cost limb down to the polluted count. All positions
        # surviving the mask share every key component computed so far.
        mask = np.ones(1 << p, dtype=bool)
        key: list[int] = []
        for energy in reversed(normalized):
            minimum = int(energy[mask].min())
            mask &= energy == minimum
            key.append(minimum)

        minimum_count = int(count_e[mask].min())
        mask &= count_e == minimum_count
        key.append(minimum_count)

        hits = np.flatnonzero(mask)
        block_key = tuple(key)
        block_first = (block << p) + int(hits[0])
        if best_key is None or block_key < best_key:
            best_key = block_key
            first_code = block_first
            second_code = (block << p) + int(hits[1]) if hits.size > 1 else -1
        elif block_key == best_key and second_code < 0:
            # Same optimum, later lexicographic block: first new hit is the
            # second distinct optimal assignment.
            second_code = block_first

    assert best_key is not None and first_code >= 0
    best_cost = 0
    for limb_value in best_key[:-1]:
        best_cost = best_cost * LIMB_BASE + limb_value
    best_combined = SCALE * best_cost + best_key[-1]
    return first_code, second_code, best_combined
