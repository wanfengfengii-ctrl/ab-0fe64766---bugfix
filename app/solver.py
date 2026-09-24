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
The lexicographic pair (cost, count) is minimized by evaluating two energy
polynomials separately:

* cost energy, weight ``c`` per observation, an integer in [0, 120 * MAX_COST]
  (<= 1.2e11);
* count energy, weight 1 per observation, an integer in [0, 120].

Both are computed in float64. Coefficients are integer sums of at most 120
terms, and a standard dot-product error bound (gamma_n = n * 2**-53) bounds the
evaluation error of the cost energy by well under 0.1 (and the count energy by
~1e-9); each block is rounded to the nearest integer and the residual is
verified against ``ROUND_TOLERANCE``. The combined scalar objective

    combined = SCALE * cost + count,   SCALE = MAX_OBSERVATIONS + 1

is then formed in exact int64 arithmetic. Since count <= MAX_OBSERVATIONS <
SCALE, minimizing it is exactly equivalent to the lexicographic pair. The
chosen assignment is finally re-evaluated with pure integer arithmetic as an
end-to-end guard.

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

#: Maximum accepted observation cost. With 120 observations the cost energy is
#: then at most 1.2e11, where the float64 block evaluation error is strictly
#: below 0.1 (see module docstring), so nearest-integer rounding is exact.
MAX_COST_VALUE = 1_000_000_000

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

    cost_energy = _QuadraticEnergy.zeros(f)
    count_energy = _QuadraticEnergy.zeros(f)
    for obs in observations:
        kwargs = dict(
            a=obs.left_rank,
            b=obs.right_rank,
            xor_value=obs.xor_value,
            fixed=fixed,
            free_pos=free_pos,
        )
        cost_energy.add_observation(float(obs.cost), **kwargs)
        count_energy.add_observation(1.0, **kwargs)

    cost_const, cost_matrix = cost_energy.matrix()
    count_const, count_matrix = count_energy.matrix()

    first_code, second_code, best_combined = _enumerate(
        cost_matrix, cost_const, count_matrix, count_const, f
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
    cost_matrix: np.ndarray,
    cost_constant: float,
    count_matrix: np.ndarray,
    count_constant: float,
    f: int,
) -> tuple[int, int, int]:
    """Scan all 2**f assignments in lexicographic order.

    Returns (first_code, second_code, best_combined); second_code is -1 when
    the optimum is unique. Variable ``pos`` is bit ``f-1-pos`` of the code, so
    ascending codes are lexicographically ordered assignments.

    The scan splits variables into slow vars (positions 0..q-1, the block
    index) and fast vars (positions q..f-1, enumerated inside a block); fast
    bit patterns and the fast/fast quadratic contributions are built once.
    Cost and count energies are evaluated and rounded independently, then the
    combined int64 objective SCALE * cost + count is formed exactly.
    """
    p = min(f, FAST_BITS)
    q = f - p

    def parts(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m_ss = matrix[:q, :q]
        m_sf = matrix[:q, q:]
        m_ff = matrix[q:, q:]
        lo_powers = np.arange(p - 1, -1, -1, dtype=np.int64)
        base_index = np.arange(1 << p, dtype=np.int64)
        fast_bits = ((base_index[:, None] >> lo_powers[None, :]) & 1).astype(np.float64)
        base = ((fast_bits @ m_ff) * fast_bits).sum(axis=1)
        return m_ss, m_sf, fast_bits, base

    c_ss, c_sf, fast_bits, cost_base = parts(cost_matrix)
    k_ss, k_sf, _, count_base = parts(count_matrix)

    best_combined: int | None = None
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

    def absorb(block: int, combined: np.ndarray) -> None:
        nonlocal best_combined, first_code, second_code
        minimum = int(combined.min())
        if best_combined is not None and minimum > best_combined:
            return
        hits = np.flatnonzero(combined == minimum)
        first = (block << p) + int(hits[0])
        if best_combined is None or minimum < best_combined:
            best_combined = minimum
            first_code = first
            second_code = (block << p) + int(hits[1]) if hits.size > 1 else -1
        elif second_code < 0:
            # Same optimum, later lexicographic block: first new hit is the
            # second distinct optimal assignment.
            second_code = first

    if q == 0:
        cost_e = round_exact(cost_constant + cost_base, "cost")
        count_e = round_exact(count_constant + count_base, "count")
        absorb(0, SCALE * cost_e + count_e)
    else:
        hi_powers = np.arange(q - 1, -1, -1, dtype=np.int64)
        for block in range(1 << q):
            slow = ((block >> hi_powers) & 1).astype(np.float64)
            cost_block_const = cost_constant + float(slow @ c_ss @ slow)
            count_block_const = count_constant + float(slow @ k_ss @ slow)
            cost_cross = 2.0 * (c_sf.T @ slow)
            count_cross = 2.0 * (k_sf.T @ slow)
            cost_e = round_exact(
                cost_block_const + cost_base + fast_bits @ cost_cross, "cost"
            )
            count_e = round_exact(
                count_block_const + count_base + fast_bits @ count_cross, "count"
            )
            absorb(block, SCALE * cost_e + count_e)

    assert best_combined is not None and first_code >= 0
    return first_code, second_code, int(best_combined)
