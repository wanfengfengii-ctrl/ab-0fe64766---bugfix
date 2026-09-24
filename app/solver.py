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
Observation costs are arbitrary positive integers with no upper bound, so
every energy coefficient is kept as an exact Python integer and the optimum
is located with one of two regimes, both exact:

* Single pass (every coefficient below 2**38, covering the whole classic
  120 x 1e9 cost regime): the cost energy (integer in [0, total cost]) and
  the count energy (integer in [0, 120]) are evaluated directly in float64.
  A standard dot-product error bound (gamma_n = n * 2**-53) keeps the
  evaluation error well under ``ROUND_TOLERANCE`` of the nearest integer, so
  rounding to it is exact.
* Component scan (larger coefficients): every coefficient is decomposed into
  signed base-2**k digits (k = 28, halved on demand down to 1) and each
  digit's energy polynomial is evaluated in float64 under the same error
  bound — digit magnitudes below 2**28 keep the error under ~0.1. Rounding
  recovers every exact integer component energy, carries are normalized
  across components, and assignments are then ordered by the exact total
  cost (most significant digit first) and finally by the polluted count.

A per-block residual check guards every rounding; a violation falls back to
the component scan with a narrower (hence even more accurate) digit width.

The combined scalar objective

    combined = SCALE * cost + count,   SCALE = MAX_OBSERVATIONS + 1

is formed in exact integer arithmetic. Since count <= MAX_OBSERVATIONS <
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

#: Number of "fast" (inner, block-local) free variables in the split scan.
#: A block holds 2**FAST_BITS assignments (~524k here, ~80 MB float64/array).
FAST_BITS = 19

#: Every float64 block energy must be within this distance of an integer.
ROUND_TOLERANCE = 0.25

#: Coefficient magnitude up to which the single-pass float64 evaluation is
#: provably exact (covers the classic 120 x 1e9 cost regime with margin).
_SINGLE_PASS_COEFF_LIMIT = 2**38

#: Digit width (bits) of the signed base-2**k decomposition used by the exact
#: component scan for larger coefficients. At k = 28 every component matrix
#: entry is below 2**35 and the float64 evaluation error of each component
#: energy is bounded by ~0.1, well under ROUND_TOLERANCE.
_COMPONENT_BITS = 28

#: Sentinel above any base-2**k digit or polluted count, for masked minima.
_LEX_INF = 1 << 60


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


class _ResidualError(SolverError):
    """A float64 block energy was too far from any integer to round exactly.

    Triggers a retry with the (slower, even more accurate) component scan or
    a narrower digit width; never silently produces an inexact result.
    """


@dataclass
class _QuadraticEnergy:
    """E(x) = constant + linear^T x + x^T pair x over the free variables.

    Coefficients are exact Python integers (observation costs have no upper
    bound), so the energy can be evaluated exactly for any input size.
    """

    constant: int
    linear: list[int]
    pair: list[list[int]]

    @classmethod
    def zeros(cls, f: int) -> "_QuadraticEnergy":
        return cls(0, [0] * f, [[0] * f for _ in range(f)])

    def add_observation(
        self,
        weight: int,
        a: int,
        b: int,
        xor_value: int,
        fixed: dict[int, int],
        free_pos: dict[int, int],
    ) -> None:
        """Add the violation indicator of one XOR observation times weight."""
        if xor_value == 0:
            # violated iff xa != xb:  xa + xb - 2 xa xb
            ca = cb = 1
            cab = -2
        else:
            # violated iff xa == xb:  1 - xa - xb + 2 xa xb
            ca = cb = -1
            cab = 2
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
                self.pair[pa][pb] += cab * weight
                self.pair[pb][pa] += cab * weight

    def max_abs_coefficient(self) -> int:
        """Largest |coefficient| among constant, linear and pair entries."""
        hi = abs(self.constant)
        for value in self.linear:
            hi = max(hi, abs(value))
        for row in self.pair:
            for value in row:
                hi = max(hi, abs(value))
        return hi

    def matrix(self) -> tuple[float, np.ndarray]:
        """Return (constant, M) with E = constant + x^T M x (linear on diag).

        The float64 conversion is exact only when every coefficient fits a
        float64 mantissa; callers use this solely in the single-pass regime
        where that is guaranteed.
        """
        f = len(self.linear)
        m = np.zeros((f, f), dtype=np.float64)
        if f:
            m = np.array(self.pair, dtype=np.float64) / 2.0
            np.fill_diagonal(m, np.array(self.linear, dtype=np.float64))
        return float(self.constant), m

    def component(self, j: int, k: int) -> tuple[int, np.ndarray]:
        """(const, M) for signed base-2**k digit ``j`` of every coefficient.

        Each coefficient c is decomposed as c = sum_j d_j * 2**(j*k) with
        signed digits |d_j| < 2**k (base-2**k digits of |c|, sign of c), so
        the exact energy is E(x) = sum_j E_j(x) * 2**(j*k) where E_j is the
        energy of the returned (const, M) pair. Entries may be half-integers
        (odd pair digits are halved symmetrically); all are exactly
        representable in float64.
        """
        shift = j * k
        mask = (1 << k) - 1

        def digit(c: int) -> int:
            d = (abs(c) >> shift) & mask
            return -d if c < 0 else d

        f = len(self.linear)
        m = np.zeros((f, f), dtype=np.float64)
        for a in range(f):
            m[a, a] = digit(self.linear[a])
            row = self.pair[a]
            for b in range(a + 1, f):
                value = row[b]
                if value:
                    half = digit(value) / 2.0
                    m[a, b] = half
                    m[b, a] = half
        return digit(self.constant), m


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
        cost_energy.add_observation(obs.cost, **kwargs)
        count_energy.add_observation(1, **kwargs)

    # Count coefficients are at most 2 * MAX_OBSERVATIONS, so the count
    # energy is always evaluated exactly in a single float64 pass.
    count_const, count_matrix = count_energy.matrix()
    total_cost = sum(obs.cost for obs in observations)

    first_code, second_code, best_combined = _enumerate(
        cost_energy, count_matrix, count_const, f, total_cost
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


def _round_exact(energies: np.ndarray, label: str) -> np.ndarray:
    """Round block energies to their exact integer values, with a guard."""
    rounded = np.rint(energies)
    residual = np.max(np.absolute(energies - rounded)) if energies.size else 0.0
    if residual > ROUND_TOLERANCE:
        raise _ResidualError(
            f"{label} energy rounding residual {residual:.6f} exceeds tolerance "
            f"{ROUND_TOLERANCE}; falling back to a narrower evaluation"
        )
    return rounded.astype(np.int64)


def _enumerate(
    cost_energy: _QuadraticEnergy,
    count_matrix: np.ndarray,
    count_constant: float,
    f: int,
    total_cost: int,
) -> tuple[int, int, int]:
    """Locate the optimum over all 2**f assignments, exactly.

    Returns (first_code, second_code, best_combined); second_code is -1 when
    the optimum is unique. Uses the single-pass float64 scan when every cost
    coefficient is small enough for it to be provably exact, and otherwise
    the exact component scan; a rounding-residual violation in either regime
    falls back to a narrower, even more accurate evaluation.
    """
    if cost_energy.max_abs_coefficient() <= _SINGLE_PASS_COEFF_LIMIT:
        cost_constant, cost_matrix = cost_energy.matrix()
        try:
            return _enumerate_single(
                cost_matrix, cost_constant, count_matrix, count_constant, f
            )
        except _ResidualError:
            pass  # fall through to the exact component scan
    k = _COMPONENT_BITS
    while True:
        try:
            return _enumerate_components(
                cost_energy, count_matrix, count_constant, f, total_cost, k
            )
        except _ResidualError:
            if k <= 1:
                raise
            k = max(1, k // 2)


def _enumerate_single(
    cost_matrix: np.ndarray,
    cost_constant: float,
    count_matrix: np.ndarray,
    count_constant: float,
    f: int,
) -> tuple[int, int, int]:
    """Scan all 2**f assignments in lexicographic order (single float64 pass).

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
        cost_e = _round_exact(cost_constant + cost_base, "cost")
        count_e = _round_exact(count_constant + count_base, "count")
        absorb(0, SCALE * cost_e + count_e)
    else:
        hi_powers = np.arange(q - 1, -1, -1, dtype=np.int64)
        for block in range(1 << q):
            slow = ((block >> hi_powers) & 1).astype(np.float64)
            cost_block_const = cost_constant + float(slow @ c_ss @ slow)
            count_block_const = count_constant + float(slow @ k_ss @ slow)
            cost_cross = 2.0 * (c_sf.T @ slow)
            count_cross = 2.0 * (k_sf.T @ slow)
            cost_e = _round_exact(
                cost_block_const + cost_base + fast_bits @ cost_cross, "cost"
            )
            count_e = _round_exact(
                count_block_const + count_base + fast_bits @ count_cross, "count"
            )
            absorb(block, SCALE * cost_e + count_e)

    assert best_combined is not None and first_code >= 0
    return first_code, second_code, int(best_combined)


def _enumerate_components(
    cost_energy: _QuadraticEnergy,
    count_matrix: np.ndarray,
    count_constant: float,
    f: int,
    total_cost: int,
    k: int,
) -> tuple[int, int, int]:
    """Scan all 2**f assignments with exact arbitrary-size integer costs.

    Every cost coefficient is split into signed base-2**k digits; each digit
    energy is evaluated with the same block pipeline as the single-pass scan
    and rounded to its exact integer value. Per assignment the digit energies
    are carry-normalized into the canonical base-2**k representation of the
    exact total cost, and assignments are ordered by (cost digits from most
    to least significant, then polluted count) — exactly the lexicographic
    (cost, count) order, for costs of any size.
    """
    hi = max(cost_energy.max_abs_coefficient(), total_cost, 1)
    # 2**(s*k) exceeds every coefficient and every attainable total cost, so
    # the digit decomposition is complete and the final carry is always zero.
    s = max(2, hi.bit_length() // k + 2)
    components = [cost_energy.component(j, k) for j in range(s)]

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

    component_parts = [
        (const, *parts(matrix)) for const, matrix in components
    ]
    n_ss, n_sf, count_base = parts(count_matrix)

    hi_powers = np.arange(q - 1, -1, -1, dtype=np.int64)
    mask = (1 << k) - 1
    n_block = 1 << p

    best_key: tuple[int, int] | None = None  # (exact cost, count)
    first_code = -1
    second_code = -1

    for block in range(1 << q):
        slow = ((block >> hi_powers) & 1).astype(np.float64)

        count_block_const = count_constant + float(slow @ n_ss @ slow)
        count_cross = 2.0 * (n_sf.T @ slow)
        count_e = _round_exact(
            count_block_const + count_base + fast_bits @ count_cross, "count"
        )

        energy_columns = []
        for const_j, c_ss, c_sf, cost_base in component_parts:
            block_const = const_j + float(slow @ c_ss @ slow)
            cross = 2.0 * (c_sf.T @ slow)
            energy_columns.append(
                _round_exact(block_const + cost_base + fast_bits @ cross, "cost")
            )

        # Carry normalization: signed component energies -> canonical
        # non-negative base-2**k digits of the exact total cost.
        carry = np.zeros(n_block, dtype=np.int64)
        digit_columns = []
        for column in energy_columns:
            v = column + carry
            digit_columns.append(v & mask)
            carry = v >> k  # arithmetic shift: floor division, exact
        if np.any(carry):
            raise SolverError("internal carry normalization failed")

        # Block minimum in (cost digits high->low, then count) order.
        alive = np.ones(n_block, dtype=bool)
        for column in reversed(digit_columns):
            minimum = np.where(alive, column, _LEX_INF).min()
            alive &= column == minimum
        minimum = np.where(alive, count_e, _LEX_INF).min()
        alive &= count_e == minimum
        hits = np.flatnonzero(alive)
        first = int(hits[0])

        block_cost = 0
        for j, column in enumerate(digit_columns):
            block_cost += int(column[first]) << (j * k)
        key = (block_cost, int(count_e[first]))
        block_first = (block << p) + first
        if best_key is None or key < best_key:
            best_key = key
            first_code = block_first
            second_code = (block << p) + int(hits[1]) if hits.size > 1 else -1
        elif key == best_key and second_code < 0:
            # Same optimum, later lexicographic block: first new hit is the
            # second distinct optimal assignment.
            second_code = block_first

    assert best_key is not None and first_code >= 0
    return first_code, second_code, SCALE * best_key[0] + best_key[1]
