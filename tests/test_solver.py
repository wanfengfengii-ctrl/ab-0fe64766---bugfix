"""Brute-force cross-checks and property tests for the exact solver."""

from __future__ import annotations

import itertools
import random

import pytest

from app.solver import SCALE, Observation, Problem, solve


def brute_force(names, observations, fixed_ranks):
    """Reference implementation: scan every assignment in Python."""
    n = len(names)
    free = [r for r in range(n) if r not in fixed_ranks]
    best = None
    best_assignments = []
    for bits in itertools.product((0, 1), repeat=len(free)):
        values = dict(fixed_ranks)
        values.update(dict(zip(free, bits)))
        cost = 0
        count = 0
        for obs in observations:
            if values[obs.left_rank] ^ values[obs.right_rank] != obs.xor_value:
                cost += obs.cost
                count += 1
        combined = SCALE * cost + count
        if best is None or combined < best:
            best = combined
            best_assignments = [dict(values)]
        elif combined == best:
            best_assignments.append(dict(values))
    return best, best_assignments


def make_problem(rng, n, m, fix_prob=0.3):
    names = tuple(f"v{k:02d}" for k in range(n))
    observations = []
    for k in range(m):
        a = rng.randrange(n)
        b = rng.randrange(n)
        observations.append(
            Observation(
                id=f"obs-{k:03d}",
                left_rank=a,
                right_rank=b,
                xor_value=rng.randrange(2),
                cost=rng.randrange(1, 50),
            )
        )
    fixed = {r: rng.randrange(2) for r in range(n) if rng.random() < fix_prob}
    return Problem(names, tuple(observations), fixed)


@pytest.mark.parametrize("seed", range(200))
def test_solver_matches_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 8)
    m = rng.randrange(0, 12)
    problem = make_problem(rng, n, m)
    solution = solve(problem)

    best_combined, best_assignments = brute_force(
        problem.variable_names, problem.observations, problem.fixed
    )

    assert SCALE * solution.optimal_cost + solution.optimal_polluted_count == best_combined

    # references are honored
    for rank, bit in problem.fixed.items():
        assert solution.assignment[problem.variable_names[rank]] == bit

    # uniqueness matches brute force
    assert solution.unique == (len(best_assignments) == 1)

    # canonical assignment is the lexicographically smallest optimum
    names = problem.variable_names
    lexicographic = sorted(best_assignments, key=lambda x: tuple(x[r] for r in range(n)))
    expected = {names[r]: lexicographic[0][r] for r in range(n)}
    assert solution.assignment == expected

    if not solution.unique:
        assert solution.witness is not None
        witness_ranks = {names.index(name): bit for name, bit in solution.witness.items()}
        assert witness_ranks in best_assignments
        assert solution.witness != solution.assignment
        # witness is the second lexicographic optimum
        expected_second = {names[r]: lexicographic[1][r] for r in range(n)}
        assert solution.witness == expected_second
        assert (
            SCALE * (solution.witness_polluted_cost or 0)
            + (solution.witness_polluted_count or 0)
            == best_combined
        )
        assert solution.witness_polluted_cost == solution.optimal_cost
        assert solution.witness_polluted_count == solution.optimal_polluted_count
    else:
        assert solution.witness is None
        assert solution.witness_violated_ids is None


def test_polluted_ids_sorted_and_consistent():
    problem = make_problem(random.Random(7), 6, 10, fix_prob=0.0)
    solution = solve(problem)
    ids = list(solution.violated_ids)
    assert ids == sorted(ids)
    by_id = {o.id: o for o in problem.observations}
    for oid in ids:
        obs = by_id[oid]
        names = problem.variable_names
        assert (
            solution.assignment[names[obs.left_rank]]
            ^ solution.assignment[names[obs.right_rank]]
            != obs.xor_value
        )
    assert solution.optimal_polluted_count == len(ids)
    assert solution.optimal_cost == sum(by_id[oid].cost for oid in ids)


def test_order_independence_shuffled_observations():
    problem = make_problem(random.Random(99), 10, 30, fix_prob=0.2)
    s1 = solve(problem)
    shuffled = list(problem.observations)
    random.Random(1).shuffle(shuffled)
    s2 = solve(Problem(problem.variable_names, tuple(shuffled), dict(problem.fixed)))
    assert s1.assignment == s2.assignment
    assert s1.violated_ids == s2.violated_ids
    assert s1.unique == s2.unique
    assert s1.optimal_cost == s2.optimal_cost
    assert s1.optimal_polluted_count == s2.optimal_polluted_count
    assert s1.witness == s2.witness


def test_self_referencing_observations():
    names = ("z",)
    # z XOR z == 1 is impossible: always violated; parity 0 always satisfied
    obs_impossible = (Observation("c", 0, 0, 1, 7),)
    sol = solve(Problem(names, obs_impossible, {}))
    assert sol.optimal_cost == 7
    assert sol.optimal_polluted_count == 1
    assert sol.unique is False  # z free: both assignments are optimal

    obs_free = (Observation("ok", 0, 0, 0, 9),)
    sol = solve(Problem(names, obs_free, {}))
    assert sol.optimal_cost == 0
    assert sol.violated_ids == ()

    sol = solve(Problem(names, obs_impossible, {0: 0}))
    assert sol.unique is True
    assert sol.assignment == {"z": 0}
    assert sol.violated_ids == ("c",)


def test_combined_objective_cost_first():
    # expensive and cheap2 are self-contradictions (always violated); cheap1 is
    # avoidable. Minimum cost 101 is reached by (0,0) and (1,1): the solver
    # reports the lexicographically smallest, witness the other.
    names = ("a", "b")
    observations = (
        Observation("expensive", 0, 0, 1, 100),   # always violated: cost 100
        Observation("cheap1", 0, 1, 0, 1),        # violated iff a != b
        Observation("cheap2", 1, 1, 1, 1),        # always violated: cost 1
    )
    sol = solve(Problem(names, observations, {}))
    assert sol.optimal_cost == 101
    assert sol.optimal_polluted_count == 2
    assert sol.unique is False
    assert sol.assignment == {"a": 0, "b": 0}
    assert sol.witness == {"a": 1, "b": 1}


def _enumerate_levels(problem):
    """Return per-assignment (cost, count) pairs keyed by rank-bit tuple."""
    n = len(problem.variable_names)
    out = {}
    for bits in itertools.product((0, 1), repeat=n):
        if any(bits[r] != v for r, v in problem.fixed.items()):
            continue
        cost = count = 0
        for obs in problem.observations:
            if bits[obs.left_rank] ^ bits[obs.right_rank] != obs.xor_value:
                cost += obs.cost
                count += 1
        out[bits] = (cost, count)
    return out


def test_count_is_second_priority_and_assignment_lexicographic():
    # Search random small instances until finding one where the minimum-cost
    # assignments contain different polluted counts; the solver must then pick
    # the smallest count, and among those the lexicographically smallest bits.
    rng = random.Random(2024)
    found = None
    for _ in range(3000):
        problem = make_problem(rng, n=4, m=6, fix_prob=0.25)
        table = _enumerate_levels(problem)
        min_cost = min(c for c, _ in table.values())
        minimizers = {a: (c, k) for a, (c, k) in table.items() if c == min_cost}
        if len({k for _, k in minimizers.values()}) > 1:
            found = (problem, table)
            break
    assert found is not None, "随机搜索应能找到同代价不同条数的实例"
    problem, table = found

    sol = solve(problem)
    min_cost = min(c for c, _ in table.values())
    min_count = min(k for c, k in table.values() if c == min_cost)
    assert sol.optimal_cost == min_cost
    assert sol.optimal_polluted_count == min_count

    ordered = sorted(
        (a for a, (c, k) in table.items() if c == min_cost and k == min_count)
    )
    names = problem.variable_names
    assert sol.assignment == {names[r]: ordered[0][r] for r in range(len(names))}
    if len(ordered) > 1:
        assert sol.unique is False
        assert sol.witness == {names[r]: ordered[1][r] for r in range(len(names))}
    else:
        assert sol.unique is True


def test_all_fixed_assignment():
    names = ("a", "b")
    observations = (Observation("o1", 0, 1, 1, 5),)
    sol = solve(Problem(names, observations, {0: 1, 1: 1}))
    assert sol.unique is True
    assert sol.assignment == {"a": 1, "b": 1}
    assert sol.optimal_cost == 5
    assert sol.violated_ids == ("o1",)


def test_no_observations_fast_path():
    names = ("a", "b", "c")
    sol = solve(Problem(names, (), {}))
    assert sol.optimal_cost == 0
    assert sol.optimal_polluted_count == 0
    assert sol.unique is False
    assert sol.assignment == {"a": 0, "b": 0, "c": 0}
    assert sol.witness == {"a": 0, "b": 0, "c": 1}
    assert sol.violated_ids == ()
    assert sol.witness_violated_ids == ()

    sol_fixed = solve(Problem(names, (), {0: 1, 1: 0, 2: 1}))
    assert sol_fixed.unique is True
    assert sol_fixed.witness is None
    assert sol_fixed.assignment == {"a": 1, "b": 0, "c": 1}

    # one free variable with no observations: two all-zero-cost optima
    sol_one = solve(Problem(names, (), {0: 1, 1: 0}))
    assert sol_one.unique is False
    assert sol_one.assignment == {"a": 1, "b": 0, "c": 0}
    assert sol_one.witness == {"a": 1, "b": 0, "c": 1}
