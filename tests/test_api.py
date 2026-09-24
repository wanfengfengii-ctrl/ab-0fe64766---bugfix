"""End-to-end API tests including response recomputation and order invariance."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_payload(n=4, m=8, seed=0, budget=10_000, fix=None, xor_mode="random"):
    rng = random.Random(seed)
    variables = [f"v{i}" for i in range(n)]
    observations = []
    for k in range(m):
        a = rng.randrange(n)
        b = rng.randrange(n)
        if xor_mode == "all_zero":
            xor_value = 0
        else:
            xor_value = rng.randrange(2)
        observations.append(
            {
                "id": f"obs-{k:03d}",
                "left": variables[a],
                "right": variables[b],
                "xor_value": xor_value,
                "cost": rng.randrange(1, 20),
            }
        )
    references = [{"variable": v, "value": b} for v, b in (fix or {}).items()]
    return {
        "variables": variables,
        "observations": observations,
        "references": references,
        "budget": budget,
    }


def recompute_from_response(payload, response_json, which="canonical"):
    """Independently recompute cost/count/equations purely from the response.

    Returns (cost, count, combined, per_equation_ok).
    """
    scale = response_json["audit"]["objective_scale"]
    adjudication = response_json["adjudication"]
    if which == "canonical":
        assignment = adjudication["assignment"]
        polluted_ids = adjudication["polluted_observation_ids"]
    else:
        assignment = adjudication["witness"]["assignment"]
        polluted_ids = adjudication["witness"]["polluted_observation_ids"]

    cost = 0
    count = 0
    equations_ok = True
    polluted_from_audit = []
    for entry in response_json["audit"]["observations"]:
        view = entry[which]
        left_value = assignment[entry["left"]]
        right_value = assignment[entry["right"]]
        actual = left_value ^ right_value
        if (
            view["left_value"] != left_value
            or view["right_value"] != right_value
            or view["actual_xor"] != actual
        ):
            equations_ok = False
        satisfied = actual == entry["xor_value"]
        if view["satisfied"] != satisfied:
            equations_ok = False
        if not satisfied:
            cost += entry["cost"]
            count += 1
            polluted_from_audit.append(entry["id"])

    ids_match = polluted_from_audit == polluted_ids
    combined = scale * cost + count
    return cost, count, combined, equations_ok and ids_match


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_accepted_and_recomputable():
    payload = make_payload()
    response = client.post("/api/v1/adjudicate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["decision"] == "accepted"
    assert data["budget"]["within_budget"] is True

    cost, count, combined, ok = recompute_from_response(payload, data)
    assert ok
    assert cost == data["adjudication"]["optimal"]["polluted_cost"]
    assert count == data["adjudication"]["optimal"]["polluted_count"]
    assert combined == data["adjudication"]["optimal"]["combined_objective"]
    assert data["budget"]["slack"] == payload["budget"] - cost

    if not data["adjudication"]["unique_optimum"]:
        wcost, wcount, wcombined, wok = recompute_from_response(
            payload, data, which="witness"
        )
        assert wok
        witness = data["adjudication"]["witness"]
        assert wcost == witness["polluted_cost"] == cost
        assert wcount == witness["polluted_count"] == count
        assert wcombined == witness["combined_objective"] == combined
        assert witness["assignment"] != data["adjudication"]["assignment"]
    else:
        assert data["adjudication"]["witness"] is None


def test_rejected_when_over_budget():
    payload = make_payload(n=1, m=1, seed=3, budget=0)
    # force an unavoidable violation: v XOR v == 1
    payload["variables"] = ["only"]
    payload["observations"] = [
        {"id": "x", "left": "only", "right": "only", "xor_value": 1, "cost": 5}
    ]
    response = client.post("/api/v1/adjudicate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "rejected"
    assert data["reason"] == "optimal_polluted_cost_exceeds_budget"
    assert data["adjudication"]["optimal"]["polluted_cost"] == 5
    assert data["budget"]["within_budget"] is False
    assert data["budget"]["excess"] == 5
    assert data["budget"]["slack"] is None


def test_budget_zero_satisfiable_accepted():
    payload = {
        "variables": ["a", "b"],
        "observations": [
            {"id": "o1", "left": "a", "right": "b", "xor_value": 1, "cost": 10}
        ],
        "references": [{"variable": "a", "value": 1}],
        "budget": 0,
    }
    data = client.post("/api/v1/adjudicate", json=payload).json()
    assert data["decision"] == "accepted"
    assert data["adjudication"]["assignment"] == {"a": 1, "b": 0}
    assert data["adjudication"]["optimal"]["polluted_cost"] == 0


def test_reference_honored_and_unique():
    payload = {
        "variables": ["p", "q"],
        "observations": [
            {"id": "o1", "left": "p", "right": "q", "xor_value": 1, "cost": 3}
        ],
        "references": [
            {"variable": "p", "value": 1},
            {"variable": "q", "value": 1},
        ],
        "budget": 100,
    }
    data = client.post("/api/v1/adjudicate", json=payload).json()
    assert data["adjudication"]["unique_optimum"] is True
    assert data["adjudication"]["assignment"] == {"p": 1, "q": 1}
    assert data["adjudication"]["polluted_observation_ids"] == ["o1"]


def test_non_unique_gives_two_witnesses_symmetric_chain():
    # parity-0 chain with no reference has global-flip symmetry: never unique
    payload = {
        "variables": ["v0", "v1", "v2", "v3"],
        "observations": [
            {"id": f"e{i}", "left": f"v{i}", "right": f"v{i+1}", "xor_value": 0, "cost": 1}
            for i in range(3)
        ],
        "references": [],
        "budget": 0,
    }
    data = client.post("/api/v1/adjudicate", json=payload).json()
    assert data["decision"] == "accepted"
    assert data["adjudication"]["unique_optimum"] is False
    assert data["adjudication"]["assignment"] == {"v0": 0, "v1": 0, "v2": 0, "v3": 0}
    witness = data["adjudication"]["witness"]
    assert witness["assignment"] == {"v0": 1, "v1": 1, "v2": 1, "v3": 1}
    assert witness["polluted_cost"] == 0
    assert witness["polluted_count"] == 0


def test_duplicate_id_and_illegal_reference_and_contradiction_returned_together():
    payload = {
        "variables": ["a"],
        "observations": [
            {"id": "dup", "left": "a", "right": "a", "xor_value": 0, "cost": 1},
            {"id": "dup", "left": "ghost", "right": "a", "xor_value": 0, "cost": 1},
            {"id": "badref", "left": "a", "right": "specter", "xor_value": 1, "cost": 2},
        ],
        "references": [
            {"variable": "a", "value": 0},
            {"variable": "a", "value": 1},
            {"variable": "phantom", "value": 0},
        ],
        "budget": 10,
    }
    response = client.post("/api/v1/adjudicate", json=payload)
    assert response.status_code == 400
    details = response.json()["error"]["details"]
    pointers = {d["pointer"] for d in details}
    assert "/observations/1/id" in pointers  # duplicate id
    assert "/observations/1/left" in pointers  # undeclared ghost
    assert "/observations/2/right" in pointers  # undeclared specter
    assert "/references/0/value" in pointers  # contradiction
    assert "/references/1/value" in pointers
    assert "/references/2/variable" in pointers  # undeclared phantom


def test_field_level_type_errors():
    payload = {
        "variables": ["a", "a", 123],
        "observations": [
            {"id": "o1", "left": "a", "right": "a", "xor_value": 2, "cost": 0},
            {"id": 9, "left": None, "right": "a", "xor_value": "0", "cost": -3, "not_a_field": 1},
        ],
        "references": [{"variable": "a", "value": True}],
        "budget": -5,
    }
    response = client.post("/api/v1/adjudicate", json=payload)
    assert response.status_code == 400
    pointers = {d["pointer"] for d in response.json()["error"]["details"]}
    assert "/variables/1" in pointers  # duplicate
    assert "/variables/2" in pointers  # bad name
    assert "/observations/0/xor_value" in pointers
    assert "/observations/0/cost" in pointers  # zero cost
    assert "/observations/1/id" in pointers
    assert "/observations/1/left" in pointers
    assert "/observations/1/xor_value" in pointers
    assert "/observations/1/cost" in pointers
    assert "/observations/1/not_a_field" in pointers
    assert "/references/0/value" in pointers  # bool rejected
    assert "/budget" in pointers


def test_size_limits():
    too_many_vars = {
        "variables": [f"v{i}" for i in range(27)],
        "observations": [],
        "references": [],
        "budget": 0,
    }
    r = client.post("/api/v1/adjudicate", json=too_many_vars)
    assert r.status_code == 400
    assert any(d["pointer"] == "/variables" for d in r.json()["error"]["details"])

    too_many_obs = {
        "variables": ["a"],
        "observations": [
            {"id": f"o{i}", "left": "a", "right": "a", "xor_value": 0, "cost": 1}
            for i in range(121)
        ],
        "references": [],
        "budget": 0,
    }
    r = client.post("/api/v1/adjudicate", json=too_many_obs)
    assert r.status_code == 400
    assert any(d["pointer"] == "/observations" for d in r.json()["error"]["details"])


def test_missing_fields_and_malformed_json():
    r = client.post("/api/v1/adjudicate", json={"variables": []})
    assert r.status_code == 400
    pointers = {d["pointer"] for d in r.json()["error"]["details"]}
    assert "/variables" in pointers
    assert "/observations" in pointers
    assert "/references" in pointers
    assert "/budget" in pointers

    r = client.post("/api/v1/adjudicate", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "malformed_json"

    r = client.post("/api/v1/adjudicate", content=b"", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_body"


def test_observation_order_does_not_change_result():
    payload = make_payload(n=8, m=40, seed=1234)
    r1 = client.post("/api/v1/adjudicate", json=payload).json()

    shuffled = dict(payload)
    obs = list(payload["observations"])
    random.Random(5).shuffle(obs)
    shuffled["observations"] = obs
    r2 = client.post("/api/v1/adjudicate", json=shuffled).json()

    assert r2["adjudication"]["assignment"] == r1["adjudication"]["assignment"]
    assert (
        r2["adjudication"]["polluted_observation_ids"]
        == r1["adjudication"]["polluted_observation_ids"]
    )
    assert r2["adjudication"]["unique_optimum"] == r1["adjudication"]["unique_optimum"]
    assert r2["decision"] == r1["decision"]
    if r1["adjudication"]["witness"]:
        assert (
            r2["adjudication"]["witness"]["assignment"]
            == r1["adjudication"]["witness"]["assignment"]
        )

    # audit entries themselves are emitted in id order regardless of input order
    ids = [e["id"] for e in r2["audit"]["observations"]]
    assert ids == sorted(ids)


def test_variable_submission_order_does_not_change_result():
    payload = make_payload(n=6, m=20, seed=77)
    r1 = client.post("/api/v1/adjudicate", json=payload).json()
    reordered = dict(payload)
    variables = list(payload["variables"])
    random.Random(8).shuffle(variables)
    reordered["variables"] = variables
    r2 = client.post("/api/v1/adjudicate", json=reordered).json()
    assert r2["adjudication"]["assignment"] == r1["adjudication"]["assignment"]
    assert r2["decision"] == r1["decision"]


def test_consistent_duplicate_references_allowed():
    # Same variable fixed to the same value twice is redundant, not contradictory.
    payload = {
        "variables": ["a"],
        "observations": [
            {"id": "o1", "left": "a", "right": "a", "xor_value": 0, "cost": 1}
        ],
        "references": [
            {"variable": "a", "value": 1},
            {"variable": "a", "value": 1},
        ],
        "budget": 0,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["adjudication"]["assignment"] == {"a": 1}


def test_cost_has_no_numeric_upper_limit():
    # The contract bounds cost only to positive integers; 1_000_000_001 must
    # be accepted and adjudicated exactly (optimum equals the budget).
    payload = {
        "variables": ["a"],
        "observations": [
            {"id": "o1", "left": "a", "right": "a", "xor_value": 1, "cost": 1_000_000_001}
        ],
        "references": [{"variable": "a", "value": 0}],
        "budget": 1_000_000_001,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] == "accepted"
    assert data["adjudication"]["unique_optimum"] is True
    assert data["adjudication"]["assignment"] == {"a": 0}
    assert data["adjudication"]["optimal"]["polluted_cost"] == 1_000_000_001
    assert data["adjudication"]["optimal"]["polluted_count"] == 1
    assert data["adjudication"]["polluted_observation_ids"] == ["o1"]
    assert data["budget"]["within_budget"] is True
    assert data["budget"]["slack"] == 0


def test_budget_has_no_numeric_upper_limit():
    # The contract bounds budget only to non-negative integers.
    payload = {
        "variables": ["a"],
        "observations": [],
        "references": [{"variable": "a", "value": 0}],
        "budget": 1_000_000_000_001,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] == "accepted"
    assert data["adjudication"]["unique_optimum"] is True
    assert data["adjudication"]["optimal"]["polluted_cost"] == 0
    assert data["budget"]["limit"] == 1_000_000_000_001
    assert data["budget"]["slack"] == 1_000_000_000_001


def test_large_costs_near_equal_exact_adjudication():
    # Optimum decided by a margin of 2 at ~2e16 (below float64 resolution of
    # the combined energy); the response must remain exactly recomputable.
    payload = {
        "variables": ["a", "b"],
        "observations": [
            {"id": "const", "left": "a", "right": "a", "xor_value": 1, "cost": 10**16 + 1},
            {"id": "neq", "left": "a", "right": "b", "xor_value": 0, "cost": 10**16},
            {"id": "eq", "left": "a", "right": "b", "xor_value": 1, "cost": 10**16 + 2},
        ],
        "references": [],
        "budget": 3 * 10**16,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] == "accepted"
    adjudication = data["adjudication"]
    assert adjudication["optimal"]["polluted_cost"] == 2 * 10**16 + 1
    assert adjudication["optimal"]["polluted_count"] == 2
    assert adjudication["unique_optimum"] is False
    assert adjudication["assignment"] == {"a": 0, "b": 1}
    assert adjudication["witness"]["assignment"] == {"a": 1, "b": 0}

    cost, count, combined, ok = recompute_from_response(payload, data)
    assert ok
    assert cost == 2 * 10**16 + 1
    assert count == 2
    assert combined == adjudication["optimal"]["combined_objective"]
    wcost, wcount, wcombined, wok = recompute_from_response(payload, data, which="witness")
    assert wok
    assert (wcost, wcount, wcombined) == (cost, count, combined)


def test_large_costs_cancellation_and_budget_rejection():
    # Equal-cost complementary observations: every assignment pays exactly
    # 10**25 + 7; a budget one below that must reject with the exact excess.
    c = 10**25 + 7
    payload = {
        "variables": ["a", "b"],
        "observations": [
            {"id": "eq", "left": "a", "right": "b", "xor_value": 1, "cost": c},
            {"id": "neq", "left": "a", "right": "b", "xor_value": 0, "cost": c},
        ],
        "references": [],
        "budget": c - 1,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] == "rejected"
    assert data["reason"] == "optimal_polluted_cost_exceeds_budget"
    assert data["adjudication"]["optimal"]["polluted_cost"] == c
    assert data["adjudication"]["optimal"]["polluted_count"] == 1
    assert data["adjudication"]["unique_optimum"] is False
    assert data["adjudication"]["assignment"] == {"a": 0, "b": 0}
    assert data["adjudication"]["witness"]["assignment"] == {"a": 0, "b": 1}
    assert data["budget"]["excess"] == 1
    assert data["budget"]["slack"] is None


@pytest.mark.slow
def test_maximum_size_deterministic_and_recomputable():
    rng = random.Random(42)
    variables = [f"v{i:02d}" for i in range(26)]
    observations = []
    used_pairs = set()
    k = 0
    while k < 120:
        a, b = rng.randrange(26), rng.randrange(26)
        if a == b:
            continue
        if (a, b) in used_pairs or (b, a) in used_pairs:
            continue
        used_pairs.add((a, b))
        observations.append(
            {
                "id": f"obs-{k:03d}",
                "left": variables[a],
                "right": variables[b],
                "xor_value": rng.randrange(2),
                "cost": rng.randrange(1, 1_000_000_000),
            }
        )
        k += 1
    payload = {
        "variables": variables,
        "observations": observations,
        "references": [{"variable": "v00", "value": 1}],
        "budget": 10**12,
    }
    r = client.post("/api/v1/adjudicate", json=payload, timeout=120)
    assert r.status_code == 200, r.text
    data = r.json()
    cost, count, combined, ok = recompute_from_response(payload, data)
    assert ok
    assert cost == data["adjudication"]["optimal"]["polluted_cost"]
    assert count == data["adjudication"]["optimal"]["polluted_count"]
    assert combined == data["adjudication"]["optimal"]["combined_objective"]
