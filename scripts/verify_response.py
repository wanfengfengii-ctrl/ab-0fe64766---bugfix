#!/usr/bin/env python3
"""验收复算脚本（仅依赖 Python 标准库）。

用法:
    python scripts/verify_response.py request.json response.json

request.json  为提交给 /api/v1/adjudicate 的请求体；
response.json 为该接口返回的响应体（accepted 或 rejected 均可复算）。

脚本完全独立于服务实现，重新计算:
  1. 每条异或观测在「字典序最小最优赋值」下的 left/right/XOR/是否满足；
  2. 污染总代价、污染条数与合成目标 SCALE*cost+count；
  3. 若存在第二份最优见证，对见证赋值重复以上计算，并核对两份目标值一致；
  4. 预算裁决 accepted/rejected 是否与最优代价相符。

全部一致时退出码 0，发现任何不符退出码 1。
"""

from __future__ import annotations

import json
import sys
from typing import Any


def fail(message: str) -> None:
    print(f"复算失败: {message}", file=sys.stderr)
    sys.exit(1)


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        fail(f"{path} 不是 JSON 对象")
    return data


def recompute(
    assignment: dict[str, str],
    observations: list[dict[str, Any]],
    scale: int,
    label: str,
) -> tuple[int, int, int, list[str]]:
    cost = 0
    count = 0
    polluted: list[str] = []
    for obs in observations:
        oid = obs["id"]
        left = obs["left"]
        right = obs["right"]
        expected_xor = obs["xor_value"]
        ocost = obs["cost"]
        if left not in assignment or right not in assignment:
            fail(f"{label}: 赋值缺少变量 {left!r} 或 {right!r}（观测 {oid}）")
        lv = assignment[left]
        rv = assignment[right]
        if lv not in (0, 1) or rv not in (0, 1):
            fail(f"{label}: 观测 {oid} 的变量取值不是 0/1")
        actual = lv ^ rv
        if actual != expected_xor:
            cost += ocost
            count += 1
            polluted.append(oid)
    combined = scale * cost + count
    return cost, count, combined, polluted


def check_audit_entries(
    response: dict[str, Any],
    assignment: dict[str, int],
    which: str,
    label: str,
) -> None:
    for entry in response["audit"]["observations"]:
        view = entry.get(which)
        if view is None:
            fail(f"audit.observations 缺少 {label} 视图: {entry['id']}")
        lv = assignment[entry["left"]]
        rv = assignment[entry["right"]]
        actual = lv ^ rv
        satisfied = actual == entry["xor_value"]
        if view["left_value"] != lv:
            fail(f"观测 {entry['id']} 的 {label} left_value 复算不一致")
        if view["right_value"] != rv:
            fail(f"观测 {entry['id']} 的 {label} right_value 复算不一致")
        if view["actual_xor"] != actual:
            fail(f"观测 {entry['id']} 的 {label} actual_xor 复算不一致")
        if view["satisfied"] != satisfied:
            fail(f"观测 {entry['id']} 的 {label} satisfied 复算不一致")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    request = load(sys.argv[1])
    response = load(sys.argv[2])

    scale = response.get("audit", {}).get("objective_scale")
    if not isinstance(scale, int) or scale <= 0:
        fail("响应缺少合法 audit.objective_scale")
    observations = request.get("observations")
    if not isinstance(observations, list):
        fail("请求缺少 observations 数组")

    adjudication = response.get("adjudication", {})
    canonical = adjudication.get("assignment")
    if not isinstance(canonical, dict):
        fail("响应缺少 adjudication.assignment")

    # ---- canonical assignment ---------------------------------------------
    check_audit_entries(response, canonical, "canonical", "canonical")
    cost, count, combined, polluted = recompute(
        canonical, observations, scale, "canonical"
    )
    optimal = adjudication.get("optimal", {})
    if cost != optimal.get("polluted_cost"):
        fail(f"污染代价复算 {cost} != 响应 {optimal.get('polluted_cost')}")
    if count != optimal.get("polluted_count"):
        fail(f"污染条数复算 {count} != 响应 {optimal.get('polluted_count')}")
    if combined != optimal.get("combined_objective"):
        fail(f"合成目标复算 {combined} != 响应 {optimal.get('combined_objective')}")
    if polluted != adjudication.get("polluted_observation_ids"):
        fail(
            f"污染编号复算 {polluted} != 响应 {adjudication.get('polluted_observation_ids')}"
        )
    if polluted != sorted(polluted):
        fail("污染编号未按编号排序")

    # ---- references honored ------------------------------------------------
    fixed = {ref["variable"]: ref["value"] for ref in request.get("references", [])}
    for name, value in fixed.items():
        if canonical.get(name) != value:
            fail(f"参考变量 {name} 未被固定为 {value}")

    # ---- witness ------------------------------------------------------------
    witness = adjudication.get("witness")
    if adjudication.get("unique_optimum"):
        if witness is not None:
            fail("裁决为唯一最优，但响应中出现了 witness")
    else:
        if witness is None:
            fail("裁决为非唯一最优，但响应中缺少 witness")
        w_assignment = witness["assignment"]
        if w_assignment == canonical:
            fail("见证赋值与字典序最小赋值相同，不构成第二份见证")
        check_audit_entries(response, w_assignment, "witness", "witness")
        wcost, wcount, wcombined, wpolluted = recompute(
            w_assignment, observations, scale, "witness"
        )
        if wcost != witness.get("polluted_cost"):
            fail(f"见证污染代价复算 {wcost} != 响应 {witness.get('polluted_cost')}")
        if wcount != witness.get("polluted_count"):
            fail(f"见证污染条数复算 {wcount} != 响应 {witness.get('polluted_count')}")
        if wcombined != witness.get("combined_objective"):
            fail(
                f"见证合成目标复算 {wcombined} != 响应 {witness.get('combined_objective')}"
            )
        if wpolluted != witness.get("polluted_observation_ids"):
            fail("见证污染编号列表复算不一致")
        # both witnesses must attain the same two-level optimum
        if (wcost, wcount) != (cost, count):
            fail(
                f"见证两级目标 ({wcost},{wcount}) 与字典序最小赋值 ({cost},{count}) 不一致"
            )

    # ---- budget decision ----------------------------------------------------
    budget = request.get("budget")
    within = cost <= budget
    if response.get("decision") != ("accepted" if within else "rejected"):
        fail("decision 与按最优代价和预算的复算不符")
    budget_block = response.get("budget", {})
    if budget_block.get("within_budget") != within:
        fail("budget.within_budget 复算不一致")

    print(
        "复算通过: "
        f"decision={response['decision']}, cost={cost}, count={count}, "
        f"combined={combined}, unique={adjudication.get('unique_optimum')}"
    )


if __name__ == "__main__":
    main()
