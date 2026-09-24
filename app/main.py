"""FastAPI application: feeder phase annotation review service.

Pure backend service (no frontend assets). The single business endpoint
accepts a batch of binary phase variables, weighted XOR observations, fixed
reference variables and a pollution budget, and returns one deterministic
adjudication: the exact lexicographically (cost, then count) optimal phase
assignment, whether it is unique, a second optimal witness when it is not,
and a self-contained audit section allowing every XOR equation and both
witnesses' objective values to be recomputed straight from the response.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .solver import SCALE, Observation, Problem, Solution, solve
from .validation import validate_payload

SERVICE_NAME = "feeder-phase-adjudicator"
SERVICE_VERSION = "1.0.0"

#: Largest accepted request body in bytes (a maximum-size request is well
#: below 100 KB; the bound guards the exact-enumeration endpoint).
MAX_BODY_BYTES = 1_048_576

app = FastAPI(
    title="供配电馈线相位标注复核服务",
    version=SERVICE_VERSION,
    description=(
        "对至多 26 个二值相位变量、至多 120 条带正整数代价的异或观测进行精确复核："
        "在全部相位赋值中先最小化不满足观测的总代价、再最小化污染条数，"
        "并裁决最优赋值唯一性。"
    ),
)


@app.get("/health", tags=["运维"])
async def health() -> dict[str, str]:
    """容器与 HTTP 探针使用的健康检查端点。"""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/", tags=["运维"])
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "version": SERVICE_VERSION, "docs": "/docs"}


@app.post(
    "/api/v1/adjudicate",
    tags=["相位复核"],
    summary="提交一次相位复核，取得确定性裁决",
)
async def adjudicate(request: Request) -> JSONResponse:
    raw = await _read_json_body(request)
    if isinstance(raw, JSONResponse):  # malformed body / wrong content type
        return raw

    errors, problem = await run_in_threadpool(validate_payload, raw)
    if errors or problem is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "validation_failed",
                    "message": f"请求校验未通过，共 {len(errors)} 处错误，pointer 指向请求 JSON 中的位置",
                    "details": [e.to_dict() for e in errors],
                }
            },
        )

    solution = await run_in_threadpool(solve, problem)
    budget = int(raw["budget"])
    return JSONResponse(
        status_code=200,
        content=_build_response(problem, solution, budget),
    )


async def _read_json_body(request: Request) -> Any:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "payload_too_large",
                            "message": f"请求体超过 {MAX_BODY_BYTES} 字节上限",
                            "details": [
                                {"pointer": "", "message": f"content-length={content_length}"}
                            ],
                        }
                    },
                )
        except ValueError:
            pass
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "payload_too_large",
                    "message": f"请求体超过 {MAX_BODY_BYTES} 字节上限",
                    "details": [],
                }
            },
        )
    if not body:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "empty_body",
                    "message": "请求体为空，需提交 JSON 对象",
                    "details": [{"pointer": "", "message": "请求体为空"}],
                }
            },
        )
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        message = "请求体不是合法 JSON"
        if isinstance(exc, json.JSONDecodeError):
            message = f"请求体不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列附近解析失败"
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "malformed_json",
                    "message": message,
                    "details": [{"pointer": "", "message": str(exc)}],
                }
            },
        )


def _assignment_view(problem: Problem, assignment: dict[str, int]) -> dict[str, int]:
    """Emit the assignment ordered by variable name (submission order independent)."""
    return {name: assignment[name] for name in problem.variable_names}


def _audit_observation(
    obs: Observation,
    names: tuple[str, ...],
    canonical: dict[str, int],
    witness: dict[str, int] | None,
) -> dict[str, Any]:
    def view(assignment: dict[str, int]) -> dict[str, Any]:
        left_value = assignment[names[obs.left_rank]]
        right_value = assignment[names[obs.right_rank]]
        actual_xor = left_value ^ right_value
        return {
            "left_value": left_value,
            "right_value": right_value,
            "actual_xor": actual_xor,
            "satisfied": actual_xor == obs.xor_value,
        }

    entry: dict[str, Any] = {
        "id": obs.id,
        "left": names[obs.left_rank],
        "right": names[obs.right_rank],
        "xor_value": obs.xor_value,
        "cost": obs.cost,
        "canonical": view(canonical),
    }
    if witness is not None:
        entry["witness"] = view(witness)
    return entry


def _build_response(problem: Problem, solution: Solution, budget: int) -> dict[str, Any]:
    within_budget = solution.optimal_cost <= budget
    ordered_observations = sorted(problem.observations, key=lambda o: o.id)

    audit = [
        _audit_observation(
            obs,
            problem.variable_names,
            solution.assignment,
            solution.witness,
        )
        for obs in ordered_observations
    ]

    witness_block = None
    if solution.witness is not None:
        witness_block = {
            "assignment": _assignment_view(problem, solution.witness),
            "polluted_cost": solution.witness_polluted_cost,
            "polluted_count": solution.witness_polluted_count,
            "polluted_observation_ids": list(solution.witness_violated_ids or []),
            "combined_objective": SCALE * (solution.witness_polluted_cost or 0)
            + (solution.witness_polluted_count or 0),
        }

    references = [
        {"variable": problem.variable_names[rank], "value": value}
        for rank, value in sorted(problem.fixed.items(), key=lambda item: problem.variable_names[item[0]])
    ]

    return {
        "decision": "accepted" if within_budget else "rejected",
        "reason": None if within_budget else "optimal_polluted_cost_exceeds_budget",
        "adjudication": {
            "unique_optimum": solution.unique,
            "assignment": _assignment_view(problem, solution.assignment),
            "optimal": {
                "polluted_cost": solution.optimal_cost,
                "polluted_count": solution.optimal_polluted_count,
                # combined scalar objective: SCALE*cost + count
                "combined_objective": SCALE * solution.optimal_cost
                + solution.optimal_polluted_count,
            },
            "polluted_observation_ids": list(solution.violated_ids),
            "witness": witness_block,
            "fixed_references": references,
        },
        "budget": {
            "limit": budget,
            "optimal_polluted_cost": solution.optimal_cost,
            "within_budget": within_budget,
            "slack": budget - solution.optimal_cost if within_budget else None,
            "excess": solution.optimal_cost - budget if not within_budget else None,
        },
        "audit": {
            "objective_scale": SCALE,
            "objective_rule": f"combined_objective = {SCALE} * polluted_cost + polluted_count; "
            "在全部赋值中先最小化 polluted_cost，再最小化 polluted_count",
            "observations": audit,
        },
    }


def main() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
