"""Request validation for the adjudication endpoint.

Validation collects *every* problem with a submission in one pass and reports
each with a JSON Pointer (RFC 6901) into the request document, so the
engineering team can fix an illegal reference, a duplicate observation id and
a contradictory reference value in one round trip instead of one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .solver import MAX_COST_VALUE, MAX_OBSERVATIONS, MAX_VARIABLES, Observation, Problem

#: Budgets above this are rejected (optimal polluted cost can never reach it,
#: since total observation cost is bounded by MAX_OBSERVATIONS * MAX_COST_VALUE).
MAX_BUDGET = 1_000_000_000_000
MAX_NAME_LENGTH = 64

_TOP_LEVEL_KEYS = {"variables", "observations", "references", "budget"}
_OBSERVATION_KEYS = {"id", "left", "right", "xor_value", "cost"}
_REFERENCE_KEYS = {"variable", "value"}


@dataclass(frozen=True)
class ApiError:
    pointer: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"pointer": self.pointer, "message": self.message}


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _ptr(*tokens: object) -> str:
    if not tokens:
        return ""
    return "/" + "/".join(_escape(str(t)) for t in tokens)


def _is_plain_int(value: object) -> bool:
    """True for ints but not bools (bool is a subclass of int in Python)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_NAME_LENGTH
        and not any(ch.isspace() or ord(ch) < 0x20 for ch in value)
    )


def validate_payload(payload: Any) -> tuple[list[ApiError], Problem | None]:
    errors: list[ApiError] = []

    if not isinstance(payload, dict):
        return [ApiError("", "请求体必须是 JSON 对象")], None

    for key in payload:
        if key not in _TOP_LEVEL_KEYS:
            errors.append(
                ApiError(_ptr(key), "未知字段，允许的字段为 variables/observations/references/budget")
            )

    # ---- variables ---------------------------------------------------------
    raw_variables = payload.get("variables")
    variable_names: list[str] = []
    name_set: set[str] = set()
    if "variables" not in payload:
        errors.append(ApiError(_ptr("variables"), "缺少必填字段 variables"))
    elif not isinstance(raw_variables, list):
        errors.append(ApiError(_ptr("variables"), "variables 必须是字符串数组"))
    else:
        if not 1 <= len(raw_variables) <= MAX_VARIABLES:
            errors.append(
                ApiError(
                    _ptr("variables"),
                    f"相位变量数量必须在 1 到 {MAX_VARIABLES} 之间，当前为 {len(raw_variables)}",
                )
            )
        first_index: dict[str, int] = {}
        for i, name in enumerate(raw_variables):
            if not _valid_name(name):
                errors.append(
                    ApiError(
                        _ptr("variables", i),
                        f"变量名必须是长度 1..{MAX_NAME_LENGTH} 且不含空白或控制字符的字符串",
                    )
                )
                continue
            if name in first_index:
                errors.append(
                    ApiError(
                        _ptr("variables", i),
                        f"变量名 {name!r} 重复，首次出现于索引 {first_index[name]}",
                    )
                )
            else:
                first_index[name] = i
                name_set.add(name)
                variable_names.append(name)

    # ---- budget ------------------------------------------------------------
    if "budget" not in payload:
        errors.append(ApiError(_ptr("budget"), "缺少必填字段 budget"))
    else:
        budget = payload["budget"]
        if not _is_plain_int(budget):
            errors.append(ApiError(_ptr("budget"), "budget 必须是非负整数"))
        elif not 0 <= budget <= MAX_BUDGET:
            errors.append(ApiError(_ptr("budget"), f"budget 必须在 0 到 {MAX_BUDGET} 之间"))

    # ---- observations ------------------------------------------------------
    raw_observations = payload.get("observations")
    parsed_observations: list[tuple[int, str, str, str, int, int]] = []
    # tuple: (raw_index, id, left, right, xor_value, cost)
    id_occurrences: dict[str, list[int]] = {}
    if "observations" not in payload:
        errors.append(ApiError(_ptr("observations"), "缺少必填字段 observations"))
    elif not isinstance(raw_observations, list):
        errors.append(ApiError(_ptr("observations"), "observations 必须是数组"))
    else:
        if len(raw_observations) > MAX_OBSERVATIONS:
            errors.append(
                ApiError(
                    _ptr("observations"),
                    f"异或观测数量不得超过 {MAX_OBSERVATIONS}，当前为 {len(raw_observations)}",
                )
            )
        for i, item in enumerate(raw_observations):
            base = _ptr("observations", i)
            if not isinstance(item, dict):
                errors.append(ApiError(base, "每条观测必须是对象"))
                continue
            for key in item:
                if key not in _OBSERVATION_KEYS:
                    errors.append(ApiError(_ptr("observations", i, key), "观测中的未知字段"))

            oid = item.get("id")
            oid_ok = "id" in item and _valid_name(oid)
            if "id" not in item:
                errors.append(ApiError(base + "/id", "缺少必填字段 id"))
            elif not _valid_name(oid):
                errors.append(
                    ApiError(base + "/id", f"观测编号必须是长度 1..{MAX_NAME_LENGTH} 且不含空白或控制字符的字符串")
                )
            else:
                id_occurrences.setdefault(oid, []).append(i)

            left = item.get("left")
            left_ok = "left" in item and _valid_name(left)
            if "left" not in item:
                errors.append(ApiError(base + "/left", "缺少必填字段 left"))
            elif not _valid_name(left):
                errors.append(ApiError(base + "/left", "left 必须是合法变量名字符串"))

            right = item.get("right")
            right_ok = "right" in item and _valid_name(right)
            if "right" not in item:
                errors.append(ApiError(base + "/right", "缺少必填字段 right"))
            elif not _valid_name(right):
                errors.append(ApiError(base + "/right", "right 必须是合法变量名字符串"))

            xor_value = item.get("xor_value")
            xor_ok = _is_plain_int(xor_value) and xor_value in (0, 1)
            if "xor_value" not in item:
                errors.append(ApiError(base + "/xor_value", "缺少必填字段 xor_value"))
            elif not xor_ok:
                errors.append(ApiError(base + "/xor_value", "xor_value 必须是 0 或 1"))

            cost = item.get("cost")
            cost_ok = _is_plain_int(cost) and 1 <= cost <= MAX_COST_VALUE
            if "cost" not in item:
                errors.append(ApiError(base + "/cost", "缺少必填字段 cost"))
            elif not _is_plain_int(cost) or cost < 1:
                errors.append(ApiError(base + "/cost", "cost 必须是正整数"))
            elif cost > MAX_COST_VALUE:
                errors.append(ApiError(base + "/cost", f"cost 不得超过 {MAX_COST_VALUE}"))

            if oid_ok and left_ok and right_ok and xor_ok and cost_ok:
                parsed_observations.append((i, oid, left, right, xor_value, cost))

        for oid, indexes in id_occurrences.items():
            if len(indexes) > 1:
                first = indexes[0]
                for duplicate_index in indexes[1:]:
                    errors.append(
                        ApiError(
                            _ptr("observations", duplicate_index, "id"),
                            f"观测编号 {oid!r} 重复，首次出现于索引 {first}",
                        )
                    )

    # ---- references --------------------------------------------------------
    raw_references = payload.get("references")
    parsed_references: list[tuple[int, str, int]] = []  # (raw_index, variable, value)
    if "references" not in payload:
        errors.append(ApiError(_ptr("references"), "缺少必填字段 references"))
    elif not isinstance(raw_references, list):
        errors.append(ApiError(_ptr("references"), "references 必须是数组"))
    else:
        for i, item in enumerate(raw_references):
            base = _ptr("references", i)
            if not isinstance(item, dict):
                errors.append(ApiError(base, "每条参考值必须是对象"))
                continue
            for key in item:
                if key not in _REFERENCE_KEYS:
                    errors.append(ApiError(_ptr("references", i, key), "参考值中的未知字段"))

            variable = item.get("variable")
            var_ok = "variable" in item and _valid_name(variable)
            if "variable" not in item:
                errors.append(ApiError(base + "/variable", "缺少必填字段 variable"))
            elif not _valid_name(variable):
                errors.append(ApiError(base + "/variable", "variable 必须是合法变量名字符串"))

            value = item.get("value")
            value_ok = _is_plain_int(value) and value in (0, 1)
            if "value" not in item:
                errors.append(ApiError(base + "/value", "缺少必填字段 value"))
            elif not value_ok:
                errors.append(ApiError(base + "/value", "参考值必须是 0 或 1"))

            if var_ok and value_ok:
                parsed_references.append((i, variable, value))

    # ---- cross-field checks (only over structurally valid entries) ---------
    for raw_index, _oid, left, right, _xor_value, _cost in parsed_observations:
        if left not in name_set:
            errors.append(
                ApiError(
                    _ptr("observations", raw_index, "left"),
                    f"变量 {left!r} 未在 variables 中声明",
                )
            )
        if right not in name_set:
            errors.append(
                ApiError(
                    _ptr("observations", raw_index, "right"),
                    f"变量 {right!r} 未在 variables 中声明",
                )
            )

    # references: undeclared variables are reported above; gather per-variable
    # reference indexes to detect contradictory fixed values.
    ref_indexes_by_variable: dict[str, list[int]] = {}
    ref_value_by_index: dict[int, int] = {}
    for raw_index, variable, value in parsed_references:
        if variable not in name_set:
            errors.append(
                ApiError(
                    _ptr("references", raw_index, "variable"),
                    f"变量 {variable!r} 未在 variables 中声明",
                )
            )
            continue
        ref_indexes_by_variable.setdefault(variable, []).append(raw_index)
        ref_value_by_index[raw_index] = value

    for variable, indexes in ref_indexes_by_variable.items():
        distinct = {ref_value_by_index[i] for i in indexes}
        if len(distinct) > 1:
            for raw_index in indexes:
                errors.append(
                    ApiError(
                        _ptr("references", raw_index, "value"),
                        f"变量 {variable!r} 的参考值矛盾：索引 "
                        + ", ".join(str(i) for i in indexes)
                        + " 处被固定为不同的 0/1 值",
                    )
                )

    if errors:
        errors.sort(key=lambda e: (e.pointer, e.message))
        return errors, None

    # Ranks follow sorted variable names so that the lexicographic ordering of
    # assignments (and hence every result) is independent of submission order.
    variable_names.sort()
    name_to_rank = {name: rank for rank, name in enumerate(variable_names)}
    observations = tuple(
        Observation(
            id=oid,
            left_rank=name_to_rank[left],
            right_rank=name_to_rank[right],
            xor_value=xor_value,
            cost=cost,
        )
        for _raw_index, oid, left, right, xor_value, cost in parsed_observations
    )
    fixed_ranks = {
        name_to_rank[variable]: ref_value_by_index[indexes[0]]
        for variable, indexes in ref_indexes_by_variable.items()
    }
    problem = Problem(
        variable_names=tuple(variable_names),
        observations=observations,
        fixed=fixed_ranks,
    )
    return [], problem
