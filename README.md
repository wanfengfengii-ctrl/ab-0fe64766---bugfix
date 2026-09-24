# 供配电馈线相位标注复核服务（纯后端）

供配电测量团队复核馈线相位标注的纯后端服务。工程师提交二值相位变量、带唯一编号与正整数代价的异或（XOR）观测、固定参考变量及可接受污染代价（预算），服务返回一次**确定性裁决**：在全部相位赋值中精确最小化不满足观测的总代价，再最小化污染条数，并判断达到这两级最优值的赋值是否唯一；不唯一时返回字典序最小赋值及另一份最优见证；最优代价超出预算则明确拒绝。

本仓库**不创建任何前端资源**，只有 JSON API、健康检查与复算工具。

## 1. 目录结构

```
app/                 服务源码（Python 3.13 / FastAPI）
  solver.py          精确求解器（全赋值枚举，numpy BLAS）
  validation.py      一次性收集所有可定位错误（JSON Pointer）
  main.py            FastAPI 路由与响应/审计结构
scripts/verify_response.py   独立复算工具（仅标准库）
tests/               测试：求解器暴力枚举对照、API、顺序无关、边界（221 个用例）
examples/request.json
Dockerfile           Python 3.13-slim，自带 HEALTHCHECK
docker-compose.yml   一键启动，HOST_PORT 可配置
requirements*.txt
```

## 2. 快速开始（Docker Compose）

```bash
cp .env.example .env          # 可改为 HOST_PORT=9000 等
docker compose up --build -d
curl http://localhost:${HOST_PORT:-8000}/health
# {"status":"ok",...}

curl -s -X POST http://localhost:${HOST_PORT:-8000}/api/v1/adjudicate \
  -H 'content-type: application/json' \
  --data @examples/request.json | python -m json.tool
```

宿主机端口通过 `HOST_PORT` 配置（默认 8000），也可以直接覆盖：

```bash
HOST_PORT=9000 docker compose up --build -d
```

容器与 Compose 均配置了 `/health` 健康检查（start-period 10s、每 10s 一次、失败 3 次标记不健康）。

## 3. 请求 / 响应契约

### 请求

| 字段 | 说明 |
| --- | --- |
| `variables` | 1..26 个二值相位变量名（非空字符串，最长 64，不含空白/控制字符），不可重名 |
| `observations` | 0..120 条异或观测 |
| `observations[].id` | 观测唯一编号（同变量名规则），不可重复 |
| `observations[].left` / `right` | 所涉变量名，必须在 `variables` 中声明 |
| `observations[].xor_value` | 0 或 1，表示要求 `left XOR right` 等于该值 |
| `observations[].cost` | 该观测不满足（被污染）时的代价，**正整数** |
| `references` | 固定参考变量条目，可空；同一变量可出现多次，但值必须一致 |
| `references[].variable` / `value` | 变量名（须已声明）/ 0 或 1 |
| `budget` | 可接受的污染总代价上限，非负整数 |

允许观测的 `left == right`（自指观测）：`x XOR x == 0` 恒成立，`x XOR x == 1` 恒矛盾。

### 响应（200）

- `decision`: **accepted** / **rejected**（`rejected` 时 `reason` 为 `optimal_polluted_cost_exceeds_budget`，仍给出最优值与赋值，供分析）
- `adjudication`:
  - `unique_optimum`、字典序最小最优赋值 `assignment`（按变量名排序的字典）
  - `optimal`: `polluted_cost`、`polluted_count`、`combined_objective`
  - `polluted_observation_ids`（按编号升序）
  - `witness`: 非唯一时另一份最优赋值及其代价/条数/污染编号/目标值；唯一时为 `null`
  - `fixed_references`: 实际生效的参考值
- `budget`: `limit`、`optimal_polluted_cost`、`within_budget`、`slack`/`excess`
- `audit.observations[]`: 每条异或式在两份赋值下逐位的 `left_value/right_value/actual_xor/satisfied`，可从响应直接复算

`combined_objective = 121 * polluted_cost + polluted_count`（121 = 120 条上限 + 1）。因 `polluted_count <= 120 < 121`，最小化该标量等价于"先最小化代价、再最小化条数"。

### 错误（400）

所有问题一次性返回，`details[].pointer` 为指向请求 JSON 的 [RFC 6901](https://datatracker.ietf.org/doc/html/rfc6901) JSON Pointer（如 `/observations/3/left`、`/references/1/value`），可直接定位到字段：

```json
{
  "error": {
    "code": "validation_failed",
    "message": "请求校验未通过，共 N 处错误，pointer 指向请求 JSON 中的位置",
    "details": [{"pointer": "/observations/1/id", "message": "观测编号 'o1' 重复，首次出现于索引 0"},
                {"pointer": "/observations/2/left", "message": "变量 'ghost' 未在 variables 中声明"}]
  }
}
```

覆盖：非法引用、重复编号、矛盾参考值，以及类型/越界/缺失字段/重名变量/数量超限等。同一变量被固定为不同值时，每条相关 reference 都会被标记。

## 4. 验收可复算

一键启动后的基础验收通过一次性 Compose 服务执行，成功时报告 `VERIFY_OK` 并以退出码 0 结束：

```bash
docker compose run --rm verify
```

独立于服务实现、仅用 Python 标准库的响应复算脚本可另外用于具体请求：

```bash
python scripts/verify_response.py <请求.json> <响应.json>
```

它会：
1. 用响应中的赋值对**每条异或式**独立计算 `left XOR right` 与是否满足，核对 `audit` 中每个字段；
2. 汇总污染代价、条数、`121*cost+count` 及污染编号顺序；
3. 有见证时对**两份见证**分别复算，并核对二者两级目标值相同、赋值不同；
4. 按预算独立判定 accepted/rejected，核对参考变量是否被遵守。

全部一致退出码 0，不一致退出码 1。

## 5. 顺序无关性与唯一性的规范

- 变量按**名称排序**后再编号（rank），因此字典序赋值与变量提交顺序无关；
- 观测按 **id 排序**处理与输出，污染编号列表按编号升序，与观测提交顺序无关；
- 赋值按排名的字典序（变量名字典序）从全零开始扫描，首个达最优者即字典序最小赋值；若全局仅一份赋值达两级最优则 `unique_optimum=true`，否则返回第二字典序的最优见证。

## 6. 精确性与性能

问题本质是加权 Max-2-XOR / 带符号 MaxCut（NP 难）。n ≤ 26 时对全部 2^n 个赋值做**精确枚举**（非近似、非随机）。异或违反指示函数是二值变量的二次多项式，故用 numpy 分块 BLAS 矩阵乘一次评估 2^19 个赋值（n=26 时共 ~6700 万）。

数值契约以请求中的整数为准，不另设代价或预算的数值上限；最优值、赋值和见证应按同一整数顺序给出确定结果。

变量与观测数量仍按前述规模限制；服务在 AnyIO 线程池中运行求解，不阻塞事件循环。

## 7. 本地开发与测试

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest                 # 221 个用例（含 200 组随机暴力对照）
python -m pytest -m slow         # 含 26×120 最大规模端到端用例
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 8. 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/` | 服务信息 |
| POST | `/api/v1/adjudicate` | 提交复核，取得确定性裁决 |

另提供自动生成的 OpenAPI 文档：`/docs`（Swagger UI）与 `/openapi.json`。
