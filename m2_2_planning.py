"""
[M2.2] Planning 机制
显式计划生成 + 交互式审查 + 动态重规划

架构：
  Phase 1: generate_plan()  → 调一次 LLM，返回结构化 JSON 计划
           validate_plan()  → 校验计划合法性（字段/工具名/依赖/占位符）
           review_plan()    → 打印计划，等待人工确认或修改
             └─ 有修改意见 → revise → 回到 review_plan()
  Phase 2: execute_plan()   → 直接调 execute_tool，代码驱动执行
           summarize()      → 执行完毕后调一次 LLM 生成自然语言总结
                              （这是整个执行阶段唯一的 LLM 调用）

设计原则："能用代码控制的就不要交给模型"
  - 模型负责：开头规划 + 结尾总结
  - 代码负责：中间所有执行步骤（直接调 execute_tool，不过模型）

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m2_2_planning.py
"""

import json
import os
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与 M2.1 完全相同）
# ══════════════════════════════════════════════════════════════════════════════

API_URL  = "https://api.anthropic.com/v1/messages"
MODEL    = "claude-haiku-4-5-20251001"
HEADERS  = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}
MAX_TURNS = 10


def add_message(messages: list, role: str, content) -> list:
    return messages + [{"role": role, "content": content}]


def send_request(
    messages:    list,
    tools:       list  = None,
    system:      str   = "",
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
) -> dict:
    body = {
        "model":       MODEL,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
        "messages":    messages,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    response = requests.post(API_URL, headers=HEADERS, json=body)
    response.raise_for_status()
    return response.json()


# ── 工具定义（与 M2.1 相同）─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "获取指定城市的当前天气信息。"
            "返回温度（摄氏度）、天气状况（晴/多云/小雨/大雨）和湿度（%）。"
            "仅支持中国大陆主要城市。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'北京'、'上海'、'广州'",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "compare_weather",
        "description": (
            "比较两个城市的天气，给出哪个城市更适合户外活动的建议。"
            "输入两个城市的天气数据（由 get_weather 获取），"
            "返回对比结果和推荐理由。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city_a":    {"type": "string", "description": "第一个城市名称"},
                "city_b":    {"type": "string", "description": "第二个城市名称"},
                "weather_a": {"type": "object", "description": "city_a 的天气数据"},
                "weather_b": {"type": "object", "description": "city_b 的天气数据"},
            },
            "required": ["city_a", "city_b", "weather_a", "weather_b"],
        },
    },
]

_WEATHER_DB = {
    "北京": {"temp": 12, "condition": "多云", "humidity": 45},
    "上海": {"temp": 18, "condition": "小雨", "humidity": 80},
    "广州": {"temp": 26, "condition": "晴",   "humidity": 65},
    "成都": {"temp": 15, "condition": "多云", "humidity": 70},
    "杭州": {"temp": 17, "condition": "小雨", "humidity": 78},
}

def _get_weather(city: str) -> dict:
    if city not in _WEATHER_DB:
        raise ValueError(f"不支持的城市：{city}。支持的城市：{list(_WEATHER_DB.keys())}")
    data = _WEATHER_DB[city]
    return {"city": city, "temp": data["temp"],
            "condition": data["condition"], "humidity": data["humidity"], "unit": "celsius"}

def _compare_weather(city_a: str, city_b: str, weather_a: dict, weather_b: dict) -> dict:
    condition_score = {"晴": 3, "多云": 2, "小雨": 1, "大雨": 0}
    def score(w):
        s = condition_score.get(w["condition"], 0)
        if w["humidity"] < 60: s += 1
        if 15 <= w["temp"] <= 25: s += 1
        return s
    sa, sb = score(weather_a), score(weather_b)
    if sa > sb:   rec, reason = city_a, f"{city_a}天气更适宜（评分 {sa} vs {sb}）"
    elif sb > sa: rec, reason = city_b, f"{city_b}天气更适宜（评分 {sb} vs {sa}）"
    else:         rec, reason = "两城市相当", f"两城市评分相同（均为 {sa}）"
    return {"recommendation": rec, "reason": reason, "score_a": sa, "score_b": sb}

_TOOL_REGISTRY = {
    "get_weather":     _get_weather,
    "compare_weather": _compare_weather,
}

def execute_tool(name: str, input_data: dict) -> dict:
    """执行工具，失败返回 error dict，不抛异常。"""
    if name not in _TOOL_REGISTRY:
        return {"status": "error", "message": f"未知工具：{name}"}
    try:
        result = _TOOL_REGISTRY[name](**input_data)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# M2.2：Planning 四件套
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """你是一个任务规划器。
给定用户任务和可用工具列表，生成一个结构化的执行计划。

可用工具：
- get_weather(city: str) -> 获取单个城市天气
- compare_weather(city_a, city_b, weather_a, weather_b) -> 比较两城市天气

输出格式：严格输出 JSON 数组，不要任何其他文字，不要 markdown 代码块。
每个步骤包含以下字段：
  step       : 步骤编号（整数，从1开始）
  action     : 工具名称（字符串）
  params     : 工具参数（对象）
  purpose    : 本步骤目的（字符串，给人看的说明）
  depends_on : 依赖的步骤编号列表（整数数组，无依赖时为空数组）

重要约定：当某个参数需要引用前一步骤的结果时，使用占位符 "__step_N_result__"
（N 为步骤编号）。执行层会在运行时替换为实际结果。"""

SUMMARIZER_SYSTEM = """你是一个天气助手。
根据工具执行结果，用中文给用户一个简洁、友好的最终回答。
如果某个城市查询失败，需要明确说明，并基于成功的数据给出力所能及的建议。"""


# ── 0. helpers ───────────────────────────────────────────────────────────────

def extract_json(raw_text: str) -> any:
    """Strip optional markdown code fences and parse JSON."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


# ── 1. generate_plan ─────────────────────────────────────────────────────────

def generate_plan(task: str) -> list[dict]:
    """
    给定用户任务，调一次 LLM 生成结构化执行计划，并立即校验。
    返回：validated plan（list of step dicts）
    """
    print("\n📋 正在生成执行计划...")

    messages = add_message([], "user", f"任务：{task}")
    response = send_request(messages, system=PLANNER_SYSTEM, temperature=0.0)

    raw_text = "".join(
        b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"
    )

    try:
        plan = extract_json(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"计划解析失败。原始输出：\n{raw_text}\n错误：{e}")

    # 解析完立刻校验——fail fast 原则
    # 类比 Go：Unmarshal 之后立刻做 schema validation，不等到运行时才爆
    validate_plan(plan)
    return plan


# ── 2. validate_plan ─────────────────────────────────────────────────────────

def validate_plan(plan: list[dict]) -> None:
    """
    校验计划的结构合法性，有问题直接 raise。

    检查项（按发现成本从低到高排序，越早报错越好）：
    1. 必填字段是否齐全
    2. action 是否在工具注册表里（防止模型幻觉出不存在的工具名）
    3. depends_on 里的步骤编号是否合法
    4. 占位符格式是否正确，且引用的步骤确实存在

    🐍 Python 插播：集合推导式 {s["step"] for s in plan}
    类比 Go 里用 map[int]struct{} 构建一个"存在性集合"：
      valid := make(map[int]struct{})
      for _, s := range plan { valid[s.Step] = struct{}{} }
    Python 的 set 更简洁，in 操作也是 O(1)。
    """
    valid_step_nums = {s["step"] for s in plan}
    required_fields = {"step", "action", "params", "purpose", "depends_on"}

    for step in plan:
        step_num = step.get("step", "?")

        # 检查1：必填字段
        missing = required_fields - set(step.keys())
        if missing:
            raise ValueError(f"Step {step_num} 缺少必填字段：{missing}")

        # 检查2：action 必须在工具注册表里
        if step["action"] not in _TOOL_REGISTRY:
            raise ValueError(
                f"Step {step_num} 的 action '{step['action']}' 不在工具注册表中。"
                f"可用工具：{set(_TOOL_REGISTRY.keys())}"
            )

        # 检查3：depends_on 里的步骤编号必须合法
        for dep in step["depends_on"]:
            if dep not in valid_step_nums:
                raise ValueError(
                    f"Step {step_num} 依赖了不存在的步骤 {dep}。"
                    f"合法步骤编号：{valid_step_nums}"
                )

        # 检查4：params 里的占位符格式
        for key, value in step["params"].items():
            if not isinstance(value, str):
                continue
            if value.startswith("__step_") and value.endswith("_result__"):
                try:
                    ref_num = int(value[7:-9])   # "__step_1_result__" → 1
                except ValueError:
                    raise ValueError(
                        f"Step {step_num} 参数 '{key}' 占位符格式错误：'{value}'。"
                        f"正确格式：__step_N_result__（N 为整数）"
                    )
                if ref_num not in valid_step_nums:
                    raise ValueError(
                        f"Step {step_num} 参数 '{key}' 引用了不存在的步骤 {ref_num}。"
                        f"合法步骤编号：{valid_step_nums}"
                    )


# ── 3. review_plan ───────────────────────────────────────────────────────────

def print_plan(plan: list[dict]) -> None:
    print("\n" + "═" * 60)
    print("📋 执行计划（共 {} 步）".format(len(plan)))
    print("═" * 60)
    for step in plan:
        dep_str = f"依赖步骤 {step['depends_on']}" if step["depends_on"] else "无依赖（可并行）"
        print(f"\n  Step {step['step']}: {step['purpose']}")
        print(f"    action     : {step['action']}")
        print(f"    params     : {json.dumps(step['params'], ensure_ascii=False)}")
        print(f"    depends_on : {dep_str}")
    print("\n" + "═" * 60)


def review_plan(plan: list[dict], task: str) -> tuple[list[dict], bool]:
    """
    交互式审查循环：打印计划 → 等待用户输入 → 有修改意见则重新生成。

    Returns (final_plan, modified) where modified=True means the user changed
    the plan at least once, so the original task string may no longer match.

    修复点（相较于初版）：
    - revised_task 里带入当前 plan 的 JSON
    - 模型能看到"现在的计划长什么样"，基于它做局部修改
    - 第二次、第三次修改时，始终基于最新版计划调整，不丢失历史修改
    类比 Git：patch 比 full rewrite 更精确。
    """
    modified = False
    while True:
        print_plan(plan)
        user_input = input(
            "\n✋ 请审查以上计划。\n"
            "  直接回车 → 确认执行\n"
            "  输入修改意见 → 重新生成计划\n"
            "> "
        ).strip()

        if not user_input:
            print("✅ 计划已确认，开始执行...")
            return plan, modified

        print("\n🔄 收到修改意见，重新生成计划...")
        modified = True

        # 关键：把当前 plan 也传给模型，保证多轮修改的连续性
        revised_task = (
            f"原始任务：{task}\n\n"
            f"当前计划：\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"用户修改意见：{user_input}\n\n"
            f"请基于当前计划和修改意见，生成修订后的完整执行计划。"
        )
        messages = add_message([], "user", revised_task)
        response = send_request(messages, system=PLANNER_SYSTEM, temperature=0.0)
        raw_text = "".join(
            b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"
        )

        try:
            new_plan = extract_json(raw_text)
            validate_plan(new_plan)   # 修改后也要校验
            plan = new_plan
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  新计划有问题，保留当前版本。错误：{e}")
        # 回到循环顶部，打印（可能已更新的）计划


# ── 4. execute_plan + summarize ──────────────────────────────────────────────

def resolve_params(params: dict, step_results: dict) -> dict:
    """替换 params 中的占位符 __step_N_result__ 为实际执行结果。"""
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("__step_") and value.endswith("_result__"):
            step_num = int(value[7:-9])
            if step_num not in step_results:
                raise RuntimeError(
                    f"参数 '{key}' 引用步骤 {step_num} 的结果，但该步骤尚未执行。"
                    f"已完成：{list(step_results.keys())}"
                )
            resolved[key] = step_results[step_num]
        else:
            resolved[key] = value
    return resolved


def execute_plan(plan: list[dict], task: str | None) -> str:
    """
    Phase 2：代码驱动执行，直接调 execute_tool，不经过模型。

    LLM 调用次数：0（执行阶段）+ 1（summarize，在最后）= 1次

    错误处理策略（三层）：
    1. validate_plan 已过滤掉结构性错误（工具名不存在等）
    2. 运行时工具失败 → error dict 存入 step_results，继续执行后续步骤
    3. summarize() 时模型看到完整执行报告，自主生成降级回答
    """
    print("\n" + "═" * 60)
    print("🚀 开始执行计划（代码驱动，执行阶段 0 次 LLM 调用）")
    print("═" * 60)

    step_results: dict = {}

    for step in plan:
        step_num = step["step"]
        action   = step["action"]
        purpose  = step["purpose"]

        print(f"\n▶ Step {step_num}: {purpose}")

        try:
            resolved = resolve_params(step["params"], step_results)
        except RuntimeError as e:
            print(f"  ❌ 参数解析失败：{e}")
            step_results[step_num] = {"status": "error", "message": str(e)}
            continue

        # 直接执行——不过模型，代码完全控制
        result = execute_tool(action, resolved)

        status_icon = "✅" if result["status"] == "ok" else "❌"
        print(f"  {status_icon} {json.dumps(result, ensure_ascii=False)}")

        # 成功时存 result 字段（结构化数据），失败时存整个 error dict
        # 统一格式：后续步骤的 resolve_params 拿到什么，summarize 就看到什么
        step_results[step_num] = result["result"] if result["status"] == "ok" else result

    print("\n" + "═" * 60)
    print("📊 执行完毕，调用 LLM 生成最终总结（本次唯一的 LLM 调用）")
    print("═" * 60)

    return summarize(task, plan, step_results)


def summarize(task: str | None, plan: list[dict], step_results: dict) -> str:
    """
    执行阶段唯一的 LLM 调用：把结构化执行结果转成自然语言回答。

    task=None when the plan was modified by the user — the original task
    string may no longer match the actual plan, so we omit it to avoid
    misleading the model.

    传入 plan 的目的：让模型知道"step 1 是做什么的"，
    否则只有结果数据，模型无法生成有意义的总结。
    """
    execution_report = [
        {
            "step":    step["step"],
            "purpose": step["purpose"],
            "result":  step_results.get(
                step["step"],
                {"status": "error", "message": "步骤未执行"}
            ),
        }
        for step in plan
    ]

    if task:
        summary_prompt = (
            f"用户任务：{task}\n\n"
            f"执行报告：\n{json.dumps(execution_report, ensure_ascii=False, indent=2)}\n\n"
            f"请根据以上执行报告，给用户一个简洁友好的最终回答。"
        )
    else:
        summary_prompt = (
            f"执行报告：\n{json.dumps(execution_report, ensure_ascii=False, indent=2)}\n\n"
            f"请根据以上执行报告，给用户一个简洁友好的最终回答。"
        )

    messages = add_message([], "user", summary_prompt)
    response = send_request(messages, system=SUMMARIZER_SYSTEM, temperature=0.3)

    return "".join(
        b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 实验入口
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 60)
    print("M2.2 Planning 实验")
    print("=" * 60)

    # ── 实验 A：正常流程（北京 vs 上海）────────────────────────────────────
    print("\n【实验 A】显式规划 + 交互式审查：北京 vs 上海哪个适合周末出行？")
    task_a  = "查询北京和上海的天气，比较哪个城市更适合这个周末出行，给出推荐理由。"
    plan_a, modified_a = review_plan(generate_plan(task_a), task_a)    # ← 人工审批节点
    final_a = execute_plan(plan_a, None if modified_a else task_a)
    print(f"\n🏁 最终答案：\n{final_a}")

    # ── 实验 B：错误处理（含不支持城市）────────────────────────────────────
    # 纽约不在 _WEATHER_DB 里 → get_weather 返回 error dict
    # summarize() 时模型看到失败报告，自主给出降级回答
    print("\n\n【实验 B】错误处理：计划含不支持城市时如何响应？")
    task_b  = "查询成都和纽约的天气，比较哪个城市更适合这个周末出行。"
    plan_b, modified_b = review_plan(generate_plan(task_b), task_b)
    final_b = execute_plan(plan_b, None if modified_b else task_b)
    print(f"\n🏁 最终答案：\n{final_b}")


if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    run_experiment()
