"""
[M2.2] Planning 机制
显式计划生成 + 交互式审查 + 动态重规划

架构：
  Phase 1: generate_plan()  → 调一次 LLM，返回结构化 JSON 计划
           review_plan()    → 打印计划，等待人工确认或修改
             └─ 有修改意见 → revise_plan() → 回到 review_plan()
  Phase 2: execute_plan()   → 驱动执行，复用 M2.1 的 tool_use_loop 逻辑
             └─ 模型读到 tool_result 中的 error 后自主调整后续步骤

复用 M2.1 的：send_request / execute_tool / tool_use_loop / TOOLS / SYSTEM

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m2_2_planning.py
"""

import json
import os
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与 M2.1 完全相同，原封不动）
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
    if name not in _TOOL_REGISTRY:
        return {"status": "error", "message": f"未知工具：{name}"}
    try:
        result = _TOOL_REGISTRY[name](**input_data)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Tool-use Loop（与 M2.1 相同，但去掉了打印轮次的噪音）────────────────────

def tool_use_loop(initial_messages: list, tools: list, system: str = "") -> str:
    messages = initial_messages
    for turn in range(MAX_TURNS):
        response    = send_request(messages, tools=tools, system=system)
        stop_reason = response.get("stop_reason")
        content     = response.get("content", [])

        if stop_reason == "end_turn":
            return "".join(b.get("text", "") for b in content if b.get("type") == "text")

        if stop_reason == "tool_use":
            messages     = add_message(messages, "assistant", content)
            tool_results = []

            for block in content:
                if block.get("type") != "tool_use":
                    continue
                name, input_data, use_id = block["name"], block["input"], block["id"]

                print(f"    🔧 {name}({json.dumps(input_data, ensure_ascii=False)})")
                result = execute_tool(name, input_data)

                # 打印结果，错误用红色标记方便观察
                status_icon = "✅" if result["status"] == "ok" else "❌"
                print(f"    {status_icon} {json.dumps(result, ensure_ascii=False)}")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": use_id,
                    "content":     json.dumps(result, ensure_ascii=False),
                })

            messages = add_message(messages, "user", tool_results)
            continue

        raise RuntimeError(f"未处理的 stop_reason: {stop_reason}")

    raise RuntimeError(f"超过最大轮次限制（{MAX_TURNS}）")


# ══════════════════════════════════════════════════════════════════════════════
# M2.2 新增：Planning 三件套
# ══════════════════════════════════════════════════════════════════════════════

# ── 计划的数据结构 ────────────────────────────────────────────────────────────
#
# 一个 Plan 是一个步骤列表，每个步骤包含：
#   step     : 步骤编号（从1开始）
#   action   : 要调用的工具名（或 "llm_reason" 表示纯推理步骤，不调工具）
#   params   : 工具参数（dict）
#   purpose  : 这一步的目的（给人看的，帮助审查）
#   depends_on: 依赖哪些步骤的结果（list of step numbers）
#
# 🐍 Python 插播：TypedDict
# 类比 Go 的 struct——给 dict 加上类型约束，IDE 可以做类型检查。
# 这里只用注释说明结构，实际运行时仍然是普通 dict（Python 动态类型）。
#
# Plan 示例：
# [
#   {"step": 1, "action": "get_weather", "params": {"city": "北京"},
#    "purpose": "获取北京天气数据", "depends_on": []},
#   {"step": 2, "action": "get_weather", "params": {"city": "上海"},
#    "purpose": "获取上海天气数据", "depends_on": []},
#   {"step": 3, "action": "compare_weather",
#    "params": {"city_a": "北京", "city_b": "上海",
#               "weather_a": "__step_1_result__", "weather_b": "__step_2_result__"},
#    "purpose": "对比两城市，给出推荐", "depends_on": [1, 2]},
# ]

# ── System Prompt：规划模式 ───────────────────────────────────────────────────
#
# 注意这里的 prompt 工程技巧：
# 1. 明确告诉模型"只输出 JSON，不要其他文字"——否则模型会加前缀"好的，以下是计划："
# 2. 给出精确的字段定义——模型输出的 JSON 必须可被代码解析，字段名不能随意
# 3. 给出占位符约定（__step_N_result__）——这是我们和模型之间的"协议"

PLANNER_SYSTEM = """你是一个任务规划器。
给定用户任务和可用工具列表，生成一个结构化的执行计划。

可用工具：
- get_weather(city: str) → 获取单个城市天气
- compare_weather(city_a, city_b, weather_a, weather_b) → 比较两城市天气

输出格式：严格输出 JSON 数组，不要任何其他文字，不要 markdown 代码块。
每个步骤包含以下字段：
  step       : 步骤编号（整数，从1开始）
  action     : 工具名称（字符串）
  params     : 工具参数（对象）
  purpose    : 本步骤目的（字符串，给人看的说明）
  depends_on : 依赖的步骤编号列表（整数数组，无依赖时为空数组）

重要约定：当某个参数需要引用前一步骤的结果时，使用占位符 "__step_N_result__"
（N 为步骤编号）。执行层会在运行时替换为实际结果。"""


def generate_plan(task: str) -> list[dict]:
    """
    Phase 1 核心函数：给定用户任务，调一次 LLM 生成结构化执行计划。

    返回：plan（list of step dicts）

    设计要点：
    - 使用独立的 PLANNER_SYSTEM，与执行阶段的 system prompt 完全分离
    - temperature=0 确保计划输出稳定可重复
    - 用 strip() + json.loads() 解析，不依赖模型完美格式化

    🐍 Python 插播：list[dict] 是 Python 3.9+ 的类型注解语法
    类比 Go 的 []map[string]interface{}
    只是给 IDE 看的提示，运行时不做检查。
    """
    print("\n📋 正在生成执行计划...")

    messages = add_message([], "user", f"任务：{task}")
    response = send_request(messages, system=PLANNER_SYSTEM, temperature=0.0)

    # 从响应中提取文本
    raw_text = ""
    for block in response.get("content", []):
        if block.get("type") == "text":
            raw_text += block.get("text", "")

    # 解析 JSON
    # 🐍 Python 插播：strip() 去掉首尾空白字符（含换行），类比 Go 的 strings.TrimSpace()
    try:
        plan = json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        # 如果解析失败，把原始文本打印出来方便调试
        raise RuntimeError(f"计划解析失败。原始输出：\n{raw_text}\n错误：{e}")

    return plan


def print_plan(plan: list[dict]) -> None:
    """
    可视化打印计划，供人工审查。

    格式设计考虑：
    - 用缩进表示层级（步骤 → 字段）
    - 高亮 depends_on，方便看出步骤间的依赖关系
    - purpose 字段放在最显眼的位置——这是给人看的
    """
    print("\n" + "═" * 60)
    print("📋 执行计划（共 {} 步）".format(len(plan)))
    print("═" * 60)

    for step in plan:
        dep_str = (
            f"依赖步骤 {step['depends_on']}" if step["depends_on"] else "无依赖（可并行）"
        )
        print(f"\n  Step {step['step']}: {step['purpose']}")
        print(f"    action     : {step['action']}")
        print(f"    params     : {json.dumps(step['params'], ensure_ascii=False)}")
        print(f"    depends_on : {dep_str}")

    print("\n" + "═" * 60)


def review_plan(plan: list[dict], task: str) -> list[dict]:
    """
    交互式审查循环：打印计划 → 等待用户输入 → 有修改意见则重新生成。

    返回最终确认的计划（可能经过0次或多次修订）。

    这里实现的就是架构图里的"人工审批节点"：
    - 直接回车 → 确认，进入执行阶段
    - 输入修改意见 → 调 LLM 重新生成计划，再次审查

    类比 Go 里的交互式 CLI：
      scanner := bufio.NewScanner(os.Stdin)
      scanner.Scan()
      input := scanner.Text()

    Python 里用内置 input() 函数，更简洁。
    """
    while True:
        print_plan(plan)

        # 🐍 Python 插播：input() 打印提示语并等待用户输入，返回字符串
        # 类比 Go 的 fmt.Scan() 或 bufio.Scanner
        user_input = input(
            "\n✋ 请审查以上计划。\n"
            "  直接回车 → 确认执行\n"
            "  输入修改意见 → 重新生成计划\n"
            "> "
        ).strip()

        # 用户确认，退出审查循环
        if not user_input:
            print("✅ 计划已确认，开始执行...")
            return plan

        # 用户有修改意见，调 LLM 重新规划
        print(f"\n🔄 收到修改意见：「{user_input}」，重新生成计划...")

        # 把原始任务 + 修改意见一起传给 LLM
        # 注意：我们不直接修改 plan，而是让模型重新生成——保持计划的一致性
        revised_task = (
            f"原始任务：{task}\n\n"
            f"初始计划已生成，但用户提出了修改意见：{user_input}\n\n"
            f"请根据修改意见重新生成完整的执行计划。"
        )
        messages  = add_message([], "user", revised_task)
        response  = send_request(messages, system=PLANNER_SYSTEM, temperature=0.0)
        raw_text  = "".join(
            b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"
        )

        try:
            plan = json.loads(raw_text.strip())
        except json.JSONDecodeError as e:
            print(f"⚠️  重新生成的计划解析失败，保留原计划。错误：{e}")
            # 解析失败时不崩溃，保留上一版计划继续审查循环

        # 回到循环顶部，再次打印新计划供审查


# ── 占位符替换：把 __step_N_result__ 替换为实际执行结果 ──────────────────────
#
# 这是 Planning 里最关键的"胶水"逻辑。
# 问题：计划生成时，step 3 的 weather_a 参数值是 "__step_1_result__"，
#       执行 step 3 之前必须把这个占位符替换成 step 1 的实际返回值。
#
# 实现思路：
#   step_results = {1: {...}, 2: {...}}  ← 已执行步骤的结果字典
#   遍历 params，如果值是 "__step_N_result__" 字符串，
#   就用 step_results[N] 替换。

def resolve_params(params: dict, step_results: dict) -> dict:
    """
    替换 params 中的占位符，返回可直接传给工具的参数 dict。

    step_results: {step_number: result_value}
    """
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("__step_") and value.endswith("_result__"):
            # 提取步骤编号："__step_1_result__" → 1
            # 🐍 Python 插播：字符串切片 s[7:-9] 去掉前缀"__step_"(7字符)和后缀"_result__"(9字符)
            step_num = int(value[7:-9])
            if step_num not in step_results:
                raise RuntimeError(
                    f"参数 '{key}' 引用了步骤 {step_num} 的结果，"
                    f"但该步骤尚未执行。已完成步骤：{list(step_results.keys())}"
                )
            resolved[key] = step_results[step_num]
        else:
            resolved[key] = value
    return resolved


def execute_plan(plan: list[dict], tools: list, system: str = "") -> str:
    """
    Phase 2 核心函数：按计划顺序执行每个步骤。

    设计要点：
    1. 维护 step_results dict，保存每步的执行结果（供后续步骤引用）
    2. 执行每步前先 resolve_params，替换占位符
    3. 每个步骤实际上是一次 tool_use_loop 调用——复用 M2.1 的逻辑
       但每步只做一次工具调用（因为计划已经把任务分解好了）
    4. 模型读到 error 后可以在当前步骤内自主决策（重试/降级/报错）

    为什么每步用独立的 tool_use_loop 而不是一个大循环？
    → 隔离性：每步的 messages 独立，避免 context 无限膨胀
    → 可观测：每步开始/结束有清晰的边界，方便调试
    → 可中断：未来可以在步骤间加检查点（checkpoint）

    🐍 Python 插播：enumerate()
    类比 Go 的 for i, v := range slice
    enumerate(plan) 返回 (index, value) 对，
    这里我们用 step["step"] 作为编号（从1开始），不用 enumerate 的 index。
    """
    print("\n" + "═" * 60)
    print("🚀 开始执行计划")
    print("═" * 60)

    step_results: dict = {}   # {step_number: actual_result}

    for step in plan:
        step_num = step["step"]
        action   = step["action"]
        purpose  = step["purpose"]

        print(f"\n▶ Step {step_num}: {purpose}")

        # 替换参数中的占位符
        try:
            resolved = resolve_params(step["params"], step_results)
        except RuntimeError as e:
            print(f"  ❌ 参数解析失败：{e}")
            break

        # 构造这一步的 prompt：告诉模型要做什么、用什么工具、用什么参数
        # 注意：我们明确指定了工具和参数——这是"显式规划"的体现。
        # 模型不需要自己决定调什么，只需要执行我们告诉它的步骤。
        step_prompt = (
            f"请执行以下步骤：{purpose}\n"
            f"调用工具：{action}\n"
            f"使用参数：{json.dumps(resolved, ensure_ascii=False)}\n"
            f"如果工具返回错误，请说明原因并给出替代方案。"
        )

        messages = add_message([], "user", step_prompt)

        # 执行这一步（内部会处理 tool_use loop）
        result_text = tool_use_loop(messages, tools, system=system)

        # 保存这一步的结果，供后续步骤引用
        # 注意：我们保存的是 execute_tool 返回的 dict（通过 tool_results 传递给模型）
        # 这里需要从 step_results 里取工具的实际返回值
        # 设计选择：我们保存"工具的 result 字段"，而不是模型的文字总结
        # 原因：后续步骤的参数需要结构化数据，文字总结无法被 compare_weather 使用
        #
        # 实现方式：在 tool_use_loop 里工具执行后，结果通过 tool_result content 传给模型。
        # 这里我们用一个小技巧：直接调用 execute_tool 再执行一次，拿到结构化结果。
        # （在实际生产中，更好的做法是让 tool_use_loop 返回工具结果，这是一个架构改进点）
        tool_result = execute_tool(action, resolved)
        if tool_result["status"] == "ok":
            step_results[step_num] = tool_result["result"]
        else:
            # 工具失败时，把错误信息存入 step_results
            # 后续步骤如果引用这个结果，会拿到错误 dict——让模型处理
            step_results[step_num] = tool_result
            print(f"  ⚠️  步骤 {step_num} 工具执行失败，已记录错误，继续执行后续步骤")

        print(f"  💾 步骤 {step_num} 结果已保存")

    print("\n" + "═" * 60)
    print("📊 计划执行完毕")
    print("═" * 60)

    # 返回最后一步的模型文字输出作为最终答案
    return result_text


# ══════════════════════════════════════════════════════════════════════════════
# 实验入口
# ══════════════════════════════════════════════════════════════════════════════

EXECUTOR_SYSTEM = """你是一个天气助手，负责执行具体的工具调用步骤。
按照用户指定的步骤和参数执行操作，用中文简洁报告结果。
如果工具返回错误，请明确说明错误原因，并建议替代方案。"""


def run_experiment():
    print("=" * 60)
    print("M2.2 Planning 实验")
    print("=" * 60)

    # ── 实验 A：正常流程（北京 vs 上海）────────────────────────────────────
    print("\n【实验 A】显式规划 + 交互式审查：北京 vs 上海哪个适合周末出行？")
    task_a = "查询北京和上海的天气，比较哪个城市更适合这个周末出行，给出推荐理由。"

    plan_a = generate_plan(task_a)
    plan_a = review_plan(plan_a, task_a)      # ← 人工审查节点
    final_a = execute_plan(plan_a, TOOLS, system=EXECUTOR_SYSTEM)
    print(f"\n🏁 最终答案：\n{final_a}")

    # ── 实验 B：动态重规划（包含不支持的城市）──────────────────────────────
    # 计划里会出现"纽约"（不支持的城市），观察模型如何在 error 后调整
    print("\n\n【实验 B】动态重规划：计划含不支持城市时模型如何响应？")
    task_b = "查询成都和纽约的天气，比较哪个城市更适合这个周末出行。"

    plan_b = generate_plan(task_b)
    plan_b = review_plan(plan_b, task_b)
    final_b = execute_plan(plan_b, TOOLS, system=EXECUTOR_SYSTEM)
    print(f"\n🏁 最终答案：\n{final_b}")


if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    run_experiment()
