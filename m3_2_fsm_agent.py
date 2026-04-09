"""
[M3.2] 状态机模式（State Machine Pattern）

核心思想：
  把 Agent 执行流程建模为有限状态机（FSM）。
  每个状态有明确的职责边界，状态转移规则由代码控制（确定性），
  模型的自由度只存在于每个状态的内部逻辑。

状态转移图：
  PLANNING ──▶ EXECUTING ──▶ EVALUATING ──┬─(passed)──▶ RESPONDING ──▶ DONE
                                           │
                                           └─(failed, retry available)──▶ PLANNING
                                           │
                                           └─(failed, no retry)──▶ FAILED

与 ReAct (M3.1) 的核心差异：
  - ReAct : 流程由 stop_reason 隐式驱动，模型决定何时结束
  - FSM   : 流程由状态机显式控制，代码决定状态转移，模型只在每个状态内工作

运行方式：
  pip install tiktoken jieba requests
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m3_2_fsm_agent.py
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum, auto

import jieba
import requests
import tiktoken

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与 M3.1 完全相同）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"
HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

MAX_RETRIES = 2   # EVALUATING 判定失败后最多重规划几次

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


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


def extract_json(text: str) -> dict | list | None:
    """
    从模型输出中提取 JSON。
    模型经常把 JSON 包在 ```json ... ``` 里，需要先剥掉 markdown 包裹。
    这个问题在 M2.2 里已经踩过坑，这里直接用健壮版。
    """
    # 先尝试剥掉 markdown 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ★ 核心：状态机定义
# ══════════════════════════════════════════════════════════════════════════════

# 🐍 Python 插播：Enum + auto()
# Enum 是枚举类型，类比 Go 的 iota 常量组：
#   Go:     type State int; const (Idle State = iota; Active; ...)
#   Python: class State(Enum): IDLE = auto(); ACTIVE = auto()
# auto() 自动赋值，不需要手写数字。
# 用 Enum 而不是字符串常量的好处：IDE 能做类型检查，打错字会报错。

class AgentState(Enum):
    PLANNING   = auto()   # 规划阶段：生成结构化执行计划
    EXECUTING  = auto()   # 执行阶段：按计划逐条调用工具（零 LLM 调用）
    EVALUATING = auto()   # 评估阶段：LLM 判断结果是否满足需求
    RESPONDING = auto()   # 响应阶段：生成最终自然语言回答
    DONE       = auto()   # 终止态：正常完成
    FAILED     = auto()   # 终止态：超过重试上限或不可恢复错误


# 静态转移表：表达"无条件"的转移路径
# 有条件的转移（如 EVALUATING 的分叉）用函数处理，不在这里
STATIC_TRANSITIONS: dict[AgentState, AgentState] = {
    AgentState.PLANNING:   AgentState.EXECUTING,
    AgentState.EXECUTING:  AgentState.EVALUATING,
    AgentState.RESPONDING: AgentState.DONE,
}

# 终止态集合：进入后不再转移
TERMINAL_STATES: set[AgentState] = {AgentState.DONE, AgentState.FAILED}


# ══════════════════════════════════════════════════════════════════════════════
# 工具模块（与 M3.1 完全相同，直接复用）
# ══════════════════════════════════════════════════════════════════════════════

_WEATHER_DB = {
    "北京": {"temp": 15, "condition": "晴",   "humidity": 30, "wind": "北风3级"},
    "上海": {"temp": 22, "condition": "多云", "humidity": 65, "wind": "东风2级"},
    "广州": {"temp": 28, "condition": "小雨", "humidity": 80, "wind": "南风1级"},
    "成都": {"temp": 18, "condition": "阴",   "humidity": 70, "wind": "无风"},
    "杭州": {"temp": 20, "condition": "晴",   "humidity": 55, "wind": "东南风2级"},
}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的当前天气。支持城市：北京、上海、广州、成都、杭州。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称（中文）"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "compare_weather",
        "description": "比较两个城市的天气，给出出行建议。支持城市：北京、上海、广州、成都、杭州。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city_a": {"type": "string", "description": "第一个城市名称（中文）"},
                "city_b": {"type": "string", "description": "第二个城市名称（中文）"},
            },
            "required": ["city_a", "city_b"],
        },
    },
]

# 工具注册表：name -> 执行函数
TOOL_REGISTRY: dict[str, callable] = {}


def register_tool(name: str):
    """装饰器：把函数注册到 TOOL_REGISTRY。"""
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator


@register_tool("get_weather")
def get_weather(city: str) -> dict:
    if city not in _WEATHER_DB:
        return {"status": "error", "message": f"不支持的城市：{city}"}
    w = _WEATHER_DB[city]
    return {
        "status":    "ok",
        "city":      city,
        "temp":      w["temp"],
        "condition": w["condition"],
        "humidity":  w["humidity"],
        "wind":      w["wind"],
    }


@register_tool("compare_weather")
def compare_weather(city_a: str, city_b: str) -> dict:
    wa = _WEATHER_DB.get(city_a)
    wb = _WEATHER_DB.get(city_b)
    if not wa:
        return {"status": "error", "message": f"不支持的城市：{city_a}"}
    if not wb:
        return {"status": "error", "message": f"不支持的城市：{city_b}"}

    # 评分：晴>多云>阴>小雨>大雨；湿度低优先；温度适中优先
    score_map = {"晴": 5, "多云": 4, "阴": 3, "小雨": 2, "大雨": 1}
    score_a = score_map.get(wa["condition"], 3) - abs(wa["temp"] - 22) * 0.1
    score_b = score_map.get(wb["condition"], 3) - abs(wb["temp"] - 22) * 0.1
    better = city_a if score_a >= score_b else city_b

    return {
        "status":       "ok",
        "city_a":       city_a,
        "city_a_info":  wa,
        "city_b":       city_b,
        "city_b_info":  wb,
        "recommendation": better,
        "reason":       f"{better}天气更适合出行（综合气温、天气状况评分）",
    }


def execute_tool(name: str, args: dict) -> dict:
    """统一工具执行入口，与 M2.4/M3.1 的 ToolRegistry.execute() 等价。"""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"status": "error", "message": f"未知工具：{name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"status": "error", "message": f"执行异常：{e}"}


# ══════════════════════════════════════════════════════════════════════════════
# ★ FSM Agent 核心
# ══════════════════════════════════════════════════════════════════════════════

class FSMAgent:
    """
    有限状态机 Agent。

    架构要点：
      - self.state     : 当前状态，是唯一的"真相来源"
      - self.ctx       : 跨状态共享数据（plan / tool_results / evaluation）
      - self._handlers : 状态 → 执行函数的映射（开闭原则）
      - retry_count    : 记录重规划次数，用于 EVALUATING 的条件转移

    主循环极其简单：
      while not terminal:
          handler = self._handlers[self.state]
          handler()
          self.state = self._get_next_state()
    """

    def __init__(self):
        self.state:       AgentState = AgentState.PLANNING
        self.retry_count: int        = 0

        # 跨状态共享数据容器
        # 每个状态只写自己负责的 key，形成隐式契约：
        #   PLANNING   写：ctx["user_request"], ctx["plan"]
        #   EXECUTING  写：ctx["tool_results"]
        #   EVALUATING 写：ctx["evaluation_passed"], ctx["evaluation_reason"]
        #   RESPONDING 写：ctx["final_response"]
        self.ctx: dict = {}

        # 🐍 Python 插播：方法引用
        # Python 中可以把实例方法当作值存入 dict。
        # self._do_planning 是一个 bound method（已绑定 self），
        # 调用 handler() 等价于调用 self._do_planning()。
        # 类比 Go 的函数值：handlers[state] = agent.DoPlanning
        self._handlers: dict[AgentState, callable] = {
            AgentState.PLANNING:   self._do_planning,
            AgentState.EXECUTING:  self._do_executing,
            AgentState.EVALUATING: self._do_evaluating,
            AgentState.RESPONDING: self._do_responding,
        }

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def handle(self, user_request: str) -> str:
        """
        处理一次用户请求，驱动状态机运转直到终止。
        返回最终回答字符串。

        这是整个 FSM Agent 的心跳：
          初始化 → while 非终止态: 执行当前状态的 handler → 转移状态
        """
        # 初始化每次请求的上下文
        self.state       = AgentState.PLANNING
        self.retry_count = 0
        self.ctx         = {"user_request": user_request}

        self._log_state_enter()

        while self.state not in TERMINAL_STATES:
            # 执行当前状态的 handler
            handler = self._handlers[self.state]
            handler()

            # 计算下一个状态
            next_state = self._get_next_state()
            if next_state != self.state:
                self._log_state_transition(next_state)
            self.state = next_state

            if self.state not in TERMINAL_STATES:
                self._log_state_enter()

        # 从 ctx 取结果返回
        if self.state == AgentState.DONE:
            return self.ctx.get("final_response", "（无回答）")
        else:
            reason = self.ctx.get("evaluation_reason", "未知原因")
            return f"❌ 任务失败：{reason}"

    # ── 状态转移逻辑 ──────────────────────────────────────────────────────────

    def _get_next_state(self) -> AgentState:
        """
        计算当前状态执行完后应转移到的下一个状态。

        设计：
          - 优先查 STATIC_TRANSITIONS（无条件路径）
          - EVALUATING 是唯一有条件分叉的状态，用专函数处理
          - 终止态返回自身（while 循环的退出条件）
        """
        # 终止态不再转移
        if self.state in TERMINAL_STATES:
            return self.state

        # EVALUATING 是唯一需要条件判断的状态
        if self.state == AgentState.EVALUATING:
            return self._evaluate_transition()

        # 其他状态查静态转移表
        return STATIC_TRANSITIONS.get(self.state, AgentState.FAILED)

    def _evaluate_transition(self) -> AgentState:
        """
        EVALUATING 状态的条件转移逻辑。

        这是整个 FSM 里唯一的"分叉点"——也是状态机比 ReAct 更可控的核心体现：
        重试次数、失败处理，全部由这里的代码（而非模型）决定。
        """
        if self.ctx.get("evaluation_passed"):
            return AgentState.RESPONDING

        # 评估失败：检查重试次数
        if self.retry_count < MAX_RETRIES:
            self.retry_count += 1
            reason = self.ctx.get("evaluation_reason", "")
            print(f"\n  🔄 评估未通过，触发重规划 (第 {self.retry_count}/{MAX_RETRIES} 次)")
            print(f"     原因：{reason}")
            return AgentState.PLANNING   # 回到 PLANNING 重来

        # 超过重试上限
        return AgentState.FAILED

    # ── 各状态 Handler ────────────────────────────────────────────────────────

    def _do_planning(self) -> None:
        """
        PLANNING 状态：调用 LLM，生成结构化 JSON 执行计划。

        输出写入 ctx["plan"]，格式：
          [
            {"tool": "get_weather", "args": {"city": "北京"}, "purpose": "..."},
            {"tool": "compare_weather", "args": {...}, "purpose": "..."},
          ]

        如果本次是重规划（retry_count > 0），把上次的失败原因也告知模型，
        让它生成不同的计划。
        """
        user_request = self.ctx["user_request"]

        # 重规划时，把失败原因附上，让模型知道为什么上次不够好
        retry_context = ""
        if self.retry_count > 0:
            reason = self.ctx.get("evaluation_reason", "结果不满足用户需求")
            prev_results = self.ctx.get("tool_results", [])
            retry_context = f"""
【重规划背景】
上一次执行后评估失败，原因：{reason}
上一次工具执行结果：{json.dumps(prev_results, ensure_ascii=False)}
请基于以上信息，生成一个更好的执行计划。
"""

        system = f"""你是一个任务规划器。
根据用户请求，生成一个结构化的工具调用计划。

可用工具：
{json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)}

输出格式（严格 JSON，不要加任何注释或 markdown）：
[
  {{
    "step": 1,
    "tool": "工具名",
    "args": {{"参数名": "参数值"}},
    "purpose": "这一步的目的"
  }}
]

规则：
1. 只输出 JSON 数组，不要输出任何其他内容
2. tool 字段必须是可用工具之一
3. args 字段必须匹配工具的参数定义
4. 如果任务不需要调用任何工具（如简单问候），输出空数组 []
{retry_context}"""

        response = send_request(
            messages=[{"role": "user", "content": user_request}],
            system=system,
            temperature=0.0,
        )

        raw_text = "".join(
            b.get("text", "") for b in response.get("content", [])
            if b.get("type") == "text"
        )

        plan = extract_json(raw_text)

        # 容错：如果解析失败，用空计划（让 EVALUATING 判断是否满足需求）
        if not isinstance(plan, list):
            print(f"  ⚠️  计划解析失败，原始输出：{raw_text[:200]}")
            plan = []

        self.ctx["plan"] = plan
        self._log_plan(plan)

    def _do_executing(self) -> None:
        """
        EXECUTING 状态：按 ctx["plan"] 逐条执行工具调用。

        ★ 关键设计：这个状态零 LLM 调用。
        所有决策（调哪个工具、传什么参数）全部来自 PLANNING 阶段的 plan。
        这是"代码控制执行，模型负责规划"原则的体现。

        输出写入 ctx["tool_results"]，格式：
          [{"step": 1, "tool": "...", "result": {...}}, ...]
        """
        plan = self.ctx.get("plan", [])

        if not plan:
            # 空计划：没有工具需要调用，直接跳过执行阶段
            print("  ℹ️  计划为空，跳过工具执行")
            self.ctx["tool_results"] = []
            return

        tool_results = []
        for step in plan:
            step_num = step.get("step", "?")
            tool     = step.get("tool", "")
            args     = step.get("args", {})
            purpose  = step.get("purpose", "")

            print(f"\n  ⚡ Step {step_num}: {tool}({json.dumps(args, ensure_ascii=False)})")
            print(f"     目的：{purpose}")

            result = execute_tool(tool, args)

            status_icon = "✅" if result.get("status") == "ok" else "❌"
            print(f"  {status_icon} 结果：{json.dumps(result, ensure_ascii=False)}")

            tool_results.append({
                "step":   step_num,
                "tool":   tool,
                "args":   args,
                "result": result,
            })

        self.ctx["tool_results"] = tool_results

    def _do_evaluating(self) -> None:
        """
        EVALUATING 状态：调用 LLM 判断工具执行结果是否满足用户需求。

        设计思路：
          - 把"结果够不够好"这个判断交给模型（非确定性）
          - 但"够不够"之后怎么办（重试/失败/继续）由代码决定（确定性）
          - 这是"模型判断，代码控制"的典型分工

        输出写入：
          ctx["evaluation_passed"] : bool
          ctx["evaluation_reason"] : str（未通过时的原因）
        """
        user_request = self.ctx["user_request"]
        tool_results = self.ctx.get("tool_results", [])

        # 如果没有工具调用结果，且任务是简单问候类，直接通过
        if not tool_results:
            self.ctx["evaluation_passed"] = True
            self.ctx["evaluation_reason"] = "无需工具调用，直接响应"
            return

        system = """你是一个结果评估器。
判断工具执行结果是否足以回答用户的问题。

输出格式（严格 JSON）：
{
  "passed": true 或 false,
  "reason": "评估理由（一句话）"
}

评估标准：
- 所有必要的工具都已调用且成功返回结果 → passed: true
- 有工具调用失败，且失败影响了核心需求 → passed: false
- 返回的信息足以回答用户问题 → passed: true"""

        eval_prompt = f"""用户问题：{user_request}

工具执行结果：
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

请评估以上结果是否能充分回答用户的问题。"""

        response = send_request(
            messages=[{"role": "user", "content": eval_prompt}],
            system=system,
            temperature=0.0,
        )

        raw_text = "".join(
            b.get("text", "") for b in response.get("content", [])
            if b.get("type") == "text"
        )

        evaluation = extract_json(raw_text)

        if not isinstance(evaluation, dict):
            # 解析失败：保守地认为通过（避免无限重试）
            print(f"  ⚠️  评估结果解析失败，默认通过")
            self.ctx["evaluation_passed"] = True
            self.ctx["evaluation_reason"] = "评估解析失败，保守通过"
            return

        passed = evaluation.get("passed", True)
        reason = evaluation.get("reason", "")

        self.ctx["evaluation_passed"] = passed
        self.ctx["evaluation_reason"] = reason

        icon = "✅" if passed else "❌"
        print(f"\n  {icon} 评估结果：{'通过' if passed else '未通过'}")
        print(f"     理由：{reason}")

    def _do_responding(self) -> None:
        """
        RESPONDING 状态：整合所有信息，生成最终自然语言回答。

        ★ 约束：这个状态只能生成回答，不能发起工具调用。
        这是状态机"约束行为"的体现——RESPONDING 的 system prompt
        里没有 tools 参数，从 API 层面就封死了工具调用的可能。

        输出写入 ctx["final_response"]。
        """
        user_request = self.ctx["user_request"]
        tool_results = self.ctx.get("tool_results", [])

        if not tool_results:
            # 无工具调用：直接让模型根据知识回答
            context = "（本次任务无需调用工具）"
        else:
            context = json.dumps(tool_results, ensure_ascii=False, indent=2)

        system = """你是一个友好的 AI 助手。
根据工具执行结果，用自然语言回答用户的问题。
- 语言：中文，简洁友好
- 不要提及"工具调用"、"执行结果"等技术细节
- 直接给出用户需要的信息"""

        response_prompt = f"""用户问题：{user_request}

工具执行结果（供参考，不要在回答中提及这些技术细节）：
{context}

请直接回答用户的问题。"""

        response = send_request(
            messages=[{"role": "user", "content": response_prompt}],
            system=system,
            temperature=0.3,   # 稍微高一点，回答更自然
        )

        final_response = "".join(
            b.get("text", "") for b in response.get("content", [])
            if b.get("type") == "text"
        )

        self.ctx["final_response"] = final_response

    # ── 日志辅助方法 ──────────────────────────────────────────────────────────

    def _log_state_enter(self) -> None:
        """打印进入新状态的日志。状态机的可观测性核心。"""
        icons = {
            AgentState.PLANNING:   "🧠",
            AgentState.EXECUTING:  "⚙️ ",
            AgentState.EVALUATING: "🔍",
            AgentState.RESPONDING: "💬",
            AgentState.DONE:       "✅",
            AgentState.FAILED:     "❌",
        }
        icon = icons.get(self.state, "❓")
        print(f"\n{'─'*50}")
        print(f"  {icon}  进入状态：{self.state.name}")
        print(f"{'─'*50}")

    def _log_state_transition(self, next_state: AgentState) -> None:
        """打印状态转移日志。"""
        print(f"\n  ➡️  状态转移：{self.state.name} → {next_state.name}")

    def _log_plan(self, plan: list) -> None:
        """打印生成的计划。"""
        if not plan:
            print("  📋 计划：（空——无需工具调用）")
            return
        print(f"  📋 生成计划（{len(plan)} 步）：")
        for step in plan:
            print(f"     Step {step.get('step', '?')}: {step.get('tool')}({step.get('args')}) — {step.get('purpose', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🤖 FSM Agent 启动")
    print(f"   可用工具：{list(TOOL_REGISTRY.keys())}")
    print(f"   最大重规划次数：{MAX_RETRIES}")
    print("\n   内置命令：quit")
    print("=" * 60)

    agent = FSMAgent()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if user_input.lower() == "quit":
            print("👋 再见！")
            break
        if not user_input:
            continue

        response = agent.handle(user_input)
        print(f"\n{'='*50}")
        print(f"助手: {response}")
        print(f"{'='*50}")


if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)
    main()
