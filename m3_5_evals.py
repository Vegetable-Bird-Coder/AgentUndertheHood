"""
[M3.5] Agent Evals — Eval-Driven Development

三要素实现：
  Dataset  : 8 个测试用例，覆盖工具选择、参数准确性、多步推理、答案质量、拒绝能力
  Grader   : DeterministicGrader（确定性检查）+ LLMJudge（LLM-as-Judge，CoT 模式）
  Harness  : EvalHarness（运行框架，支持分数汇总、对比实验）

关键设计决策：
  1. MockToolRegistry 注入：替换真实工具执行，消除外部 I/O 噪音，保证可重复性
  2. 调用日志（call_log）：在 Mock 层记录所有调用，DeterministicGrader 消费
  3. CoT Grading：Judge 先输出 reasoning 再给分，抑制直接打分的随机性
  4. 对比实验：Baseline Prompt vs ReAct Prompt，唯一变量是 system prompt 内容
  5. 依赖注入：run_agent_once 接受 mock_registry 参数，而非内部创建——
     这是"可测试性"的最小代价：在设计时留一个缝，让测试可以把 mock 塞进去

数据一致性说明：
  MockToolRegistry 使用与 m2_4_mini_agent._WEATHER_DB 完全相同的天气数值，
  确保 eval 场景与真实场景行为一致。

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m3_5_evals.py
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, stdev

import requests

# ── 复用已有基础设施 ──────────────────────────────────────────────────────────
# send_request：统一 HTTP 请求入口，不重复实现
# _WEATHER_SCHEMAS / _MEMORY_SCHEMAS：工具的 API 描述，供 LLM 理解工具能力
# 注：前缀 _ 在 Python 中是"约定私有"，不是强制隔离，可以跨文件 import
from m2_4_mini_agent import (
    _MEMORY_SCHEMAS,
    _WEATHER_SCHEMAS,
    send_request,
)

# ══════════════════════════════════════════════════════════════════════════════
# Mock 数据 — 与 m2_4_mini_agent._WEATHER_DB 完全一致
#
# 为什么保持一致：eval mock 代表的是"如果工具真实运行会返回什么"，
# 数据不同 → Agent 的行为和真实场景不同 → eval 分数失去参考意义
# ══════════════════════════════════════════════════════════════════════════════

_MOCK_WEATHER_DB = {
    "北京": {"status": "ok", "city": "北京", "temp": 12, "condition": "多云", "humidity": 45, "unit": "celsius"},
    "上海": {"status": "ok", "city": "上海", "temp": 18, "condition": "小雨", "humidity": 80, "unit": "celsius"},
    "广州": {"status": "ok", "city": "广州", "temp": 26, "condition": "晴",   "humidity": 65, "unit": "celsius"},
    "成都": {"status": "ok", "city": "成都", "temp": 15, "condition": "多云", "humidity": 70, "unit": "celsius"},
    "杭州": {"status": "ok", "city": "杭州", "temp": 17, "condition": "小雨", "humidity": 78, "unit": "celsius"},
}


# ══════════════════════════════════════════════════════════════════════════════
# MockToolRegistry — 工具执行层的桩实现
#
# 核心职责：
#   1. 暴露与真实 ToolRegistry 相同的接口（schemas + execute）
#   2. execute() 返回固定 Mock 数据，不发真实 HTTP/IO 请求
#   3. _call_log 记录所有调用，供 DeterministicGrader 消费
#
# 类比 Go 的 mock interface：
#   type ToolExecutor interface { Execute(name string, input map[string]any) map[string]any }
#   // MockToolExecutor 实现同一接口，但注入固定数据而非真实逻辑
# ══════════════════════════════════════════════════════════════════════════════

class MockToolRegistry:
    """
    工具注册表的 Mock 实现。

    每个 EvalCase 运行前调用 reset_log() 清空状态，
    确保不同 case 之间的工具调用记录互不干扰。
    """

    def __init__(self):
        # schema 与真实 ToolRegistry 一致：LLM 看到相同的工具描述
        self._schemas: list[dict] = _WEATHER_SCHEMAS + _MEMORY_SCHEMAS
        # 调用日志：记录 {tool, args, result}
        self._call_log: list[dict] = []
        # 轻量内存 KV 模拟 FactStore，避免磁盘副作用（每个 case 独立）
        self._memory: list[dict] = []

    @property
    def schemas(self) -> list[dict]:
        return list(self._schemas)

    def reset_log(self) -> None:
        """每个 EvalCase 开始前清空，保证 case 间隔离。"""
        self._call_log.clear()
        self._memory.clear()

    def execute(self, name: str, input_data: dict) -> dict:
        """
        路由并记录工具调用。

        顺序：先记录 → 再返回
        这样即使 _dispatch 出错，调用行为也被捕捉到了。
        """
        result = self._dispatch(name, input_data)
        self._call_log.append({"tool": name, "args": input_data, "result": result})
        return result

    def _dispatch(self, name: str, input_data: dict) -> dict:
        """按工具名路由到具体的 Mock 实现。"""

        if name == "get_weather":
            city = input_data.get("city", "")
            return _MOCK_WEATHER_DB.get(
                city,
                {"status": "error", "message": f"不支持的城市：{city}。支持：{list(_MOCK_WEATHER_DB.keys())}"},
            )

        elif name == "compare_weather":
            # 复用与原始相同的评分逻辑，保持行为一致性
            city_a, city_b = input_data.get("city_a", ""), input_data.get("city_b", "")
            wa, wb = input_data.get("weather_a", {}), input_data.get("weather_b", {})

            score_map = {"晴": 3, "多云": 2, "小雨": 1, "大雨": 0}

            def score(w: dict) -> int:
                s = score_map.get(w.get("condition", ""), 0)
                if w.get("humidity", 100) < 60: s += 1
                if 15 <= w.get("temp", 0) <= 25: s += 1
                return s

            sa, sb = score(wa), score(wb)
            if   sa > sb: rec, reason = city_a, f"{city_a}更适宜（{sa} vs {sb}分）"
            elif sb > sa: rec, reason = city_b, f"{city_b}更适宜（{sb} vs {sa}分）"
            else:         rec, reason = "两城市相当", f"评分相同（均 {sa} 分）"

            return {"status": "ok", "recommendation": rec, "reason": reason,
                    "score_a": sa, "score_b": sb}

        elif name == "save_fact":
            fact = {"content": input_data.get("content", ""), "tags": input_data.get("tags", [])}
            self._memory.append(fact)
            return {"status": "ok", "message": f"已保存：{fact['content']}"}

        elif name == "recall_facts":
            query_words = input_data.get("query", "").lower().split()
            matches = [
                f for f in self._memory
                if any(w in f["content"].lower() for w in query_words)
            ]
            if matches:
                return {"status": "ok", "facts": matches}
            return {"status": "ok", "facts": [], "message": "未找到相关记忆"}

        return {"status": "error", "message": f"未知工具：{name}"}


# ══════════════════════════════════════════════════════════════════════════════
# run_agent_once — 无交互版的 Agent 主循环
#
# 从 MiniAgent._agent_turn 提取出的纯函数版本。
# 关键差异：接受 mock_registry 作为参数（依赖注入），而非内部创建
#
# 类比 Go：
#   func RunAgentOnce(input string, prompt string, exec ToolExecutor) (string, []Call) { ... }
#   // ToolExecutor 是 interface，可以注入 MockExecutor 或 RealExecutor
# ══════════════════════════════════════════════════════════════════════════════

_MAX_LOOP_TURNS = 10


def run_agent_once(
    user_input:    str,
    system_prompt: str,
    mock_registry: MockToolRegistry,
) -> tuple[str, list[dict]]:
    """
    对单条用户输入运行完整的 Tool-use Loop，返回结果。

    Args:
        user_input    : 用户问题 / 指令
        system_prompt : 被测的 prompt 变体（Baseline / ReAct 等）
        mock_registry : 注入的 Mock 工具注册表（每次调用前已 reset）

    Returns:
        reply    : Agent 最终的自然语言回复
        call_log : 本次运行所有工具调用的列表，供 Grader 消费
    """
    mock_registry.reset_log()

    # 从干净的单条消息开始，不带任何对话历史
    # 这保证 case 间的 context 完全隔离
    messages: list[dict] = [{"role": "user", "content": user_input}]

    for _ in range(_MAX_LOOP_TURNS):
        response    = send_request(messages, tools=mock_registry.schemas,
                                   system=system_prompt)
        stop_reason = response.get("stop_reason")
        content     = response.get("content", [])

        # ── 情况一：模型完成回答 ─────────────────────────────────────────
        if stop_reason == "end_turn":
            reply = "".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
            return reply, mock_registry._call_log

        # ── 情况二：模型请求工具调用 ─────────────────────────────────────
        if stop_reason == "tool_use":
            # assistant 完整内容（含 tool_use blocks）必须进 messages
            # ⚠️ API 约束：tool_result 前必须有对应的 assistant turn
            messages.append({"role": "assistant", "content": content})

            tool_results = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                result = mock_registry.execute(block["name"], block["input"])
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block["id"],
                    "content":     json.dumps(result, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})
            continue

        break  # 未预期的 stop_reason

    return "[超过最大轮次，未完成]", mock_registry._call_log


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — 测试用例集
#
# 设计原则：每个 case 覆盖一个"最小能力单元"，
# 这样分数变化时能精确定位是哪项能力退步了。
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    id:          str
    description: str         # 人类可读的用例说明（出现在报告里）
    user_input:  str         # 发给 Agent 的原始输入

    # ── 确定性检查字段 ────────────────────────────────────────────────────
    # 期望调用的工具列表（顺序无关，支持重复同一工具）
    expected_tools: list[str] = field(default_factory=list)

    # 期望最终回复包含的关键词；至少命中 keyword_min_match 个才算通过
    expected_keywords:  list[str] = field(default_factory=list)
    keyword_min_match:  int = 1

    # ── LLM-as-Judge 字段 ─────────────────────────────────────────────────
    # rubric 要具体可操作，不能写"答案是否好"这种模糊描述
    rubric: str = ""


# 天气数据速查（写 expected_keywords 时参考）：
# 北京 12°多云45%  |  上海 18°小雨80%  |  广州 26°晴65%
# 成都 15°多云70%  |  杭州 17°小雨78%
#
# compare_weather 评分规则（同 m2_4_mini_agent）：
#   晴=3 多云=2 小雨=1  |  湿度<60 +1  |  温度15~25°C +1
# 所以：北京(多云2 +湿度1 =3) > 上海(小雨1 +温度1 =2) → 北京更适宜

EVAL_DATASET: list[EvalCase] = [
    # ── 基础能力：单工具调用 ────────────────────────────────────────────
    EvalCase(
        id="T01_single_weather",
        description="单城市天气查询（最基础的工具调用）",
        user_input="北京现在天气怎么样？",
        expected_tools=["get_weather"],
        expected_keywords=["北京", "12", "多云"],
        keyword_min_match=2,
        rubric=(
            "回答是否包含北京的具体天气数据（温度12°C、天气状况多云）？"
            "信息是否来自工具返回而非捏造？"
        ),
    ),
    EvalCase(
        id="T02_unsupported_city",
        description="不支持城市的优雅拒绝（边界处理）",
        user_input="纽约今天天气如何？",
        expected_tools=[],   # 可能调用 get_weather 但应得到 error 后向用户说明
        expected_keywords=["不支持", "无法", "纽约"],
        keyword_min_match=1,
        rubric=(
            "Agent 是否清楚告知无法查询纽约天气，且没有编造虚假天气数据？"
            "是否说明了支持的城市范围？"
        ),
    ),
    # ── 核心能力：多步推理 ───────────────────────────────────────────────
    EvalCase(
        id="T03_weather_comparison",
        description="两城市天气对比（需要 get_weather×2 + compare_weather，共3次工具调用）",
        user_input="北京和上海，哪个城市的天气更适合今天出游？",
        expected_tools=["get_weather", "get_weather", "compare_weather"],
        expected_keywords=["北京", "上海", "适宜", "推荐"],
        keyword_min_match=2,
        rubric=(
            "是否分别查询了北京和上海的天气？"
            "是否调用了 compare_weather 进行比较？"
            "结论是否正确（北京多云12°湿度低=3分 优于 上海小雨18°湿度高=2分）？"
        ),
    ),
    EvalCase(
        id="T04_hottest_city",
        description="最热城市判断（纯温度对比，不需要 compare_weather）",
        user_input="北京和广州哪个城市今天温度更高？",
        expected_tools=["get_weather", "get_weather"],
        expected_keywords=["广州", "26", "高"],
        keyword_min_match=2,
        rubric=(
            "是否正确指出广州（26°C）温度高于北京（12°C）？"
            "是否给出了具体温度数据作为依据？没有数据的结论不得分。"
        ),
    ),
    # ── 记忆能力 ─────────────────────────────────────────────────────────
    EvalCase(
        id="T05_save_fact",
        description="保存用户信息到长期记忆",
        user_input="请记住：我是一名 Go 开发者，喜欢在晴天出游。",
        expected_tools=["save_fact"],
        expected_keywords=["记住", "保存", "Go"],
        keyword_min_match=1,
        rubric=(
            "Agent 是否调用了 save_fact 工具？"
            "是否向用户确认已成功保存信息？"
            "内容是否完整（职业 + 偏好两个要点都保存了）？"
        ),
    ),
    EvalCase(
        id="T06_recall_empty_memory",
        description="记忆为空时的诚实应答（空记忆边界）",
        user_input="我之前告诉你我是做什么工作的吗？",
        expected_tools=["recall_facts"],
        expected_keywords=["记忆", "找到", "没有"],
        keyword_min_match=1,
        rubric=(
            "Agent 是否尝试调用 recall_facts 检索？"
            "在记忆为空的情况下，是否诚实告知用户未找到相关记忆，而非捏造信息？"
        ),
    ),
    # ── 边界能力 ─────────────────────────────────────────────────────────
    EvalCase(
        id="T07_out_of_scope",
        description="完全超出能力范围的任务（不应调用任何工具）",
        user_input="帮我查一下今天的苹果公司股票价格。",
        expected_tools=[],
        expected_keywords=["无法", "不支持", "股票"],
        keyword_min_match=1,
        rubric=(
            "Agent 是否明确说明自己没有股票查询能力？"
            "是否避免了调用任何工具试图完成该任务？"
            "是否没有捏造任何股价数据？"
        ),
    ),
    EvalCase(
        id="T08_cross_tool_multistep",
        description="跨工具类别的多步任务（查天气 → 保存结论，涉及两个工具模块）",
        user_input="查一下成都今天的天气，然后帮我把这个天气情况记到记忆里。",
        expected_tools=["get_weather", "save_fact"],
        expected_keywords=["成都", "15", "多云"],
        keyword_min_match=2,
        rubric=(
            "是否先查询了成都天气（15°C、多云）再保存？"
            "保存的内容是否包含了实际的天气数据而非泛泛描述？"
            "两个步骤是否都完成了？"
        ),
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# DeterministicGrader — 确定性评分器
#
# 职责边界：只回答"做了什么"，不回答"做得好不好"
# 特点：零 API 成本、毫秒级速度、结果 100% 可重复
# ══════════════════════════════════════════════════════════════════════════════

class DeterministicGrader:

    def grade_tool_selection(
        self,
        expected:     list[str],
        actual_calls: list[dict],
    ) -> float:
        """
        检查工具选择是否符合预期。

        评分逻辑（顺序无关，Counter 处理重复调用）：
          - expected 为空 → 期望"不调用工具"：actual 也为空得 1.0，否则得 0.0
          - 否则：每个期望工具命中 +1，未命中 0；多余的调用不扣分（容错）
            score = 命中数 / 期望总数

        🐍 Python 插播：Counter 是 dict 的子类，自动统计元素频次
          Counter(["a", "a", "b"]) → Counter({"a": 2, "b": 1})
          类比 Go：map[string]int 手动 ++
        """
        actual_names = [c["tool"] for c in actual_calls]

        if not expected:
            # 期望零工具调用
            return 1.0 if not actual_names else 0.0

        exp_counter = Counter(expected)
        act_counter = Counter(actual_names)

        matched = sum(
            min(exp_counter[tool], act_counter[tool])
            for tool in exp_counter
        )
        return matched / len(expected)

    def grade_keywords(
        self,
        expected_keywords: list[str],
        actual_output:     str,
        min_match:         int,
    ) -> float:
        """
        检查最终回复是否包含关键词（关键数据是否被引用）。

        min_match 是"最低及格线"：未达到则直接返回 0.0，
        达到后按命中比例给分（激励尽量多命中）。
        """
        if not expected_keywords:
            return 1.0

        matched = sum(1 for kw in expected_keywords if kw in actual_output)
        if matched < min_match:
            return 0.0
        return matched / len(expected_keywords)


# ══════════════════════════════════════════════════════════════════════════════
# LLMJudge — LLM-as-Judge 评分器
#
# 两个可靠性设计：
#   1. CoT Grading（先 reasoning 后 score）：
#      强迫模型在打分前先生成推理链，抑制直接打分时的随机波动。
#      类比 Chain-of-Thought：让模型"说出思考"比"直接给答案"更稳定。
#   2. 结构化 JSON 输出：避免自由文本解析的脆弱性。
#
# 已知局限（可靠性边界）：
#   - 事实正确性：Judge 可能和 Agent 有相同的幻觉，无法发现事实错误
#   - 自我评分偏差：同一模型评估自己的输出会偏高（当前 baseline 未规避）
#   - 生产建议：用更强的模型（Sonnet/Opus）做 Judge，避免 self-grading bias
# ══════════════════════════════════════════════════════════════════════════════

class LLMJudge:

    # Judge 的 system prompt：严格定义评分协议
    _JUDGE_SYSTEM = """\
你是一名专业的 AI Agent 输出质量评估员。
你的任务是根据给定的评分标准（rubric），客观评估 Agent 回复的质量。

【评分规则】
- 分数范围：0.0（完全不达标）到 1.0（完全达标）
- 常用中间值：0.3（勉强达标）/ 0.5（部分达标）/ 0.7（基本达标）/ 0.9（近乎完美）
- 必须先写出 2-3 句评分理由，再给出分数（顺序不可颠倒）
- 严格对照 rubric 评分，不要被回复的文字数量或语气影响

【输出格式】
严格输出以下 JSON，不要有任何其他内容（包括 markdown 代码块）：
{"reasoning": "你的评分理由（2-3句）", "score": 0.8}
"""

    def grade(self, case: EvalCase, actual_output: str) -> tuple[float, str]:
        """
        对 Agent 回复打分。

        Returns:
            (score, reasoning)  — score 已 clamp 到 [0.0, 1.0]
        """
        user_msg = (
            f"评分标准（rubric）：\n{case.rubric}\n\n"
            f"Agent 的实际回复：\n{actual_output}\n\n"
            "请按要求输出 JSON 格式的评分。"
        )

        try:
            resp = send_request(
                messages=[{"role": "user", "content": user_msg}],
                system=self._JUDGE_SYSTEM,
                temperature=0.0,   # Judge 要确定性，不要创意
                max_tokens=256,
            )
            raw = "".join(
                b.get("text", "") for b in resp.get("content", [])
                if b.get("type") == "text"
            )
            # 容错解析：有时模型会用 ```json ... ``` 包裹
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                parsed    = json.loads(m.group())
                score     = float(parsed.get("score", 0.0))
                reasoning = parsed.get("reasoning", "")
                return max(0.0, min(1.0, score)), reasoning

        except Exception as exc:
            return 0.0, f"Judge 调用失败：{exc}"

        return 0.0, "Judge 解析失败"


# ══════════════════════════════════════════════════════════════════════════════
# CaseResult — 单个用例的完整运行结果
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    case_id:              str
    description:          str
    tool_selection_score: float   # DeterministicGrader 输出
    keyword_score:        float   # DeterministicGrader 输出
    answer_quality_score: float   # LLMJudge 输出
    total_score:          float   # 加权平均
    actual_tool_calls:    list    # 调试信息：实际调用了哪些工具
    actual_output:        str     # Agent 完整回复（调试用）
    judge_reasoning:      str     # Judge 的评分理由（调试用）


# 评分权重配置
# 工具选择（0.4）> 答案质量（0.4）> 关键词（0.2）
# 工具选择权重高：过程正确比答案美观更重要，特别是在多步任务中
_SCORE_WEIGHTS = {
    "tool_selection": 0.4,
    "keyword":        0.2,
    "answer_quality": 0.4,
}


def _compute_total(r: CaseResult) -> float:
    return (
        r.tool_selection_score  * _SCORE_WEIGHTS["tool_selection"] +
        r.keyword_score         * _SCORE_WEIGHTS["keyword"] +
        r.answer_quality_score  * _SCORE_WEIGHTS["answer_quality"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# EvalHarness — 运行框架
#
# 类比 Go 的 testing 包：
#   Dataset 中每个 EvalCase ≈ func TestXxx(t *testing.T)
#   run_all() ≈ go test -v ./...
#   compare() ≈ go test -bench=. 的 benchmark 对比
# ══════════════════════════════════════════════════════════════════════════════

class EvalHarness:

    def __init__(self):
        self.det_grader = DeterministicGrader()
        self.judge      = LLMJudge()
        self.mock_reg   = MockToolRegistry()

    def run_case(
        self,
        case:          EvalCase,
        system_prompt: str,
        verbose:       bool = True,
    ) -> CaseResult:
        """运行单个 EvalCase，返回 CaseResult。"""
        if verbose:
            print(f"\n  ▶ [{case.id}] {case.description}")

        # ── Step 1: 运行 Agent ────────────────────────────────────────────
        reply, call_log = run_agent_once(case.user_input, system_prompt, self.mock_reg)

        if verbose:
            tools_called = [c["tool"] for c in call_log]
            print(f"     工具调用序列：{tools_called}")
            print(f"     回复（前100字）：{reply[:100].replace(chr(10), ' ')}...")

        # ── Step 2: 确定性评分（免费、快速）──────────────────────────────
        tool_score = self.det_grader.grade_tool_selection(
            case.expected_tools, call_log
        )
        kw_score = self.det_grader.grade_keywords(
            case.expected_keywords, reply, case.keyword_min_match
        )

        # ── Step 3: LLM-as-Judge 评分（付费、慢、但能评语义）─────────────
        judge_score, judge_reason = self.judge.grade(case, reply)

        # ── Step 4: 汇总 ─────────────────────────────────────────────────
        result = CaseResult(
            case_id=case.id,
            description=case.description,
            tool_selection_score=tool_score,
            keyword_score=kw_score,
            answer_quality_score=judge_score,
            total_score=0.0,          # 先占位，下一行计算
            actual_tool_calls=call_log,
            actual_output=reply,
            judge_reasoning=judge_reason,
        )
        result.total_score = _compute_total(result)

        if verbose:
            status = "✅" if result.total_score >= 0.6 else "❌"
            print(
                f"     {status} 工具:{tool_score:.2f} "
                f"关键词:{kw_score:.2f} "
                f"Judge:{judge_score:.2f} "
                f"总分:{result.total_score:.2f}"
            )
            if judge_reason:
                print(f"     Judge说：{judge_reason[:80]}...")

        return result

    def run_all(
        self,
        dataset:       list[EvalCase],
        system_prompt: str,
        label:         str = "Eval Run",
        verbose:       bool = True,
    ) -> list[CaseResult]:
        """运行全量 dataset，返回结果列表并打印汇总报告。"""
        print(f"\n{'═'*65}")
        print(f"🧪  {label}")
        print(f"{'═'*65}")

        results = []
        for case in dataset:
            result = self.run_case(case, system_prompt, verbose=verbose)
            results.append(result)
            # 避免触发 API 速率限制
            time.sleep(0.8)

        self._print_summary(results, label)
        return results

    def _print_summary(self, results: list[CaseResult], label: str) -> None:
        scores = [r.total_score for r in results]

        print(f"\n{'─'*65}")
        print(f"📊  {label} — 汇总报告")
        print(f"{'─'*65}")
        print(f"  {'用例ID':<28} {'工具':>5} {'关键词':>6} {'Judge':>6} {'总分':>6}")
        print(f"  {'─'*60}")

        for r in results:
            icon = "✅" if r.total_score >= 0.6 else "❌"
            print(
                f"  {icon} {r.case_id:<26} "
                f"{r.tool_selection_score:>5.2f} "
                f"{r.keyword_score:>6.2f} "
                f"{r.answer_quality_score:>6.2f} "
                f"{r.total_score:>6.2f}"
            )

        print(f"  {'─'*60}")
        avg = mean(scores)
        print(f"  平均总分：{avg:.3f}")
        if len(scores) > 1:
            print(f"  标准差：  {stdev(scores):.3f}")
        pass_rate = sum(1 for s in scores if s >= 0.6) / len(scores)
        print(f"  通过率（≥0.6）：{pass_rate:.0%}  ({sum(1 for s in scores if s >= 0.6)}/{len(scores)})")

    def compare(
        self,
        results_a: list[CaseResult],
        results_b: list[CaseResult],
        label_a:   str = "Baseline",
        label_b:   str = "Variant",
    ) -> None:
        """
        逐用例对比两组实验结果。

        Δ > +0.05 → ↑（Variant 更好）
        Δ < -0.05 → ↓（Variant 更差）
        否则      → →（无显著差异）

        注意：LLM-as-Judge 本身有随机性，Δ < 0.1 的差异在统计上不显著。
        建议：重要决策前，n_runs ≥ 3 取均值。
        """
        print(f"\n{'═'*65}")
        print(f"⚖️   对比实验：{label_a}  vs  {label_b}")
        print(f"{'═'*65}")
        print(f"  {'用例ID':<28} {label_a:>8} {label_b:>8} {'Δ':>8}")
        print(f"  {'─'*55}")

        map_a = {r.case_id: r for r in results_a}
        map_b = {r.case_id: r for r in results_b}
        deltas = []

        for cid in sorted(set(map_a) | set(map_b)):
            sa = map_a[cid].total_score if cid in map_a else 0.0
            sb = map_b[cid].total_score if cid in map_b else 0.0
            d  = sb - sa
            deltas.append(d)
            arrow = "↑" if d > 0.05 else ("↓" if d < -0.05 else "→")
            print(f"  {cid:<28} {sa:>8.3f} {sb:>8.3f} {arrow}{d:>+7.3f}")

        avg_a = mean(r.total_score for r in results_a)
        avg_b = mean(r.total_score for r in results_b)
        delta = avg_b - avg_a

        print(f"  {'─'*55}")
        print(f"  {'平均':<28} {avg_a:>8.3f} {avg_b:>8.3f} {delta:>+8.3f}")

        verdict = (
            f"✅ {label_b} 显著更好" if delta > 0.05 else
            f"❌ {label_b} 显著更差" if delta < -0.05 else
            f"→ 两者无显著差异"
        )
        print(f"\n  结论：{verdict}  (Δ = {delta:+.3f})")
        print("  ⚠️  LLM-as-Judge 有随机性，Δ < 0.1 的结论仅供参考，建议多次运行取均值。")


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt 变体 — 对比实验的两个被测对象
# ══════════════════════════════════════════════════════════════════════════════

# Baseline：与 M2.4 Mini Agent 相同（控制变量）
BASELINE_SYSTEM = """\
你是一个具备长期记忆和工具调用能力的智能助手。

【能力】
- 天气查询：可查询北京、上海、广州、成都、杭州的天气，并进行城市间比较
- 长期记忆：可保存和检索用户告知的重要信息

【工作原则】
1. 遇到需要工具的任务，直接调用，不要询问是否需要
2. 遇到多步骤任务（如"查天气并比较"），先按顺序完成所有步骤，再综合回答
3. 遇到超出能力的任务，诚实说明无法完成，不要编造答案
4. 回答要简洁、有依据，关键数据要体现在回答中
"""

# ReAct 变体：在 Baseline 基础上增加显式推理格式要求
# 唯一变量：增加了"思考/行动/观察总结"格式
REACT_SYSTEM = """\
你是一个具备长期记忆和工具调用能力的智能助手。

【能力】
- 天气查询：可查询北京、上海、广州、成都、杭州的天气，并进行城市间比较
- 长期记忆：可保存和检索用户告知的重要信息

【工作原则】
1. 遇到需要工具的任务，直接调用，不要询问是否需要
2. 遇到多步骤任务（如"查天气并比较"），先按顺序完成所有步骤，再综合回答
3. 遇到超出能力的任务，诚实说明无法完成，不要编造答案
4. 回答要简洁、有依据，关键数据要体现在回答中

【推理格式（ReAct 模式）】
在每次调用工具前，先说明你的推理（无需等待批准，直接调用）：
  思考：[分析当前任务，我需要做什么，为什么]
  行动：调用 xxx，因为...

在给出最终答案前，总结观察结果：
  观察总结：[基于工具返回的数据得出的结论]
"""


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    harness = EvalHarness()

    # ── 实验一：Baseline ───────────────────────────────────────────────────
    results_baseline = harness.run_all(
        EVAL_DATASET,
        system_prompt=BASELINE_SYSTEM,
        label="Baseline（M2.4 原版 Prompt）",
        verbose=True,
    )

    print("\n⏳ 等待 5 秒后运行 ReAct 变体...")
    time.sleep(5)

    # ── 实验二：ReAct Prompt 变体 ──────────────────────────────────────────
    results_react = harness.run_all(
        EVAL_DATASET,
        system_prompt=REACT_SYSTEM,
        label="ReAct（显式推理格式 Prompt）",
        verbose=True,
    )

    # ── 对比实验 ───────────────────────────────────────────────────────────
    harness.compare(
        results_baseline, results_react,
        label_a="Baseline", label_b="ReAct",
    )
