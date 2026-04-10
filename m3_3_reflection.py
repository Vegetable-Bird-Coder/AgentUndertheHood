"""
[M3.3] 反思机制（Reflection / Self-Critique）

核心思想：
  给任意 Agent 套上一层"自我 review"能力。
  ReflectionAgent 是一个装饰器（B-1 方案）：
    - 接受任意 callable 作为 inner_fn（可以是 FSMAgent.handle，也可以是任何函数）
    - 不修改 inner_fn 的内部逻辑，只在其输出上叠加 Reflection 层

完整执行流程：
  handle(request):
    1. recall_lessons(request)       → 检索历史教训，注入 context（冷启动为空）
    2. inner_fn(augmented_request)   → 生成 draft（利用历史教训）
    3. critique(draft, request)      → LLM 审查 draft，输出 issues
    4a. 无 issues  → 直接返回 draft
    4b. 有 issues  → save_lesson(...)  先保存教训（日志优先）
                   → revise(draft, issues)  再生成修订版
                   → 返回修订版

与 M3.2 FSMAgent 的关系：
  - FSMAgent.EVALUATING 验收的是"工具执行结果对不对"（工具层）
  - ReflectionAgent 审查的是"最终答案好不好"（输出层）
  - 两者职责不同，可以组合：ReflectionAgent(inner_fn=fsm_agent.handle)

Extended Thinking：
  - 通过 anthropic-beta: interleaved-thinking-2025-05-14 开启
  - 触发条件：启发式规则（任务包含推理关键词 OR token 超阈值）OR 用户显式传 flag
  - 用于 critique 阶段：让模型做更深入的审查

运行方式：
  pip install tiktoken jieba requests
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m3_3_reflection.py
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Callable

import jieba
import requests
import tiktoken

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与 M3.2 相同）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"
HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

# Extended Thinking 需要额外的 beta header
HEADERS_THINKING = {
    **HEADERS,
    "anthropic-beta": "interleaved-thinking-2025-05-14",
}

_ENCODER = tiktoken.get_encoding("cl100k_base")

# 触发 Extended Thinking 的 token 阈值（请求超过这个长度认为是复杂任务）
THINKING_TOKEN_THRESHOLD = 50
# Extended Thinking 的预算 token 数（模型用于内部推理的 token 上限）
THINKING_BUDGET_TOKENS   = 2000

# 触发 Extended Thinking 的关键词（命中任意一个则认为是推理型任务）
REASONING_KEYWORDS = [
    "为什么", "分析", "推断", "比较优劣", "如果", "应该",
    "怎么选", "哪个更好", "原因", "建议", "利弊", "评估",
]

# ReflectionStore 持久化路径
REFLECTION_STORE_PATH = os.path.join(os.path.dirname(__file__), "reflections.json")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def send_request(
    messages:    list,
    tools:       list  = None,
    system:      str   = "",
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
    use_thinking: bool = False,
) -> dict:
    """
    统一请求入口。use_thinking=True 时启用 Extended Thinking。

    Extended Thinking 的 API 差异：
      - headers 多一个 anthropic-beta
      - body 多一个 thinking block：{"type": "thinking", "budget_tokens": N}
      - temperature 必须 >= 1（API 强制要求，thinking 模式下不支持低温）
      - 响应 content 里会多出 type="thinking" 的 block
    """
    headers = HEADERS_THINKING if use_thinking else HEADERS

    body = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "stream":     False,
        "messages":   messages,
    }

    if use_thinking:
        # Extended Thinking 模式：temperature 固定为 1（API 要求）
        body["thinking"] = {
            "type":         "enabled",
            "budget_tokens": THINKING_BUDGET_TOKENS,
        }
        body["temperature"] = 1
    else:
        body["temperature"] = temperature

    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools

    response = requests.post(API_URL, headers=headers, json=body)
    response.raise_for_status()
    return response.json()


def extract_text(response: dict) -> str:
    """从 API 响应中提取所有 text block 拼接成字符串。"""
    return "".join(
        b.get("text", "") for b in response.get("content", [])
        if b.get("type") == "text"
    )


def extract_thinking_content(response: dict) -> str:
    """
    从 Extended Thinking 响应中提取 thinking block 内容。

    Extended Thinking 响应的 content 结构：
      [
        {"type": "thinking", "thinking": "模型的内部推理过程..."},
        {"type": "text",     "text":     "最终输出..."},
      ]

    thinking block 对用户不可见，但我们可以打印出来用于调试和教学。
    """
    return "".join(
        b.get("thinking", "") for b in response.get("content", [])
        if b.get("type") == "thinking"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ★ ReflectionStore：持久化 Agent 的历史失败经验
# ══════════════════════════════════════════════════════════════════════════════

class ReflectionStore:
    """
    存储 Agent 自我反思后提炼的经验教训。

    与 FactStore 的核心区别：
      FactStore  → 存用户世界的事实（"用户住杭州"）
      ReflectionStore → 存 Agent 自身行为的教训（"比较任务必须给结论"）

    Schema:
      {
        "id":        str,   # 唯一标识
        "failure":   str,   # 这次犯的错误描述（critique 原文）
        "lesson":    str,   # 提炼的行动指南（注入给下次 draft 生成）
        "timestamp": str,   # 写入时间
      }

    去掉 task_type 的原因：
      - task_type 只用于检索时的字段匹配
      - 但 recall_lessons 直接在 lesson 全文做分词匹配，task_type 冗余
      - 去掉后：schema 更简单，检索逻辑更直接，无精度损失
    """

    def __init__(self, path: str = REFLECTION_STORE_PATH):
        self._path    = path
        self._lessons: list[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                self._lessons = json.load(f)

    def _persist(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._lessons, f, ensure_ascii=False, indent=2)

    def save_lesson(self, failure: str, lesson: str) -> dict:
        """
        保存一条经验教训。

        注意：调用时机是 critique 发现问题之后、revise 之前。
        原因：先记日志再处理，保证即使 revise 失败教训也不丢失。
        """
        entry = {
            "id":        str(uuid.uuid4())[:8],
            "failure":   failure,
            "lesson":    lesson,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._lessons.append(entry)
        self._persist()
        return entry

    def recall_lessons(self, query: str, max_lessons: int = 3) -> list[dict]:
        """
        检索与当前任务相关的历史教训。

        检索时机：在生成 draft 之前。
        检索逻辑：jieba 分词 OR 匹配，直接在 lesson 全文里匹配。
          去掉 task_type 后，检索更直接：lesson 字段本身就包含了任务语义。
          例如 lesson="应该在比较天气时给出明确推荐" 里包含"比较"、"天气"、"推荐"，
          用户问"北京上海哪个更好"时，"比较"会自然命中。
        """
        if not self._lessons:
            return []

        query_words = set(jieba.cut(query))

        matched = []
        for entry in self._lessons:
            # 只在 lesson 全文里检索（去掉 task_type 拼接）
            lesson_words = set(jieba.cut(entry["lesson"]))
            if query_words & lesson_words:
                matched.append(entry)

        matched.sort(key=lambda x: x["timestamp"], reverse=True)
        return matched[:max_lessons]

    def all_lessons(self) -> list[dict]:
        return list(self._lessons)


# ══════════════════════════════════════════════════════════════════════════════
# ★ Extended Thinking 触发判断
# ══════════════════════════════════════════════════════════════════════════════

def should_use_extended_thinking(request: str, force: bool = False) -> bool:
    """
    判断是否为当前请求开启 Extended Thinking。

    优先级链（高到低）：
      1. force=True  → 用户显式开启，直接返回 True
      2. 包含推理关键词 → 启发式命中，返回 True
      3. 请求超过 token 阈值 → 复杂任务，返回 True
      4. 其余 → 不开启

    为什么不用 LLM 自判断：
      LLM 判断本身就是一次调用。如果它能判断"这个任务很难"，
      那它已经在思考了——不如直接用 extended thinking 做正事。
    """
    if force:
        return True

    # 规则 1：推理关键词命中
    if any(kw in request for kw in REASONING_KEYWORDS):
        return True

    # 规则 2：请求长度超阈值
    if count_tokens(request) > THINKING_TOKEN_THRESHOLD:
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# ★ ReflectionAgent：核心实现
# ══════════════════════════════════════════════════════════════════════════════

class ReflectionAgent:
    """
    反思装饰器（Decorator Pattern）。

    接受任意 callable 作为 inner_fn，在其输出上叠加反思层。
    不修改 inner_fn 的内部逻辑——这是"可插拔"的关键。

    用法：
      fsm_agent = FSMAgent()
      agent = ReflectionAgent(inner_fn=fsm_agent.handle)
      response = agent.handle("北京今天适合出行吗？")

    或者包装任意函数：
      def simple_llm(request): ...
      agent = ReflectionAgent(inner_fn=simple_llm)
    """

    def __init__(
        self,
        inner_fn: Callable[[str], str],
        store_path: str = REFLECTION_STORE_PATH,
    ):
        # inner_fn 是任意接受 str 返回 str 的 callable
        # 类比 Go 的函数类型：type HandlerFn func(string) string
        self.inner_fn = inner_fn
        self.store    = ReflectionStore(path=store_path)

    def handle(self, request: str, force_thinking: bool = False) -> str:
        """
        主入口：带反思的完整执行流程。

        force_thinking=True 可由用户在交互时显式触发 Extended Thinking。
        """
        use_thinking = should_use_extended_thinking(request, force=force_thinking)

        print(f"\n{'─'*50}")
        print(f"  🔮 Extended Thinking: {'开启' if use_thinking else '关闭'}")
        print(f"{'─'*50}")

        # ── Step 1：检索历史教训 ──────────────────────────────────────────────
        lessons = self.store.recall_lessons(request)
        augmented_request = self._augment_with_lessons(request, lessons)

        if lessons:
            print(f"\n  📚 注入历史教训（{len(lessons)} 条）：")
            for l in lessons:
                print(f"     {l['lesson']}")

        # ── Step 2：调用 inner_fn 生成 draft ─────────────────────────────────
        print(f"\n  📝 生成 Draft...")
        draft = self.inner_fn(augmented_request)
        print(f"\n  Draft: {draft}")

        # ── Step 3：Critique ──────────────────────────────────────────────────
        print(f"\n  🔍 Critique 审查中{'（Extended Thinking）' if use_thinking else ''}...")
        issues, thinking_log = self._critique(draft, request, use_thinking)

        if thinking_log:
            print(f"\n  ┌─ Extended Thinking 内容")
            for line in thinking_log.split("\n")[:10]:   # 只打印前 10 行避免刷屏
                if line.strip():
                    print(f"  │  {line.strip()}")
            print(f"  └─────────────────────────────")

        # ── Step 4a：无问题，直接返回 ────────────────────────────────────────
        if not issues:
            print(f"\n  ✅ Critique 通过，无需修订")
            return draft

        # ── Step 4b：有问题，先存教训再修订 ──────────────────────────────────
        print(f"\n  ⚠️  Critique 发现问题：{issues}")

        # 合并调用：一次 LLM 同时提炼教训 + 生成修订版
        # 原来是两次调用（distill_lesson + revise），现在合并为一次
        # 权衡：略微降低每件事的专注度，换来少一次 API 调用
        print(f"\n  ✏️  提炼教训 + 修订中（合并调用）...")
        lesson, revised = self._critique_and_revise(draft, issues, request)

        # 先存教训（日志优先原则：即使后续出错，教训已落盘）
        entry = self.store.save_lesson(failure=issues, lesson=lesson)
        print(f"\n  💾 教训已保存 [{entry['id']}]：{entry['lesson']}")
        print(f"\n  修订版: {revised}")

        return revised

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _augment_with_lessons(self, request: str, lessons: list[dict]) -> str:
        """
        把历史教训注入到请求里，让 inner_fn 在生成 draft 时就能参考。

        注入方式：在请求前追加一段"注意事项"。
        这比在 inner_fn 内部修改 system prompt 更解耦——inner_fn 不需要知道
        Reflection 的存在，只是看到了一个"更详细的请求"。
        """
        if not lessons:
            return request

        notes = "\n".join(
            f"  - {l['lesson']}"
            for l in lessons
        )
        return (
            f"{request}\n\n"
            f"【历史经验注意事项（请在回答时参考）】\n{notes}"
        )

    def _critique(
        self, draft: str, original_request: str, use_thinking: bool
    ) -> tuple[str, str]:
        """
        让 LLM 审查 draft，输出发现的问题。

        返回：(issues_str, thinking_content)
          - issues_str：空字符串表示无问题；非空描述具体问题
          - thinking_content：Extended Thinking 的内部推理（普通模式为空）

        Critic 的 System Prompt 设计原则：
          1. 明确告知审查维度（完整性、准确性、有无明确结论）
          2. 无问题时输出固定字符串"无问题"，方便代码判断
          3. 有问题时简洁描述，不要冗长解释
        """
        system = """\
你是一个严格的质量审查员。
审查给定的回答是否存在以下问题：
  1. 信息不完整（遗漏了用户需要的关键信息）
  2. 逻辑错误或前后矛盾
  3. 比较类问题没有给出明确结论/推荐
  4. 回答与用户问题不符

输出格式：
  - 如果回答质量合格，只输出：无问题
  - 如果有问题，一句话描述最主要的问题（不超过50字）
不要输出任何其他内容。"""

        prompt = f"用户问题：{original_request}\n\n待审查回答：{draft}"

        response = send_request(
            messages     = [{"role": "user", "content": prompt}],
            system       = system,
            use_thinking = use_thinking,
            max_tokens   = THINKING_BUDGET_TOKENS + 256 if use_thinking else 256,
        )

        verdict      = extract_text(response).strip()
        thinking_log = extract_thinking_content(response) if use_thinking else ""

        # "无问题" 视为通过，其余视为有问题
        issues = "" if verdict == "无问题" else verdict
        return issues, thinking_log

    def _critique_and_revise(
        self, draft: str, issues: str, original_request: str
    ) -> tuple[str, str]:
        """
        合并调用：一次 LLM 同时完成两件事：
          1. 把 critique 发现的问题提炼为可复用的行动指南（lesson）
          2. 生成修订版回答（revised）

        返回：(lesson, revised)

        为什么合并：
          原来 _distill_lesson + _revise 是两次独立调用，输入几乎相同（都是 issues）。
          合并后少一次 API 调用，速度更快，成本更低。
          代价：模型同时做两件事，每件事的输出质量可能略低于专注单任务。
          在这个场景下影响可忽略——lesson 只需 30 字，revised 只需修复已知问题。

        输出格式约定为 JSON，用 extract_json 解析（和 M2.2/M3.2 一致的健壮处理）：
          {
            "lesson":  "应该...",      # 行动指南，30字内
            "revised": "修订后的回答"   # 完整修订版
          }
        """
        system = """\
你是一个智能助手，同时负责经验提炼和回答修订。
根据给定的错误描述，完成两件事：
  1. lesson：把错误提炼为一条行动指南（"应该..."开头，不超过30字）
  2. revised：修订原始回答，修复错误，给出准确、完整、有明确结论的回答

严格输出 JSON，不要加任何 markdown 或注释：
{"lesson": "...", "revised": "..."}"""

        prompt = (
            f"用户问题：{original_request}\n\n"
            f"原始回答：{draft}\n\n"
            f"发现的错误：{issues}\n\n"
            f"请输出 JSON。"
        )

        response = send_request(
            messages    = [{"role": "user", "content": prompt}],
            system      = system,
            temperature = 0.3,
            max_tokens  = 600,
        )
        raw = extract_text(response).strip()

        # 健壮解析：剥掉可能的 markdown 包裹再 JSON 解析
        # 模型有时会忘记 "严格输出 JSON" 的指令，套一层 ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
            lesson  = parsed.get("lesson",  f"应该避免：{issues[:20]}")
            revised = parsed.get("revised", draft)
        except json.JSONDecodeError:
            # 解析失败时降级：lesson 用 issues 截断，revised 保留原 draft
            # 不抛异常——教训存储和回答输出都不应因格式问题中断
            lesson  = f"应该避免：{issues[:20]}"
            revised = draft

        return lesson, revised

    def show_lessons(self) -> None:
        """打印所有历史教训，用于调试和观察学习效果。"""
        lessons = self.store.all_lessons()
        if not lessons:
            print("  📚 暂无历史教训")
            return
        print(f"\n  📚 历史教训（{len(lessons)} 条）：")
        for l in lessons:
            print(f"     [{l['id']}] {l['lesson']}")
            print(f"            失败原因：{l['failure']}")
            print(f"            时间：{l['timestamp'][:19]}")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 用的简单 inner_fn（不依赖 FSMAgent，方便独立运行测试）
# ══════════════════════════════════════════════════════════════════════════════

# 模拟天气数据（和 M3.2 相同）
_WEATHER_DB = {
    "北京": {"temp": 15, "condition": "晴",   "humidity": 30, "wind": "北风3级"},
    "上海": {"temp": 22, "condition": "多云", "humidity": 65, "wind": "东风2级"},
    "广州": {"temp": 28, "condition": "小雨", "humidity": 80, "wind": "南风1级"},
    "成都": {"temp": 18, "condition": "阴",   "humidity": 70, "wind": "无风"},
    "杭州": {"temp": 20, "condition": "晴",   "humidity": 55, "wind": "东南风2级"},
}


def simple_weather_agent(request: str) -> str:
    """
    一个故意写得"不完整"的简单 Agent，用于演示 Reflection 能发现并修复什么。

    故意的缺陷：
      - 比较类问题只给数据，不给明确推荐（Reflection 应该能发现这个）
      - 不分析风力对出行的影响
    """
    system = """\
你是一个天气助手。根据以下天气数据回答用户问题。
只报告数据，不给推荐意见。

天气数据：
""" + json.dumps(_WEATHER_DB, ensure_ascii=False, indent=2)

    response = send_request(
        messages = [{"role": "user", "content": request}],
        system   = system,
        max_tokens = 256,
    )
    return extract_text(response).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🤖 Reflection Agent 启动")
    print("   inner_fn: simple_weather_agent（故意设计有缺陷）")
    print(f"   Extended Thinking 阈值: {THINKING_TOKEN_THRESHOLD} tokens / 推理关键词")
    print("\n   内置命令：")
    print("     quit      → 退出")
    print("     lessons   → 查看所有历史教训")
    print("     think: XXX → 强制开启 Extended Thinking 回答 XXX")
    print("=" * 60)

    agent = ReflectionAgent(inner_fn=simple_weather_agent)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("👋 再见！")
            break
        if user_input.lower() == "lessons":
            agent.show_lessons()
            continue

        # 支持 "think: XXX" 语法强制开启 Extended Thinking
        force_thinking = False
        if user_input.lower().startswith("think:"):
            force_thinking = True
            user_input = user_input[6:].strip()
            print(f"  （强制开启 Extended Thinking）")

        result = agent.handle(user_input, force_thinking=force_thinking)

        print(f"\n{'='*50}")
        print(f"最终回答: {result}")
        print(f"{'='*50}")


if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)
    main()
