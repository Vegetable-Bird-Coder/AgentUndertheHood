"""
[M3.1] ReAct Agent — Reasoning + Acting 范式

核心改动（相对于 M2.4 Mini Agent）：
  1. System Prompt  : 加入 ReAct 格式要求，要求模型在每次 Action 前输出 <thinking>
  2. 响应解析        : extract_thinking() 从 text block 中提取 <thinking> 内容
  3. 日志输出        : 每一步打印 Thought / Action / Observation，可观测性大幅提升

架构不变：
  - Tool-use Loop 结构与 M2.4 完全相同
  - ConversationBuffer / FactStore / ToolRegistry 原封不动复用
  - 工具模块（天气 + 记忆）完全不变

运行方式：
  pip install tiktoken jieba requests
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m3_1_react.py
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

import jieba
import requests
import tiktoken

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与 M2.4 相同）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"
HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

MAX_LOOP_TURNS = 10

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


# ══════════════════════════════════════════════════════════════════════════════
# Memory 组件（从 M2.3 / M2.4 原封不动复制）
# ══════════════════════════════════════════════════════════════════════════════

MAX_BUFFER_TOKENS  = 2000
MAX_SUMMARY_CHUNKS = 3
FACT_STORE_PATH    = os.path.join(os.path.dirname(__file__), "facts_react.json")

_COMPRESSOR_SYSTEM = """你是一个对话摘要器。
将以下对话历史压缩成一段简洁的摘要，保留所有关键信息：用户的问题、助手的回答要点、提到的关键事实。
输出格式：纯文本，不超过200字，不需要任何前缀或标题。"""


class ConversationBuffer:
    def __init__(self):
        self._messages: list[dict] = []
        self._summary_chunks: list[str] = []
        self._summary: str = ""

    def add(self, role: str, content) -> None:
        self._messages.append({"role": role, "content": content})
        self._compress_if_needed()

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def get_summary(self) -> str:
        return self._summary

    def total_tokens(self) -> int:
        return sum(
            count_tokens(
                m["content"] if isinstance(m["content"], str)
                else json.dumps(m["content"], ensure_ascii=False)
            )
            for m in self._messages
        )

    def _find_split_index(self) -> int:
        total, target, accumulated = self.total_tokens(), self.total_tokens() // 2, 0
        for i, m in enumerate(self._messages):
            accumulated += count_tokens(
                m["content"] if isinstance(m["content"], str)
                else json.dumps(m["content"], ensure_ascii=False)
            )
            if accumulated >= target:
                return i + 1
        return len(self._messages)

    def _compress_if_needed(self) -> None:
        if self.total_tokens() <= MAX_BUFFER_TOKENS:
            return
        split = self._find_split_index()
        to_compress = self._messages[:split]
        self._messages = self._messages[split:]
        history_text = "\n".join(
            f"{m['role']}: {m['content'] if isinstance(m['content'], str) else json.dumps(m['content'], ensure_ascii=False)}"
            for m in to_compress
        )
        resp = send_request(
            [{"role": "user", "content": history_text}],
            system=_COMPRESSOR_SYSTEM,
        )
        new_chunk = "".join(
            b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text"
        )
        self._summary_chunks.append(new_chunk)
        if len(self._summary_chunks) > MAX_SUMMARY_CHUNKS:
            self._summary_chunks.pop(0)
        self._summary = " | ".join(self._summary_chunks)


class FactStore:
    def __init__(self, path: str = FACT_STORE_PATH):
        self._path = path
        self._facts: list[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                self._facts = json.load(f)

    def _persist(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, ensure_ascii=False, indent=2)

    def save(self, content: str, tags: list[str] = None) -> dict:
        fact = {
            "id":         str(uuid.uuid4())[:8],
            "content":    content,
            "tags":       tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding":  None,
        }
        self._facts.append(fact)
        self._persist()
        return fact

    def recall(self, query: str, max_results: int = 5) -> list[dict]:
        if not self._facts:
            return []
        query_words = set(jieba.cut(query.lower()))
        scored = []
        for fact in self._facts:
            fact_words = set(jieba.cut(fact["content"].lower()))
            tag_words  = set(w.lower() for t in fact["tags"] for w in jieba.cut(t))
            score = len(query_words & (fact_words | tag_words))
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:max_results]]

    def all_facts(self) -> list[dict]:
        return list(self._facts)

    def as_context_string(self, max_facts: int = 20) -> str:
        if not self._facts:
            return ""
        recent = self._facts[-max_facts:]
        lines = ["[已知信息]"] + [f"- {f['content']}" for f in recent]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 工具模块（与 M2.4 完全相同，不改动）
# ══════════════════════════════════════════════════════════════════════════════

_WEATHER_DB = {
    "北京": {"temp": 15, "condition": "晴",   "humidity": 30, "wind": "北风3级"},
    "上海": {"temp": 22, "condition": "多云", "humidity": 65, "wind": "东风2级"},
    "广州": {"temp": 28, "condition": "小雨", "humidity": 80, "wind": "南风1级"},
    "成都": {"temp": 18, "condition": "阴",   "humidity": 70, "wind": "无风"},
    "杭州": {"temp": 20, "condition": "晴",   "humidity": 55, "wind": "东南风2级"},
}

_WEATHER_SCHEMAS = [
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

_MEMORY_SCHEMAS = [
    {
        "name": "save_fact",
        "description": "将重要信息永久保存到长期记忆。用于保存用户透露的个人信息、偏好、重要事实。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要保存的事实内容"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签列表，用于后续检索（如 ['天气', '北京']）",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_facts",
        "description": "从长期记忆中检索与查询相关的信息。在回答涉及用户个人信息的问题前调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题描述"}
            },
            "required": ["query"],
        },
    },
]


def weather_module() -> tuple[list, dict]:
    def _exec_get_weather(city: str) -> dict:
        if city not in _WEATHER_DB:
            return {"status": "error", "message": f"不支持的城市：{city}"}
        data = _WEATHER_DB[city]
        return {"status": "ok", "city": city, **data}

    def _exec_compare_weather(city_a: str, city_b: str) -> dict:
        if city_a not in _WEATHER_DB:
            return {"status": "error", "message": f"不支持的城市：{city_a}"}
        if city_b not in _WEATHER_DB:
            return {"status": "error", "message": f"不支持的城市：{city_b}"}
        a, b = _WEATHER_DB[city_a], _WEATHER_DB[city_b]
        better = city_a if a["condition"] in ["晴", "多云"] and b["condition"] not in ["晴", "多云"] \
            else city_b if b["condition"] in ["晴", "多云"] \
            else city_a
        return {
            "status":     "ok",
            "city_a":     city_a, "weather_a": a,
            "city_b":     city_b, "weather_b": b,
            "suggestion": f"{better} 天气更适合出行",
        }

    return _WEATHER_SCHEMAS, {
        "get_weather":     _exec_get_weather,
        "compare_weather": _exec_compare_weather,
    }


def memory_module(fact_store: FactStore) -> tuple[list, dict]:
    def _exec_save_fact(content: str, tags: list = None) -> dict:
        fact = fact_store.save(content, tags or [])
        return {"status": "ok", "message": f"已保存：{fact['content']}", "id": fact["id"]}

    def _exec_recall_facts(query: str) -> dict:
        facts = fact_store.recall(query)
        if not facts:
            return {"status": "ok", "facts": [], "message": "未找到相关记忆"}
        return {
            "status": "ok",
            "facts": [{"content": f["content"], "tags": f["tags"]} for f in facts],
        }

    return _MEMORY_SCHEMAS, {
        "save_fact":    _exec_save_fact,
        "recall_facts": _exec_recall_facts,
    }


class ToolRegistry:
    def __init__(self):
        self._schemas:  list[dict]          = []
        self._impl_map: dict[str, callable] = {}

    def register(self, schemas: list[dict], impl_map: dict) -> None:
        for schema in schemas:
            name = schema["name"]
            if name in self._impl_map:
                raise ValueError(f"工具名冲突：'{name}' 已注册")
        self._schemas.extend(schemas)
        self._impl_map.update(impl_map)

    @property
    def schemas(self) -> list[dict]:
        return list(self._schemas)

    def execute(self, name: str, input_data: dict) -> dict:
        if name not in self._impl_map:
            return {"status": "error",
                    "message": f"未知工具：{name}。可用：{list(self._impl_map.keys())}"}
        try:
            return self._impl_map[name](**input_data)
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# ★ ReAct 核心：提取 <thinking> 内容
# ══════════════════════════════════════════════════════════════════════════════

def extract_thinking(text: str) -> tuple[str, str]:
    """
    从模型输出的 text block 中分离出 <thinking> 和正文。

    返回：(thinking_content, remaining_text)

    示例：
      输入: "<thinking>需要先查北京天气</thinking>\n好的，我来查询。"
      输出: ("需要先查北京天气", "好的，我来查询。")

    为什么用正则而不是 XML 解析？
      - 模型输出不保证严格 XML 合法性（可能有多余空格/换行）
      - re.DOTALL 让 . 匹配换行，处理多行 thinking
      - 类比 Go: regexp.MustCompile(`(?s)<thinking>(.*?)</thinking>`)
    """
    pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
    match   = pattern.search(text)
    if not match:
        return "", text.strip()

    thinking = match.group(1).strip()
    remaining = pattern.sub("", text).strip()
    return thinking, remaining


# ══════════════════════════════════════════════════════════════════════════════
# ★ ReAct System Prompt
# ══════════════════════════════════════════════════════════════════════════════

# 与 M2.4 的区别：加入了 ReAct 格式要求（第一项行为规范）
# 其余能力描述、记忆规范保持不变——最小化改动，方便对比效果
_REACT_PERSONA = """\
你是一个具备长期记忆和工具调用能力的智能助手，采用 ReAct 推理模式。

【ReAct 格式要求】
在每次调用工具之前，必须先输出你的推理过程：

<thinking>
[在这里写你的推理：为什么需要这个工具？期望得到什么？这一步对整体任务有什么意义？]
</thinking>

在获得工具结果后、给出最终回答之前，也输出你的分析：

<thinking>
[分析工具返回的结果：是否符合预期？是否需要继续调用其他工具？如何根据结果调整计划？]
</thinking>

注意：<thinking> 只写推理过程，不写给用户看的内容。最终回答写在 <thinking> 块之外。

【能力】
- 天气查询：可查询北京、上海、广州、成都、杭州的天气，并进行城市间比较
- 长期记忆：可保存和检索跨对话的重要信息

【行为规范】
1. 严格遵守 ReAct 格式：工具调用前必须有 <thinking>
2. 用户透露个人信息（职业/偏好/城市/背景）时，主动调用 save_fact 保存
3. 用户询问你之前了解的信息时，先调用 recall_facts 检索再回答
4. 工具调用失败时，在 <thinking> 中分析失败原因，诚实告知用户
5. 用中文回答，语言自然友好，简洁直接\
"""


def build_system_prompt(buffer: ConversationBuffer, fact_store: FactStore) -> str:
    parts = [_REACT_PERSONA]
    facts_str = fact_store.as_context_string()
    if facts_str:
        parts.append(facts_str)
    summary = buffer.get_summary()
    if summary:
        parts.append(f"[历史对话摘要]\n{summary}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ★ ReAct Agent 主循环
# ══════════════════════════════════════════════════════════════════════════════

class ReActAgent:
    """
    ReAct Agent：在 Mini Agent 基础上，让推理过程可见。

    与 M2.4 MiniAgent 的核心差异（只有两处）：
      1. System Prompt 加入 ReAct 格式要求
      2. _agent_turn() 里增加 _print_react_step() 打印推理轨迹

    其余逻辑（buffer / fact_store / registry / 工具执行）完全相同。
    这证明了 ReAct 是一个"格式层"改造，而非架构重写。
    """

    def __init__(self, fact_store_path: str = FACT_STORE_PATH):
        self.buffer     = ConversationBuffer()
        self.fact_store = FactStore(path=fact_store_path)
        self.registry   = ToolRegistry()
        self.registry.register(*weather_module())
        self.registry.register(*memory_module(self.fact_store))

        # ReAct 轨迹计数器：用于打印 Step N 标签，让执行过程更清晰
        self._step = 0

    def run(self) -> None:
        self._print_startup_info()

        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if user_input.lower() == "quit":
                print("👋 再见！")
                break
            if user_input.lower() == "facts":
                self._show_facts()
                continue
            if user_input.lower() == "status":
                self._show_status()
                continue
            if not user_input:
                continue

            # 每次新用户输入，重置步骤计数器
            self._step = 0
            self.buffer.add("user", user_input)
            self._agent_turn()

    def _agent_turn(self) -> None:
        """
        ReAct 版 Agent Turn。

        与 M2.4 _agent_turn() 相比，只增加了两处：
          A. end_turn 时：提取并打印最终 thinking（如果有）
          B. tool_use 时：提取并打印每次调用前的 thinking

        工具执行逻辑、buffer 管理、循环结构完全不变。
        """
        for turn in range(MAX_LOOP_TURNS):
            system_prompt = build_system_prompt(self.buffer, self.fact_store)
            messages      = self.buffer.get_messages()
            response      = send_request(messages, tools=self.registry.schemas,
                                         system=system_prompt)
            stop_reason = response.get("stop_reason")
            content     = response.get("content", [])

            # ── 情况一：正常结束 ────────────────────────────────────────────
            if stop_reason == "end_turn":
                full_text = "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                # ★ ReAct 新增：提取 thinking 和正文
                thinking, reply = extract_thinking(full_text)

                if thinking:
                    self._step += 1
                    self._print_thinking(thinking)

                print(f"\n助手: {reply}")
                # buffer 只存正文，不存 <thinking> 标签
                # 原因：<thinking> 是调试信息，不应污染对话历史
                self.buffer.add("assistant", reply)
                return

            # ── 情况二：需要调用工具 ────────────────────────────────────────
            if stop_reason == "tool_use":
                # ★ ReAct 新增：提取工具调用前的 thinking
                full_text = "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                thinking, _ = extract_thinking(full_text)
                if thinking:
                    self._step += 1
                    self._print_thinking(thinking)

                # assistant 完整回复（含 tool_use blocks）存入 buffer
                # ⚠️ 注意：存原始 content（含 <thinking> 标签），因为 API 要求严格配对
                # buffer 里的 assistant turn 必须和 tool_result 的 tool_use_id 对应
                self.buffer.add("assistant", content)

                # 执行所有工具调用
                tool_results = []
                for block in content:
                    if block.get("type") != "tool_use":
                        continue

                    name        = block["name"]
                    input_data  = block["input"]
                    tool_use_id = block["id"]

                    # ★ ReAct 日志：打印 Action
                    self._print_action(name, input_data)

                    result = self.registry.execute(name, input_data)

                    # ★ ReAct 日志：打印 Observation
                    self._print_observation(result)

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_use_id,
                        "content":     json.dumps(result, ensure_ascii=False),
                    })

                self.buffer.add("user", tool_results)
                continue

            # ── 情况三：未预期的 stop_reason ───────────────────────────────
            print(f"  ⚠️  未处理的 stop_reason: {stop_reason}")
            return

        print(f"  ⚠️  超过最大工具调用轮次（{MAX_LOOP_TURNS}），本轮终止")

    # ── ReAct 可视化打印 ──────────────────────────────────────────────────────

    def _print_thinking(self, thinking: str) -> None:
        """
        打印 Thought 步骤。

        格式设计：
          [Step N | Thought] 让用户能追踪推理链的顺序
          缩进 + 颜色分隔，区别于 Action/Observation
        """
        print(f"\n  ┌─ Step {self._step} | 💭 Thought")
        for line in thinking.split("\n"):
            if line.strip():
                print(f"  │  {line.strip()}")
        print(f"  └─────────────────────────────")

    def _print_action(self, name: str, input_data: dict) -> None:
        """打印 Action 步骤（工具调用）。"""
        args_str = json.dumps(input_data, ensure_ascii=False)
        print(f"  ⚡ Action  → {name}({args_str})")

    def _print_observation(self, result: dict) -> None:
        """打印 Observation 步骤（工具返回结果）。"""
        status_icon = "✅" if result.get("status") == "ok" else "❌"
        result_str  = json.dumps(result, ensure_ascii=False)
        print(f"  {status_icon} Observe → {result_str}")

    # ── 调试辅助方法（与 M2.4 相同）────────────────────────────────────────

    def _print_startup_info(self) -> None:
        existing = self.fact_store.all_facts()
        tools    = [s["name"] for s in self.registry.schemas]
        print("\n" + "=" * 60)
        print("🤖 ReAct Agent 启动")
        print(f"   已注册工具：{tools}")
        if existing:
            print(f"   从磁盘加载了 {len(existing)} 条历史记忆：")
            for f in existing:
                print(f"     - {f['content']}")
        else:
            print("   记忆为空（首次启动）")
        print()
        print("   内置命令：facts | status | quit")
        print("=" * 60)

    def _show_facts(self) -> None:
        facts = self.fact_store.all_facts()
        if facts:
            print(f"\n📚 当前记忆（{len(facts)} 条）：")
            for f in facts:
                print(f"  [{f['id']}] {f['content']} | 标签: {f['tags']}")
        else:
            print("\n📚 记忆为空")

    def _show_status(self) -> None:
        print(f"\n📊 Buffer 状态：")
        print(f"   消息数：{len(self.buffer.get_messages())}")
        print(f"   Token 数：{self.buffer.total_tokens()}")
        print(f"   摘要长度：{len(self.buffer.get_summary())} 字符")
        print(f"   记忆条数：{len(self.fact_store.all_facts())}")


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    agent = ReActAgent()
    agent.run()
