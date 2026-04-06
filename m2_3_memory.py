"""
[M2.3] Memory 机制
Module A: ConversationBuffer  — 对话缓冲 + 摘要压缩
Module B: FactStore           — 持久化事实存储 + 关键词检索

架构原则："记忆"的本质是 外化存储 → 按需检索 → 注入 Context
  - 模型看不到 FactStore 的内部，只能通过工具 save_fact / recall_facts 与之交互
  - ConversationBuffer 对模型完全透明，由代码自动管理

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m2_3_memory.py
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

import requests
import tiktoken

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与 M2.1 / M2.2 相同）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"
HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


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
        "model":      MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream":     False,
        "messages":   messages,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    response = requests.post(API_URL, headers=HEADERS, json=body)
    response.raise_for_status()
    return response.json()


# ══════════════════════════════════════════════════════════════════════════════
# Module A：ConversationBuffer
# ══════════════════════════════════════════════════════════════════════════════

# 触发摘要压缩的 token 阈值
# 超过这个数字，把最早的一半对话压缩成摘要
MAX_BUFFER_TOKENS = 2000

# 压缩时调用 LLM 生成摘要，用这个 system prompt
_COMPRESSOR_SYSTEM = """你是一个对话摘要器。
将以下对话历史压缩成一段简洁的摘要，保留所有关键信息：用户的问题、助手的回答要点、提到的关键事实。
输出格式：纯文本，不超过200字，不需要任何前缀或标题。"""


class ConversationBuffer:
    """
    对话缓冲区：维护最近 N 轮完整对话 + 更早对话的压缩摘要。

    内部结构：
      _messages : list of {"role": ..., "content": ...}  ← 完整对话，直接喂给 API
      _summary  : str                                     ← 历史摘要，注入 System Prompt

    类比 Go：
      类似 container/ring 的环形缓冲区，但不是简单丢弃旧数据——
      而是先"精馏"（压缩摘要）再丢弃，像 WAL（Write-Ahead Log）的归档机制。
    """

    def __init__(self):
        self._messages: list[dict] = []
        self._summary: str = ""          # 历史摘要，初始为空

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def add(self, role: str, content) -> None:
        """
        追加一条消息，追加后检查是否需要压缩。

        content 可以是 str 或 list（tool_use / tool_result 场景）。
        token 计数时统一转成字符串处理——计数允许有误差，不需要精确。
        """
        self._messages.append({"role": role, "content": content})
        self._compress_if_needed()

    def get_messages(self) -> list[dict]:
        """返回当前 buffer 里的完整消息列表，直接传给 API 的 messages 字段。"""
        return list(self._messages)   # 返回副本，防止调用方意外修改

    def get_summary(self) -> str:
        """返回历史摘要字符串，调用方负责把它注入 System Prompt。"""
        return self._summary

    def total_tokens(self) -> int:
        """当前 buffer 的总 token 数（用于调试和监控）。"""
        return sum(
            count_tokens(
                m["content"] if isinstance(m["content"], str)
                else json.dumps(m["content"], ensure_ascii=False)
            )
            for m in self._messages
        )

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _compress_if_needed(self) -> None:
        """
        如果 buffer 超过阈值，把前一半消息压缩成摘要，只保留后一半。

        为什么是"前一半"而不是"最早的几条"？
        保证压缩比例稳定：无论对话多长，压缩后 buffer 大小都回到阈值的约 50%，
        不会因为"压缩了 2 条但每条都很长"而导致压缩效果不稳定。

        🐍 Python 插播：列表切片
        self._messages[:mid]  → 前一半（待压缩）
        self._messages[mid:]  → 后一半（保留）
        类比 Go 的 slice[0:mid] 和 slice[mid:]，语法完全一致。
        """
        if self.total_tokens() <= MAX_BUFFER_TOKENS:
            return

        mid = len(self._messages) // 2
        if mid == 0:
            return   # 只有1条消息时不压缩，避免死循环

        to_compress = self._messages[:mid]
        to_keep     = self._messages[mid:]

        print(f"\n  [Buffer] 触发压缩：{len(self._messages)} 条消息，压缩前 {mid} 条...")

        new_summary = self._summarize(to_compress)

        # 把新摘要和已有摘要合并
        # 如果已有摘要，新摘要追加在后面（时间顺序）
        if self._summary:
            self._summary = self._summary + "\n" + new_summary
        else:
            self._summary = new_summary

        self._messages = to_keep
        print(f"  [Buffer] 压缩完成，保留 {len(self._messages)} 条，摘要长度：{len(self._summary)} 字符")

    def _summarize(self, messages: list[dict]) -> str:
        """
        调用 LLM 把一段对话压缩成摘要。

        注意：这是 Memory 模块内部的 LLM 调用，与业务对话循环无关。
        类比 Go 里的后台 goroutine 做数据归档——对主流程透明。
        """
        # 把消息列表格式化成可读文本
        dialogue = "\n".join(
            f"{m['role'].upper()}: "
            + (m["content"] if isinstance(m["content"], str)
               else json.dumps(m["content"], ensure_ascii=False))
            for m in messages
        )

        user_prompt = f"请压缩以下对话：\n\n{dialogue}"
        response = send_request(
            add_message([], "user", user_prompt),
            system=_COMPRESSOR_SYSTEM,
            temperature=0.0,
            max_tokens=300,
        )

        return "".join(
            b.get("text", "") for b in response.get("content", [])
            if b.get("type") == "text"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module B：FactStore
# ══════════════════════════════════════════════════════════════════════════════

# Fact 持久化文件路径（同目录下）
FACT_STORE_PATH = os.path.join(os.path.dirname(__file__), "facts.json")


class FactStore:
    """
    持久化事实存储：跨对话保存结构化知识，支持关键词检索。

    存储格式（facts.json）：
      [
        {
          "id":        "f_<timestamp>_<uuid4_short>",
          "content":   "用户是 Go/C++ 开发者",
          "tags":      ["语言", "背景"],
          "timestamp": "2026-04-06T10:00:00+00:00",
          "embedding": null       ← 预留字段，以后换语义检索时填入
        },
        ...
      ]

    类比 Go：
      类似一个带 Load() / Save() 的结构体，序列化到磁盘。
      内存里是 []Fact，磁盘上是 JSON 文件。
    """

    def __init__(self, path: str = FACT_STORE_PATH):
        self._path  = path
        self._facts = self._load()

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def save(self, content: str, tags: list[str] = None) -> dict:
        """
        存入一条新 fact，持久化到磁盘，返回保存的 fact dict。

        tags 是可选的关键词列表，用于扩展检索面。
        """
        fact = {
            "id":        self._new_id(),
            "content":   content,
            "tags":      tags or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embedding": None,    # 预留：未来填入 embedding 向量
        }
        self._facts.append(fact)
        self._persist()
        return fact

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """
        关键词检索：把 query 按空格拆成词，在 content + tags 里匹配。
        返回命中的 facts，按时间倒序（最近的在前）。

        匹配逻辑：query 中任意一个词命中，就算相关。
        这是 OR 语义，比 AND 更宽松——宁可多返回，不要漏掉。

        🐍 Python 插播：列表推导式 + any()
          [f for f in self._facts if any(w in f["content"] for w in words)]
          类比 Go 里的 for 循环 + 内层 for 循环 + break，但更简洁。
          any() 是短路求值：只要找到一个 True 就立刻返回，不继续遍历。
        """
        if not query.strip():
            return []

        words = query.lower().split()

        def matches(fact: dict) -> bool:
            # 在 content 和 tags 里搜索（都转成小写做大小写不敏感匹配）
            search_text = fact["content"].lower() + " " + " ".join(fact["tags"]).lower()
            return any(w in search_text for w in words)

        matched = [f for f in self._facts if matches(f)]

        # 按时间倒序，取前 top_k 条
        matched.sort(key=lambda f: f["timestamp"], reverse=True)
        return matched[:top_k]

    def all_facts(self) -> list[dict]:
        """返回所有 facts（用于调试和全量注入 context 场景）。"""
        return list(self._facts)

    def as_context_string(self) -> str:
        """
        把所有 facts 格式化成可读字符串，用于注入 System Prompt。

        格式示例：
          [已知事实]
          - 用户是 Go/C++ 开发者 (标签: 语言, 背景)
          - 用户偏好城市是北京 (标签: 偏好)
        """
        if not self._facts:
            return ""

        lines = ["[已知事实]"]
        # 按时间顺序显示（最早的在前，符合阅读习惯）
        for f in sorted(self._facts, key=lambda x: x["timestamp"]):
            tag_str = ", ".join(f["tags"]) if f["tags"] else "无标签"
            lines.append(f"  - {f['content']} (标签: {tag_str})")

        return "\n".join(lines)

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _new_id(self) -> str:
        """生成唯一 ID：f_<unix_timestamp>_<uuid4前8位>"""
        ts  = int(time.time())
        uid = uuid.uuid4().hex[:8]
        return f"f_{ts}_{uid}"

    def _load(self) -> list[dict]:
        """从磁盘加载 facts。文件不存在时返回空列表（首次运行）。"""
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [FactStore] 加载失败，从空白开始：{e}")
            return []

    def _persist(self) -> None:
        """把当前 facts 写回磁盘（全量覆写）。"""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 工具定义：把 FactStore 暴露给模型
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_TOOLS = [
    {
        "name": "save_fact",
        "description": (
            "把一条重要事实或用户信息保存到长期记忆中。"
            "适用场景：用户提到了关于自己的信息（职业、偏好、背景）、"
            "任务执行中发现的重要结论、需要跨对话记住的关键数据。"
            "不要保存临时性的中间结果。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "事实内容，一句完整的陈述句，如'用户是 Go/C++ 开发者'",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词标签列表，如 ['语言', '背景']，用于检索",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_facts",
        "description": (
            "从长期记忆中检索与查询相关的事实。"
            "适用场景：用户问到你之前了解的信息、需要参考历史数据做决策。"
            "返回最相关的若干条事实。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，如'用户语言背景'、'偏好城市'",
                },
            },
            "required": ["query"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 工具执行层：连接模型调用和 FactStore 实例
# ══════════════════════════════════════════════════════════════════════════════

def make_tool_executor(fact_store: FactStore):
    """
    工厂函数：返回一个绑定了特定 FactStore 实例的 execute_tool 函数。

    为什么用工厂函数而不是全局变量？
    因为 FactStore 是有状态的（持有文件路径），测试时可以传入不同实例。
    类比 Go 里给 handler 注入依赖（dependency injection），而不是用全局变量。

    🐍 Python 插播：闭包（closure）
    内部函数 execute_tool 可以访问外层函数的变量 fact_store——
    这就是闭包。类比 Go 里的函数字面量捕获外层变量：
      executor := func(name string, input map[string]any) map[string]any {
          // 这里可以访问外层的 factStore
      }
    """
    def execute_tool(name: str, input_data: dict) -> dict:
        try:
            if name == "save_fact":
                fact = fact_store.save(
                    content=input_data["content"],
                    tags=input_data.get("tags", []),
                )
                return {
                    "status": "ok",
                    "message": f"已保存：{fact['content']}",
                    "id": fact["id"],
                }

            elif name == "recall_facts":
                facts = fact_store.recall(query=input_data["query"])
                if not facts:
                    return {"status": "ok", "facts": [], "message": "未找到相关记忆"}
                return {
                    "status": "ok",
                    "facts": [
                        {"content": f["content"], "tags": f["tags"], "timestamp": f["timestamp"]}
                        for f in facts
                    ],
                }

            else:
                return {"status": "error", "message": f"未知工具：{name}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    return execute_tool


# ══════════════════════════════════════════════════════════════════════════════
# Memory Agent Loop：把两个模块串起来
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(buffer: ConversationBuffer, fact_store: FactStore) -> str:
    """
    每轮对话前动态构建 System Prompt，把记忆信息注入进去。

    注入顺序（从稳定到动态）：
      1. 行为规范（Procedural Memory，固定）
      2. 已知事实（Semantic Memory，跨对话持久）
      3. 历史摘要（Episodic Memory，压缩后的历史）
    """
    parts = [
        # Procedural：行为规范
        "你是一个具备长期记忆的助手。\n"
        "你有两个记忆工具：\n"
        "  - save_fact：把重要信息保存到长期记忆\n"
        "  - recall_facts：从长期记忆中检索信息\n"
        "当用户透露个人信息或重要偏好时，主动调用 save_fact 保存。\n"
        "当用户询问你之前了解的信息时，先调用 recall_facts 检索再回答。\n"
        "用中文回答，语言自然友好。",
    ]

    # Semantic：已知事实（如果有）
    facts_str = fact_store.as_context_string()
    if facts_str:
        parts.append(facts_str)

    # Episodic：历史摘要（如果有）
    summary = buffer.get_summary()
    if summary:
        parts.append(f"[历史对话摘要]\n{summary}")

    return "\n\n".join(parts)


MAX_AGENT_TURNS = 10   # 单轮用户输入最多触发几次 LLM 调用（含工具调用）


def memory_agent_loop(buffer: ConversationBuffer, fact_store: FactStore) -> None:
    """
    带记忆的对话主循环。

    与 M2.1 tool_use_loop 的核心区别：
      1. messages 不再是每轮新建，而是从 buffer.get_messages() 获取历史
      2. System Prompt 每轮动态构建（含最新的 facts 和摘要）
      3. 每轮对话结束后把新消息 add 到 buffer（触发自动压缩检查）
    """
    execute_tool = make_tool_executor(fact_store)

    print("\n" + "=" * 60)
    print("Memory Agent 启动（输入 'quit' 退出，'facts' 查看所有记忆）")
    print("=" * 60)

    while True:
        # ── 获取用户输入 ────────────────────────────────────────────────────
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if user_input.lower() == "quit":
            print("👋 再见！")
            break

        # 调试命令：打印当前所有 facts
        if user_input.lower() == "facts":
            facts = fact_store.all_facts()
            if facts:
                print(f"\n📚 当前记忆（{len(facts)} 条）：")
                for f in facts:
                    print(f"  [{f['id']}] {f['content']} | 标签: {f['tags']}")
            else:
                print("\n📚 记忆为空")
            continue

        if not user_input:
            continue

        # ── 把用户输入加入 buffer ───────────────────────────────────────────
        buffer.add("user", user_input)

        # ── 内层循环：处理可能的多次工具调用 ────────────────────────────────
        # （和 M2.1 的 tool_use_loop 逻辑相同，但 messages 来自 buffer）
        for turn in range(MAX_AGENT_TURNS):
            system_prompt = build_system_prompt(buffer, fact_store)
            messages      = buffer.get_messages()

            response    = send_request(messages, tools=MEMORY_TOOLS, system=system_prompt)
            stop_reason = response.get("stop_reason")
            content     = response.get("content", [])

            if stop_reason == "end_turn":
                # 提取文本回复，打印并存入 buffer
                reply = "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                print(f"\n助手: {reply}")
                buffer.add("assistant", reply)
                break

            if stop_reason == "tool_use":
                # 把 assistant 的完整回复（含 tool_use blocks）存入 buffer
                buffer.add("assistant", content)

                # 执行所有工具调用
                tool_results = []
                for block in content:
                    if block.get("type") != "tool_use":
                        continue

                    tool_name   = block["name"]
                    tool_input  = block["input"]
                    tool_use_id = block["id"]

                    print(f"\n  🔧 调用工具: {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")
                    result = execute_tool(tool_name, tool_input)
                    print(f"  ✅ 结果: {json.dumps(result, ensure_ascii=False)}")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_use_id,
                        "content":     json.dumps(result, ensure_ascii=False),
                    })

                # 把 tool_results 存入 buffer（作为 user 消息）
                buffer.add("user", tool_results)
                continue

            # 未预期的 stop_reason
            print(f"  ⚠️  未处理的 stop_reason: {stop_reason}")
            break

        else:
            print(f"  ⚠️  超过最大轮次 ({MAX_AGENT_TURNS})，跳过本次输入")


# ══════════════════════════════════════════════════════════════════════════════
# 实验入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    # 初始化两个记忆模块
    buffer     = ConversationBuffer()
    fact_store = FactStore()   # 自动从 facts.json 加载历史记忆

    # 告知用户当前记忆状态
    existing = fact_store.all_facts()
    if existing:
        print(f"\n📚 从磁盘加载了 {len(existing)} 条历史记忆：")
        for f in existing:
            print(f"  - {f['content']}")

    # 进入对话循环
    memory_agent_loop(buffer, fact_store)
