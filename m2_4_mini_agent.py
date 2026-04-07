"""
[M2.4] Mini Agent — Tool-use + Planning + Memory 综合实战
把 M2.1 / M2.2 / M2.3 的三个组件组合成一个完整的交互式 Agent

架构决策（来自 Step 2 讨论）：
  - 工具注册表：分模块定义 (schemas, executor)，主程序动态合并
  - Planning：内化到 System Prompt，无显式审批节点（交互式对话定位）
  - 错误处理：M2.1 策略——error dict 立即注入 messages，模型当场决策

组件复用：
  - ConversationBuffer : 直接从 m2_3_memory.py 复制（对话缓冲 + 摘要压缩）
  - FactStore          : 直接从 m2_3_memory.py 复制（持久化事实存储）
  - 天气工具            : 从 m2_1_tool_use.py 迁移，包装成模块接口
  - 记忆工具            : 从 m2_3_memory.py 迁移，包装成模块接口

运行方式：
  pip install tiktoken jieba requests
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m2_4_mini_agent.py
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

import jieba
import requests
import tiktoken

# ══════════════════════════════════════════════════════════════════════════════
# 基础设施（与前序模块相同，集中放这里避免跨文件导入）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"
HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

MAX_LOOP_TURNS = 10   # 单次用户输入最多触发几次 LLM 调用（防止工具调用无限循环）

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
# Memory 组件（从 M2.3 原封不动复制，保持独立性）
# ══════════════════════════════════════════════════════════════════════════════

MAX_BUFFER_TOKENS  = 2000
MAX_SUMMARY_CHUNKS = 3
FACT_STORE_PATH    = os.path.join(os.path.dirname(__file__), "facts.json")

_COMPRESSOR_SYSTEM = """你是一个对话摘要器。
将以下对话历史压缩成一段简洁的摘要，保留所有关键信息：用户的问题、助手的回答要点、提到的关键事实。
输出格式：纯文本，不超过200字，不需要任何前缀或标题。"""


class ConversationBuffer:
    """对话缓冲区：维护最近 N 轮完整对话 + 更早对话的压缩摘要。"""

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
        return max(1, len(self._messages) // 2)

    def _compress_if_needed(self) -> None:
        if self.total_tokens() <= MAX_BUFFER_TOKENS:
            return
        mid = self._find_split_index()
        if mid == 0:
            return
        to_compress, to_keep = self._messages[:mid], self._messages[mid:]
        print(f"\n  [Buffer] 触发压缩：{len(self._messages)} 条 → 压缩前 {mid} 条...")
        new_summary = self._summarize(to_compress)
        self._summary_chunks.append(new_summary)
        if len(self._summary_chunks) > MAX_SUMMARY_CHUNKS:
            self._summary_chunks.pop(0)
        self._summary = "\n---\n".join(self._summary_chunks)
        self._messages = to_keep
        print(f"  [Buffer] 压缩完成，保留 {len(self._messages)} 条")

    def _summarize(self, messages: list[dict]) -> str:
        dialogue = "\n".join(
            f"{m['role'].upper()}: "
            + (m["content"] if isinstance(m["content"], str)
               else json.dumps(m["content"], ensure_ascii=False))
            for m in messages
        )
        response = send_request(
            [{"role": "user", "content": f"请压缩以下对话：\n\n{dialogue}"}],
            system=_COMPRESSOR_SYSTEM,
            temperature=0.0,
            max_tokens=300,
        )
        return "".join(
            b.get("text", "") for b in response.get("content", [])
            if b.get("type") == "text"
        )


class FactStore:
    """持久化事实存储：跨对话保存结构化知识，支持关键词检索。"""

    def __init__(self, path: str = FACT_STORE_PATH):
        self._path  = path
        self._facts = self._load()

    def save(self, content: str, tags: list[str] = None) -> dict:
        fact = {
            "id":        f"f_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "content":   content,
            "tags":      tags or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embedding": None,
        }
        self._facts.append(fact)
        self._persist()
        return fact

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        words = list(set(
            t.strip() for t in jieba.cut(query.lower(), cut_all=False) if t.strip()
        ))
        def matches(f):
            text = f["content"].lower() + " " + " ".join(f["tags"]).lower()
            return any(w in text for w in words)
        matched = sorted(
            [f for f in self._facts if matches(f)],
            key=lambda f: f["timestamp"], reverse=True
        )
        return matched[:top_k]

    def all_facts(self) -> list[dict]:
        return list(self._facts)

    def as_context_string(self, max_facts: int = 20) -> str:
        if not self._facts:
            return ""
        recent = sorted(self._facts, key=lambda x: x["timestamp"])[-max_facts:]
        hidden = len(self._facts) - len(recent)
        lines  = [f"[已知事实（显示最近 {len(recent)} 条，共 {len(self._facts)} 条）]"]
        for f in recent:
            tag_str = ", ".join(f["tags"]) if f["tags"] else "无标签"
            lines.append(f"  - {f['content']} (标签: {tag_str})")
        if hidden > 0:
            lines.append(f"  （另有 {hidden} 条较早的记忆未显示，可用 recall_facts 检索）")
        return "\n".join(lines)

    def _load(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _persist(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 工具模块层
#
# 架构约定：每个模块暴露一个工厂函数，返回 (schemas, executor) 元组。
#   schemas  : list[dict]  — 传给 API 的 tools 参数
#   executor : callable    — execute(name, input_data) -> dict
#
# 主程序只做两件事：
#   all_schemas  = module_a_schemas + module_b_schemas
#   all_executors = {**module_a_executors, **module_b_executors}
#
# 🐍 Python 插播：元组（tuple）
# (schemas, executor) 是一个不可变的二元组，类比 Go 的多返回值：
#   func weatherModule() ([]Schema, Executor) { ... }
# Python 没有多返回值，但元组在语义上完全等价。
# ══════════════════════════════════════════════════════════════════════════════

# ── 天气工具模块 ──────────────────────────────────────────────────────────────

_WEATHER_DB = {
    "北京": {"temp": 12, "condition": "多云", "humidity": 45},
    "上海": {"temp": 18, "condition": "小雨", "humidity": 80},
    "广州": {"temp": 26, "condition": "晴",   "humidity": 65},
    "成都": {"temp": 15, "condition": "多云", "humidity": 70},
    "杭州": {"temp": 17, "condition": "小雨", "humidity": 78},
}

_WEATHER_SCHEMAS = [
    {
        "name": "get_weather",
        "description": (
            "获取指定城市的当前天气信息。"
            "返回温度（摄氏度）、天气状况（晴/多云/小雨/大雨）和湿度（%）。"
            "仅支持中国大陆主要城市：北京、上海、广州、成都、杭州。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如'北京'、'上海'"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "compare_weather",
        "description": (
            "比较两个城市的天气，给出哪个更适合户外活动的建议。"
            "需要先用 get_weather 获取两个城市的天气数据，再传入本工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city_a":    {"type": "string"},
                "city_b":    {"type": "string"},
                "weather_a": {"type": "object", "description": "city_a 的 get_weather 返回结果"},
                "weather_b": {"type": "object", "description": "city_b 的 get_weather 返回结果"},
            },
            "required": ["city_a", "city_b", "weather_a", "weather_b"],
        },
    },
]


def _exec_get_weather(city: str) -> dict:
    if city not in _WEATHER_DB:
        raise ValueError(f"不支持的城市：{city}。支持：{list(_WEATHER_DB.keys())}")
    d = _WEATHER_DB[city]
    return {"city": city, "temp": d["temp"], "condition": d["condition"],
            "humidity": d["humidity"], "unit": "celsius"}


def _exec_compare_weather(city_a, city_b, weather_a, weather_b) -> dict:
    score_map = {"晴": 3, "多云": 2, "小雨": 1, "大雨": 0}
    def score(w):
        s = score_map.get(w["condition"], 0)
        if w["humidity"] < 60: s += 1
        if 15 <= w["temp"] <= 25: s += 1
        return s
    sa, sb = score(weather_a), score(weather_b)
    if sa > sb:   rec, reason = city_a, f"{city_a}更适宜（{sa} vs {sb}分）"
    elif sb > sa: rec, reason = city_b, f"{city_b}更适宜（{sb} vs {sa}分）"
    else:         rec, reason = "两城市相当", f"评分相同（均 {sa} 分）"
    return {"recommendation": rec, "reason": reason, "score_a": sa, "score_b": sb}


_WEATHER_IMPL = {
    "get_weather":     _exec_get_weather,
    "compare_weather": _exec_compare_weather,
}


def weather_module() -> tuple[list, dict]:
    """
    天气工具模块工厂函数。

    返回 (schemas, impl_map)：
      schemas  : 传给 API 的 tool 定义列表
      impl_map : name → 实现函数的字典（由 ToolRegistry 消费）
    """
    return _WEATHER_SCHEMAS, _WEATHER_IMPL


# ── 记忆工具模块 ──────────────────────────────────────────────────────────────

_MEMORY_SCHEMAS = [
    {
        "name": "save_fact",
        "description": (
            "把一条重要事实或用户信息保存到长期记忆中。"
            "适用：用户提到个人信息（职业/偏好/背景）、跨对话需要记住的关键结论。"
            "不要保存临时性中间结果。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "完整陈述句，如'用户是 Go 开发者'"},
                "tags":    {"type": "array", "items": {"type": "string"},
                            "description": "关键词标签，如 ['语言', '背景']"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_facts",
        "description": (
            "从长期记忆中检索与查询相关的事实。"
            "适用：用户问起之前了解的信息、需要参考历史数据。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "检索关键词，空格分隔，如'用户 语言 背景'"},
            },
            "required": ["query"],
        },
    },
]


def memory_module(fact_store: FactStore) -> tuple[list, dict]:
    """
    记忆工具模块工厂函数。

    接收 FactStore 实例（依赖注入），返回 (schemas, impl_map)。
    impl_map 里的函数通过闭包持有 fact_store 引用——
    这样外部无法直接访问 fact_store，只能通过工具接口操作。

    类比 Go：
      func memoryModule(store *FactStore) ([]Schema, map[string]Handler) {
          return schemas, map[string]Handler{
              "save_fact":    func(input) { store.Save(...) },
              "recall_facts": func(input) { store.Recall(...) },
          }
      }
    """
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

    impl_map = {
        "save_fact":    _exec_save_fact,
        "recall_facts": _exec_recall_facts,
    }
    return _MEMORY_SCHEMAS, impl_map


# ══════════════════════════════════════════════════════════════════════════════
# ToolRegistry：统一的工具注册表
#
# 职责：
#   1. 接收若干个 (schemas, impl_map) 模块
#   2. 合并 schemas → 传给 API
#   3. 合并 impl_map → 分发 execute 调用
#   4. execute() 失败时返回 error dict，不抛异常（M2.1 约定）
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    工具注册表：把多个工具模块聚合成统一接口。

    类比 Go 的 http.ServeMux：
      mux.Handle("/weather", weatherHandler)
      mux.Handle("/memory",  memoryHandler)
      // 之后 mux.ServeHTTP() 自动路由

    ToolRegistry 做同样的事：
      registry.register(*weather_module())
      registry.register(*memory_module(fact_store))
      // 之后 registry.execute(name, input) 自动路由
    """

    def __init__(self):
        self._schemas:  list[dict]       = []
        self._impl_map: dict[str, callable] = {}

    def register(self, schemas: list[dict], impl_map: dict) -> None:
        """
        注册一个工具模块。

        🐍 Python 插播：*registry.register(*weather_module())
        weather_module() 返回 (schemas, impl_map) 元组，
        * 运算符把元组"解包"成位置参数——
        等价于 registry.register(schemas, impl_map)。
        类比 Go 的多返回值：s, m := weatherModule(); registry.Register(s, m)
        """
        # 检查工具名冲突（fail fast，类比 Go 里重复注册 HTTP handler 会 panic）
        for schema in schemas:
            name = schema["name"]
            if name in self._impl_map:
                raise ValueError(f"工具名冲突：'{name}' 已注册，请检查模块是否重复加载")

        self._schemas.extend(schemas)
        self._impl_map.update(impl_map)

    @property
    def schemas(self) -> list[dict]:
        """返回所有工具的 schema 列表，直接传给 API 的 tools 参数。"""
        return list(self._schemas)

    def execute(self, name: str, input_data: dict) -> dict:
        """
        执行工具调用。失败时返回 error dict，不抛异常。

        设计意图（M2.1 约定）：
        - 工具失败不等于 Agent 失败
        - error dict 注入 messages → 模型当场决策：重试/告知用户/跳过
        - 类比 Go：return nil, err  而不是 panic(err)
        """
        if name not in self._impl_map:
            return {"status": "error",
                    "message": f"未知工具：{name}。可用：{list(self._impl_map.keys())}"}
        try:
            result = self._impl_map[name](**input_data)
            # 实现函数可能直接返回结果 dict，也可能已经带 status 字段
            # 统一包装成 {"status": "ok", "result": ...} 格式
            if isinstance(result, dict) and "status" in result:
                return result   # 记忆工具已经自带 status，直接透传
            return {"status": "ok", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt 构建
# ══════════════════════════════════════════════════════════════════════════════

# Mini Agent 的行为规范（Procedural Memory，固定不变）
# 关键：Planning 内化在这里——"遇到多步任务，先说明步骤再执行"
_AGENT_PERSONA = """\
你是一个具备长期记忆和工具调用能力的智能助手。

【能力】
- 天气查询：可查询北京、上海、广州、成都、杭州的天气，并进行城市间比较
- 长期记忆：可保存和检索跨对话的重要信息

【行为规范】
1. 遇到需要多步工具调用的任务，先简要说明"我需要先做A，再做B"，再逐步执行
   （这是轻量级 Planning：不需要用户审批，但让用户知道你的思路）
2. 用户透露个人信息（职业/偏好/城市/背景）时，主动调用 save_fact 保存
3. 用户询问你之前了解的信息时，先调用 recall_facts 检索再回答
4. 工具调用失败时，诚实告知用户并说明原因，不要编造数据
5. 用中文回答，语言自然友好，简洁直接\
"""


def build_system_prompt(buffer: ConversationBuffer, fact_store: FactStore) -> str:
    """
    每轮对话前动态构建 System Prompt。

    注入顺序（从稳定到动态，类比 HTTP 请求的 Header 优先级）：
      1. Procedural Memory：行为规范（固定）
      2. Semantic Memory  ：已知事实（跨对话持久，会增长）
      3. Episodic Memory  ：历史摘要（本次对话的压缩历史）
    """
    parts = [_AGENT_PERSONA]

    facts_str = fact_store.as_context_string()
    if facts_str:
        parts.append(facts_str)

    summary = buffer.get_summary()
    if summary:
        parts.append(f"[历史对话摘要]\n{summary}")

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Mini Agent 主循环
# ══════════════════════════════════════════════════════════════════════════════

class MiniAgent:
    """
    Mini Agent：把三个组件组合成完整的交互式对话 Agent。

    内部状态：
      buffer    : ConversationBuffer  — 对话历史（自动压缩）
      fact_store: FactStore           — 跨对话持久记忆
      registry  : ToolRegistry        — 统一工具注册表

    主循环逻辑：
      while True:
        用户输入 → buffer.add("user", ...)
        内层循环（最多 MAX_LOOP_TURNS 轮）：
          send_request → end_turn  → 打印 + buffer.add("assistant", ...) → break
                       → tool_use → execute → 注入结果 → 继续内层循环
    """

    def __init__(self, fact_store_path: str = FACT_STORE_PATH):
        self.buffer     = ConversationBuffer()
        self.fact_store = FactStore(path=fact_store_path)
        self.registry   = ToolRegistry()

        # 注册所有工具模块
        # 🐍 Python 插播：* 解包元组作为位置参数
        # weather_module() 返回 (schemas, impl_map)，
        # *weather_module() 等价于把元组展开成两个独立参数
        self.registry.register(*weather_module())
        self.registry.register(*memory_module(self.fact_store))

    def run(self) -> None:
        """启动交互式对话主循环。"""
        self._print_startup_info()

        while True:
            # ── 获取用户输入 ────────────────────────────────────────────────
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            # 内置调试命令
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

            # ── 用户输入存入 buffer ─────────────────────────────────────────
            self.buffer.add("user", user_input)

            # ── 内层循环：处理工具调用链 ────────────────────────────────────
            self._agent_turn()

    def _agent_turn(self) -> None:
        """
        处理一次用户输入对应的完整 Agent 响应（可能包含多次工具调用）。

        这是 Tool-use Loop 的核心，逻辑与 M2.1 完全相同，
        区别在于 messages 来自 buffer，tools 来自 registry。
        """
        for turn in range(MAX_LOOP_TURNS):
            system_prompt = build_system_prompt(self.buffer, self.fact_store)
            messages      = self.buffer.get_messages()

            response    = send_request(messages, tools=self.registry.schemas,
                                       system=system_prompt)
            stop_reason = response.get("stop_reason")
            content     = response.get("content", [])

            # ── 情况一：正常结束 ────────────────────────────────────────────
            if stop_reason == "end_turn":
                reply = "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                print(f"\n助手: {reply}")
                self.buffer.add("assistant", reply)
                return

            # ── 情况二：需要调用工具 ────────────────────────────────────────
            if stop_reason == "tool_use":
                # Step A：assistant 的完整回复（含 tool_use blocks）存入 buffer
                # ⚠️ 不能省略：API 要求 tool_result 前必须有对应的 assistant turn
                self.buffer.add("assistant", content)

                # Step B：执行所有工具调用，收集结果
                tool_results = []
                for block in content:
                    if block.get("type") != "tool_use":
                        continue

                    name        = block["name"]
                    input_data  = block["input"]
                    tool_use_id = block["id"]

                    print(f"\n  🔧 {name}({json.dumps(input_data, ensure_ascii=False)})")
                    result = self.registry.execute(name, input_data)
                    status_icon = "✅" if result.get("status") == "ok" else "❌"
                    print(f"  {status_icon} {json.dumps(result, ensure_ascii=False)}")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_use_id,
                        "content":     json.dumps(result, ensure_ascii=False),
                    })

                # Step C：tool_results 作为 user 消息注入 buffer，进入下一轮
                self.buffer.add("user", tool_results)
                continue

            # ── 情况三：未预期的 stop_reason ───────────────────────────────
            print(f"  ⚠️  未处理的 stop_reason: {stop_reason}")
            return

        # 超过最大轮次
        print(f"  ⚠️  超过最大工具调用轮次（{MAX_LOOP_TURNS}），本轮终止")

    # ── 调试辅助方法 ──────────────────────────────────────────────────────────

    def _print_startup_info(self) -> None:
        existing = self.fact_store.all_facts()
        tools    = [s["name"] for s in self.registry.schemas]

        print("\n" + "=" * 60)
        print("🤖 Mini Agent 启动")
        print(f"   已注册工具：{tools}")
        if existing:
            print(f"   从磁盘加载了 {len(existing)} 条历史记忆：")
            for f in existing:
                print(f"     - {f['content']}")
        else:
            print("   记忆为空（首次启动）")
        print()
        print("   内置命令：facts（查看所有记忆）| status（查看 buffer 状态）| quit（退出）")
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

    agent = MiniAgent()
    agent.run()
