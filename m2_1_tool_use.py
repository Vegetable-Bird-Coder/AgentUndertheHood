"""
[M2.1] Tool-use 机制
用纯 HTTP 实现完整的 Tool-use Loop，不使用 Anthropic SDK

架构设计：
  - execute_tool()   : 工具注册表 + 本地执行（失败返回 error dict，不抛异常）
  - send_request()   : 带 tools 参数的 HTTP 请求
  - tool_use_loop()  : 核心循环（发送 → 解析 → 执行 → 注入 → 再发送）
  - main()           : 组装入口

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m2_1_tool_use.py
"""

import json
import os
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 复用 M1 的底层基础设施（原封不动）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"

HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

MAX_TURNS = 10  # 防止无限循环的硬上限


def add_message(messages: list, role: str, content) -> list:
    """
    content 可以是 str（普通文本）或 list（content blocks）。

    🐍 Python 插播：Python 是动态类型语言，函数参数可以接受多种类型。
    这里 content 既可能是 str（普通回复），也可能是 list（tool_use blocks）。
    类比 Go 里用 interface{} 或泛型处理多态——Python 里直接传就行，不需要声明类型。
    """
    return messages + [{"role": role, "content": content}]


def parse_sse_stream(response: requests.Response) -> str:
    """
    解析 SSE 流，返回完整的响应文本（拼接所有 token）。

    注意：tool_use 响应不走 text_delta，而是走 input_json_delta。
    这里我们只收集 text 部分——tool_use 的结构化数据从最终 JSON 里取，
    不从流里逐字节拼 JSON（容易出错）。
    """
    full_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                token = delta.get("text", "")
                print(token, end="", flush=True)
                full_text += token
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# M2.1 新增：带 tools 参数的请求函数
# ══════════════════════════════════════════════════════════════════════════════

def send_request(
    messages:    list,
    tools:       list  = None,
    system:      str   = "",
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
) -> dict:
    """
    发送请求，返回完整的响应 JSON（而非 Response 对象）。

    与 M1 的 send_request 的核心区别：
    1. 新增 tools 参数
    2. 返回 dict（解析后的 JSON），而不是 Response 对象
       原因：tool_use 的结构化数据需要从完整 JSON 中取，
             而不能只从 SSE 流里拼文本——流式响应中 tool input 是
             逐字节的 JSON 片段，拼接容易出错。

    🐍 Python 插播：`tools = None` 是默认参数
    类比 Go 的函数重载——Python 没有重载，用默认参数代替。
    调用时不传 tools，就等价于"不使用工具"的普通请求。
    """
    body = {
        "model":       MODEL,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,   # ← 关键：tool_use 场景关闭流式，拿完整 JSON
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
# 工具定义：Schema（给模型看）+ 实现（本地执行）
# ══════════════════════════════════════════════════════════════════════════════

# ── Tool Schema：告诉模型有哪些工具、怎么调用 ────────────────────────────────
#
# 这就是"API 文档"——写得越清晰，模型调用越准确。
# 注意 description 的写法：说清楚"返回什么"、"限制是什么"，不只是"做什么"。

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
                "city_a": {
                    "type": "string",
                    "description": "第一个城市名称",
                },
                "city_b": {
                    "type": "string",
                    "description": "第二个城市名称",
                },
                "weather_a": {
                    "type": "object",
                    "description": "city_a 的天气数据（get_weather 的返回结果）",
                },
                "weather_b": {
                    "type": "object",
                    "description": "city_b 的天气数据（get_weather 的返回结果）",
                },
            },
            "required": ["city_a", "city_b", "weather_a", "weather_b"],
        },
    },
]


# ── 工具实现：模拟数据（生产环境替换为真实 API 调用）────────────────────────

# 🐍 Python 插播：dict 字面量
# 类比 Go 的 map[string]interface{}{"key": value}
# Python 直接写 {"key": value}，更简洁

_WEATHER_DB = {
    "北京": {"temp": 12, "condition": "多云", "humidity": 45},
    "上海": {"temp": 18, "condition": "小雨", "humidity": 80},
    "广州": {"temp": 26, "condition": "晴",   "humidity": 65},
    "成都": {"temp": 15, "condition": "多云", "humidity": 70},
    "杭州": {"temp": 17, "condition": "小雨", "humidity": 78},
}

def _get_weather(city: str) -> dict:
    """
    模拟天气查询。
    生产环境：替换为真实 HTTP 请求（如和风天气、高德天气 API）。
    """
    if city not in _WEATHER_DB:
        # 故意让不支持的城市报错，用来演示错误处理流程
        raise ValueError(f"不支持的城市：{city}。支持的城市：{list(_WEATHER_DB.keys())}")

    data = _WEATHER_DB[city]
    return {
        "city":      city,
        "temp":      data["temp"],
        "condition": data["condition"],
        "humidity":  data["humidity"],
        "unit":      "celsius",
    }


def _compare_weather(city_a: str, city_b: str, weather_a: dict, weather_b: dict) -> dict:
    """
    比较两城市天气，计算"户外活动适宜度"分数。

    评分规则（简化版）：
    - 天气状况：晴=3分，多云=2分，小雨=1分，大雨=0分
    - 湿度：< 60% 加1分
    - 温度：15-25°C 加1分
    """
    condition_score = {"晴": 3, "多云": 2, "小雨": 1, "大雨": 0}

    def score(w: dict) -> int:
        s = condition_score.get(w["condition"], 0)
        if w["humidity"] < 60:
            s += 1
        if 15 <= w["temp"] <= 25:
            s += 1
        return s

    score_a = score(weather_a)
    score_b = score(weather_b)

    if score_a > score_b:
        recommendation = city_a
        reason = f"{city_a}天气更适宜（评分 {score_a} vs {score_b}）"
    elif score_b > score_a:
        recommendation = city_b
        reason = f"{city_b}天气更适宜（评分 {score_b} vs {score_a}）"
    else:
        recommendation = "两城市相当"
        reason = f"两城市评分相同（均为 {score_a}）"

    return {
        "recommendation": recommendation,
        "reason":         reason,
        "score_a":        score_a,
        "score_b":        score_b,
    }


# ── 工具注册表：name → 实现函数 ───────────────────────────────────────────────
#
# 🐍 Python 插播：函数是一等公民（first-class citizen）
# 类比 Go 的 func 类型——Python 里函数可以像变量一样存进 dict。
# 这个 dict 就是"工具注册表"：通过 name 字符串找到对应的执行函数。
# 类比 Go 里的 map[string]func(...) interface{}

_TOOL_REGISTRY = {
    "get_weather":     _get_weather,
    "compare_weather": _compare_weather,
}


def execute_tool(name: str, input_data: dict) -> dict:
    """
    查找并执行工具。

    返回约定（和模型约定好的格式）：
    - 成功：{"status": "ok",    "result": {...}}
    - 失败：{"status": "error", "message": "..."}

    关键设计：失败时不抛异常，而是返回 error dict。
    这样错误信息能通过正常的 tool_result 通道传给模型，
    模型可以根据错误决定：换参数重试？告知用户？放弃？
    """
    if name not in _TOOL_REGISTRY:
        return {
            "status":  "error",
            "message": f"未知工具：{name}。可用工具：{list(_TOOL_REGISTRY.keys())}",
        }

    try:
        result = _TOOL_REGISTRY[name](**input_data)
        return {"status": "ok", "result": result}

    # 🐍 Python 插播：except Exception as e
    # 类比 Go 的 if err != nil { ... }
    # Python 用 try/except 捕获异常，e 是异常对象，str(e) 取错误信息
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 核心：Tool-use Loop
# ══════════════════════════════════════════════════════════════════════════════

def tool_use_loop(
    initial_messages: list,
    tools:            list,
    system:           str = "",
) -> str:
    """
    Tool-use 核心循环。

    循环逻辑：
      发送请求
        → stop_reason == "end_turn"  : 正常结束，返回最终文本
        → stop_reason == "tool_use"  : 执行工具，结果注入 messages，继续循环
        → 超过 MAX_TURNS             : 抛出异常，终止

    messages 的演化过程（以两轮 tool_use 为例）：

      初始：[user: "查天气"]
      第1轮发送后：
        append assistant: [{tool_use: get_weather}]
        append user:      [{tool_result: "25°C..."}]
      第2轮发送后：
        append assistant: "北京今天25度..."   ← end_turn，结束
    """
    messages = initial_messages

    for turn in range(MAX_TURNS):
        print(f"\n{'─' * 50}")
        print(f"🔄 第 {turn + 1} 轮请求")
        print(f"{'─' * 50}")

        response = send_request(messages, tools=tools, system=system)

        stop_reason = response.get("stop_reason")
        content     = response.get("content", [])

        print(f"stop_reason: {stop_reason}")

        # ── 情况一：正常结束 ────────────────────────────────────────────────
        if stop_reason == "end_turn":
            # 提取文本内容（content 是 list，找 type==text 的 block）
            final_text = ""
            for block in content:
                if block.get("type") == "text":
                    final_text += block.get("text", "")
            print(f"\n✅ 模型最终回答：\n{final_text}")
            return final_text

        # ── 情况二：需要调用工具 ────────────────────────────────────────────
        if stop_reason == "tool_use":

            # Step A：把 assistant 的完整回复（含 tool_use blocks）加入 messages
            # ⚠️ 这一步不能省：Anthropic API 要求 tool_result 前必须有对应的 assistant turn
            messages = add_message(messages, "assistant", content)

            # Step B：遍历所有 tool_use blocks，逐个执行
            # （一次回复里模型可能同时调用多个工具）
            tool_results = []

            for block in content:
                if block.get("type") != "tool_use":
                    continue

                tool_name    = block["name"]
                tool_input   = block["input"]
                tool_use_id  = block["id"]

                print(f"\n🔧 调用工具：{tool_name}")
                print(f"   参数：{json.dumps(tool_input, ensure_ascii=False)}")

                # 本地执行
                result = execute_tool(tool_name, tool_input)

                print(f"   结果：{json.dumps(result, ensure_ascii=False)}")

                # 构造 tool_result block
                # content 字段必须是字符串——把 dict 序列化为 JSON string
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,   # 配对键，原样带回
                    "content":     json.dumps(result, ensure_ascii=False),
                })

            # Step C：把所有 tool_result 作为一条 user 消息注入 messages
            # ⚠️ 必须是 user role，且 content 是 list（多个 tool_result 合并进一条消息）
            messages = add_message(messages, "user", tool_results)
            continue  # 进入下一轮循环

        # ── 情况三：未知的 stop_reason（如 max_tokens、stop_sequence 等）──
        raise RuntimeError(
            f"未处理的 stop_reason: {stop_reason}。"
            f"完整响应：{json.dumps(response, ensure_ascii=False, indent=2)}"
        )

    # 超过最大轮次，硬退出
    raise RuntimeError(
        f"超过最大轮次限制（{MAX_TURNS} 轮），任务未完成。"
        f"当前 messages 长度：{len(messages)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 实验入口
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM = """你是一个天气助手。
你有两个工具：get_weather（查询单个城市天气）和 compare_weather（比较两城市天气）。
回答用中文，语言简洁直接。"""


def run_experiment():
    print("=" * 60)
    print("M2.1 Tool-use 实验")
    print("=" * 60)

    # ── 实验 A：单工具调用 ─────────────────────────────────────────────────
    print("\n【实验 A】单工具调用：查询北京天气")
    messages = [{"role": "user", "content": "北京今天天气怎么样？"}]
    tool_use_loop(messages, TOOLS, system=SYSTEM)

    # ── 实验 B：多步工具调用（串行）────────────────────────────────────────
    # 模型需要先查两个城市天气，再调用 compare_weather 综合比较
    print("\n\n【实验 B】多步工具调用：比较北京和广州，哪个更适合周末出行？")
    messages = [{"role": "user", "content": "帮我比较北京和广州的天气，哪个城市更适合这个周末出行？"}]
    tool_use_loop(messages, TOOLS, system=SYSTEM)

    # ── 实验 C：错误处理 ────────────────────────────────────────────────────
    # 故意查一个不支持的城市，观察模型如何响应 error 信息
    print("\n\n【实验 C】错误处理：查询不支持的城市（纽约）")
    messages = [{"role": "user", "content": "纽约今天天气怎么样？"}]
    tool_use_loop(messages, TOOLS, system=SYSTEM)


if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    run_experiment()
