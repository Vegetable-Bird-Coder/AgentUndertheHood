"""
[M1.1] LLM API 底层机制 - 纯 HTTP 实现
不使用 Anthropic SDK，用 requests 库直接构造请求

运行前提：设置环境变量 ANTHROPIC_API_KEY
  export ANTHROPIC_API_KEY="sk-ant-..."

运行方式：
  python m1_1_llm_api_raw.py
"""

import json
import os
import requests

# ── 常量 ──────────────────────────────────────────────────────────────────────

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"

# Anthropic API 要求这两个 Header，缺一不可
HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",   # API 版本，固定填这个
    "content-type":      "application/json",
}


# ── 函数一：构建消息 ──────────────────────────────────────────────────────────
# 对应你说的"内容构建函数"
# 注意：messages 作为参数传入，函数返回新列表——不修改外部状态
# （类比 Go 里返回新 slice 而不是 append 到全局变量）

def add_message(messages: list, role: str, content: str) -> list:
    """
    向对话历史追加一条消息，返回新列表。

    role: "user" 或 "assistant"
    """
    return messages + [{"role": role, "content": content}]


# ── 函数二：解析 SSE 流 ───────────────────────────────────────────────────────
# 对应你说的"单独解析逻辑"
# 职责：只负责从 HTTP 响应流中提取 token 文本，不做任何打印

def parse_sse_stream(response: requests.Response):
    """
    生成器函数：逐行读取 SSE 响应，yield 每个 token 文本片段。

    SSE 协议：每行格式为 "data: {json}"
    我们关心的 event 类型：content_block_delta（携带实际 token）

    🐍 Python 插播：yield 把函数变成"生成器"
    类比 Go 的 channel——调用方用 for 循环逐个消费，
    函数内部每 yield 一次就暂停，等调用方取走再继续。
    """
    for line in response.iter_lines():  # iter_lines() 逐行读，不等全部到达
        if not line:
            continue  # 跳过空行（SSE 用空行分隔 event）

        # line 是 bytes，先 decode
        text = line.decode("utf-8")

        # SSE 格式：只处理 "data: ..." 行
        if not text.startswith("data: "):
            continue

        payload = text[6:]  # 去掉 "data: " 前缀，拿到 JSON 字符串

        # 流结束标记
        if payload == "[DONE]":
            break

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # Anthropic 的 SSE event 类型很多，我们只关心这一种：
        # content_block_delta → delta.text 就是新到的 token
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                yield delta.get("text", "")


# ── 函数三：发送请求 ──────────────────────────────────────────────────────────
# 对应你说的"发送函数"
# temperature 作为参数，方便做实验对比

def send_request(
    messages:    list,
    system:      str   = "",
    temperature: float = 1.0,
    max_tokens:  int   = 1024,
) -> requests.Response:
    """
    构造请求体，发起流式 POST 请求，返回 Response 对象。

    stream=True：让 requests 不立刻读完响应体，保持连接打开，
    供 parse_sse_stream() 逐行消费。
    """
    body = {
        "model":       MODEL,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,         # 开启流式响应
        "messages":    messages,
    }

    # system prompt 单独放顶层字段（Anthropic API 的规定）
    if system:
        body["system"] = system

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=body,
        stream=True,   # requests 级别也要设 stream=True，才不会提前读完
    )

    # 非 200 直接报错，暴露原始错误信息方便调试
    response.raise_for_status()
    return response


# ── 函数四：格式化输出 ────────────────────────────────────────────────────────
# 对应你说的"结果输出函数"
# 调用 parse_sse_stream()，边收边打印，最后返回完整文本（供后续使用）

def stream_and_print(response: requests.Response, label: str = "") -> str:
    """
    消费 SSE 流：实时打印 token，返回完整回复文本。

    label：实验标签，如 "temperature=0.0"，方便对比输出
    """
    if label:
        print(f"\n{'─' * 50}")
        print(f"📌 {label}")
        print(f"{'─' * 50}")

    full_text = ""
    for token in parse_sse_stream(response):
        print(token, end="", flush=True)  # flush=True：不等换行就立刻输出
        full_text += token

    print()  # 流结束后换行
    return full_text


# ── 实验：Temperature 对比 ────────────────────────────────────────────────────

def run_temperature_experiment():
    """
    用同一个 prompt，对比 temperature=0 vs temperature=1 的输出差异。
    观察：确定性任务 vs 创意任务在不同温度下的行为。
    """

    # 实验 A：确定性任务（代码生成）
    # 预期：temperature=0 和 1 输出几乎一样——因为"正确答案"唯一
    prompt_code = "用 Python 写一个函数，计算两个数的最大公约数。只输出代码，不要解释。"

    print("\n" + "=" * 60)
    print("实验 A：确定性任务（GCD 函数）")
    print("= " * 30)

    for temp in [0.0, 1.0]:
        messages = add_message([], "user", prompt_code)
        resp = send_request(messages, temperature=temp)
        stream_and_print(resp, label=f"temperature={temp}")

    # 实验 B：创意任务（自由文本）
    # 预期：temperature=0 重复输出，temperature=1 每次不同
    prompt_creative = "用一句话描述'学习编程'这件事，要有创意。"

    print("\n" + "=" * 60)
    print("实验 B：创意任务（一句话描述）")
    print("= " * 30)

    for temp in [0.0, 1.0]:
        messages = add_message([], "user", prompt_creative)
        resp = send_request(messages, temperature=temp)
        stream_and_print(resp, label=f"temperature={temp}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 检查 API Key
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    run_temperature_experiment()