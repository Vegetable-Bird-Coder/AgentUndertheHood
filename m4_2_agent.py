"""
[M4.2] MCP Agent 集成层

职责：
  把 MCP Client（协议层）和 Anthropic API（模型层）组装成一个完整的 Agent。
  这一层的代码量很少——大部分复杂逻辑已经在各自的层里封装好了。

与 M2.4 Mini Agent 的区别：
  M2.4:  tools 是硬编码的 Python 函数（本地调用）
  M4.2:  tools 从 MCP Server 动态发现，执行走 JSON-RPC（跨进程调用）
  但 Tool-use Loop 的逻辑完全相同——这正是 MCP 的价值所在。

数据流：
  ┌─── 启动阶段 ──────────────────────────────────────────────────────┐
  │  MCPClient.__enter__()                                           │
  │    └─ spawn Server 进程 → initialize 握手 → list_tools 缓存      │
  │  mcp_to_anthropic_tools(client.tools) → anthropic_tools          │
  └──────────────────────────────────────────────────────────────────┘

  ┌─── 每轮循环 ──────────────────────────────────────────────────────┐
  │  send_request(messages, tools=anthropic_tools)                   │
  │    └─ Claude 返回 tool_use blocks                                │
  │  for each tool_use:                                              │
  │    client.call_tool(name, args) → JSON-RPC → Server → 结果      │
  │    追加 tool_result 到 messages                                   │
  │  继续循环，直到 stop_reason == "end_turn"                         │
  └──────────────────────────────────────────────────────────────────┘

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m4_2_agent.py
"""

import sys
from pathlib import Path

# 复用 M2.1 的 Anthropic API 调用基础设施
from m2_1_tool_use import send_request, add_message

# M4.2 新增：MCP 协议层
from m4_2_mcp_client import MCPClient, mcp_to_anthropic_tools


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# MCP Server 脚本路径（和 Agent 脚本放在同一目录）
SERVER_SCRIPT = Path(__file__).parent / "m4_1_mcp_server.py"

MAX_TURNS = 10   # 防止无限循环

SYSTEM_PROMPT = """You are a helpful coding assistant with access to powerful tools.

You can:
- Execute Python code with `execute_python` to run calculations, process data, or test logic
- Write files with `write_file` to save results to the workspace

When executing code that writes files, use the write_file tool directly rather than 
writing file I/O code inside execute_python — it's cleaner and respects the sandbox.

Always explain what you're doing and show the results clearly."""


# ══════════════════════════════════════════════════════════════════════════════
# Agent 主循环
# ══════════════════════════════════════════════════════════════════════════════

def agent_loop(user_query: str, client: MCPClient) -> list:
    """
    核心 Tool-use Loop，和 M2.4 完全同构。
    唯一变化：工具执行从本地函数调用 → client.call_tool()（MCP 跨进程调用）

    Args:
        user_query: 用户输入
        client:     已初始化的 MCPClient（tools 已缓存）

    Returns:
        完整的 messages 历史（供调试用）
    """
    # MCP 格式 → Anthropic 格式（唯一的格式转换，放在 Agent 层）
    anthropic_tools = mcp_to_anthropic_tools(client.tools)

    messages = add_message([], "user", user_query)

    print(f"\n{'═'*60}")
    print(f"🧑 用户: {user_query}")
    print(f"🔌 可用工具: {[t['name'] for t in anthropic_tools]}")
    print(f"{'═'*60}")

    for turn in range(MAX_TURNS):
        # ── Step 1: 调 Claude ────────────────────────────────────────────────
        response = send_request(
            messages,
            tools=anthropic_tools,
            system=SYSTEM_PROMPT,
            max_tokens=2048,
        )

        stop_reason = response.get("stop_reason")
        content     = response.get("content", [])

        # 把 Claude 的回复追加到历史（不管是文本还是 tool_use，都要存）
        messages = add_message(messages, "assistant", content)

        # ── Step 2: 处理 Claude 的决策 ──────────────────────────────────────
        if stop_reason == "end_turn":
            # Claude 决定不再调工具，直接给出最终回答
            print(f"\n🤖 最终回答:")
            for block in content:
                if block.get("type") == "text":
                    print(block["text"])
            break

        if stop_reason == "tool_use":
            # Claude 决定调工具，我们代为执行并把结果喂回去
            tool_results = []

            for block in content:
                if block.get("type") != "tool_use":
                    continue

                tool_name = block["name"]
                tool_args = block["input"]
                tool_id   = block["id"]

                print(f"\n🔧 [Turn {turn+1}] 调用工具: {tool_name}")
                print(f"   参数: {tool_args}")

                # ← 这里是 M4.2 的核心替换：本地函数 → MCP 跨进程调用
                result_text = client.call_tool(tool_name, tool_args)

                print(f"   结果: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")

                # 按 Anthropic API 格式包装 tool_result
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     result_text,
                })

            # 把所有工具结果作为 user 角色注入（Anthropic API 约定）
            messages = add_message(messages, "user", tool_results)

        else:
            # 遇到未知的 stop_reason，打印并退出
            print(f"\n⚠️  Unexpected stop_reason: {stop_reason}")
            break

    else:
        print(f"\n⚠️  达到最大轮次限制 ({MAX_TURNS})")

    return messages


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    演示两个任务：
    1. 计算斐波那契数列并写入文件（需要 execute_python + write_file 协作）
    2. 验证文件内容（只需 execute_python 读文件）

    两个任务共用同一个 MCPClient 实例（同一个 Server 子进程），
    验证"进程长驻复用"的设计。
    """
    print("═"*60)
    print("M4.2 MCP Agent Demo")
    print("═"*60)

    # with 语句保证 Server 子进程一定被清理
    with MCPClient(SERVER_SCRIPT) as client:
        print(f"\n✅ MCP Server 启动成功")
        print(f"   发现 {len(client.tools)} 个工具: {[t['name'] for t in client.tools]}")

        # 任务 1：计算 + 写文件
        agent_loop(
            "计算斐波那契数列前15项，把结果写入 workspace/fib.txt，"
            "文件里每行一个数字，最后一行写上总和。",
            client,
        )

        # 任务 2：验证文件（用 execute_python 读取并展示）
        agent_loop(
            "读取 workspace/fib.txt 的内容，最后一行是所有数字的总和,"
            "验证文件是否正确写入，并计算文件中所有数字的平均值（除了最后一行）。",
            client,
        )

    print(f"\n{'═'*60}")
    print("✅ MCP Server 已关闭，演示结束")


if __name__ == "__main__":
    main()