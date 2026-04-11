"""
[M4.2] MCP Client 实现（stdio 传输层）

职责边界：
  这个文件只负责"说 MCP 协议"——把 Python 函数调用翻译成 JSON-RPC 消息，
  再把 JSON-RPC 响应翻译回 Python 对象。
  它不知道 Anthropic API，不知道 Agent 逻辑，是纯粹的协议层。

架构图：
  Agent (调用方)
      │
      │  call_tool("execute_python", {"code": "print(1)"})
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │                 MCPClient                       │
  │                                                │
  │  __enter__  → _start() + _initialize()         │
  │               + list_tools() → cache           │
  │                                                │
  │  call_tool  → _send("tools/call", ...)         │
  │               → 解析结果 → 返回 str            │
  │                                                │
  │  __exit__   → _stop() 终止子进程               │
  └──────────────────┬─────────────────────────────┘
                     │
            stdin/stdout (JSON-RPC 2.0)
            每行一个 JSON 对象
                     │
                     ▼
         ┌───────────────────────┐
         │   m4_1_mcp_server.py  │
         │   (子进程，长驻)       │
         └───────────────────────┘

关键设计决策：
  1. 上下文管理器（with 语句）确保子进程一定被清理——类比 Go 的 defer
  2. _send vs _notify 分离：有响应的请求 vs 不需要响应的通知
     混用会导致死锁（等一个永远不会来的响应）
  3. tools 列表在 __enter__ 时缓存，会话内复用
  4. 格式转换（MCP → Anthropic）由调用方 Agent 负责，Client 只返回原始 MCP 格式
     原则：谁依赖目标格式，谁负责转换
"""

import json
import subprocess
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# MCPClient
# ══════════════════════════════════════════════════════════════════════════════

class MCPClient:
    """
    与 MCP Server 通信的客户端。

    设计为上下文管理器，保证子进程生命周期和 Client 对象绑定：
        with MCPClient("m4_1_mcp_server.py") as client:
            result = client.call_tool("execute_python", {"code": "print(1)"})
        # 离开 with 块，Server 进程自动被 terminate

    🐍 Python 插播：上下文管理器协议
      __enter__ → with 块开始时调用，返回值赋给 as 后的变量
      __exit__  → with 块结束时调用（含异常退出），等价于 Go 的 defer
      这是 Python 资源管理的标准惯用法（文件、锁、数据库连接都用这个）
    """

    def __init__(self, server_script: str):
        """
        Args:
            server_script: MCP Server 的 Python 脚本路径
        """
        self._server_script = str(server_script)
        self._proc: subprocess.Popen | None = None
        self._next_id = 1        # 单调递增的请求 ID，保证每个请求唯一可追踪
        self.tools: list = []    # 缓存 MCP 格式的工具列表（原始格式，不转换）

    # ── 上下文管理器 ────────────────────────────────────────────────────────

    def __enter__(self) -> "MCPClient":
        self._start()
        self._initialize()
        self.tools = self._list_tools()
        _log(f"Client ready. Tools: {[t['name'] for t in self.tools]}")
        return self  # 这个返回值会被赋给 `with ... as client` 里的 client

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        无论正常退出还是异常退出都会执行。
        返回 False 表示不吞掉异常，让异常继续向上传播。
        """
        self._stop()
        return False  # 不吞异常

    # ── 进程管理 ─────────────────────────────────────────────────────────────

    def _start(self) -> None:
        """
        启动 Server 子进程。

        关键参数说明：
          stdin=PIPE   → 我们可以向 Server 写数据（发请求）
          stdout=PIPE  → 我们可以从 Server 读数据（收响应）
          stderr=sys.stderr → Server 的日志直接透传到我们的 stderr
                              （不拦截，方便调试；不走 stdout 是协议约束）

        🐍 Python 插播：sys.executable
          = 当前运行的 Python 解释器的绝对路径
          用它而不是硬编码 "python3"，保证虚拟环境里的 Python 也能找到
          类比 Go 的 os.Executable()
        """
        _log(f"Starting server: {self._server_script}")
        self._proc = subprocess.Popen(
            [sys.executable, self._server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # Server 日志直接透传，不拦截
        )

    def _stop(self) -> None:
        """
        终止 Server 进程。
        terminate() 发 SIGTERM（优雅关闭），wait() 等它真正退出。
        如果 Server 卡住不退出，可以改用 kill()——但我们的 Server 会正确处理 stdin EOF。
        """
        if self._proc and self._proc.poll() is None:  # poll() is None → 进程还活着
            _log("Stopping server...")
            self._proc.terminate()
            self._proc.wait()
            _log("Server stopped")

    # ── JSON-RPC 底层 ────────────────────────────────────────────────────────

    def _send(self, method: str, params: dict) -> dict:
        """
        发送一个 JSON-RPC 请求并等待响应。

        协议约定（stdio 传输）：
          - 发：向 stdin 写一行 JSON + "\n"，flush 确保数据离开缓冲区
          - 收：从 stdout 读一行，解析 JSON

        死锁风险：
          如果发的是 notification（Server 不会响应），
          不能调这个函数——会永久阻塞在 readline()。
          通知要用 _notify()。

        跳过 notification：
          Server 可能主动推送 notification（无 "id" 字段）。
          我们循环读直到拿到有 "id" 的响应，跳过无关通知。
        """
        req_id = self._next_id
        self._next_id += 1

        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._write(request)

        # 读响应：跳过没有 id 的通知消息
        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                # stdout 关闭 → Server 意外退出
                raise RuntimeError("MCP Server closed unexpectedly")

            response = json.loads(raw.decode("utf-8"))

            if "id" not in response:
                # 这是 Server 主动推送的 notification，不是我们请求的响应，跳过
                _log(f"Skipping server notification: {response.get('method', '?')}")
                continue

            if response["id"] != req_id:
                # ID 不匹配（理论上不应发生，加个防御）
                _log(f"Warning: unexpected response id {response['id']}, expected {req_id}")
                continue

            break

        # JSON-RPC 错误：抛异常，让上层决定怎么处理
        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error [{err['code']}]: {err['message']}")

        return response["result"]

    def _notify(self, method: str, params: dict = None) -> None:
        """
        发送 JSON-RPC notification（不需要响应的单向消息）。

        Notification 和 Request 的区别：没有 "id" 字段。
        Server 收到后不会写任何东西到 stdout，所以不能等响应。

        使用场景：MCP 握手里的 notifications/initialized，
        告知 Server "我准备好了，可以开始正常通信了"。
        """
        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        self._write(notification)

    def _write(self, obj: dict) -> None:
        """向 Server stdin 写一行 JSON，必须 flush，否则数据卡在缓冲区里出不去"""
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        self._proc.stdin.flush()

    # ── MCP 协议方法 ─────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """
        MCP 握手：两步完成

        Step 1 - initialize 请求（有响应）
          Client 告知自己的版本和能力
          Server 返回自己的版本和支持的能力类型

        Step 2 - notifications/initialized 通知（无响应）
          Client 告知 Server "握手完成，我准备好了"
          Server 收到后才进入正常工作状态

        这个两步握手来自 MCP spec，类比 TCP 的 SYN/SYN-ACK/ACK——
        先协商能力，再确认双方都 ready。
        """
        result = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "m4-agent", "version": "1.0.0"},
            "capabilities": {},      # Client 暂不声明特殊能力
        })
        _log(f"Server info: {result.get('serverInfo', {})}")
        _log(f"Server capabilities: {list(result.get('capabilities', {}).keys())}")

        # 通知 Server 握手完成（不等响应！Server 收到后不写 stdout）
        self._notify("notifications/initialized")

    def _list_tools(self) -> list:
        """
        获取 Server 暴露的工具列表。

        返回原始 MCP 格式（inputSchema），不做任何转换。
        格式转换由调用方（Agent）负责——谁依赖目标格式，谁来转换。

        MCP 工具格式示例：
          {
            "name": "execute_python",
            "description": "...",
            "inputSchema": {           ← 注意：MCP 用 inputSchema（驼峰）
              "type": "object",
              "properties": {...},
              "required": [...]
            }
          }
        """
        result = self._send("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        调用指定工具，返回结果文本。

        MCP tools/call 响应格式：
          {
            "content": [{"type": "text", "text": "..."}],
            "isError": false
          }

        我们把多个 content block 拼成一个字符串返回，
        这样上层 Agent 可以直接把它塞进 tool_result 的 content 字段。

        Args:
            name:      工具名（必须和 Server 声明的 name 一致）
            arguments: 工具参数 dict（对应 inputSchema 里定义的字段）

        Returns:
            工具执行结果的文本内容（isError=True 时也返回文本，让模型看到错误信息）
        """
        _log(f"call_tool: {name}({list(arguments.keys())})")
        result = self._send("tools/call", {"name": name, "arguments": arguments})

        # 拼接所有 text 类型的 content block
        contents = result.get("content", [])
        text = "\n".join(
            block.get("text", "")
            for block in contents
            if block.get("type") == "text"
        )

        is_error = result.get("isError", False)
        if is_error:
            _log(f"Tool returned error: {text[:100]}")

        return text or "(no output)"


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def mcp_to_anthropic_tools(mcp_tools: list) -> list:
    """
    将 MCP 格式的工具列表转换为 Anthropic API 格式。

    为什么放在这个文件而不是 Agent 文件？
    这个函数在逻辑上属于"协议边界转换"，和 MCPClient 紧密相关。
    但它不是 MCPClient 的方法，因为 Client 本身不应该知道 Anthropic。
    放在同一个文件作为模块级函数，是一种折中——Agent import 时顺手拿走。

    格式差异（唯一区别）：
      MCP:       "inputSchema"  （驼峰，来自 JSON Schema 标准）
      Anthropic: "input_schema" （下划线，Anthropic 的 API 约定）

    示例转换：
      MCP 输入:
        {"name": "execute_python", "description": "...",
         "inputSchema": {"type": "object", "properties": {...}}}

      Anthropic 输出:
        {"name": "execute_python", "description": "...",
         "input_schema": {"type": "object", "properties": {...}}}
    """
    return [
        {
            "name":         tool["name"],
            "description":  tool.get("description", ""),
            "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
        }
        for tool in mcp_tools
    ]


def _log(message: str) -> None:
    """Client 侧调试日志，同样走 stderr，不污染 stdout"""
    print(f"[MCP Client] {message}", file=sys.stderr)