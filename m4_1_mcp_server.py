"""
[M4.1] MCP Server 实现（stdio 传输层）

架构图：
  Claude Desktop / MCP Client
         │
         │  stdin  → JSON-RPC 2.0 请求
         │  stdout ← JSON-RPC 2.0 响应
         │
  ┌──────▼──────────────────────────────┐
  │           MCP Server                │
  │                                     │
  │  main_loop()                        │
  │    └─ dispatcher()  ← 分发器         │
  │         ├─ handle_initialize()      │
  │         ├─ handle_tools_list()      │
  │         ├─ handle_tools_call()      │
  │         │    ├─ tool: execute_python│
  │         │    └─ tool: write_file    │
  │         ├─ handle_resources_list()  │
  │         └─ handle_resources_read()  │
  │              └─ resource: file://   │
  └─────────────────────────────────────┘

设计决策：
  - stdio 传输：每行一个 JSON 对象，最简单的进程间通信
  - 工具注册表：dict[name → handler]，和 M2.1 ToolRegistry 同一个模式
  - execute_python 用 subprocess 隔离执行，有超时保护
  - 所有日志写 stderr，stdout 只输出协议消息（不能混）
  - 错误统一封装为 JSON-RPC error 对象，不抛异常到上层

运行方式（手动测试）：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m4_1_mcp_server.py
  # 然后在 stdin 粘贴 JSON-RPC 消息测试
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

SERVER_NAME    = "code-toolbox"
SERVER_VERSION = "1.0.0"

# execute_python 的执行超时（秒）
PYTHON_TIMEOUT = 10

# 工作区根目录：Resource 只能读这个目录下的文件（安全沙箱）
WORKSPACE_DIR = Path(os.getcwd()) / "workspace"


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC 2.0 帮助函数
# ══════════════════════════════════════════════════════════════════════════════

def make_result(request_id, result: dict) -> dict:
    """构造成功响应"""
    return {
        "jsonrpc": "2.0",
        "id":      request_id,
        "result":  result,
    }


def make_error(request_id, code: int, message: str) -> dict:
    """
    构造错误响应。

    JSON-RPC 2.0 标准错误码：
      -32700  Parse error      JSON 解析失败
      -32600  Invalid Request  不是合法的 Request 对象
      -32601  Method not found 方法不存在
      -32602  Invalid params   参数错误
      -32603  Internal error   服务器内部错误
    """
    return {
        "jsonrpc": "2.0",
        "id":      request_id,
        "error":   {"code": code, "message": message},
    }


def send_response(response: dict) -> None:
    """
    把响应写到 stdout。

    关键约束：stdout 只能有协议消息，日志必须写 stderr。
    原因：MCP Client 用行读取 stdout，任何非 JSON 内容都会导致解析失败。

    🐍 Python 插播：
      sys.stdout.write + flush 等价于 fmt.Fprintln(os.Stdout, ...)
      flush=True 强制立刻发送，不等缓冲区满——进程间通信必须加这个，
      否则 Client 可能一直等不到响应（数据卡在缓冲区里没出去）。
    """
    line = json.dumps(response, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    """调试日志写 stderr，绝对不能写 stdout"""
    print(f"[MCP Server] {message}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 实现
# ══════════════════════════════════════════════════════════════════════════════

def tool_execute_python(arguments: dict) -> dict:
    """
    在子进程里执行 Python 代码，返回 stdout 输出。

    安全考虑：
      - subprocess 隔离：崩溃不影响 Server 主进程
      - timeout=PYTHON_TIMEOUT：防止死循环占用资源
      - 不限制 import——这是开发工具箱，不是公开服务

    返回格式（MCP Tool 标准）：
      {"content": [{"type": "text", "text": "..."}], "isError": bool}
    """
    code = arguments.get("code", "")
    if not code.strip():
        return {
            "content": [{"type": "text", "text": "Error: empty code"}],
            "isError": True,
        }

    log(f"execute_python: {code[:50]}...")

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],  # sys.executable = 当前 Python 解释器路径
            cwd=WORKSPACE_DIR,
            capture_output = True,          # 捕获 stdout 和 stderr
            text           = True,          # 返回字符串而不是 bytes
            timeout        = PYTHON_TIMEOUT,
        )

        # stdout + stderr 都返回给调用方，方便调试
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        is_error = result.returncode != 0
        return {
            "content": [{"type": "text", "text": output or "(no output)"}],
            "isError": is_error,
        }

    except subprocess.TimeoutExpired:
        return {
            "content": [{"type": "text", "text": f"Error: execution timeout ({PYTHON_TIMEOUT}s)"}],
            "isError": True,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        }


def tool_write_file(arguments: dict) -> dict:
    """
    向工作区写入文件。

    安全沙箱：路径必须在 WORKSPACE_DIR 内，防止路径穿越攻击。
    例如 path="../../etc/passwd" 会被拒绝。

    🐍 Python 插播：
      Path.resolve() 把相对路径、../、软链接全部展开为绝对路径，
      类似 Go 的 filepath.EvalSymlinks + filepath.Abs。
      str.startswith 检查是否在沙箱目录内——简单有效。
    """
    path    = arguments.get("path", "")
    content = arguments.get("content", "")

    if not path:
        return {
            "content": [{"type": "text", "text": "Error: path is required"}],
            "isError": True,
        }

    # 路径安全检查
    target = (WORKSPACE_DIR / path).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
        return {
            "content": [{"type": "text", "text": f"Error: path outside workspace: {path}"}],
            "isError": True,
        }

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        log(f"write_file: {target}")
        return {
            "content": [{"type": "text", "text": f"OK: written {len(content)} chars to {path}"}],
            "isError": False,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 注册表
# ══════════════════════════════════════════════════════════════════════════════

# Tool 定义：description 是给模型看的，直接决定模型会不会调用这个 Tool
# （和 M2.1 的教训一致：description quality 是一等工程关注点）
TOOLS = [
    {
        "name":        "execute_python",
        "description": "Execute a Python code snippet and return stdout/stderr output. Use for calculations, data processing, or testing logic.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "code": {
                    "type":        "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name":        "write_file",
        "description": "Write content to a file in the workspace directory. Creates parent directories if needed.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "path": {
                    "type":        "string",
                    "description": "Relative file path within workspace (e.g. 'output/result.txt')",
                },
                "content": {
                    "type":        "string",
                    "description": "File content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
]

# 注册表：name → handler 函数
# 和 M2.1 ToolRegistry 同一个模式，只是这里用普通 dict 而不是类
TOOL_HANDLERS = {
    "execute_python": tool_execute_python,
    "write_file":     tool_write_file,
}


# ══════════════════════════════════════════════════════════════════════════════
# Resource 实现
# ══════════════════════════════════════════════════════════════════════════════

def list_workspace_resources() -> list[dict]:
    """
    动态列出工作区所有文件。

    这正是 Capability Discovery 的价值所在：
    文件随时在变，Server 每次动态扫描返回，Client 无需更新配置。
    """
    WORKSPACE_DIR.mkdir(exist_ok=True)
    resources = []
    for file_path in sorted(WORKSPACE_DIR.rglob("*")):
        if file_path.is_file():
            relative = file_path.relative_to(WORKSPACE_DIR)
            resources.append({
                "uri":      f"file://workspace/{relative}",
                "name":     str(relative),
                "mimeType": "text/plain",
            })
    return resources


def read_workspace_resource(uri: str) -> dict:
    """
    读取工作区文件内容。

    URI 格式：file://workspace/{relative_path}
    同样有路径安全检查，防止穿越攻击。
    """
    prefix = "file://workspace/"
    if not uri.startswith(prefix):
        return {"error": f"Unknown resource URI: {uri}"}

    relative_path = uri[len(prefix):]
    target = (WORKSPACE_DIR / relative_path).resolve()

    # 路径安全检查（和 write_file 一致）
    if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
        return {"error": f"Path outside workspace: {relative_path}"}

    if not target.exists():
        return {"error": f"File not found: {relative_path}"}

    try:
        content = target.read_text(encoding="utf-8")
        return {
            "contents": [{
                "uri":      uri,
                "mimeType": "text/plain",
                "text":     content,
            }]
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Request Handlers（对应 dispatcher 的每个分支）
# ══════════════════════════════════════════════════════════════════════════════

def handle_initialize(request_id, params: dict) -> dict:
    """
    握手第一步：Client 告知自己的版本，Server 返回自己的能力声明。

    protocolVersion 必须回应 Client 请求的版本（或降级到支持的最高版本）。
    capabilities 声明 Server 支持哪些特性——这里声明支持 tools 和 resources。
    """
    log(f"initialize: client={params.get('clientInfo', {})}")
    return make_result(request_id, {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name":    SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "capabilities": {
            "tools":     {},      # 声明支持 tools
            "resources": {},      # 声明支持 resources
        },
    })


def handle_tools_list(request_id, params: dict) -> dict:
    """返回所有 Tool 定义，Client 会把这些注入到模型的 context 里"""
    log("tools/list")
    return make_result(request_id, {"tools": TOOLS})


def handle_tools_call(request_id, params: dict) -> dict:
    """
    分发 Tool 调用到对应的 handler。

    这里是真正的 dispatcher 内层——
    外层 dispatcher 按 method 路由，内层按 tool name 路由。
    两层路由，结构对称。
    """
    name      = params.get("name", "")
    arguments = params.get("arguments", {})

    log(f"tools/call: {name}({list(arguments.keys())})")

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return make_error(request_id, -32601, f"Tool not found: {name}")

    result = handler(arguments)
    return make_result(request_id, result)


def handle_resources_list(request_id, params: dict) -> dict:
    """动态返回工作区文件列表"""
    log("resources/list")
    resources = list_workspace_resources()
    return make_result(request_id, {"resources": resources})


def handle_resources_read(request_id, params: dict) -> dict:
    """读取指定 URI 的 Resource 内容"""
    uri = params.get("uri", "")
    log(f"resources/read: {uri}")
    result = read_workspace_resource(uri)
    if "error" in result:
        return make_error(request_id, -32602, result["error"])
    return make_result(request_id, result)


# ══════════════════════════════════════════════════════════════════════════════
# 主分发器 + 主循环
# ══════════════════════════════════════════════════════════════════════════════

# method → handler 的映射表
# 🐍 Python 插播：
#   这里用 dict 替代 if-elif 链，是 Python 里常见的"表驱动分发"惯用法。
#   Go 里用 switch，Python 里用 dict——本质是同一个模式。
DISPATCHER = {
    "initialize":       handle_initialize,
    "tools/list":       handle_tools_list,
    "tools/call":       handle_tools_call,
    "resources/list":   handle_resources_list,
    "resources/read":   handle_resources_read,
    # notifications（Client 发来的通知，不需要响应）
    "notifications/initialized": None,
}


def dispatch(request: dict) -> dict | None:
    """
    解析 JSON-RPC 请求，路由到对应 handler。

    返回 None 表示这是通知（notification），不需要响应。
    通知的特征：没有 "id" 字段，或 method 以 "notifications/" 开头。
    """
    request_id = request.get("id")        # None 表示这是 notification
    method     = request.get("method", "")
    params     = request.get("params", {})

    # 通知：不需要响应
    if method.startswith("notifications/"):
        log(f"notification: {method} (ignored)")
        return None

    handler = DISPATCHER.get(method)
    if handler is None:
        return make_error(request_id, -32601, f"Method not found: {method}")

    try:
        return handler(request_id, params)
    except Exception as e:
        log(f"handler error: {e}")
        return make_error(request_id, -32603, f"Internal error: {e}")


def main_loop() -> None:
    """
    主循环：逐行读 stdin，解析 JSON-RPC，分发，写响应到 stdout。

    每行一个 JSON 对象——这是 stdio 传输层的约定。
    EOF（Client 关闭连接）时退出循环。
    """
    log(f"Server started: {SERVER_NAME} v{SERVER_VERSION}")
    log(f"Workspace: {WORKSPACE_DIR}")
    WORKSPACE_DIR.mkdir(exist_ok=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # 解析 JSON
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            send_response(make_error(None, -32700, f"Parse error: {e}"))
            continue

        # 分发
        response = dispatch(request)

        # 有响应才发（notification 返回 None，不发）
        if response is not None:
            send_response(response)

    log("Server exiting (stdin closed)")


# ══════════════════════════════════════════════════════════════════════════════
# 手动测试入口
# ══════════════════════════════════════════════════════════════════════════════

def run_self_test() -> None:
    """
    不依赖外部 Client，直接在进程内模拟几条请求，验证 Server 核心逻辑。
    用于开发阶段快速验证，不走 stdin/stdout。
    """
    print("=" * 60)
    print("MCP Server 自测")
    print("=" * 60)

    test_cases = [
        # 1. 握手
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "test"}}},

        # 2. 能力发现
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list",     "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},

        # 3. 执行 Python
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "execute_python", "arguments": {"code": "print(1 + 1)"}}},

        # 4. 写文件
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "write_file",
                    "arguments": {"path": "hello.txt", "content": "Hello from MCP!\n"}}},

        # 5. 再次列出 Resource（文件已存在）
        {"jsonrpc": "2.0", "id": 6, "method": "resources/list", "params": {}},

        # 6. 读文件
        {"jsonrpc": "2.0", "id": 7, "method": "resources/read",
         "params": {"uri": "file://workspace/hello.txt"}},

        # 7. 不存在的 Tool
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
         "params": {"name": "nonexistent", "arguments": {}}},

        # 8. 路径穿越攻击（应该被拒绝）
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
         "params": {"name": "write_file",
                    "arguments": {"path": "../../etc/passwd", "content": "hacked"}}},
    ]

    for req in test_cases:
        print(f"\n▶ Request {req['id']}: {req['method']}")
        if req['method'] == 'tools/call':
            print(f"  tool: {req['params']['name']}")

        response = dispatch(req)
        if response:
            if "result" in response:
                result = response["result"]
                # 只打印关键字段，避免刷屏
                if "tools" in result:
                    print(f"  ✅ tools: {[t['name'] for t in result['tools']]}")
                elif "resources" in result:
                    print(f"  ✅ resources: {[r['uri'] for r in result['resources']]}")
                elif "content" in result:
                    text = result["content"][0]["text"] if result["content"] else ""
                    is_error = result.get("isError", False)
                    status = "❌" if is_error else "✅"
                    print(f"  {status} output: {text[:80]}")
                elif "contents" in result:
                    text = result["contents"][0].get("text", "")[:60]
                    print(f"  ✅ content: {text!r}")
                else:
                    print(f"  ✅ {result}")
            elif "error" in response:
                print(f"  ⚠️  error: {response['error']['message']}")


if __name__ == "__main__":
    # 有 --test 参数时跑自测，否则启动真正的 stdio Server
    if "--test" in sys.argv:
        run_self_test()
    else:
        main_loop()
