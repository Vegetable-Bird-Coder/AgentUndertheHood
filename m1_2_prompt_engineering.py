"""
[M1.2] Prompt Engineering 实战 - Go Code Reviewer
对比三种策略：Zero-shot / CoT / Few-shot

复用 M1.1 的底层函数（send_request / stream_and_print / add_message）

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m1_2_prompt_engineering.py
"""

import json
import os
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 复用 M1.1 的底层基础设施（原封不动）
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"

HEADERS = {
    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}


def add_message(messages: list, role: str, content: str) -> list:
    return messages + [{"role": role, "content": content}]


def parse_sse_stream(response: requests.Response):
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
                yield delta.get("text", "")


def send_request(
    messages:    list,
    system:      str   = "",
    temperature: float = 0.0,   # Review 任务用低温度：要求稳定输出，不要创意
    max_tokens:  int   = 1024,
) -> requests.Response:
    body = {
        "model":       MODEL,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,
        "messages":    messages,
    }
    if system:
        body["system"] = system

    response = requests.post(API_URL, headers=HEADERS, json=body, stream=True)
    response.raise_for_status()
    return response


def stream_and_print(response: requests.Response, label: str = "") -> str:
    if label:
        print(f"\n{'─' * 60}")
        print(f"📌 {label}")
        print(f"{'─' * 60}")

    full_text = ""
    for token in parse_sse_stream(response):
        print(token, end="", flush=True)
        full_text += token

    print()
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# M1.2 新增内容从这里开始
# ══════════════════════════════════════════════════════════════════════════════

# ── 测试用的"有问题的 Go 代码" ────────────────────────────────────────────────
#
# 故意埋了 4 个问题，看不同策略能各找出几个：
#   1. 没有检查 err（第 13 行的 os.Open）
#   2. 用了裸 panic，生产代码不该这样做
#   3. 资源泄漏：file 打开后没有 defer file.Close()
#   4. 变量名 d 没有语义（违反 Go 命名惯例）

BUGGY_GO_CODE = '''
package main

import (
    "fmt"
    "os"
    "strings"
)

func processFile(path string) string {
    file, err := os.Open(path)
    if err != nil {
        panic(err)
    }

    d := make([]byte, 1024)
    n, _ := file.Read(d)

    result := strings.TrimSpace(string(d[:n]))
    return result
}

func main() {
    content := processFile("input.txt")
    fmt.Println(content)
}
'''


# ── System Prompt：三种策略共用 ───────────────────────────────────────────────
#
# 注意：角色定义放 System，具体代码放 User——遵循方案 A

SYSTEM_REVIEWER = """你是一位资深 Go 语言工程师，负责代码 Review。
你对 Go 的惯例（idiomatic Go）、错误处理、资源管理和命名规范有深刻理解。
请用中文回复。"""


# ── 策略一：Zero-shot（直接问）────────────────────────────────────────────────
#
# 最朴素的方式：直接丢代码，什么格式要求都不给
# 预期结果：模型会给出 review，但格式和深度不可控

def build_prompt_zero_shot(code: str) -> tuple[str, list]:
    """返回 (system, messages)"""
    user_content = f"请 review 这段 Go 代码：\n\n```go\n{code}\n```"
    messages = add_message([], "user", user_content)
    return SYSTEM_REVIEWER, messages


# ── 策略二：Chain-of-Thought ──────────────────────────────────────────────────
#
# 显式要求模型先推理再输出结论
# 关键词："先分析...，再给出..."——强制生成中间推理 Token
# 预期结果：发现更多问题，推理过程可见，更容易追溯为什么

def build_prompt_cot(code: str) -> tuple[str, list]:
    """返回 (system, messages)"""
    user_content = f"""请 review 这段 Go 代码：

```go
{code}
```

请按以下步骤思考：
1. 先逐行检查代码，列出所有你注意到的潜在问题（不要遗漏）
2. 对每个问题，解释为什么它是问题，以及在生产环境中可能导致什么后果
3. 最后给出修改建议"""

    messages = add_message([], "user", user_content)
    return SYSTEM_REVIEWER, messages


# ── 策略三：Few-shot ──────────────────────────────────────────────────────────
#
# 给两个示例，规定输出格式：severity 分级 + 位置 + 说明 + 建议
# 预期结果：输出格式高度一致，可以直接解析/入库——这是生产环境最常用的方式
#
# 🐍 Python 插播：三引号字符串（"""..."""）是多行字符串字面量
# 类比 Go 的反引号字符串 `...`，可以跨行，不需要转义

FEW_SHOT_EXAMPLES = """
以下是我期望的 review 格式示例：

【示例 1】
代码：
```go
func divide(a, b int) int {
    return a / b
}
```
Review 结果：
[HIGH] 第2行 — 除零风险
说明：当 b=0 时会触发 runtime panic，调用方无法捕获。
建议：修改签名为 func divide(a, b int) (int, error)，b==0 时返回错误。

---

【示例 2】
代码：
```go
func getName() string {
    n := "alice"
    return n
}
```
Review 结果：
[LOW] 第2行 — 无意义中间变量
说明：变量 n 没有语义，直接 return "alice" 更清晰。
建议：删除中间变量，直接 return "alice"。

---

现在请用完全相同的格式 review 以下代码（每个问题单独一条，severity 用 HIGH/MEDIUM/LOW）：
"""

def build_prompt_few_shot(code: str) -> tuple[str, list]:
    """返回 (system, messages)"""
    user_content = f"""{FEW_SHOT_EXAMPLES}
```go
{code}
```"""
    messages = add_message([], "user", user_content)
    return SYSTEM_REVIEWER, messages


# ── 实验主函数 ────────────────────────────────────────────────────────────────

def run_review_experiment():
    """
    用同一段有 bug 的 Go 代码，跑三种 prompt 策略。
    观察重点：
      - 各策略能找出几个 bug？（满分 4 个）
      - 输出格式的可控程度？
      - CoT 的推理过程是否帮助发现了更多问题？
    """
    print("=" * 60)
    print("🔍 Go Code Review 实验")
    print("测试代码故意埋了 4 个问题，看各策略能找出几个")
    print("=" * 60)

    print("\n📄 待 review 的代码：")
    print(BUGGY_GO_CODE)

    # 策略一：Zero-shot
    system, messages = build_prompt_zero_shot(BUGGY_GO_CODE)
    resp = send_request(messages, system=system)
    stream_and_print(resp, label="策略一：Zero-shot（直接问）")

    # 策略二：CoT
    system, messages = build_prompt_cot(BUGGY_GO_CODE)
    resp = send_request(messages, system=system)
    stream_and_print(resp, label="策略二：Chain-of-Thought（先分析再结论）")

    # 策略三：Few-shot
    system, messages = build_prompt_few_shot(BUGGY_GO_CODE)
    resp = send_request(messages, system=system)
    stream_and_print(resp, label="策略三：Few-shot（规定输出格式）")

    # 实验结束后的讨论引导
    print("\n" + "=" * 60)
    print("📊 实验结束，建议讨论以下问题：")
    print("  1. 三种策略各找出了几个 bug？有没有只有某种策略才发现的？")
    print("  2. Few-shot 的格式控制效果怎么样？")
    print("  3. CoT 的推理过程有没有帮助，或者带来了不必要的\"废话\"？")
    print("  4. 哪种策略适合集成进 CI/CD 流水线？为什么？")
    print("=" * 60)


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    run_review_experiment()