"""
[M1.3] Context Window 与 Token 经济学
实验 A：Tokenizer 效率对比（中英文 / 代码 / JSON）
实验 B：Needle in a Haystack（Context Rot 验证）

运行前提：
  pip install tiktoken
  export ANTHROPIC_API_KEY="sk-ant-..."

运行方式：
  python m1_3_context_window.py
"""

import json
import os
import tiktoken
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 复用底层基础设施（与 M1.1 / M1.2 相同）
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
    temperature: float = 0.0,
    max_tokens:  int   = 512,
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
# 工具函数：Token 计数
# ══════════════════════════════════════════════════════════════════════════════

# 🐍 Python 插播：模块级变量（类比 Go 的 package-level var）
# tiktoken.get_encoding() 加载分词表，比较重，只初始化一次
_ENCODER = tiktoken.get_encoding("cl100k_base")   # GPT-4 / Claude 近似编码


def count_tokens(text: str) -> int:
    """计算文本的 token 数。"""
    return len(_ENCODER.encode(text))


def token_stats(label: str, text: str) -> dict:
    """
    计算并打印一段文本的 token 统计信息。
    返回 dict，方便后续比较。

    tokens_per_char < 1 意味着"一个字符不到一个 token"，编码效率高
    tokens_per_char > 1 意味着"一个字符需要超过一个 token"，编码效率低

    🐍 Python 插播：f-string（格式化字符串）
    类比 Go 的 fmt.Sprintf，但更简洁：
      Go:     fmt.Sprintf("%.2f", value)
      Python: f"{value:.2f}"
    """
    chars  = len(text)
    tokens = count_tokens(text)
    ratio  = tokens / chars if chars > 0 else 0   # tokens per char

    print(f"  {label:<30} | chars={chars:>5} | tokens={tokens:>5} | ratio={ratio:.3f} tok/char")

    return {"label": label, "chars": chars, "tokens": tokens, "ratio": ratio}


# ══════════════════════════════════════════════════════════════════════════════
# 实验 A：Tokenizer 效率对比
# ══════════════════════════════════════════════════════════════════════════════

# ── A1: 同一内容，中英文对比 ──
# 目的：分离两个变量——
#   变量1：中文本身更简洁（字符数少）
#   变量2：tokenizer 对中文编码效率低（每字符 token 数更多）
# 通过对比 ratio（tok/char），我们能看到变量2 的纯粹影响

EN_SENTENCES = [
    # (中文含义, 英文句子)
    (
        "【描述排序算法】",
        "Bubble sort repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order.",
    ),
    (
        "【描述错误处理】",
        "In Go, errors are values, and functions return errors as the last return value, which the caller must explicitly check.",
    ),
    (
        "【描述网络请求】",
        "The HTTP client sends a POST request with a JSON body to the API endpoint, then parses the streaming response line by line.",
    ),
]

ZH_SENTENCES = [
    # 与上面英文对应的中文翻译（语义完全相同）
    (
        "【描述排序算法】",
        "冒泡排序反复遍历列表，比较相邻元素，如果顺序错误就交换它们的位置。",
    ),
    (
        "【描述错误处理】",
        "在 Go 中，错误是值，函数将错误作为最后一个返回值返回，调用方必须显式检查。",
    ),
    (
        "【描述网络请求】",
        "HTTP 客户端向 API 端点发送带有 JSON 请求体的 POST 请求，然后逐行解析流式响应。",
    ),
]

# ── A2: 不同文本类型对比（固定英文，测类型差异）──
# 代码 / JSON / 普通文字 的 token 效率是不同的

CODE_SAMPLE = """\
func processFile(path string) (string, error) {
    file, err := os.Open(path)
    if err != nil {
        return "", fmt.Errorf("open file: %w", err)
    }
    defer file.Close()
    buf := make([]byte, 1024)
    n, err := file.Read(buf)
    if err != nil {
        return "", fmt.Errorf("read file: %w", err)
    }
    return strings.TrimSpace(string(buf[:n])), nil
}"""

JSON_SAMPLE = """\
{
  "model": "claude-haiku-4-5-20251001",
  "max_tokens": 1024,
  "temperature": 0.7,
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Explain context windows in LLMs."}
  ]
}"""

PROSE_SAMPLE = """\
Large language models process text by converting it into tokens before feeding it \
into the neural network. The context window defines how many tokens the model can \
attend to simultaneously. When the context is full, older tokens are dropped and \
the model loses access to that information, which can degrade response quality."""


def run_experiment_a():
    print("\n" + "=" * 70)
    print("实验 A：Tokenizer 效率对比")
    print("=" * 70)

    # ── A1：同一语义，中英文 ratio 对比 ──
    print("\n【A1】同一语义：英文 vs 中文的 tok/char 对比")
    print("  关注 ratio 列：ratio 越高 = 每个字符消耗 token 越多")
    print()

    en_stats = []
    zh_stats = []

    for (en_label, en_text), (zh_label, zh_text) in zip(EN_SENTENCES, ZH_SENTENCES):
        en_stat = token_stats(f"EN {en_label}", en_text)
        zh_stat = token_stats(f"ZH {zh_label}", zh_text)
        en_stats.append(en_stat)
        zh_stats.append(zh_stat)
        print()

    # 汇总：计算平均 ratio，得出结论
    avg_en_ratio = sum(s["ratio"] for s in en_stats) / len(en_stats)
    avg_zh_ratio = sum(s["ratio"] for s in zh_stats) / len(zh_stats)

    print(f"  英文平均 tok/char：{avg_en_ratio:.3f}")
    print(f"  中文平均 tok/char：{avg_zh_ratio:.3f}")
    print(f"  结论：中文 tokenizer 效率比英文低 {avg_zh_ratio / avg_en_ratio:.1f}x")
    print()
    print("  📌 工程含义：中文 prompt 消耗的 token 是英文的约 N 倍，")
    print("     但中文字符数通常更少。两者叠加，实际 token 消耗差距不如直觉中那么大。")

    # ── A2：不同文本类型 ──
    print("\n【A2】不同文本类型的 token 效率")
    print("  (同为英文，测量代码 / JSON / 普通文字 的差异)")
    print()

    token_stats("普通文字 (prose)",  PROSE_SAMPLE)
    token_stats("Go 代码 (code)",    CODE_SAMPLE)
    token_stats("JSON 数据 (json)",  JSON_SAMPLE)

    print()
    print("  📌 工程含义：代码和 JSON 里有大量标点、缩进、换行，")
    print("     tokenizer 效率低于普通文字。把大块 JSON 塞进 context 要特别注意 token 消耗。")


# ══════════════════════════════════════════════════════════════════════════════
# 实验 B：Needle in a Haystack
# ══════════════════════════════════════════════════════════════════════════════

# ── 干草堆：一段中性的技术背景文字，本身不包含"答案" ──
# 重复拼接到约 3000 token（不需要精确）

_HAYSTACK_UNIT = """\
The history of computer science spans several decades of innovation and discovery. \
Early computers were room-sized machines that required teams of engineers to operate. \
As transistors replaced vacuum tubes, computers became smaller and more reliable. \
The invention of the integrated circuit further miniaturized computing hardware. \
Programming languages evolved from machine code to assembly, then to high-level languages. \
Operating systems emerged to manage hardware resources and provide abstractions for developers. \
Networking protocols enabled computers to communicate across long distances. \
The internet connected millions of machines into a global information network. \
Databases provided structured storage and efficient retrieval of large datasets. \
Object-oriented programming introduced new paradigms for organizing complex software. \
"""

def build_haystack(target_tokens: int = 3000) -> str:
    """
    重复拼接 _HAYSTACK_UNIT 直到达到目标 token 数。

    🐍 Python 插播：while 循环和字符串拼接
    Python 字符串拼接用 += 或 join()，
    大量拼接时 join() 更高效（类比 Go 的 strings.Builder）。
    这里为了可读性用 +=，token 数不大，性能无所谓。
    """
    text = ""
    while count_tokens(text) < target_tokens:
        text += _HAYSTACK_UNIT
    return text

# 针（关键事实）——这是模型需要"召回"的信息
NEEDLE = "The secret configuration key for the production database is PROD-XK-7729."

# 召回问题
RECALL_QUESTION = "What is the secret configuration key for the production database?"

def insert_needle(haystack: str, needle: str, position: float) -> str:
    """
    在 haystack 的 position 比例处（0.0 = 开头, 1.0 = 结尾）插入 needle。

    position=0.0 → 针在最前面
    position=0.5 → 针在正中间
    position=1.0 → 针在最后面

    🐍 Python 插播：字符串切片
    text[start:end] 类比 Go 的 text[start:end]，语法完全一致。
    text[:i]  = text[0:i]（前 i 个字符）
    text[i:]  = text[i:len(text)]（从 i 到末尾）
    """
    insert_at = int(len(haystack) * position)

    # 找到最近的换行符，避免切断句子中间（让插入点更自然）
    newline_pos = haystack.rfind("\n", 0, insert_at)
    if newline_pos != -1:
        insert_at = newline_pos + 1

    before = haystack[:insert_at]
    after  = haystack[insert_at:]

    # 针的前后各加一个空行，让它在文本中更像"正常段落"而不是明显标记
    return before + "\n" + needle + "\n\n" + after


def ask_needle(full_text: str, question: str, position_label: str) -> str:
    """
    把完整文本（含针）发给模型，问它能否召回关键信息。
    返回模型的回答，用于后续判断是否正确。
    """
    # 注意 prompt 的设计：
    # - 文档在前，问题在后（这是 RAG 场景的标准做法）
    # - 明确要求"只基于文档回答"，排除模型用内部知识猜测
    user_content = f"""Below is a document. Read it carefully and answer the question at the end.
Only use information from the document to answer. If the answer is not in the document, say "Not found".

<document>
{full_text}
</document>

Question: {question}
Answer:"""

    messages = add_message([], "user", user_content)
    resp = send_request(messages, temperature=0.0, max_tokens=100)

    return stream_and_print(resp, label=f"针的位置：{position_label}")


def run_experiment_b():
    print("\n" + "=" * 70)
    print("实验 B：Needle in a Haystack（Context Rot 验证）")
    print("=" * 70)

    # 生成干草堆
    haystack = build_haystack(target_tokens=3000)
    haystack_tokens = count_tokens(haystack)

    print(f"\n干草堆统计：{len(haystack)} 字符 / {haystack_tokens} tokens")
    print(f"针（关键事实）：{NEEDLE}")
    print(f"召回问题：{RECALL_QUESTION}")
    print()
    print("现在把针插入文本的不同位置，看模型能否准确召回...")

    # 测试 5 个位置
    # 🐍 Python 插播：list of tuples（元组列表）
    # 类比 Go 的 []struct{ label string; pos float64 }
    positions = [
        ("开头（0%）",   0.0),
        ("1/4 处（25%）", 0.25),
        ("中间（50%）",  0.5),
        ("3/4 处（75%）", 0.75),
        ("结尾（100%）", 1.0),
    ]

    results = []
    for label, pos in positions:
        full_text = insert_needle(haystack, NEEDLE, pos)
        total_tokens = count_tokens(full_text)
        print(f"\n[总 token 数：{total_tokens}]")

        answer = ask_needle(full_text, RECALL_QUESTION, label)

        # 判断是否正确召回（检查关键字符串是否出现在回答里）
        recalled = "PROD-XK-7729".lower() in answer.lower()
        results.append((label, recalled))

    # 汇总
    print("\n" + "=" * 70)
    print("实验 B 结果汇总")
    print("=" * 70)
    for label, recalled in results:
        status = "✅ 召回成功" if recalled else "❌ 召回失败"
        print(f"  {label:<20} → {status}")

    print()
    print("📌 讨论：")
    print("  - 位置对召回率的影响是否符合「Lost in the Middle」理论？")
    print("  - 中间位置失败的原因是什么（Attention 权重分布的影响）？")
    print("  - Agent 设计中如何利用这个结论？（重要信息放哪里？）")


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HEADERS["x-api-key"]:
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    run_experiment_a()   # 纯本地计算，不调 API
    run_experiment_b()   # 调用 API，共 5 次请求
