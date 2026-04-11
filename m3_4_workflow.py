"""
[M3.4] 工作流编排模式（Workflow Orchestration）

实现三种核心模式的组合：
  ┌─────────────────────────────────────────────────┐
  │  用户粘贴代码                                     │
  │       │                                          │
  │       ▼                                          │
  │  [Stage 1: Routing]  ← Haiku 快速分类            │
  │    判断语言 → Gate（不支持则短路返回）              │
  │       │                                          │
  │       ▼                                          │
  │  [Stage 2: Parallelization]  ← asyncio.gather   │
  │    安全审查 ─┐                                    │
  │    性能审查 ─┼→ 三个 Worker 并发跑                │
  │    可读性   ─┘                                    │
  │       │                                          │
  │       ▼  （代码拼接三份报告）                      │
  │  [Stage 3: Chaining]  ← 再调一次 LLM             │
  │    综合报告 → 优先级排序的改进建议                  │
  └─────────────────────────────────────────────────┘

设计决策记录：
  - Gate 封装在 pipeline 内部：调用方无感知，符合封装原则
  - Worker 参数化（aspect 字段）：prompt 结构相同，只有维度不同
  - 三份报告用 XML 标签分隔：减少 LLM 歧义解析负担
  - asyncio.to_thread 包装同步 send_request：复用已有基础设施

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m3_4_workflow.py
"""

import asyncio
import time

# ── 直接复用 M3.3 的基础设施，不重复写 ────────────────────────────────────
# send_request：统一 HTTP 请求入口
# extract_text：从响应中提取 text block
from m3_3_reflection import send_request, extract_text

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# 支持的语言列表，Router 只需从这里选
SUPPORTED_LANGUAGES = ["python", "go"]

# 审查维度配置：aspect_key → (中文名, prompt 描述)
# 用 dict 而非硬编码三个函数，方便后续增加维度
REVIEW_ASPECTS = {
    "security": (
        "安全性",
        "检查代码中的安全漏洞：SQL注入、路径穿越、敏感信息泄露、不安全的反序列化、"
        "权限校验缺失等。如无明显问题，说明'未发现明显安全问题'。",
    ),
    "performance": (
        "性能",
        "检查代码中的性能问题：不必要的循环、重复计算、内存泄漏风险、低效数据结构、"
        "阻塞操作等。如无明显问题，说明'未发现明显性能问题'。",
    ),
    "readability": (
        "可读性",
        "检查代码的可读性：命名是否清晰、函数是否过长、注释是否充分、"
        "错误处理是否明确、代码结构是否清晰。",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1：Routing — 语言分类
# ══════════════════════════════════════════════════════════════════════════════

async def route_language(code: str) -> str:
    """
    用 LLM 判断代码语言，返回 "python" / "go" / "other"。

    为什么单独一次 LLM 调用？
      - Router 的职责单一：只分类，不处理。符合单一职责原则。
      - 使用更快的 Haiku（通过 temperature=0 保证确定性输出）。
      - Gate 逻辑在调用方（pipeline）处理，Router 本身不做决策。

    asyncio.to_thread：
      send_request 是同步阻塞调用（requests 库）。
      包装进线程池后，event loop 可以在等待期间调度其他 coroutine。
      类比 Go：go func() { result <- blockingCall() }()
    """
    system = (
        "你是代码语言分类器。\n"
        "只输出以下之一：python / go / other\n"
        "不要输出任何其他内容，不要加解释。"
    )
    messages = [{"role": "user", "content": f"判断以下代码的编程语言：\n\n{code}"}]

    # 🐍 asyncio.to_thread：把同步函数包装成可 await 的协程
    #    第一个参数是函数，后面是位置参数，用 keyword 参数传具名参数
    response = await asyncio.to_thread(
        send_request,
        messages,           # positional: messages
        None,               # positional: tools
        system,             # positional: system
        0.0,                # positional: temperature（确定性分类）
        16,                 # positional: max_tokens（只需输出一个词）
    )

    lang = extract_text(response).strip().lower()

    # 健壮处理：模型偶尔会输出 "Python" 或 "这是 python 代码"
    for supported in SUPPORTED_LANGUAGES:
        if supported in lang:
            return supported
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2：Parallelization — 并行审查 Worker
# ══════════════════════════════════════════════════════════════════════════════

async def review(code: str, lang: str, aspect: str) -> str:
    """
    单个审查 Worker：针对一个维度审查代码，返回审查报告字符串。

    参数化设计：三个维度共用这一个函数，通过 aspect 键区分。
    REVIEW_ASPECTS[aspect] 取出 (中文名, prompt描述)，动态构造 prompt。

    为什么是 async？
      本函数会被 asyncio.gather 并发调用。
      async 声明让 event loop 知道"这个函数可以被挂起和恢复"。
      实际的 I/O 等待发生在 asyncio.to_thread 内部。
    """
    aspect_name, aspect_desc = REVIEW_ASPECTS[aspect]

    system = (
        f"你是一名资深 {lang.capitalize()} 开发者，专注于代码 {aspect_name} 审查。\n"
        f"审查要点：{aspect_desc}\n\n"
        "输出格式：\n"
        "  - 每个问题单独一行，格式：[严重程度: 高/中/低] 问题描述\n"
        "  - 如无问题，输出：未发现明显问题\n"
        "不超过 5 条，只列最重要的问题。"
    )
    messages = [{"role": "user", "content": f"请审查以下 {lang} 代码：\n\n```{lang}\n{code}\n```"}]

    response = await asyncio.to_thread(
        send_request,
        messages,
        None,
        system,
        0.2,   # 低 temperature：审查结论要稳定，但允许少量表达变化
        512,
    )
    return extract_text(response).strip()


async def parallel_review(code: str, lang: str) -> dict[str, str]:
    """
    并发跑三个审查维度，返回 {aspect: report} 字典。

    asyncio.gather：
      同时启动所有 coroutine，等所有完成后一起返回。
      类比 Go 的：
        var wg sync.WaitGroup
        results := make([]string, 3)
        for i, aspect := range aspects {
            wg.Add(1)
            go func(i int, aspect string) {
                defer wg.Done()
                results[i] = review(code, lang, aspect)
            }(i, aspect)
        }
        wg.Wait()

    实际耗时 ≈ max(单个 Worker 耗时)，而非 sum。
    """
    aspects = list(REVIEW_ASPECTS.keys())  # ["security", "performance", "readability"]

    # gather 返回列表，顺序与输入 coroutine 顺序一致
    results = await asyncio.gather(
        *[review(code, lang, aspect) for aspect in aspects]
    )

    # 🐍 zip：把两个列表对应元素配对，类似 Go 里用下标对齐两个 slice
    return dict(zip(aspects, results))


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3：Chaining — 综合报告
# ══════════════════════════════════════════════════════════════════════════════

async def synthesize(code: str, lang: str, reviews: dict[str, str]) -> str:
    """
    接收三份独立审查报告，综合生成优先级排序的改进建议。

    为什么需要再调一次 LLM？
      三份报告可能有交叉（安全问题和性能问题同一根源），
      或者需要权衡（修性能可能降低可读性）。
      这种跨维度推理只有 LLM 能做，代码字符串拼接做不到。

    XML 标签分隔三份报告：
      让模型清晰知道每个 block 的语义边界，减少混淆。
      Claude 系列对 XML 结构理解很好（训练数据里大量 XML）。
    """
    # 用 XML 标签把三份报告结构化，语义边界清晰
    reviews_xml = "\n".join(
        f"<{aspect}_review>\n{report}\n</{aspect}_review>"
        for aspect, report in reviews.items()
    )

    system = (
        "你是资深代码架构师，负责综合多维度审查报告，生成清晰的改进建议。\n\n"
        "输出要求：\n"
        "  1. 先给出 1-2 句总体评价\n"
        "  2. 列出改进建议，按优先级从高到低排序\n"
        "  3. 每条建议格式：[P1/P2/P3] 建议内容（P1=必须修复，P2=建议修复，P3=可选优化）\n"
        "  4. 不超过 6 条建议\n"
        "  5. 语言简洁，每条不超过 50 字"
    )

    messages = [{
        "role": "user",
        "content": (
            f"以下是一段 {lang} 代码的三维度审查结果，请综合生成改进建议：\n\n"
            f"{reviews_xml}\n\n"
            f"原始代码供参考：\n```{lang}\n{code}\n```"
        ),
    }]

    response = await asyncio.to_thread(
        send_request,
        messages,
        None,
        system,
        0.3,
        768,
    )
    return extract_text(response).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline：三阶段组装
# ══════════════════════════════════════════════════════════════════════════════

async def pipeline(code: str) -> str:
    """
    完整的代码审查流水线，封装三个 Stage。

    Gate 在内部处理，调用方只需：
        result = await pipeline(code)
    不需要知道内部有多少个检查点。

    计时设计：
      整体计时 vs 分段计时，帮助建立"并行真的省时间"的直觉。
    """
    print("\n" + "═" * 55)
    print("  🔍 代码审查流水线启动")
    print("═" * 55)

    t_start = time.time()

    # ── Stage 1：Routing ──────────────────────────────────────────────────────
    print("\n  [Stage 1] Routing：识别语言...")
    t1 = time.time()
    lang = await route_language(code)
    print(f"           → 语言：{lang}  ({time.time()-t1:.1f}s)")

    # Gate：不支持的语言直接短路，不进入后续两个 Stage
    if lang == "other":
        print("\n  ❌ Gate：不支持的语言，流水线终止")
        return "❌ 仅支持 Python 和 Go 代码，请检查输入。"

    # ── Stage 2：Parallelization ──────────────────────────────────────────────
    print(f"\n  [Stage 2] Parallel Review：三维度并发审查...")
    t2 = time.time()
    reviews = await parallel_review(code, lang)
    elapsed2 = time.time() - t2
    print(f"           → 完成  ({elapsed2:.1f}s，三个 Worker 并发)")

    # 打印各维度摘要（取第一行）
    for aspect, report in reviews.items():
        name = REVIEW_ASPECTS[aspect][0]
        first_line = report.split("\n")[0]
        print(f"           · {name}：{first_line[:60]}...")

    # ── Stage 3：Chaining — Synthesis ────────────────────────────────────────
    print(f"\n  [Stage 3] Synthesis：综合报告生成...")
    t3 = time.time()
    final = await synthesize(code, lang, reviews)
    print(f"           → 完成  ({time.time()-t3:.1f}s)")

    total = time.time() - t_start
    print(f"\n  ✅ 流水线完成  总耗时 {total:.1f}s")
    print("═" * 55)

    return final


# ══════════════════════════════════════════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════════════════════════════════════════

# 故意埋了几个问题：SQL 注入、列表重复拼接（性能）、函数过长（可读性）
SAMPLE_PYTHON = '''
import sqlite3

def get_user_orders(username, db_path="orders.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT * FROM orders WHERE username = '{username}'"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    result = ""
    for row in rows:
        result = result + str(row) + "\\n"

    return result

def process_all(users):
    output = []
    for u in users:
        data = get_user_orders(u)
        if data:
            output.append(data)
        else:
            output.append("no orders")
    return output
'''

SAMPLE_GO = '''
package main

import (
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
)

func readSecret() string {
    secret := os.Getenv("API_SECRET")
    fmt.Println("Secret:", secret)
    return secret
}

func fetchData(url string) string {
    resp, _ := http.Get(url)
    body, _ := ioutil.ReadAll(resp.Body)
    return string(body)
}
'''

SAMPLE_UNSUPPORTED = "SELECT * FROM users WHERE id = 1;"


async def main():
    print("=" * 55)
    print("  M3.4 工作流编排模式 — 代码审查流水线")
    print("=" * 55)

    cases = [
        ("Python 案例（含 SQL 注入 + 性能问题）", SAMPLE_PYTHON),
        ("Go 案例（含安全 + 错误处理问题）", SAMPLE_GO),
        ("不支持语言（Gate 测试）", SAMPLE_UNSUPPORTED),
    ]

    for title, code in cases:
        print(f"\n\n{'#'*55}")
        print(f"  案例：{title}")
        print(f"{'#'*55}")
        result = await pipeline(code)
        print(f"\n  📋 最终建议：\n")
        for line in result.split("\n"):
            print(f"     {line}")
        input("\n  [按 Enter 运行下一个案例...]")


if __name__ == "__main__":
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ 请先设置环境变量：export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)
    asyncio.run(main())
