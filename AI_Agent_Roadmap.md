# 🗺️ AI Agent 从零到一：专属学习路线图

> **学员画像**：Go/C++ 后端开发者（1年经验） · Python 基础语法 · 会用 Claude/Gemini · 有 MCP 配置经验  
> **短期目标**：打破 Agent 黑盒，理解底层机制  
> **长期目标**：转型 AI Agent 研发工程师  
> **核心原则**：第一性原理 · 从零造轮子 · 不依赖高度封装框架  
> **最后更新**：2026-04-04 · *本文档为 Living Document，随学习进展持续迭代*

---

## 全局架构：五大里程碑

```
M1              M2              M3              M4            M4.5            M5
地基层  ───▶  核心层  ───▶  设计层  ───▶  协议层  ───▶  实践层  ───▶  系统层
LLM 基础      Agent 组件    设计模式       MCP 深潜     工程实践拆解   Multi-Agent
(2~3周)       (3~4周)       (3~4周)       (2~3周)      (2~3周)       (3~4周)
```

> **Python 能力**不单独设立里程碑，而是作为"伴随技能"融入每个模块的代码实践环节。每当用到 Python 高级特性时，会以 `🐍 Python 插播` 的形式做极简讲解。

---

## 🤝 协作模式：讨论驱动 + AI 实现 + 人工审查

AI 时代的学习方式不应该是"从零手敲每一行代码"，而是**理解架构 → 验证逻辑 → 审查实现**。我们约定以下五步协作流程：

```
Step 1          Step 2          Step 3          Step 4          Step 5
概念拆解  ───▶  架构讨论  ───▶  代码实现  ───▶  审查质疑  ───▶  运行实验
(导师讲解)     (共同讨论)     (导师编写)     (你来 Review)   (改参数/做实验)
```

| 步骤 | 谁主导 | 做什么 |
|------|--------|--------|
| **概念拆解** | 导师 | 用类比和图解把机制讲透，你确认理解 |
| **架构讨论** | 共同 | 讨论实现需要哪些模块、数据怎么流动——锻炼你最强的架构直觉 |
| **代码实现** | 导师 | 你确认逻辑框架后，导师编写代码，关键位置加注释 |
| **审查质疑** | 你 | 读代码，提出"这里为什么这样做"、"这个地方我觉得有问题" |
| **运行实验** | 共同 | 跑起来看效果，改参数做实验，建立直觉 |

> **为什么这样设计**：这个流程模拟的是你未来作为 AI Agent 研发工程师的真实工作场景——你设计架构、Review AI 写的代码、做技术决策，而不是从零手敲每一行。**能判断代码对不对、好不好，比能写出来更重要。**

---

## M1 · 地基层：大模型基础

**目标**：从一个 HTTP 请求开始，彻底理解你每天在用的对话模型背后发生了什么。

### 1.1 LLM API 的底层机制

**核心概念**：

- **Token 与 Tokenizer**：文本不是以字符为单位送入模型的，而是被切成 Token。这类似你在 Go 中做协议解析时先 `Unmarshal` 一样——模型也有自己的"序列化协议"。
- **API 调用的本质**：一次 Chat Completion 请求本质上是一次无状态的 HTTP POST。模型不记忆上下文——所谓的"多轮对话"只是客户端把历史消息全部拼进 `messages` 数组再发一次。你可以类比 Go 中的 REST API：服务端无状态，状态由客户端（或中间层）管理。
- **Temperature / Top-p 采样**：模型输出的每个 Token 是从一个概率分布中"采样"出来的。Temperature 控制分布的锐度（类比信号处理中的增益），Top-p 做截断。
- **流式响应（Streaming）**：Server-Sent Events (SSE) 逐 Token 推送。你在 Go 里用 `bufio.Scanner` 逐行读 SSE 流就能实现一个最小客户端。

**实战环节**（协作模式：讨论 → 导师实现 → 你 Review）：
- 我们先讨论 API 请求的结构设计，然后我用纯 Python `requests` 库（不用任何 SDK）直接调 Anthropic/OpenAI 的 Messages API，手动构造 JSON 请求体，解析流式响应。你来审查代码逻辑是否清晰。
- 对比同一 prompt 在不同 Temperature 下的输出差异，共同分析结果，建立直觉。

### 1.2 Prompt Engineering 的本质

**核心概念**：

- **System Prompt vs User Prompt**：System Prompt 定义行为边界（类比 Go 中为一个 goroutine 设定的 context），User Prompt 是具体指令。
- **Few-shot Prompting**：在 prompt 中给出输入/输出示例，本质是在做"运行时的模式匹配"——模型通过示例推断你期望的输出格式和风格。
- **Chain-of-Thought (CoT)**：让模型"先想后答"。这不是魔法，而是通过增加中间推理 Token 来提高最终答案的准确率。

**实战环节**：
- 我们共同构造一个"让模型扮演 Code Reviewer"的 System Prompt，输入一段有 bug 的 Go 代码，观察不同 prompt 策略（直接问 vs CoT vs Few-shot）的效果差异，讨论为什么会有这些差异。

### 1.3 Context Window 与 Token 经济学

**核心概念**：

- **Context Window 的物理含义**：模型的"工作内存"。类比你在 Go 中处理一个固定大小的 `[]byte` buffer——超了就溢出（被截断）。Claude Sonnet 4.5 的 context window 达到了约 200K tokens，但更长不等于更好。
- **Context Rot（上下文腐烂）**：研究发现，随着 context 中 token 数量增加，模型对早期信息的"注意力"下降。这就是为什么"把所有东西塞进去"是个坏策略。
- **从 Prompt Engineering 到 Context Engineering**：2025年，业界共识已从"如何写好 prompt"转向"如何管理好整个上下文窗口"。Andrej Karpathy 的比喻很精妙：LLM 像 CPU，Context Window 像 RAM，而 Context Engineering 就是这个新操作系统的内存管理。

**实战环节**：
- 我来写一个 Python 脚本，用 `tiktoken` 库对不同长度的文本进行 tokenize，你审查后我们一起跑，直观感受 token 数与文本长度的关系。
- 实验：给模型一段超长文本，在不同位置插入一个关键事实（"Needle in a Haystack"测试），共同观察模型能否准确召回，讨论 Context Rot 的影响。

### 📚 M1 参考资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📄 官方文档 | [Anthropic API Docs](https://docs.anthropic.com) | Messages API 的权威参考 |
| 📄 官方文档 | [OpenAI API Reference](https://platform.openai.com/docs/api-reference) | 对照学习，理解行业通用模式 |
| 📄 博客 | [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Anthropic 官方出品，Context Engineering 必读 |
| 📄 博客 | [LangChain: Context Engineering](https://blog.langchain.com/context-engineering-for-agents/) | Write/Select/Compress/Isolate 四大策略 |
| 🎥 视频 | [Andrej Karpathy: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) | 从底层讲解 LLM 的工作原理 |
| 📄 指南 | [Prompt Engineering Guide](https://www.promptingguide.ai/) | 全面的 Prompt 技术参考 |
| 📄 框架 | [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) | 从软件工程角度理解 Agent 构建原则 |

---

## M2 · 核心层：Agent 三大组件

**目标**：拆解 Agent 的"大脑"——理解 Planning、Memory、Tool-use 三大核心机制，并通过协作实现一个最小可用 Agent。

> **类比入口**：如果你把 Agent 想象成一个长期运行的后台服务（Go 中的 daemon），那么：
> - **Planning** = 任务调度器（决定做什么、按什么顺序做）
> - **Memory** = 状态存储（保持跨请求的上下文）
> - **Tool-use** = RPC 调用（调用外部服务完成具体动作）

### 2.1 Tool-use（工具调用）机制

**核心概念**：

- **Function Calling 的本质**：模型并不真正"执行"工具。它只是根据你提供的工具描述（JSON Schema），生成一个结构化的 JSON 输出，表达"我想调用这个函数，参数是这些"。然后由你的代码（Host）去执行，并把结果喂回给模型。
- **这就是"结构化输出"**：Tool-use 本质上和让模型输出 JSON 是一回事——模型在 Token 层面生成符合特定 Schema 的文本。这也是为什么 12-Factor Agents 说"Tools Are Just Structured Outputs"。
- **Tool Description 就是 API 文档**：你在 Go 中写 gRPC 的 `.proto` 定义时有多讲究，给模型写 tool description 就该有多讲究。描述不清晰 = 调用出错。

**实战环节**：
- **Step 1**：我们先讨论 tool_use 的数据流设计，然后我用纯 Python 调用 Anthropic API 的 tool_use 能力，定义一个 `get_weather(city: str)` 工具。你来 Review 代码，重点看模型返回的 `tool_use` content block 结构。
- **Step 2**：我们讨论完整 Tool-use Loop 的架构后，我来实现：发送请求 → 解析 `tool_use` 响应 → 本地执行函数 → 将 `tool_result` 拼入 messages → 再次发送。你审查这个循环的逻辑——它就是 Agent 的心跳。

### 2.2 Planning（规划）机制

**核心概念**：

- **单步规划 vs 多步规划**：简单任务模型一步到位；复杂任务需要模型先分解子任务再逐步执行——这就是 Planning。
- **基于 Prompt 的规划**：最朴素的实现就是在 System Prompt 中要求模型"先制定计划，再逐步执行"。CoT 本质上就是最简单的 Planning。
- **动态重规划**：好的 Agent 不是一条路走到黑——当某个步骤失败或获得新信息时，它应该能调整计划。类比 Go 中你不会在 goroutine 里硬编码所有逻辑，而是用 `select` 监听多个 channel 动态响应。

**实战环节**：
- 在 Step 2 的基础上，我们共同设计一个复杂任务（如"查询北京和上海的天气，然后比较哪个更适合周末出行"），跑起来观察模型如何自行分解步骤、串联多次工具调用，讨论它的规划是否合理。

### 2.3 Memory（记忆）机制

**核心概念**：

- **短期记忆 = Conversation History**：就是你在 M1 中学到的——把所有历史消息塞进 `messages` 数组。但随着对话变长，你会遇到 context window 的上限。
- **长期记忆 = 外部存储 + 检索**：把重要信息持久化到数据库/文件中，在需要时检索回来注入 context。这本质上就是 RAG（Retrieval-Augmented Generation）的思想。
- **Memory 类型学**：
  - **Episodic Memory**（情景记忆）：过去的对话记录、任务执行轨迹。
  - **Semantic Memory**（语义记忆）：抽取出的知识/事实。
  - **Procedural Memory**（程序性记忆）：如何做某事的规则和偏好。
- **Context Window 管理策略**：Summarization（摘要压缩）、Trimming（裁剪旧消息）、Scratchpad（外部笔记本）。

**实战环节**：
- 我们先讨论记忆模块的数据结构设计（用什么格式存？怎么检索？），然后我来实现：用一个 JSON 文件作为 "long-term memory"，Agent 可以将重要信息 `save` 到文件，也可以在新对话中 `load` 回来。你 Review 实现是否合理。

### 2.4 🔧 综合实战：协作构建 Mini Agent

把上面三个组件组合起来，通过架构讨论 → 我来实现 → 你来 Review 的流程，完成一个约 200 行 Python 代码的完整 Mini Agent：

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
              ┌────▶│   LLM (Brain)   │◀───┐
              │     └────────┬────────┘    │
              │              │             │
         tool_result    tool_use?      memory
              │              │          recall
              │              ▼             │
              │     ┌─────────────────┐    │
              └─────│  Tool Executor  │    │
                    └─────────────────┘    │
                             │             │
                    ┌─────────────────┐    │
                    │  Memory Store   │────┘
                    └─────────────────┘
```

**架构核心就是一个 `while True` 循环**——这和 Go 中的 `for { select { ... } }` 事件循环本质一样。

### 📚 M2 参考资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📄 Paper | [Tool Learning with Foundation Models (Qin et al., 2023)](https://arxiv.org/abs/2304.08354) | 工具调用的系统性综述 |
| 📄 Paper | [A Survey on Large Language Model based Autonomous Agents (Wang et al., 2023)](https://arxiv.org/abs/2308.11432) | Agent 架构的经典综述 |
| 💻 GitHub | [GenAI Agents](https://github.com/NirDiamant/GenAI_Agents) | 各种 Agent 模式的 Notebook 实现集合 |
| 💻 GitHub | [AI Agents for Beginners (Microsoft)](https://github.com/microsoft/ai-agents-for-beginners) | 微软出品的入门教程，覆盖面广 |
| 📄 官方文档 | [Anthropic: Tool Use Docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) | Claude Tool Use 的权威指南 |
| 📄 博客 | [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | Anthropic 官方的 Agent 构建指南 |
| 📄 博客 | [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | 必读经典，Agent 组件全景图 |

---

## M3 · 设计层：Agent 设计模式与工作流

**目标**：掌握业界已验证的 Agent 架构范式，理解何时用哪种模式，并通过协作实现关键模式。

### 3.1 ReAct 范式（Reasoning + Acting）

**核心概念**：

- **论文核心思想**：Yao et al. (2022) 提出的 ReAct 框架，让 LLM 在生成推理轨迹（Thought）和执行动作（Action）之间交替进行。每一步的循环是：**Thought → Action → Observation → Thought → ...**
- **为什么有效**：推理轨迹帮助模型追踪计划和处理异常，动作让模型能与外部环境交互获取信息。两者协同产生的效果远大于各自独立运行。
- **与你在 M2 构建的 Agent 的关系**：你的 Mini Agent 已经有了 ReAct 的雏形！区别在于 ReAct 显式地要求模型在每一步先输出 Thought（推理过程），让决策过程可解释、可追溯。

**实战环节**：
- 我们讨论如何改造 Mini Agent 使其符合 ReAct 范式，然后我来修改代码——在 System Prompt 中加入 ReAct 格式要求，让模型在每次 tool_use 前先输出 `<thinking>` 块。你对比改造前后的行为差异，判断 ReAct 在哪些场景下更有优势。

### 3.2 状态机模式（State Machine Pattern）

**核心概念**：

- **Agent 即状态机**：Agent 的执行流程可以建模为一个有限状态机（FSM）。这对 Go 开发者来说应该非常熟悉——你在设计网络协议处理器、任务编排器时大概率用过类似的模式。
- **状态定义**：`PLANNING → EXECUTING → OBSERVING → REFLECTING → DONE`（或 `FAILED`）。
- **确定性控制 vs 模型自由度**：状态转移可以是硬编码的（确定性），也可以由模型决定（非确定性）。在生产环境中，关键路径通常用确定性转移，把模型自由度限制在每个状态内部。

**实战环节**：
- 我们先讨论状态定义和转移规则的设计，然后我用 Python 实现一个简单的 FSM Agent，用 `Enum` 定义状态，用 `dict` 定义转移规则。你来 Review 状态转移逻辑是否覆盖了所有边界情况。

```python
# 🐍 Python 插播：Enum 是 Python 的枚举类型，类似 Go 的 iota 常量组
from enum import Enum, auto

class AgentState(Enum):
    PLANNING = auto()    # auto() 自动赋值，类似 Go 的 iota
    EXECUTING = auto()
    REFLECTING = auto()
    DONE = auto()
```

### 3.3 反思机制（Reflection / Self-Critique）

**核心概念**：

- **自我校验**：让 Agent 在完成任务后回顾自己的输出，检查错误、遗漏和改进空间。这类似你在 Go 中写单元测试——不过这里是 Agent 自己给自己写测试。
- **Reflexion (Shinn et al., 2023)**：一种让 Agent 从失败中学习的框架。Agent 执行 → 评估结果 → 如果失败则生成"反思笔记" → 下一次尝试时将反思注入 context。
- **实现方式**：最简单的做法是在一次 LLM 调用完成后，再发起一次调用，prompt 为"请检查以下输出是否正确..."。

### 3.4 工作流编排模式

**核心概念**：

- **Prompt Chaining（链式调用）**：将复杂任务分解为多个 LLM 调用步骤，前一步的输出作为后一步的输入。类比 Unix Pipeline `cmd1 | cmd2 | cmd3`。
- **Routing（路由分发）**：用一个 LLM 调用对输入进行分类/路由，然后分发给不同的专用 prompt 处理。类比 Go 中的 `http.ServeMux` 路由。
- **Parallelization（并行编排）**：多个独立子任务并行调用 LLM，汇总结果。你在 Go 中用 `sync.WaitGroup` + goroutines 做的事情，在这里是用 `asyncio` 做的。
- **Orchestrator-Worker（编排者-工人）**：一个"指挥官" Agent 负责规划和分配任务，多个"工人" Agent 负责执行具体步骤。

**实战环节**：
- 我们共同设计一个 Prompt Chain 的流程：`分析需求 → 生成代码 → 自动 Review → 输出最终版本`。讨论确认后我来实现，包括用 `asyncio` 实现并行调用。你 Review 并行逻辑是否正确。

```python
# 🐍 Python 插播：asyncio 是 Python 的异步并发库
# 类比 Go 的 goroutine + channel，但语法是 async/await
import asyncio

async def call_llm(prompt: str) -> str:
    # ... 异步 HTTP 请求
    pass

# 类比 Go: go func() { ch <- result }()
results = await asyncio.gather(
    call_llm("分析任务A"),
    call_llm("分析任务B"),
    call_llm("分析任务C"),
)
```

### 3.5 Flow Engineering 与 Agentic Workflows

**核心概念**：

- **Agentic Workflow 的本质**：不是让一个万能 Agent 解决所有问题，而是设计精巧的工作流，让多个简单的 LLM 调用协同完成复杂任务。
- **关键设计决策**：何时用确定性代码（if/else、for 循环）控制流程，何时让模型自主决策。经验法则——能用代码控制的就不要交给模型。
- **Error Handling 与 Self-Healing**：当工具调用失败时，将错误信息注入 context 让模型重试。类比 Go 中的错误处理哲学：`if err != nil { ... }`。

### 📚 M3 参考资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📄 Paper | [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) | Agent 设计的奠基论文，必读 |
| 📄 Paper | [Reflexion (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366) | 反思机制的经典论文 |
| 📄 Paper | [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601) | 树状推理，Planning 的进阶范式 |
| 📄 Paper | [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) | CoT 的原始论文 |
| 📄 博客 | [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | 工作流编排模式的最佳实践 |
| 💻 GitHub | [ReAct Official Repo](https://github.com/ysymyth/ReAct) | ReAct 论文的官方代码 |
| 💻 GitHub | [LangGraph (扒源码用)](https://github.com/langchain-ai/langgraph) | 理解图状态机编排的工业级实现 |

---

## M4 · 协议层：MCP 深度解析

**目标**：从"会配置 MCP"升级到"理解 MCP 的每一层设计决策"，并通过协作实现一个 MCP Server。

### 4.1 MCP 的设计哲学

**核心概念**：

- **N×M 问题**：假设有 N 个 AI 应用和 M 个外部工具/数据源。没有标准协议前，需要 N×M 个定制连接器。MCP 将其简化为 N+M 个实现——每个应用实现一个 MCP Client，每个工具实现一个 MCP Server。
- **"USB-C for AI" 类比**：就像 USB-C 统一了充电线，MCP 统一了 AI 应用与工具的连接方式。
- **与 LSP 的渊源**：MCP 借鉴了 Language Server Protocol (LSP) 的消息流设计。如果你用过 VS Code 的语言服务，你对这种"Client-Server 通过标准化消息通信"的架构模式不会陌生。
- **传输层演进**：stdio（本地进程通信）→ SSE（单向推送）→ Streamable HTTP（2025年3月引入，真正让远程 MCP Server 成为可能）。

### 4.2 MCP 架构三层模型

```
┌──────────────────────────────────────────┐
│              MCP Host                     │
│  (Claude Desktop / IDE / 你的应用)         │
│                                          │
│  ┌─────────────┐  ┌─────────────┐        │
│  │ MCP Client  │  │ MCP Client  │  ...   │
│  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼───────────────┘
          │ JSON-RPC 2.0   │ JSON-RPC 2.0
          ▼                ▼
   ┌─────────────┐  ┌─────────────┐
   │ MCP Server  │  │ MCP Server  │
   │ (GitHub)    │  │ (Database)  │
   └─────────────┘  └─────────────┘
```

- **Host**：运行 LLM 的应用（如 Claude Desktop）。负责管理多个 Client 实例。
- **Client**：与 Server 建立 1:1 连接的协议层。
- **Server**：暴露 Tools（工具）、Resources（资源）、Prompts（提示词模板）三大能力。

**关键洞察**：模型不直接和 MCP Server 通信。整个流程是：

```
模型生成 tool_use → Host 解析 → Host 通过 Client 调用 Server → 
Server 返回结果 → Host 将结果注入 messages → 模型继续推理
```

这和你在 M2 中构建的 Tool-use Loop **完全是一回事**，只不过 MCP 将其标准化了。

### 4.3 MCP 的三大能力原语

| 能力 | 控制方 | 说明 | 类比 |
|------|--------|------|------|
| **Tools** | 模型发起 | 模型决定何时调用什么工具 | 你代码中的 RPC 调用 |
| **Resources** | 应用发起 | 应用主动拉取数据注入 context | 你代码中的数据库查询 |
| **Prompts** | 用户发起 | 预定义的 prompt 模板 | 你代码中的配置模板 |

### 4.4 协议细节

- **生命周期**：`initialize` → 能力协商（Client 和 Server 告知对方各自支持什么）→ 正常通信 → `shutdown`。类比 TCP 三次握手。
- **JSON-RPC 2.0**：MCP 使用 JSON-RPC 作为消息格式。每个请求都有 `id`、`method`、`params`，响应有 `result` 或 `error`。如果你做过 JSON-RPC 或 gRPC 的工作，会非常亲切。
- **安全模型**：2025年11月的 spec 更新引入了 OAuth 2.0 授权、令牌绑定等企业级安全特性。Server 端的工具描述被视为**不可信数据**——Host 必须在执行前获取用户确认。

### 4.5 实战环节：从零实现一个 MCP Server

- **Step 1**：我们先讨论 MCP Server 需要实现哪些接口、数据怎么流动，然后我用 Python 的 `mcp` SDK 实现一个最小 MCP Server，暴露一个 `list_files(directory: str)` 工具。你 Review 代码结构。
- **Step 2**：你来操作——通过 stdio 传输方式将其接入 Claude Desktop，测试端到端流程。
- **Step 3**（进阶）：我们讨论 JSON-RPC 2.0 协议细节后，我不用 SDK，用纯 Python 实现消息的解析和响应。你对照 MCP spec 审查实现是否合规——理解协议的每一个字节。

### 📚 M4 参考资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📄 官方文档 | [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25) | 协议规范，最权威的参考 |
| 📄 官方文档 | [MCP Documentation](https://modelcontextprotocol.io) | 入门教程和概念指南 |
| 💻 GitHub | [MCP Official Repo](https://github.com/modelcontextprotocol/modelcontextprotocol) | Schema 定义和 spec 源文件 |
| 💻 GitHub | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | Python SDK 源码，值得深读 |
| 💻 GitHub | [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) | TS SDK 源码，与 Python SDK 对照阅读 |
| 📄 博客 | [MCP First Anniversary Blog](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) | 官方回顾，了解协议演进史 |
| 📄 分析 | [Why the Model Context Protocol Won (The New Stack)](https://thenewstack.io/why-the-model-context-protocol-won/) | 深度分析 MCP 为何成为行业标准 |
| 📄 安全 | [MCP Spec Security Analysis](https://forgecode.dev/blog/mcp-spec-updates/) | MCP 安全模型详解 |

---

## M4.5 · 实践层：Agent 工程实践拆解

**目标**：用前四个里程碑学到的理论，去拆解当今最前沿的 AI 开发工具的架构决策。理解"产品级 Agent"是如何将 Context Engineering、Tool-use、工作流编排等概念落地的。

> **定位说明**：这个模块不是学新理论，而是"理论验证场"——当你能看着 Claude Code 的 Skills 机制说出"这就是 Context Engineering 中的 select 策略"，说明你真正内化了前面的知识。同时，这些也是你日常开发中马上能用起来的实践。

### 4.5.1 Claude Code 架构拆解

Claude Code 是目前业界最具代表性的 Agentic Coding Tool，它的内部架构是我们学过的所有概念的集大成者：

**CLAUDE.md — 持久化 Context 注入**

- **本质**：分层的 System Prompt 管理系统。用户级（`~/.claude/CLAUDE.md`）、项目级（项目根目录）、本地级（`.claude/` 目录）三层配置，优先级从高到低合并。
- **对应理论**：M1.3 中讲的 Context Engineering。CLAUDE.md 解决的就是"如何在每次对话开始时，自动注入正确的项目上下文"。
- **类比**：类似 Go 项目中的 `.golangci.yml` + `Makefile` 的组合——项目级配置自动生效，不需要每次手动指定。

**Skills — 按需加载的知识包**

- **本质**：`SKILL.md` 文件定义了特定任务的指令集。通过 YAML frontmatter 中的 `name` 和 `description`，Claude Code 可以自动判断何时加载，也可以通过 `/slash-command` 手动触发。
- **对应理论**：Context Engineering 中的 **Select（选择）** 策略——不是把所有知识塞进 context，而是按需检索最相关的指令。Skills 的 `description` 字段本质上就是 Tool Description 的变体——写得好不好，直接决定触发是否精准。
- **三层架构**：Reference Skills（知识层）→ Task Skills（执行层）→ Domain Skills（领域层），和我们在 M2 中讨论的 Memory 类型学（Semantic / Procedural / Episodic）有直接映射。

**Hooks — 确定性控制点**

- **本质**：在 Agent 执行生命周期的关键节点（`PreToolUse`、`PostToolUse`、`PreCommit`、`Stop` 等）注入 Shell 脚本或 Python 脚本，拦截、校验或增强 Agent 的行为。
- **对应理论**：M3.2 状态机模式中的"确定性控制 vs 模型自由度"——Hooks 就是你用硬编码逻辑约束 Agent 行为的具体手段。模型负责"想"，Hooks 负责"兜底"。
- **类比**：和 Go 中的 HTTP Middleware、Git Hooks 完全同构。`PreToolUse` Hook 检查敏感文件 = `pre-commit` Hook 检查是否包含密钥。

**Sub-agents — 多 Agent 编排的产品化**

- **本质**：Claude Code 可以通过 Task Tool 生成子 Agent，每个子 Agent 有独立的 context window 和特定任务。
- **对应理论**：M5 中的 Hub-and-Spoke 多 Agent 模式的产品级实现。主 Agent 是 Orchestrator，子 Agent 是 Worker。

### 4.5.2 Agentic 开发工具全景对比

理解了 Claude Code 的架构后，横向对比其他主流工具，你会发现它们都是同一套理论的不同实现：

| 工具 | 类型 | Context 管理方式 | 安全模型 | 核心特色 |
|------|------|-----------------|----------|----------|
| **Claude Code** | 终端 Agent | CLAUDE.md + Skills 分层注入 | 应用层 Hooks（17 个事件点） | 1M context · Sub-agents · 最强推理 |
| **Cursor** | AI IDE | .cursorrules + 自动索引 | OS 级沙箱（2026 年新增） | 最快补全 · Composer 多文件编辑 |
| **Windsurf** | AI IDE | Cascade 自动上下文检索 | 用户审批 | Flow 持久记忆 · 企业级安全 |
| **Codex CLI** | 终端 Agent | AGENTS.md（Linux Foundation 标准） | 内核级沙箱（Seatbelt/Landlock） | 云端隔离容器 · 并行任务 |
| **Copilot** | VS Code 插件 | copilot-instructions.md | GitHub 权限体系 | 多模型支持 · 最大用户基数 |

**关键洞察**：每个工具的 context 配置文件（CLAUDE.md / .cursorrules / AGENTS.md）本质上都是同一个东西——**持久化的 System Prompt 注入机制**。它们的差异在于作用域管理、加载策略和安全边界的设计选择。

### 4.5.3 通用设计模式提炼

从这些工具中可以提炼出几个正在成为行业共识的 Agent 工程模式：

- **分层 Context 注入**：用户级 → 项目级 → 任务级，优先级递增，冲突时高层覆盖低层。所有主流工具都采用了这个模式。
- **Skill 按需加载**：不把所有能力一次性塞进 context，而是根据任务动态加载——这是 Context Engineering 在产品中的核心体现。
- **确定性 Guardrails**：关键操作（文件写入、命令执行、代码提交）用硬编码规则兜底，不依赖模型判断。Hooks / 沙箱 / 权限系统都是这个思路。
- **配置即代码（Config-as-Code）**：Agent 的行为规范以版本化的文件形式存在于代码仓库中，和代码一起 review、一起演进。

### 4.5.4 Harness Engineering：把它们串起来的上位概念

2026 年初，业界出现了一个新术语来统称上述所有实践——**Harness Engineering**（线具工程）。OpenAI 的 Codex 团队用纯 Agent 生成了百万行生产代码，Martin Fowler 在 ThoughtWorks 正式撰文讨论，这个概念迅速成为主流。

**核心公式：`Agent = Model + Harness`**

Harness 就是模型之外的一切——Context 管理、工具编排、Guardrails、反馈循环、可观测性。Philipp Schmid 的类比很到位：模型是 CPU，Context Window 是 RAM，而 **Harness 就是操作系统**。

**与我们 Roadmap 的映射关系**：

```
Harness Engineering 的组成部分          对应 Roadmap 位置
──────────────────────────────────     ──────────────────
Context 管理（注入/选择/压缩/隔离）  ←  M1.3 Context Engineering
工具编排与 Tool-use Loop             ←  M2.1 Tool-use 机制
任务规划与状态管理                    ←  M3.2 状态机模式
确定性 Guardrails（Hooks / 沙箱）    ←  M4.5.1 Hooks + M4.5.3 通用模式
反馈循环与自愈（Error → Retry）      ←  M3.5 Error Handling / Self-Healing
持久化指令（CLAUDE.md / Skills）     ←  M4.5.1 Claude Code 架构
多 Agent 编排                        ←  M5 Multi-Agent
```

**关键洞察**：Harness Engineering 不是一个需要从头学的新学科——你学完这份 Roadmap，就已经掌握了构建 Harness 的全部核心能力。这个概念的价值在于给你一个**统一的思维框架**，让你在面对任何 Agent 系统时都能问出正确的问题："这个 Agent 的 Harness 是怎么设计的？Context 怎么管？Guardrails 在哪？反馈循环是什么？"

**演进脉络**：

```
2024: Prompt Engineering  ──▶  写好提示词
2025: Context Engineering ──▶  管好上下文窗口
2026: Harness Engineering ──▶  造好模型周围的整个运行环境
```

每一代不是替代前一代，而是在前一代基础上扩展范围。Prompt Engineering 是 Context Engineering 的子集，Context Engineering 又是 Harness Engineering 的子集。

### 4.5.5 实战环节

- 我们共同拆解你当前的 Claude Code 使用场景，讨论哪些痛点可以通过 Skills / Hooks / CLAUDE.md 优化解决。
- 我来实现一个自定义 Skill（例如：一个 Go 代码 Review Skill，包含你团队的编码规范），你 Review 并接入你的 Claude Code 环境。
- 我们讨论并实现一个 PreToolUse Hook（例如：阻止 Agent 修改 `go.mod` 而不先运行 `go mod tidy`），你来验证效果。

### 📚 M4.5 参考资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📄 官方文档 | [Claude Code Skills Docs](https://code.claude.com/docs/en/skills) | Skills 的权威指南 |
| 📄 官方文档 | [Claude Code Hooks Docs](https://code.claude.com/docs/en/hooks) | Hooks 的权威指南 |
| 📄 博客 | [OpenAI: Harness Engineering](https://openai.com/index/harness-engineering/) | OpenAI 官方博文，Codex 团队百万行代码的 Harness 实践 |
| 📄 博客 | [Martin Fowler: Harness Engineering for Coding Agents](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html) | Guides + Sensors 框架，计算性与推理性控制的分类 |
| 📄 博客 | [Philipp Schmid: The Importance of Agent Harness](https://www.philschmid.de/agent-harness-2026) | Agent = Model + Harness 的原始阐述 |
| 💻 GitHub | [awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) | Claude Code 生态资源大全：Skills、Hooks、插件、工作流 |
| 💻 GitHub | [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) | 社区总结的最佳实践，含 37 条 Tips |
| 💻 GitHub | [Learn Claude Code (shareAI-Lab)](https://github.com/anthropics/claude-code) | 拆解 Coding Agent 的设计，用几百行 Python 重建核心架构 |
| 📄 博客 | [Inside Claude Code: Architecture Behind Tools, Memory, Hooks, and MCP](https://www.penligent.ai/hackinglabs/inside-claude-code-the-architecture-behind-tools-memory-hooks-and-mcp/) | 深度架构分析 + 安全视角 |
| 📄 博客 | [Claude Code Has a Hidden Architecture (Medium)](https://medium.com/@ai_93276/claude-code-has-a-hidden-architecture-most-engineers-never-find-it-72e7e3a299c1) | 四层架构解析：CLAUDE.md → Skills → Hooks → Agents |
| 📄 博客 | [My Claude Code Setup: MCP, Hooks, Skills](https://okhlopkov.com/claude-code-setup-mcp-hooks-skills-2026/) | 一个 24/7 自主运行 Agent 的完整配置实例 |
| 📄 对比 | [Codex CLI vs Claude Code Architecture Deep Dive](https://blakecrosley.com/blog/codex-vs-claude-code-2026) | 两大终端 Agent 的架构设计决策对比 |
| 📄 对比 | [Cursor vs Windsurf vs Claude Code (DEV)](https://dev.to/pockit_tools/cursor-vs-windsurf-vs-claude-code-in-2026-the-honest-comparison-after-using-all-three-3gof) | 三大工具的诚实对比，含 80/15/5 使用法则 |

---

## M5 · 系统层：多智能体架构

**目标**：从单 Agent 扩展到多 Agent 协同系统，理解分布式 Agent 架构的设计挑战。

### 5.1 为什么需要 Multi-Agent？

**核心概念**：

- **单体 vs 微服务**：单个万能 Agent 在面对复杂任务时会出现性能下降、context 爆炸、难以维护等问题。这和你把所有业务逻辑写进一个 Go 程序是一回事——最终你会需要拆分为微服务。
- **专业化**：每个 Agent 聚焦特定领域（Coder、Reviewer、Researcher），通过协作完成整体任务。
- **涌现行为**：多个简单 Agent 的交互可能产生单个复杂 Agent 无法实现的复杂行为模式。

### 5.2 Multi-Agent 通信模式

| 模式 | 描述 | 类比 |
|------|------|------|
| **Hub-and-Spoke** | 一个 Orchestrator 协调多个 Worker Agent | Go 中的 Worker Pool 模式 |
| **Peer-to-Peer** | Agent 之间直接通信 | Go 中 goroutine 之间通过 channel 通信 |
| **Blackboard** | 共享一块"黑板"（状态存储），所有 Agent 读写同一数据 | Go 中的 `sync.Map` 共享状态 |
| **Hierarchical** | 树状管理结构，高层 Agent 委托给低层 Agent | 微服务中的服务编排 |

### 5.3 Agent-to-Agent 协议：A2A

- Google 在 2025 年提出的 Agent-to-Agent Protocol (A2A)，与 MCP 互补——MCP 解决"Agent 到 Tool"的通信，A2A 解决"Agent 到 Agent"的通信。
- 2025年12月，A2A 同 MCP 一起被捐赠给 Linux Foundation 的 Agentic AI Foundation (AAIF)。

### 5.4 关键挑战

- **Context 隔离 vs 共享**：每个 Agent 有自己的 context window。Agent 间传递信息时，如何高效压缩？Cognition AI 使用微调模型做 Agent 边界的摘要——这说明这个环节的工程复杂度之高。
- **死锁与无限循环**：多个 Agent 互相等待或反复踢皮球。需要设置最大轮次、超时机制。类比 Go 中的 `context.WithTimeout()`。
- **一致性与冲突解决**：当多个 Agent 对同一问题给出矛盾的建议时，如何仲裁？
- **安全边界**：每个 Agent 的权限控制——"bounded autonomy"架构正在成为企业级多 Agent 系统的标准实践。

### 5.5 实战环节：构建一个双 Agent 系统

我们共同设计并实现一个 **Coder + Reviewer** 的双 Agent 协作系统：
- 先讨论整体架构：消息格式、迭代终止条件、错误处理策略。
- 我来实现代码，你 Review 以下关键点：Agent 间的消息传递是否清晰？死锁防护是否到位？
- 共同运行实验，观察两个 Agent 的交互质量。

系统组成：
- **Coder Agent**：接收需求，生成代码。
- **Reviewer Agent**：审查代码，提出改进建议。
- **Orchestrator**（Python 代码）：在两个 Agent 之间传递消息，控制迭代轮次。

```python
# 🐍 Python 插播：dataclass 是 Python 的结构体语法糖
# 类似 Go 的 struct，但自动生成 __init__、__repr__ 等方法
from dataclasses import dataclass

@dataclass
class AgentMessage:
    sender: str      # 类似 Go: Sender string
    content: str     # 类似 Go: Content string
    round_num: int   # 类似 Go: RoundNum int
```

### 📚 M5 参考资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📄 Paper | [AutoGen: Multi-Agent Conversation Framework (Wu et al., 2023)](https://arxiv.org/abs/2308.08155) | 微软的多 Agent 框架论文 |
| 📄 Paper | [MetaGPT: Multi-Agent Collaborative Framework (Hong et al., 2023)](https://arxiv.org/abs/2308.00352) | 用软件工程角色分工的多 Agent 系统 |
| 📄 Paper | [ChatDev (Qian et al., 2023)](https://arxiv.org/abs/2307.07924) | 模拟软件公司的多 Agent 开发 |
| 💻 GitHub | [AutoGen](https://github.com/microsoft/autogen) | 微软出品，扒源码首选 |
| 💻 GitHub | [CrewAI](https://github.com/crewAIInc/crewAI) | 角色扮演式多 Agent，API 设计值得学习 |
| 📄 博客 | [Anthropic: Multi-Agent Systems Guide](https://www.anthropic.com/engineering/building-effective-agents) | 涵盖多 Agent 模式的官方指南 |
| 📄 分析 | [Agentic AI Trends 2026 (MLM)](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/) | 多 Agent 系统在 2026 年的趋势分析 |

---

## 🧭 跨里程碑：持续关注的前沿方向

以下方向尚在快速演进中，建议在完成主 Roadmap 后持续跟踪：

| 方向 | 说明 | 关注资源 |
|------|------|----------|
| **Agent Evaluation** | 如何系统化地测试和评估 Agent 的表现 | [SWE-bench](https://github.com/princeton-nlp/SWE-bench), [GAIA](https://huggingface.co/gaia-benchmark) |
| **Agent Safety & Governance** | 权限控制、审计追踪、bounded autonomy | Anthropic Research Blog |
| **Browser/Computer Use Agents** | 让 Agent 操作浏览器/桌面，执行 UI 自动化 | [Browser Use (78K stars)](https://github.com/browser-use/browser-use) |
| **Context Engineering 进阶** | Context Rot 研究、动态 context 管理 | Anthropic Engineering Blog, LangChain Blog |
| **Agent-Native 开发工具** | Claude Code, Cursor, Windsurf 等背后的架构 | 各工具的 SDK/Plugin 源码 |

---

## 📅 建议的学习节奏

| 周次 | 里程碑 | 核心产出 |
|------|--------|----------|
| 1-3 | **M1** 地基层 | 能用纯 HTTP 调用 LLM API；理解 tokenization 和 context window |
| 4-7 | **M2** 核心层 | 协作完成一个 ~200 行的 Mini Agent（含 Tool-use + Memory） |
| 8-11 | **M3** 设计层 | 实现 ReAct、状态机、反思等模式；完成 Prompt Chain 编排 |
| 12-14 | **M4** 协议层 | 协作实现一个可用的 MCP Server；深读 MCP spec |
| 15-17 | **M4.5** 实践层 | 拆解 Claude Code 等工具的架构；自定义 Skills + Hooks 实战 |
| 18-21 | **M5** 系统层 | 构建 Coder+Reviewer 双 Agent 系统 |

> **节奏说明**：以上为建议节奏，非刚性约束。每个人的工作强度和学习速度不同。重要的是每个里程碑都有明确的"可交付产出"——**你能读懂并审查通过的代码，就是你最好的知识固化方式。**

---

## 🔗 持续对齐机制：边学基础，边看前沿

> 学底层不等于闭门造车。以下习惯确保你在打地基的同时，始终知道业界在发生什么。

**每周 30 分钟：前沿扫描**

关注 2-3 个高信噪比信息源即可，不求多求精：

| 信息源 | 说明 |
|--------|------|
| [Anthropic Engineering Blog](https://www.anthropic.com/engineering) | 第一手的 Agent 构建理念和 Context Engineering 实践 |
| [OpenAI Blog - Engineering](https://openai.com/index/) | Harness Engineering、Codex 架构等前沿实践 |
| [Simon Willison's Blog](https://simonwillison.net/) | AI 工程领域最高质量的独立评论，每篇都值得读 |

**每节课结束时：一句话"桥接"**

导师在每节课结束的迭代日志中，会附带一句「前沿连接」，把刚学的底层概念和当前业界热点做关联。例如：

- 学完 M2.1 Tool-use → "Claude Code 源码中的 `tool_executor.py` 就是这个 Loop 的产品级实现"
- 学完 M3.2 状态机 → "Cursor 的 Agent Mode 本质上是一个带 retry 和 human-in-the-loop 的 FSM"

**核心心态：T 型学习**

广度（每周 30min 扫描）和深度（Roadmap 主线）不冲突，反而互相加速：
```
广度（每周 30min 扫描前沿）
━━━━━━━━━━━━━━━━━━━━━━━━
         ┃
         ┃  深度（Roadmap 主线学习）
         ┃
         ┃
         ┃
```

横向保持信息灵敏度，纵向保持学习深度。你扫描到的新概念会不断印证"我学的这些基础原来被用在了这里"，而基础越扎实，扫描时的信息吸收率越高。

---

## 🔄 Roadmap 迭代日志

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-04-04 | v1.0 | 初始版本，确立五大里程碑和资源列表 |
| 2026-04-04 | v1.1 | 新增「协作模式」章节；全文实战环节统一为"讨论驱动 + AI 实现 + 人工审查"模式 |
| 2026-04-04 | v1.2 | 新增 M4.5「实践层」：Claude Code 架构拆解、Agentic 工具全景对比、Skills/Hooks 实战 |
| 2026-04-04 | v1.3 | 完成 M1.1 实战：纯 HTTP 调用 Anthropic API，验证 Temperature 对确定性/创意任务的影响差异 |
| 2026-04-04 | v1.3 | 完成 M1.2「Prompt Engineering」：Zero-shot/CoT/Few-shot 三策略对比实验，Go Code Reviewer 实战，结论：Few-shot 格式控制 > 文字约束，生产最优解为两者结合 |
| 2026-04-04 | v1.4 | 完成 M1.3「Context Window 与 Token 经济学」：Tokenizer 效率实验（中文 tok/char 是英文 4.3x）、Needle in a Haystack 实验（3K token 下 Lost in the Middle 不显著，发现 tool_result 累积是 Agent 场景最大的 context 坑） |
| 2026-04-04 | v1.5 | M4.5 新增「Harness Engineering」章节：建立 Agent=Model+Harness 概念映射，串联 Roadmap 全部理论，补充 OpenAI/Fowler/Schmid 三大核心参考 |
| 2026-04-05 | v1.6 | 完成 M2.1「Tool-use 机制」：实现完整 Tool-use Loop（工具注册表 + 错误分类 + Parallel Tool Use），实验验证模型自动并行调用独立工具、串行调用依赖工具；厘清 stream=False vs stream=True 两种响应结构（content blocks / SSE 事件序列）；确立错误处理两道防线：Tool Description 过滤无效调用，execute_tool 返回 error dict 处理运行时错误 |
| 2026-04-06 | v1.7 | 完成 M2.2「Planning 机制」：实现显式计划生成（JSON 结构化 plan）、validate_plan（fail fast 四项校验）、交互式 review_plan（人工审批节点，支持多轮修改）、代码驱动 execute_plan（执行阶段 0 次 LLM 调用）、summarize（唯一结尾 LLM 调用）；厘清规划层/执行层信息隔离原则；验证"传给模型信息越精确幻觉越少"；修复 markdown JSON 包裹、task 与 plan 不一致导致的幻觉问题 |
| 2026-04-06 | v1.8 | 完成 M2.3「Memory 机制」：实现 ConversationBuffer（按 token 触发压缩、滚动摘要窗口 MAX_SUMMARY_CHUNKS=3）和 FactStore（持久化 JSON、关键词 OR 检索、预留 embedding 字段）；厘清 as_context_string（热数据自动注入，max_facts=20）与 recall_facts（冷数据按需检索）的分工边界；确立三层检索架构：L1全量注入 → L2关键词检索 → L3向量检索（M3覆盖）；工厂函数+闭包实现依赖注入，替代全局变量 |
| 2026-04-08 | v1.9 | 完成 M2.4「Mini Agent 综合实战」：实现 ToolRegistry（分模块注册 + 统一执行接口）；确立工具接口契约：所有实现函数返回带 status 的扁平 dict，路由层无脑透传不感知内部格式；天气工具采用方案 B（扁平结构）避免 compare_weather 感知 get_weather 返回格式；Planning 内化到 System Prompt（轻量化，无审批节点）；验证跨对话记忆持久化、recall_facts 主动触发、tool description 过滤无效调用等核心机制均符合预期 |
| 2026-04-08 | v2.0 | 新增「持续对齐机制」：前沿扫描信息源 + T 型学习策略，解决"学基础是否会落伍"的焦虑 |
| 2026-04-08 | v2.1 | 完成 M3.1「ReAct 范式」：实现 ReAct Agent（extract_thinking 正则解析、Thought/Action/Observation 三段式日志）；对比实验验证 ReAct 是 Prompt 层改造而非架构重写；厘清两个认知修正：① extract_thinking 应用 findall 合并多块；② thinking 不存入最终回答的真实原因是"调试信息不属于对话内容"而非"格式污染"；确认 ReAct 核心价值是决策可审计性而非结果质量提升 |

---

> **协作约定**：本文档确认后，将作为 `.md` 文件上传至 Project Knowledge。后续每个 Milestone 开启独立 Chat 进行专项学习——以讨论为主，需要写代码时按「协作模式」执行。Chat 标题建议格式：`[M1.1] Token 与 API 底层`、`[M2.4] 构建 Mini Agent` 等。每完成一个里程碑，回来更新本文档的迭代日志。