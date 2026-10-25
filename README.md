# 多工具调用个人助手 Agent

基于 **LangGraph** 构建的轻量级多工具智能体（Agent）项目。Agent 具备**任务规划、工具调用、结果反思**的闭环能力：收到用户复杂自然语言请求后，自动拆解任务、按需调用外部工具、多步执行并输出最终答案，通过 Gradio 提供网页交互界面。

支持 **Ollama 本地模型**（免费、断网可用）与 **DeepSeek 云端 API**（更快、更强）两种接入方式，切换只需修改一个配置文件。

---

## ✨ 功能特性

- **任务规划（Planner）**：大模型根据用户问题判断"直接回答"还是"调用工具"，并输出结构化指令（JSON）
- **工具调用（Tools）**：内置 5 个工具，可自动按需调用
  - `read_file(file_path)`：读取本地文本文件（按行分片读取 + 字符数双重限制，防大文件撑爆上下文）
  - `calculator(expr)`：数学表达式计算（AST 语法树白名单过滤，防代码注入）
  - `text_summarize(text)`：文本摘要
  - `get_current_time()`：获取当前日期时间与星期
  - `write_file(file_path, content)`：安全写入本地文件（仅限 `data/` `output/` 目录，路径白名单防越权）
- **对话日志持久化**：每一轮对话轨迹自动保存到 `logs/` 目录，便于复盘
- **结果反思（Reflection）**：校验工具结果是否足够回答用户，决定继续调用还是收尾
- **防死循环**：最大执行步数限制（`MAX_STEP=5`），达到上限强制终止
- **多轮对话记忆**：历史消息自动累积，上下文感知
- **Web 交互界面**：Gradio 页面实时展示 Agent 思考日志（规划→工具→反思）+ 最终回答
- **双模型接入**：Ollama 本地 / DeepSeek 云端一键切换

---

## 🧠 核心工作流

```
用户输入
  ↓
Planner（规划节点）── 判断是否需要工具
  ├─ 不需要 → Final（直接输出答案）
  └─ 需要   → Tools（执行工具）
                ↓
              Reflection（反思节点）── 信息是否足够？
                ├─ 不足 → 回到 Planner 继续
                └─ 足够 / 达最大步数 → Final（整合输出）
```

这是 Agent 区别于普通 RAG / 链式问答的核心：**运行时动态决定调用什么工具、调用几次**。

---

## 📁 项目结构

```
多工具调用个人助手 Agent/
├── agent/                    # Agent 核心代码
│   ├── __init__.py           # 包标记
│   ├── state.py              # 状态定义（query/messages/tool_results/step）
│   ├── tools.py              # 工具集：read_file / calculator / text_summarize / get_current_time / write_file
│   ├── nodes.py              # 三个节点：planner / tools / reflection / final
│   └── graph.py              # LangGraph 工作流图（节点 + 条件路由）
├── data/
│   └── test.txt              # 测试用示例文档
├── config.py                 # 读取 .env，初始化大模型实例（自动适配两种接入）
├── main.py                   # 命令行调试入口
├── app.py                    # Gradio 网页界面
├── test_agent.py             # 三场景功能测试脚本
├── requirements.txt          # Python 依赖清单
├── .env                      # 本地配置（含密钥，禁止上传 GitHub）
├── .env.example              # 配置模板（可提交）
├── .gitignore
└── README.md
```

---

## 🛠 环境要求

- Windows / macOS / Linux
- Python 3.10+（本项目在 3.12 下开发验证）
- 方案一（Ollama）：已安装 [Ollama](https://ollama.com/download)，已拉取本地模型（如 `qwen3:4b`）
- 方案二（DeepSeek）：已注册 [DeepSeek 开放平台](https://platform.deepseek.com/) 并获取 API Key

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单：`langgraph`、`langchain`、`langchain-openai`、`gradio`、`python-dotenv`、`chromadb`

### 2. 配置模型（二选一）

复制 `.env.example` 为 `.env`，编辑其中内容。

**方案 A：Ollama 本地模型（免费，无需 Key）**

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=dummy
LLM_MODEL=qwen3:4b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=256
```

确保 Ollama 服务已启动（右下角托盘有 🦙 图标），并已拉取模型：

```bash
ollama pull qwen3:4b
```

**方案 B：DeepSeek 云端 API（更快更强，需 Key）**

```env
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的密钥
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512
```

> `config.py` 会自动适配：连接 Ollama（端口 11434）时关闭模型思考模式并限长输出；连接云端时忽略这些参数。切换只需改 `.env`，业务代码零改动。

### 3. 运行

**命令行调试**（无需网页）：

```bash
python main.py
```

**网页交互界面**：

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。

---

## 🧪 功能测试

```bash
python test_agent.py
```

覆盖三个核心场景：数学计算、文件读取+摘要（多步工具链）、闲聊直接回答。

---

## 💬 使用示例

| 输入 | Agent 行为 |
|---|---|
| `计算 123 + 456 * 7` | 调用 `calculator` → 返回 3315 |
| `读取 data/test.txt 文件，然后对文件内容做摘要` | 自动拆解：`read_file` → `text_summarize` → 输出摘要 |
| `现在几点？` | 调用 `get_current_time` → 返回当前日期时间与星期 |
| `把这句话保存到 output/note.txt：你好` | 调用 `write_file` → 写入本地文件（仅限 data/output 目录） |
| `你好，介绍一下你自己` | 判断无需工具，直接回答 |

---

## ❓ 常见问题（FAQ）

**Q1：网页报红色"错误"？**
大概率是 Ollama 服务未运行。检查右下角托盘是否有 🦙 图标，没有则双击 Ollama 启动，稍等再刷新页面。

**Q2：`ModuleNotFoundError: No module named 'langgraph'/'gradio'`？**
PyCharm 运行用了错误的 Python 解释器。在 `File → Settings → Python Interpreter` 中切换到项目虚拟环境 `.venv`（注意别选成外层目录的空 `.venv`）。

**Q3：导入 `agent` 模块报错？**
右键项目根目录 → `Mark Directory as` → `Sources Root`。

**Q4：本地模型回答很慢（几十秒）？**
Ollama 本地小模型速度受限于电脑配置；同时确保 `.env` 中 `LLM_MAX_TOKENS` 生效（限制超长输出）。追求速度建议切换 DeepSeek 云端方案。

**Q5：多轮对话报"messages format 不兼容"？**
需使用新版 `app.py`（已兼容 Gradio 5/6 的消息格式），并重新启动应用。

**Q6：切换云端后需要额外改代码吗？**
不需要。只改 `.env`，`config.py` 自动适配。

---

## 🛡 安全说明

- API Key 存放在 `.env`，已被 `.gitignore` 忽略，**禁止上传 GitHub**；对外只提供 `.env.example` 模板
- `calculator` 工具使用 AST 语法树白名单过滤，拒绝 `eval` 直接执行任意代码，防注入
- `write_file` 工具带路径白名单，仅允许写入 `data/` `output/` 目录，防越权访问系统文件
- 文件读取按行分片 + 字符数双重限制，避免超大文件消耗过多 token
- Agent 设置最大执行步数，防止无限循环调用工具

---

## 🔮 后续扩展方向（简历加分点）

- 增加工具：网页搜索、JSON/CSV 解析、代码执行沙箱
- 接入记忆持久化（如 SQLite / 向量库）实现长期记忆
- 用 Gradio 增加文件上传组件，直接上传文档问答
- 封装为 FastAPI 服务，提供 HTTP 接口

---

## 📄 License

仅供学习与简历项目展示使用。
