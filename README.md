# Data Management Agent

Data Management Agent 是一个面向文件、知识和数据工作空间的 Agent 项目，而不是单纯的 RAG Demo。

项目将以 Python 3.11、LangGraph、Pydantic 和 uv 为基础，逐步构建可扩展的 Agent runtime、工具接口、检索能力、评估体系和轨迹记录机制。

当前处于第一阶段，已实现面向 DABstep 的只读最小 Agent 链路；尚未实现完整 Tool Surface，
也未配置 Agentic RL 训练环境。

## Minimal vertical slice

当前版本提供一个只读的最小链路：

`User task → LangGraph agent → workspace tools → DABstep data → final answer`

已实现的工具包括 `list_workspace`、`search_workspace`、`inspect_file`、`read_file` 和
基于 DuckDB 的 `query_data`。检索目前是内存 BM25 baseline，没有 embedding、向量数据库或
reranker。

先下载共享 workspace 数据：

```bash
uv run python scripts/download_dabstep.py
```

复制环境变量模板，并在本地 `.env` 中填写 API key：

```bash
cp .env.example .env
```

```dotenv
DATA_WORKSPACE_PATH=data/external/dabstep
LOG_LEVEL=INFO
OPENAI_API_KEY=your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
```

当前通过 LangChain 的 OpenAI-compatible 适配器连接 DeepSeek，因此环境变量仍使用
`OPENAI_*` 命名。复制 `.env.example` 后只需填写 `OPENAI_API_KEY`。

运行 Agent：

```bash
uv run python scripts/run_agent.py \
  "Which issuing country has the highest number of transactions?" \
  --guidelines "Answer must be just the country code."
```

本地开发调试也可以使用 Streamlit 聊天界面：

```bash
uv run --group dev streamlit run scripts/streamlit_app.py
```

每轮回答下面会显示可展开的 tool trajectory，包括参数、结果、错误、延迟和模型调用的
token usage。该 UI 只调用 `WorkspaceAgent.run()`，不包含独立的 tool routing 或 workspace
访问逻辑。

## DABstep benchmark v0

下载共享 workspace 和官方 dev task manifest（不会下载 submissions 或 leaderboard 数据）：

```bash
uv run python scripts/download_dabstep.py --include-dev-tasks
```

顺序运行前 3 个 dev tasks：

```bash
uv run python scripts/run_benchmark.py --limit 3
```

或者运行指定 task IDs：

```bash
uv run python scripts/run_benchmark.py --task-ids 5 49 1305
```

每次运行写入 `data/external/dabstep_benchmark/results/*.jsonl`，与 Agent 可见的 workspace 隔离。
每条记录包含 task、conversation、thread 和 turn 标识，以及 question、reference/final answer、
完整 message trajectory、tool calls、tool errors、延迟、token usage 和
`normalized_exact_match_v0` correctness。Streamlit 侧边栏的 `Benchmark Results` 页面可以查看
汇总和单任务详情。

`.env` 和 `data/external/` 均不会提交到 Git。没有 API key 时仍可运行 Workspace、Tool 和
离线 smoke tests，但无法调用真实 chat model。
