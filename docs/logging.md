# 本地系统日志

CLI、Streamlit Chat 和 benchmark 现在默认启用 `RunJournal`，运行命令不变。
日志在运行过程中逐事件写入，而不是只在成功返回后保存。`LOG_LEVEL` 不影响这份
结构化审计日志。直接使用 `WorkspaceAgent` 的调用者需要传入 `journal=RunJournal(root)`。

## 文件与版本

- `logs/index.jsonl`：运行开始/结束的轻量索引；包含运行标识、版本、状态和事件文件路径。
- `logs/runs/<execution_id>.jsonl`：本次调用的完整事件流。
- `logs/versions/<version_id>.json`：首次运行该版本时保存的版本快照。
- `docs/journal/`：开发助手撰写的文字日记，随代码纳入版本管理。

版本快照包含 Git commit/dirty、`src/` 和 `scripts/` Python 源码及 pyproject/lock 文件
SHA-256、Python/关键依赖版本，以及模型名、endpoint、temperature、重试设置、system
prompt、tool schema、workspace ID 和图步数配置。`version_id` 是该快照的哈希。
因此未提交源码和 prompt/schema 改动也能区分，不只是记录一个 Git commit。
这里记录的是源码指纹，不是源码备份；仍需 Git/日记保留可恢复版本。
workspace ID 标识位置，不是数据内容版本；目前不对大型数据集做逐次哈希。

## 事件和关联字段

`schema_version=agent-events-v1`；每条事件有 UTC 时间、sequence、elapsed_ms、version_id，
以及以下关联字段：

| 字段 | 含义 |
| --- | --- |
| run_id | 一次普通运行，或整个 benchmark 批次 |
| execution_id | 一次 Agent 调用，对应一个事件文件 |
| conversation_id / thread_id | 逻辑会话/线程；仅标识，不加载历史消息或 checkpoint |
| task_id | benchmark 官方任务 ID；普通聊天则生成 UUID |
| turn_id / turn_index | 本次轮次，索引从 0 开始 |
| source | cli / streamlit / benchmark / api 等入口 |

UI 同一会话共享 conversation/thread，每个问题拥有独立 execution/task/turn；清空会话后
生成新 ID，但不会删除磁盘日志。Agent 仍是单轮，不会把 UI 的旧问题传给模型。
benchmark 结果新增可选 `execution_id`，可与事件文件关联；旧 JSONL 仍可加载。

事件顺序：`run_started → llm_started → llm_finished → tool_started → tool_finished
→ … → run_finished`。模型异常为 `llm_failed`，然后保存失败的 `run_finished`。

- `run_started.data`：问题、guidelines、实际运行配置。
- `llm_finished.data`：完整 AIMessage（文本及 tool calls）、latency_ms、token_usage。
- `tool_started.data`：call_id、name、arguments。
- `tool_finished.data.message`：完整 ToolMessage；content 为工具结果/错误 JSON，status
  区分 success/error，artifact.latency_ms 为工具耗时，通过 tool_call_id 与参数关联。
- `run_finished.data`：状态、最终答案或异常类型/信息、总 latency、LLM/tool 调用数、
  tool error 数和总 token usage。失败时还保留可获得的图消息快照。

用每次 LLM 响应的 usage 求和，不按 tool 次数重复累计；`usage_reports` 表示提供 usage
的响应数，缺失的 usage 不冒充 0。细分 cache/reasoning 等字段保留在原始响应事件中。
这些数字是模型服务报告的用量，不是独立计费审计。没有记录模型未公开的内部推理。

## 边界与隐私

日志逐事件关闭文件；普通模型/工具/图异常保留已发生的过程。强杀进程时可能只有开始
而没有结束事件；这代表未完成，不代表成功。磁盘写入失败会警告并继续运行，不能承诺
磁盘不可写时仍能保存日志。进程启动、模型/workspace 初始化阶段的错误仍在终端报告。
当前无日志轮转、保留期管理或跨进程事务；长期开启需要自行留意磁盘占用。

不序列化 `.env` 或模型客户端。过滤已加载到环境的 API key/HF token 值及常见敏感键，
但这不是通用隐私脱敏器：问题、答案、工具结果可能包含原始业务数据，日志只供本地调试。
`logs/`、`.env`、`data/external/` 均被 Git 忽略，分享前需人工检查。

这里的会话日志指项目 Agent 的会话，不是 Codex/ChatGPT 开发聊天的自动全文归档；
开发决策由助手写入日记。旧 CLI/UI 运行无法追溯补齐，既有 benchmark JSONL 保留原样。
