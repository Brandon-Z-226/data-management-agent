# Architecture Principles

## Project scope

本项目面向文件、知识和数据工作空间，目标是构建能够理解工作空间、调用工具并完成数据管理任务的 Agent，而不是单纯的 RAG Demo。

## Orchestration

- 使用 LangGraph 负责 Agent orchestration 和运行流程管理。
- 第一版采用“单主 Agent + Tools”的设计，避免过早引入大量 multi-agent 协作和调度复杂度。

## Component boundaries

以下能力保持相互解耦，并通过明确接口协作：

- Retrieval
- Tools
- Agent Runtime
- Evaluation
- Trajectory Logging

具体实现可以独立演进，不应把检索、工具执行、运行时状态和评估逻辑耦合在单一模块中。

## Stable interfaces

项目后续会加入 benchmark 和 Agentic RL。因此，从第一版开始就需要保持稳定、可版本化的 tool interface，并为 trajectory schema 保留清晰的演进边界。工具输入、工具输出、运行事件和评估数据应能够被可靠记录与重放。

## Current phase

当前阶段只开发 Agent 工程基础和运行能力，不配置 RL 训练环境，也不引入训练框架或大模型训练依赖。
