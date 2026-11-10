---
name: langchain-middleware
description: LangChain v1 orchestration, built-in LangChain middlewares, SQLite URL sessions, and Phase 2 LangGraph progression.
---

# LangChain + Built-in Middleware + LangGraph Skill

## Rule: Always Use Built-in LangChain Utilities First
Prefer `langchain.agents.middleware` built-ins over custom boilerplate:
- `PIIMiddleware` & `RedactionRule` (email, credit_card, ip, custom regex like API keys/SSN/phone)
- `SummarizationMiddleware` (automatic conversation history summarization)
- `ModelCallLimitMiddleware` & `ToolCallLimitMiddleware` (loop & budget caps)
- `ToolRetryMiddleware` / `ModelRetryMiddleware` (resilient retries)
- `HumanInTheLoopMiddleware` (when interactive tool-approval interrupts are needed with a checkpointer)

Write custom `AgentMiddleware` hooks (`before_agent`, `after_agent`) only for RAG-specific rules that LangChain does not bundle (e.g., prompt-injection blocking and RAG citation/groundedness auditing).

## Phase 1 (Default): LangChain Orchestration
- Build chat models via `infrastructure/llm_factory.py` (`ChatGroq` or `ChatOpenAI(base_url=..., api_key=...)` for local Qwen).
- Use modular `ChatPromptTemplate` templates from `prompts/templates.py`.
- Register RAG `@tool` functions (`search_knowledge_base`, `list_active_sources`) with `langchain.agents.create_agent` and the built-in + RAG middleware stack.
- Persist conversations in SQLite (`SQLiteLangChainChatHistory(BaseChatMessageHistory)` + `sessions` table) keyed by a ChatGPT-style UUID `session_id` synced with `st.query_params["session_id"]`.

## Phase 2: LangGraph Progression
When `ORCHESTRATION_ENGINE=langgraph`, `orchestration/langgraph_agent.py` demonstrates an explicit `StateGraph` (`route -> retrieve -> grade_documents -> rewrite_query -> generate`) backed by `SqliteSaver` while reusing the exact same guardrail pipeline and `NotebookService`.
