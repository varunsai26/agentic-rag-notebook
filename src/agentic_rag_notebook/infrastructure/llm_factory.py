"""Factory for creating Groq or Local OpenAI-compatible (Qwen) LangChain chat models."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agentic_rag_notebook.config.settings import LLMProviderType, Settings


@dataclass(frozen=True)
class RuntimeLLMOverrides:
    """Optional runtime overrides from UI without mutating environment settings."""

    provider: LLMProviderType | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None


class LLMFactory:
    """Builds LangChain BaseChatModel instances from Settings + optional overrides."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create(self, overrides: RuntimeLLMOverrides | None = None) -> tuple[BaseChatModel, str]:
        """Instantiate the configured chat model and return `(model, display_label)`."""
        provider: LLMProviderType = (
            overrides.provider if overrides and overrides.provider else self._settings.llm_provider
        )
        temperature = (
            overrides.temperature
            if overrides and overrides.temperature is not None
            else self._settings.llm_temperature
        )

        if provider == "groq":
            api_key = (
                overrides.api_key
                if overrides and overrides.api_key is not None
                else self._settings.groq_api_key
            )
            model_name = (
                overrides.model if overrides and overrides.model else self._settings.groq_model
            )
            if not api_key or api_key.startswith("gsk_your_"):
                raise ValueError(
                    "GROQ_API_KEY is not configured. Please set a valid Groq API key in `.env`."
                )
            chat_model: BaseChatModel = ChatGroq(
                api_key=SecretStr(api_key),
                model=model_name,
                temperature=temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
            return chat_model, f"groq/{model_name}"

        # Local OpenAI-compatible Qwen client (vLLM, Ollama, LM Studio, LiteLLM, etc.)
        base_url = (
            overrides.base_url
            if overrides and overrides.base_url
            else self._settings.openai_local_base_url
        )
        api_key = (
            overrides.api_key
            if overrides and overrides.api_key is not None
            else self._settings.openai_local_api_key
        )
        model_name = (
            overrides.model if overrides and overrides.model else self._settings.openai_local_model
        )
        if not base_url:
            raise ValueError("OPENAI_LOCAL_BASE_URL is required when LLM_PROVIDER=openai_local.")
        effective_key = api_key if api_key else "local-no-key-required"
        local_chat_model: BaseChatModel = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr(effective_key),
            model=model_name,
            temperature=temperature,
            max_completion_tokens=self._settings.llm_max_tokens,
        )
        return local_chat_model, f"openai_local/{model_name}"
