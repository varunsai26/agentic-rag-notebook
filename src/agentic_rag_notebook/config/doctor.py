"""CLI health-check utility to validate `.env` configuration safely without leaking secrets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.infrastructure.embedding_factory import EmbeddingFactory
from agentic_rag_notebook.infrastructure.llm_factory import LLMFactory


@dataclass(frozen=True)
class EnvCheckReport:
    """Structured result from checking `.env` settings and client factories."""

    ok: bool
    env_file_exists: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    collection_name: str
    warnings: list[str]
    errors: list[str]
    ping_output: str = ""


def _mask_secret_status(secret: str, *, placeholder_prefix: str = "") -> str:
    if not secret:
        return "missing"
    if placeholder_prefix and secret.startswith(placeholder_prefix):
        return "placeholder"
    if secret in {"your_local_server_api_key", "your_local_embedding_key"}:
        return "placeholder"
    return f"configured ({len(secret)} chars)"


def inspect_environment(
    settings: Settings | None = None,
    *,
    env_path: Path = Path(".env"),
    ping: bool = False,
) -> EnvCheckReport:
    """Validate `.env` fields, active provider credentials, and optional live model connectivity."""
    errors: list[str] = []
    try:
        active_settings = settings or Settings()
    except Exception as exc:
        return EnvCheckReport(
            ok=False,
            env_file_exists=env_path.exists(),
            llm_provider="unknown",
            llm_model="unknown",
            embedding_provider="unknown",
            embedding_model="unknown",
            collection_name="unknown",
            warnings=[],
            errors=[f"Failed to parse .env / environment settings: {exc}"],
        )

    warnings = active_settings.validate_active_configuration()
    ping_output = ""

    try:
        llm, llm_label = LLMFactory(active_settings).create()
        if ping:
            msg = llm.invoke("Reply with OK only.")
            ping_output = f"{llm_label} -> {str(msg.content).strip()[:80]}"
    except Exception as exc:
        errors.append(f"LLM check failed: {exc}")

    if active_settings.embedding_provider in {"openai", "openai_local"}:
        try:
            emb = EmbeddingFactory(active_settings).create()
            if ping:
                vec = emb.embed_query("healthcheck")
                ping_output = f"{ping_output} | embeddings_dim={len(vec)}".strip(" |")
        except Exception as exc:
            errors.append(f"Embedding check failed: {exc}")

    return EnvCheckReport(
        ok=not warnings and not errors,
        env_file_exists=env_path.exists(),
        llm_provider=active_settings.llm_provider,
        llm_model=active_settings.active_llm_model_name(),
        embedding_provider=active_settings.embedding_provider,
        embedding_model=active_settings.active_embedding_model_name(),
        collection_name=active_settings.embedding_collection_name(),
        warnings=warnings,
        errors=errors,
        ping_output=ping_output,
    )


def main(argv: list[str] | None = None) -> int:
    """Run CLI `.env` verification and return exit code 0 when healthy, 1 otherwise."""
    parser = argparse.ArgumentParser(
        prog="agentic-rag-check-env",
        description="Validate .env configuration for Groq / Local Qwen and embeddings.",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Send a tiny live test prompt to verify API key, base_url, and model reachability.",
    )
    args = parser.parse_args(argv)

    report = inspect_environment(ping=bool(args.ping))
    print(f".env file found     : {'yes' if report.env_file_exists else 'no (using defaults/env)'}")
    print(f"LLM provider/model  : {report.llm_provider} / {report.llm_model}")
    print(f"Embedding model     : {report.embedding_provider} / {report.embedding_model}")
    print(f"Chroma collection   : {report.collection_name}")

    if report.env_file_exists and not report.errors:
        cfg = Settings()
        if cfg.llm_provider == "groq":
            print(
                "GROQ_API_KEY        : "
                f"{_mask_secret_status(cfg.groq_api_key, placeholder_prefix='gsk_your_')}"
            )
        else:
            print(f"OPENAI_LOCAL_URL    : {cfg.openai_local_base_url or 'missing'}")
            print(f"OPENAI_LOCAL_KEY    : {_mask_secret_status(cfg.openai_local_api_key)}")

    for warn in report.warnings:
        print(f"WARNING             : {warn}")
    for err in report.errors:
        print(f"ERROR               : {err}")
    if report.ping_output:
        print(f"Live ping           : {report.ping_output}")

    if report.ok:
        print("STATUS              : OK (.env configuration is valid)")
        return 0
    print("STATUS              : INVALID (fix warnings/errors above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
