"""Wikipedia loader built on LangChain's WikipediaLoader + RecursiveCharacterTextSplitter."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag_notebook.domain.models import ParsedChunk, ScalarMetaValue

WIKIPEDIA_USER_AGENT = "AgenticRAGNotebook/0.1.0 (Python-requests; local-rag-research-client)"
_MIN_REQUEST_INTERVAL_SECONDS = 0.25
_REQUEST_TIMEOUT_SECONDS = 15.0
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 0.6
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)


def _parse_retry_after(header_value: str | None, fallback: float) -> float:
    """Parse numeric Retry-After header bounded to a safe wait window."""
    if not header_value:
        return fallback
    try:
        seconds = float(header_value.strip())
    except ValueError:
        return fallback
    return min(max(seconds, 0.1), 5.0)


def _configure_wikipedia_client(lang: str) -> None:
    """Configure Wikimedia-compliant User-Agent, HTTPS endpoint, and resilient request retries."""
    try:
        import wikipedia
        import wikipedia.wikipedia as wiki_mod

        wikipedia.set_user_agent(WIKIPEDIA_USER_AGENT)
        if not getattr(wikipedia.set_lang, "_https_patched", False):
            orig_set_lang = wikipedia.set_lang

            def _https_set_lang(prefix: str) -> None:
                orig_set_lang(prefix)
                https_url = f"https://{prefix.lower()}.wikipedia.org/w/api.php"
                wiki_mod.API_URL = https_url
                wikipedia.API_URL = https_url

            _https_set_lang._https_patched = True  # type: ignore[attr-defined]
            wikipedia.set_lang = _https_set_lang

        if not getattr(wiki_mod._wiki_request, "_resilient_patched", False):
            session = requests.Session()
            last_call_ts = 0.0

            def _resilient_wiki_request(params: dict[str, Any]) -> dict[str, Any]:
                nonlocal last_call_ts
                req_params = dict(params)
                req_params["format"] = "json"
                req_params.setdefault("action", "query")

                headers = {
                    "User-Agent": wiki_mod.USER_AGENT,
                    "Accept": "application/json",
                }
                url = wiki_mod.API_URL
                last_error: Exception | None = None

                for attempt in range(_MAX_RETRIES):
                    elapsed = time.monotonic() - last_call_ts
                    if last_call_ts > 0.0 and elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
                        time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)

                    try:
                        response = session.get(
                            url,
                            params=req_params,
                            headers=headers,
                            timeout=_REQUEST_TIMEOUT_SECONDS,
                        )
                        last_call_ts = time.monotonic()
                    except requests.RequestException as exc:
                        last_call_ts = time.monotonic()
                        last_error = exc
                        if attempt + 1 < _MAX_RETRIES:
                            time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
                            continue
                        raise RuntimeError(
                            f"Network error contacting Wikipedia API: {exc}"
                        ) from exc

                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        wait = _parse_retry_after(
                            response.headers.get("Retry-After"),
                            _RETRY_BACKOFF_SECONDS * (2**attempt),
                        )
                        last_error = RuntimeError(
                            f"Wikipedia API ({url}) returned HTTP {response.status_code}."
                        )
                        if attempt + 1 < _MAX_RETRIES:
                            logger.warning(
                                "Wikipedia API retry attempt=%d/%d status=%d wait=%.2fs",
                                attempt + 1,
                                _MAX_RETRIES,
                                response.status_code,
                                wait,
                            )
                            time.sleep(wait)
                            continue
                        raise last_error

                    if response.status_code >= 400:
                        snippet = " ".join(response.text.split())[:160]
                        detail = f": {snippet}" if snippet else ""
                        raise RuntimeError(
                            f"Wikipedia API ({url}) returned HTTP {response.status_code}{detail}"
                        )

                    body = response.text.strip()
                    if not body:
                        last_error = RuntimeError(
                            f"Wikipedia API ({url}) returned an empty response "
                            f"(HTTP {response.status_code})."
                        )
                        if attempt + 1 < _MAX_RETRIES:
                            time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
                            continue
                        raise last_error

                    try:
                        payload: dict[str, Any] = response.json()
                        return payload
                    except json.JSONDecodeError as exc:
                        snippet = " ".join(body.split())[:160]
                        last_error = RuntimeError(
                            f"Wikipedia API ({url}) returned a non-JSON response "
                            f"(HTTP {response.status_code}): {snippet}"
                        )
                        if attempt + 1 < _MAX_RETRIES:
                            time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
                            continue
                        raise last_error from exc

                raise RuntimeError(f"Wikipedia API ({url}) request failed: {last_error}")

            _resilient_wiki_request._resilient_patched = True  # type: ignore[attr-defined]
            wiki_mod._wiki_request = _resilient_wiki_request

        wikipedia.set_lang(lang)
    except ImportError:
        pass


def normalize_wikipedia_query(query_or_url: str) -> tuple[str, str]:
    """Extract a clean Wikipedia search query and language code from a topic or URL."""
    cleaned = query_or_url.strip()
    if not cleaned:
        raise ValueError("Please provide a Wikipedia topic or URL.")

    if cleaned.startswith(("http://", "https://")):
        parsed = urlparse(cleaned)
        host_parts = parsed.netloc.split(".")
        candidate = (
            host_parts[0].lower() if len(host_parts) >= 3 and len(host_parts[0]) <= 5 else "en"
        )
        lang = "en" if candidate == "www" else candidate
        if "/wiki/" in parsed.path:
            slug = parsed.path.split("/wiki/", maxsplit=1)[1]
            topic = unquote(slug).replace("_", " ").strip()
            if topic:
                return topic, lang
    return cleaned, "en"


def _flatten_scalar_wiki_meta(meta: dict[str, Any]) -> dict[str, ScalarMetaValue]:
    """Flatten WikipediaLoader's extended metadata into Chroma-safe scalar fields."""
    flat: dict[str, ScalarMetaValue] = {}
    if summary := meta.get("summary"):
        summary_str = " ".join(str(summary).split())
        flat["wiki_summary"] = summary_str[:500]
    if categories := meta.get("categories"):
        if isinstance(categories, list):
            flat["wiki_categories"] = ", ".join(str(c) for c in categories[:8])
        else:
            flat["wiki_categories"] = str(categories)[:300]
    if (sections := meta.get("sections")) and isinstance(sections, list):
        flat["wiki_sections"] = ", ".join(str(s) for s in sections[:10])
    if isinstance(meta.get("revision_id"), int):
        flat["wiki_revision_id"] = int(meta["revision_id"])
    return flat


def _make_wiki_source_key(title: str) -> str:
    slug = re.sub(r"[\s/\\]+", "_", title.strip())
    return f"wiki:{slug}"


class LangChainWikipediaLoader:
    """Implements WikipediaTopicLoader using LangChain's built-in WikipediaLoader."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        doc_content_chars_max: int = 8000,
        load_all_available_meta: bool = False,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._doc_content_chars_max = doc_content_chars_max
        self._load_all_available_meta = load_all_available_meta

    def load(
        self,
        query_or_url: str,
        *,
        session_id: str = "default",
        load_max_docs: int = 2,
    ) -> list[ParsedChunk]:
        """Fetch Wikipedia articles, extract URL/title metadata, split, and tag with session_id."""
        topic, lang = normalize_wikipedia_query(query_or_url)
        _configure_wikipedia_client(lang)
        loader = WikipediaLoader(
            query=topic,
            lang=lang,
            load_max_docs=load_max_docs,
            load_all_available_meta=self._load_all_available_meta,
            doc_content_chars_max=self._doc_content_chars_max,
        )
        logger.info(
            "Loading Wikipedia topic=%r lang=%s max_docs=%d session_id=%s",
            topic,
            lang,
            load_max_docs,
            session_id,
        )
        try:
            raw_docs = loader.load()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Wikipedia API (https://{lang}.wikipedia.org/w/api.php) returned an empty or "
                "non-JSON response. Verify internet/proxy access to wikipedia.org or retry shortly."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Network error contacting Wikipedia API: {exc}") from exc
        if not raw_docs:
            return []

        split_docs = self._splitter.split_documents(raw_docs)
        parsed_chunks: list[ParsedChunk] = []
        for idx, doc in enumerate(split_docs):
            content = doc.page_content.strip()
            if not content:
                continue

            title = str(doc.metadata.get("title") or topic).strip()
            wiki_url = str(doc.metadata.get("page_url") or doc.metadata.get("source") or "").strip()
            source_key = _make_wiki_source_key(title)
            extra_meta = _flatten_scalar_wiki_meta(doc.metadata)

            parsed_chunks.append(
                ParsedChunk(
                    text=content,
                    source=source_key,
                    source_type="wikipedia",
                    session_id=session_id,
                    filename="",
                    url=wiki_url,
                    title=title,
                    heading=title,
                    chunk_index=idx,
                    metadata=extra_meta,
                )
            )
        logger.info(
            "Loaded Wikipedia topic=%r articles=%d chunks=%d session_id=%s",
            topic,
            len(raw_docs),
            len(parsed_chunks),
            session_id,
        )
        return parsed_chunks
