"""LangExtract-powered documentation crawler for generating JSON/JSONL corpora.

This module stitches together the existing :class:`WebScraper` with the
`langextract` project (https://github.com/kennethleungty/LangExtract-Gemma-Structured-Extraction)
to convert raw HTML documentation pages into structured chunks that can be used
in the QLoRA and RAG pipelines.

The implementation prefers the langextract library when it is available but
gracefully falls back to lightweight BeautifulSoup parsing so unit tests remain
hermetic even when the optional dependency is missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import jsonlines

from .web_scraper import WebScraper, BeautifulSoup

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from langextract import LangExtract  # type: ignore

    _LANGEXTRACT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LangExtract = None  # type: ignore
    _LANGEXTRACT_AVAILABLE = False


@dataclass
class TopicConfig:
    """Configuration for a single documentation topic crawl."""

    topic: str
    urls: List[str]
    profile: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


DEFAULT_TOPIC_CONFIGS: List[TopicConfig] = [
    TopicConfig(
        topic="TypeScript",
        profile="language",
        urls=[
            "https://www.typescriptlang.org/docs/handbook/intro.html",
            "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
        ],
    ),
    TopicConfig(
        topic="Drizzle ORM",
        profile="orm",
        urls=[
            "https://orm.drizzle.team/docs/overview",
            "https://orm.drizzle.team/docs/sql-schema-declaration",
        ],
    ),
    TopicConfig(
        topic="UnoCSS",
        profile="styling",
        urls=[
            "https://unocss.dev/guide/",
            "https://unocss.dev/config/",
        ],
    ),
]


class LangExtractAdapter:
    """Thin wrapper around the optional langextract dependency."""

    def __init__(self, model_name: str = "embeddinggemma:latest"):
        self.model_name = model_name
        self._extractor = None

        if _LANGEXTRACT_AVAILABLE:
            try:  # pragma: no cover - requires external dependency
                self._extractor = LangExtract(model_name=model_name)  # type: ignore
                logger.info("LangExtract initialised with model %s", model_name)
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning(
                    "Failed to initialise LangExtract (%s). Falling back to heuristic parser.",
                    exc,
                )
                self._extractor = None
        else:
            logger.info("LangExtract library not available. Using heuristic fallback parser.")

    def extract(self, html: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured records from HTML."""

        if self._extractor is not None:
            for candidate in ("extract", "__call__", "run"):
                method = getattr(self._extractor, candidate, None)
                if callable(method):  # pragma: no branch
                    try:  # pragma: no cover - depends on optional dependency
                        result = method(html, metadata=metadata)
                        return self._normalise_result(result, metadata)
                    except TypeError:
                        # Some versions expect only HTML without metadata
                        try:
                            result = method(html)
                            return self._normalise_result(result, metadata)
                        except Exception as exc:  # pragma: no cover
                            logger.warning("LangExtract call failed: %s", exc)
                            break
                    except Exception as exc:  # pragma: no cover
                        logger.warning("LangExtract extraction error: %s", exc)
                        break

        # Fallback heuristic: split by paragraphs and headings
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title else metadata.get("title", "")
        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all(["p", "li", "code", "pre"])
            if p.get_text(strip=True)
        ]
        if not paragraphs:
            paragraphs = [soup.get_text(strip=True)]

        records = []
        for idx, paragraph in enumerate(paragraphs):
            records.append(
                {
                    "chunk_id": f"{metadata.get('topic', 'doc')}-{idx}",
                    "title": title,
                    "content": paragraph,
                    "metadata": metadata,
                }
            )

        return records

    def _normalise_result(
        self, result: Any, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:  # pragma: no cover - depends on optional dependency
        """Normalise langextract outputs into a list of dictionaries."""

        if result is None:
            return []
        if isinstance(result, dict):
            result = [result]
        normalised: List[Dict[str, Any]] = []
        for idx, item in enumerate(result):
            if isinstance(item, str):
                payload = {"content": item}
            elif isinstance(item, dict):
                payload = item
            else:
                payload = {"content": str(item)}

            payload.setdefault("chunk_id", f"{metadata.get('topic', 'doc')}-{idx}")
            payload.setdefault("metadata", metadata)
            normalised.append(payload)

        return normalised


class DocumentationCrawler:
    """Agentic workflow for crawling and extracting documentation corpora."""

    def __init__(
        self,
        scraper: Optional[WebScraper] = None,
        extractor: Optional[LangExtractAdapter] = None,
    ) -> None:
        self.scraper = scraper or WebScraper()
        self.extractor = extractor or LangExtractAdapter()

    def crawl_topics(
        self,
        topics: Iterable[TopicConfig],
        output_dir: Path,
        json_name: str = "documentation_corpus.json",
        jsonl_name: str = "documentation_corpus.jsonl",
    ) -> List[Dict[str, Any]]:
        """Crawl and extract documentation for the provided topics."""

        output_dir.mkdir(parents=True, exist_ok=True)
        aggregate: List[Dict[str, Any]] = []

        for topic in topics:
            logger.info("Crawling topic '%s'", topic.topic)
            scraped = self.scraper.scrape_urls(topic.urls)
            for page in scraped:
                html = page.get("raw_html") or ""
                metadata = {
                    "topic": topic.topic,
                    "profile": topic.profile,
                    "url": page.get("resolved_url", page.get("url")),
                    "title": page.get("title", ""),
                }
                metadata.update(topic.metadata)

                extracted = self.extractor.extract(html, metadata)
                for idx, chunk in enumerate(extracted):
                    content = chunk.get("content") or chunk.get("text") or ""
                    if not content:
                        continue

                    record = {
                        "chunk_id": chunk.get("chunk_id")
                        or f"{topic.topic}-{idx}",
                        "topic": topic.topic,
                        "title": chunk.get("title")
                        or page.get("title")
                        or metadata.get("title"),
                        "url": metadata.get("url"),
                        "profile": topic.profile,
                        "content": content,
                        "metadata": {**metadata, **chunk.get("metadata", {})},
                    }
                    aggregate.append(record)

        json_path = output_dir / json_name
        jsonl_path = output_dir / jsonl_name
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, ensure_ascii=False)

        with jsonlines.open(jsonl_path, "w") as writer:
            for record in aggregate:
                writer.write(record)

        logger.info("Wrote %s records to %s", len(aggregate), jsonl_path)
        return aggregate


def crawl_default_corpus(output_dir: Path) -> List[Dict[str, Any]]:
    """Utility wrapper for generating the default documentation corpus."""

    crawler = DocumentationCrawler()
    return crawler.crawl_topics(DEFAULT_TOPIC_CONFIGS, output_dir)

