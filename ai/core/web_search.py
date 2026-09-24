"""Looking something up on the web, when he has allowed it.

North star rule 6 keeps the default path local, so this is off unless
ALICE_ENABLE_WEB_SEARCH=1. With it on, the model can search when a question
depends on what is true today (a score, a price, the news) instead of answering
from memory that may be months old. No account or API key is needed.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

_ENDPOINT = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
}
_TITLE_RE = re.compile(r'<a[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_TITLE_RE_HREF_FIRST = re.compile(
    r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>(.*?)</a>', re.S | re.I
)
_SNIPPET_RE = re.compile(r'class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</(?:a|div|td)>', re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def web_search_enabled() -> bool:
    return str(os.getenv("ALICE_ENABLE_WEB_SEARCH", "")).strip().lower() in {"1", "true", "yes", "on"}


def _text(fragment: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", fragment or "")).split())


def _target(href: str) -> str:
    """The result's own address; DuckDuckGo wraps each one in a redirect."""
    link = html.unescape(href or "")
    if link.startswith("//"):
        link = "https:" + link
    wrapped = parse_qs(urlparse(link).query).get("uddg")
    return wrapped[0] if wrapped else link


def parse_results(page: str, limit: int = 5) -> List[Dict[str, str]]:
    titles = sorted(
        list(_TITLE_RE.finditer(page or "")) + list(_TITLE_RE_HREF_FIRST.finditer(page or "")),
        key=lambda m: m.start(),
    )
    seen: set = set()
    results: List[Dict[str, str]] = []
    for index, match in enumerate(titles):
        if match.start() in seen:
            continue
        seen.add(match.start())
        end = titles[index + 1].start() if index + 1 < len(titles) else len(page)
        snippet = _SNIPPET_RE.search(page, match.end(), end)
        result = {
            "title": _text(match.group(2)),
            "url": _target(match.group(1)),
            "snippet": _text(snippet.group(1)) if snippet else "",
        }
        if result["title"] and result["url"].startswith("http"):
            results.append(result)
        if len(results) >= limit:
            break
    return results


def search_web(query: str, limit: int = 5, timeout: float = 8.0) -> Dict[str, Any]:
    """Titles, snippets and links for ``query``, or why there are none."""
    if not web_search_enabled():
        return {"success": False, "error": "Web search is off. Set ALICE_ENABLE_WEB_SEARCH=1 to allow it."}
    query = str(query or "").strip()
    if not query:
        return {"success": False, "error": "Nothing to search for."}
    try:
        import requests

        response = requests.post(_ENDPOINT, data={"q": query}, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        return {"success": False, "error": f"Couldn't reach the web: {exc}"}
    results = parse_results(response.text, limit)
    content = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results)
    return {"success": True, "query": query, "results": results, "content": content}
