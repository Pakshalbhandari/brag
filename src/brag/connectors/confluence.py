"""Confluence Cloud spaces: pages (as HTML) and their attachments.

Uses the REST API v2 with basic auth: set CONFLUENCE_EMAIL and
CONFLUENCE_API_TOKEN (an Atlassian API token for that account).
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from .base import TIMEOUT, ConnectorError, RemoteDoc, check, stream_to, supported


class ConfluenceConnector:
    def __init__(self, name: str, *, base_url: str, space: str, attachments: bool = True,
                 client: httpx.AsyncClient | None = None):
        self.name, self.space, self.attachments = name, space, attachments
        self.base = base_url.rstrip("/")  # https://your-site.atlassian.net/wiki
        url = urlparse(self.base)
        self.origin = f"{url.scheme}://{url.netloc}"
        if client is None:
            email, token = os.getenv("CONFLUENCE_EMAIL"), os.getenv("CONFLUENCE_API_TOKEN")
            if not (email and token):
                raise ConnectorError("Set CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN for Confluence access")
            client = httpx.AsyncClient(timeout=TIMEOUT, auth=(email, token), follow_redirects=True)
        self.client = client

    async def _pages_of(self, first_url: str, params: dict, what: str):
        url, query = first_url, params
        while url:
            page = check(await self.client.get(url, params=query), what).json()
            yield page
            nxt = (page.get("_links") or {}).get("next")
            # "next" is relative to the site root (it starts with /wiki/...).
            url, query = (urljoin(self.origin, nxt), None) if nxt else (None, None)

    async def list(self) -> list[RemoteDoc]:
        spaces = check(await self.client.get(f"{self.base}/api/v2/spaces", params={"keys": self.space}),
                       "Space lookup").json().get("results", [])
        if not spaces:
            raise ConnectorError(f"{self.name}: no space with key {self.space!r}")
        space_id = spaces[0]["id"]
        docs: list[RemoteDoc] = []
        async for page in self._pages_of(f"{self.base}/api/v2/spaces/{space_id}/pages",
                                         {"body-format": "storage", "limit": 250}, "Listing pages"):
            for p in page.get("results", []):
                title = p.get("title") or p["id"]
                body = ((p.get("body") or {}).get("storage") or {}).get("value", "")
                version = str((p.get("version") or {}).get("number", ""))
                docs.append(RemoteDoc(f"page-{p['id']}", f"{title}.html", version, extra={"title": title, "body": body}))
                if self.attachments:
                    docs += await self._attachments(p["id"], title)
        return docs

    async def _attachments(self, page_id: str, page_title: str) -> list[RemoteDoc]:
        out = []
        async for page in self._pages_of(f"{self.base}/api/v2/pages/{page_id}/attachments", {"limit": 250},
                                         "Listing attachments"):
            base = (page.get("_links") or {}).get("base") or self.base
            for a in page.get("results", []):
                title = a.get("title", "")
                if not supported(title) or not a.get("downloadLink"):
                    continue
                version = str((a.get("version") or {}).get("number", ""))
                out.append(RemoteDoc(f"attachment-{a['id']}", f"{page_title}/{title}", version, a.get("fileSize"),
                                     extra={"url": base.rstrip("/") + a["downloadLink"]}))
        return out

    async def download(self, doc: RemoteDoc, dest: Path) -> None:
        if "body" in doc.extra:
            title = html.escape(doc.extra["title"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
                f"<body><h1>{title}</h1>{doc.extra['body']}</body></html>",
                encoding="utf-8",
            )
            return
        await stream_to(self.client.stream("GET", doc.extra["url"]), dest)

    async def aclose(self) -> None:
        await self.client.aclose()
