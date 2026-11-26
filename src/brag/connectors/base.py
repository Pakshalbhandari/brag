"""What every connector provides, and helpers they share."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..config import SUPPORTED_EXTENSIONS

TIMEOUT = httpx.Timeout(60.0, connect=15.0)


class ConnectorError(Exception):
    pass


@dataclass
class RemoteDoc:
    id: str  # stable ID in the source system
    path: str  # source-relative path with "/" separators; becomes part of the citation label
    version: str  # changes whenever the content changes (eTag, generation, version number, ...)
    size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # connector-private data for download()


class Connector(Protocol):
    name: str

    async def list(self) -> list[RemoteDoc]: ...

    async def download(self, doc: RemoteDoc, dest: Path) -> None: ...

    async def aclose(self) -> None: ...


def supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS and not name.startswith(("~$", "."))


_UNSAFE = re.compile(r'[<>:"|?*\x00-\x1f]')


def clean_path(path: str) -> str:
    """A safe relative path: no absolute paths, no '..', no characters Windows rejects."""
    parts = []
    for part in posixpath.normpath("/" + path.replace("\\", "/")).split("/"):
        part = _UNSAFE.sub("_", part).strip().rstrip(".")
        if part and part not in (".", ".."):
            parts.append(part[:150])
    if not parts:
        raise ConnectorError(f"Unusable document path: {path!r}")
    return "/".join(parts)


def check(response: httpx.Response, what: str) -> httpx.Response:
    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        raise ConnectorError(f"{what} failed: HTTP {response.status_code} {detail}")
    return response


async def stream_to(response_cm, dest: Path) -> None:
    """Write a streamed httpx response to dest through a temporary file."""
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with response_cm as response:
            check(response, f"Download of {dest.name}")
            with tmp.open("wb") as f:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    f.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
