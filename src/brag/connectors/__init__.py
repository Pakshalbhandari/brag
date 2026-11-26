"""Document sources: SharePoint/OneDrive, Google Drive, Google Cloud Storage, Confluence.

sources.yaml:

    sources:
      - name: finance-sp                 # used in citation labels: finance-sp/Reports/q3.docx
        type: sharepoint
        site_url: https://contoso.sharepoint.com/sites/Finance
        drive: Documents                 # library name (default "Documents")
        folder: Reports                  # optional
        index: finance                   # optional: the service index (workspace) to sync into
      - name: ceo-onedrive
        type: onedrive
        user: ceo@contoso.com
      - name: hr-drive
        type: gdrive
        folder_id: 1AbCdEf...
      - name: eng-wiki
        type: confluence
        base_url: https://acme.atlassian.net/wiki
        space: ENG
        attachments: true
      - name: archive
        type: gcs
        bucket: acme-docs
        prefix: reports/
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import Connector, ConnectorError, RemoteDoc  # noqa: F401
from .sync import SyncResult, mirror_paths, sync_source  # noqa: F401

SOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TYPES = {
    "sharepoint": {"site_url", "drive", "folder"},
    "onedrive": {"user", "drive", "folder"},
    "gdrive": {"folder_id", "credentials_file"},
    "gcs": {"bucket", "prefix", "credentials_file"},
    "confluence": {"base_url", "space", "attachments"},
}
REQUIRED = {"sharepoint": {"site_url"}, "onedrive": {"user"}, "gdrive": {"folder_id"},
            "gcs": {"bucket"}, "confluence": {"base_url", "space"}}


@dataclass
class SourceConfig:
    name: str
    type: str
    index: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


def load_sources(path: str | Path) -> list[SourceConfig]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sources, names = [], set()
    for raw in data.get("sources") or []:
        raw = dict(raw)
        name, kind, index = str(raw.pop("name", "")), str(raw.pop("type", "")), raw.pop("index", None)
        if not SOURCE_NAME.match(name):
            raise ValueError(f"Source name {name!r}: letters, digits, '.', '-' and '_' (max 64)")
        if name in names:
            raise ValueError(f"Duplicate source name {name!r}")
        if kind not in TYPES:
            raise ValueError(f"Source {name!r}: type must be one of {', '.join(TYPES)}")
        unknown = set(raw) - TYPES[kind]
        if unknown:
            raise ValueError(f"Source {name!r}: unknown option(s) {', '.join(sorted(unknown))}")
        missing = REQUIRED[kind] - set(raw)
        if missing:
            raise ValueError(f"Source {name!r}: missing {', '.join(sorted(missing))}")
        names.add(name)
        sources.append(SourceConfig(name, kind, str(index) if index else None, raw))
    if not sources:
        raise ValueError(f"{path} defines no sources")
    return sources


def make_connector(source: SourceConfig) -> Connector:
    o = source.options
    if source.type in ("sharepoint", "onedrive"):
        from .microsoft import GraphConnector

        return GraphConnector(source.name, site_url=o.get("site_url"), user=o.get("user"),
                              drive=o.get("drive"), folder=str(o.get("folder") or ""))
    if source.type == "gdrive":
        from .google import DriveConnector

        return DriveConnector(source.name, folder_id=str(o["folder_id"]), credentials_file=o.get("credentials_file"))
    if source.type == "gcs":
        from .google import GCSConnector

        return GCSConnector(source.name, bucket=str(o["bucket"]), prefix=str(o.get("prefix") or ""),
                            credentials_file=o.get("credentials_file"))
    from .confluence import ConfluenceConnector

    return ConfluenceConnector(source.name, base_url=str(o["base_url"]), space=str(o["space"]),
                               attachments=bool(o.get("attachments", True)))


async def sync(pipeline, source: SourceConfig) -> SyncResult:
    connector = make_connector(source)
    try:
        mirror, state = mirror_paths(pipeline.settings, source.name)
        return await sync_source(pipeline, connector, mirror, state)
    finally:
        await connector.aclose()
