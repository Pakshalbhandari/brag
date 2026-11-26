"""SharePoint document libraries and OneDrive, through Microsoft Graph.

App-only access: register an app in Entra ID with the application permissions
Sites.Read.All (site lookup) and Files.Read.All (files), grant admin consent,
and set MS_TENANT_ID, MS_CLIENT_ID and MS_CLIENT_SECRET.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx

from .base import TIMEOUT, ConnectorError, RemoteDoc, check, stream_to

GRAPH = "https://graph.microsoft.com/v1.0"


class GraphConnector:
    def __init__(self, name: str, *, site_url: str | None = None, user: str | None = None,
                 drive: str | None = None, folder: str = "", client: httpx.AsyncClient | None = None):
        if bool(site_url) == bool(user):
            raise ConnectorError(f"{name}: set either site_url (SharePoint) or user (OneDrive)")
        self.name, self.site_url, self.user, self.drive_name = name, site_url, user, drive
        self.folder = folder.strip("/")
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT)
        self._token: tuple[str, float] | None = None
        self._drive_id: str | None = None

    async def _auth(self) -> dict[str, str]:
        if self._token is None or self._token[1] < time.time() + 60:
            tenant = _env("MS_TENANT_ID")
            r = check(await self.client.post(
                f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                data={
                    "client_id": _env("MS_CLIENT_ID"),
                    "client_secret": _env("MS_CLIENT_SECRET"),
                    "scope": "https://graph.microsoft.com/.default",
                    "grant_type": "client_credentials",
                },
            ), "Microsoft sign-in")
            body = r.json()
            self._token = (body["access_token"], time.time() + int(body.get("expires_in", 3600)))
        return {"Authorization": f"Bearer {self._token[0]}"}

    async def _get(self, url: str, what: str) -> dict:
        return check(await self.client.get(url, headers=await self._auth()), what).json()

    async def _drive(self) -> str:
        if self._drive_id:
            return self._drive_id
        if self.user:
            drives = (await self._get(f"{GRAPH}/users/{quote(self.user)}/drives", "OneDrive lookup"))["value"]
        else:
            url = urlparse(self.site_url)
            site = await self._get(f"{GRAPH}/sites/{url.hostname}:{url.path.rstrip('/') or '/'}", "SharePoint site lookup")
            drives = (await self._get(f"{GRAPH}/sites/{site['id']}/drives", "Document library lookup"))["value"]
        if not drives:
            raise ConnectorError(f"{self.name}: no document libraries found")
        wanted = (self.drive_name or "Documents").lower()
        match = next((d for d in drives if d.get("name", "").lower() == wanted), None)
        if match is None and self.drive_name:
            names = ", ".join(d.get("name", "?") for d in drives)
            raise ConnectorError(f"{self.name}: no library named {self.drive_name!r} (found: {names})")
        self._drive_id = (match or drives[0])["id"]
        return self._drive_id

    async def list(self) -> list[RemoteDoc]:
        drive = await self._drive()
        start = (
            f"{GRAPH}/drives/{drive}/root:/{quote(self.folder)}:/children" if self.folder
            else f"{GRAPH}/drives/{drive}/root/children"
        )
        docs: list[RemoteDoc] = []
        pending = [(start, "")]
        while pending:
            url, prefix = pending.pop()
            while url:
                page = await self._get(url, "Listing files")
                for item in page.get("value", []):
                    path = f"{prefix}{item['name']}"
                    if "folder" in item:
                        pending.append((f"{GRAPH}/drives/{drive}/items/{item['id']}/children", path + "/"))
                    elif "file" in item:
                        version = item.get("cTag") or item.get("eTag") or item.get("lastModifiedDateTime", "")
                        docs.append(RemoteDoc(item["id"], path, version, item.get("size")))
                url = page.get("@odata.nextLink")
        return docs

    async def download(self, doc: RemoteDoc, dest: Path) -> None:
        drive = await self._drive()
        r = await self.client.get(f"{GRAPH}/drives/{drive}/items/{doc.id}/content", headers=await self._auth())
        if r.status_code in (301, 302, 303, 307, 308):
            # A pre-authenticated download URL: it must not receive our token.
            await stream_to(self.client.stream("GET", r.headers["Location"]), dest)
            return
        check(r, f"Download of {doc.path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)

    async def aclose(self) -> None:
        await self.client.aclose()


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConnectorError(f"Set {name} for SharePoint/OneDrive access")
    return value
