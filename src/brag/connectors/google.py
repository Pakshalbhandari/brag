"""Google Drive folders and Google Cloud Storage buckets.

Credentials come from Application Default Credentials (`gcloud auth
application-default login`, GOOGLE_APPLICATION_CREDENTIALS pointing at a
service-account key, or the metadata server on Google Cloud), or from a
service-account key file named in the source's `credentials_file`. For Drive,
share the folder with the service account. Needs `pip install 'brag-rag[google]'`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote

import httpx

from .base import TIMEOUT, ConnectorError, RemoteDoc, check, stream_to

DRIVE = "https://www.googleapis.com/drive/v3"
STORAGE = "https://storage.googleapis.com/storage/v1"
FOLDER = "application/vnd.google-apps.folder"
# Google-native files are exported to the matching Office format (limit: 10 MB per export).
EXPORTS = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
}


class GoogleAuth:
    def __init__(self, scopes: list[str], credentials_file: str | None = None):
        self.scopes, self.credentials_file = scopes, credentials_file
        self._credentials = None

    def _refresh(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
            from google.oauth2 import service_account
        except ImportError as e:
            raise ConnectorError("Google sources need extra packages: pip install 'brag-rag[google]'") from e
        if self._credentials is None:
            if self.credentials_file:
                self._credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file, scopes=self.scopes)
            else:
                self._credentials, _ = google.auth.default(scopes=self.scopes)
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        return self._credentials.token

    async def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await asyncio.to_thread(self._refresh)}"}


class DriveConnector:
    def __init__(self, name: str, *, folder_id: str, credentials_file: str | None = None,
                 client: httpx.AsyncClient | None = None, auth=None):
        self.name, self.folder_id = name, folder_id
        self.auth = auth or GoogleAuth(["https://www.googleapis.com/auth/drive.readonly"], credentials_file)
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT)

    async def list(self) -> list[RemoteDoc]:
        docs: list[RemoteDoc] = []
        pending = [(self.folder_id, "")]
        while pending:
            folder, prefix = pending.pop()
            token = None
            while True:
                params = {
                    "q": f"'{folder}' in parents and trashed = false",
                    "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, version, md5Checksum, size)",
                    "pageSize": 1000,
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                }
                if token:
                    params["pageToken"] = token
                page = check(await self.client.get(f"{DRIVE}/files", params=params, headers=await self.auth.headers()),
                             "Listing Drive files").json()
                for f in page.get("files", []):
                    mime = f.get("mimeType", "")
                    if mime == FOLDER:
                        pending.append((f["id"], f"{prefix}{f['name']}/"))
                    elif mime in EXPORTS:
                        export_mime, ext = EXPORTS[mime]
                        docs.append(RemoteDoc(f["id"], f"{prefix}{f['name']}{ext}", f.get("version") or f.get("modifiedTime", ""),
                                              extra={"export": export_mime}))
                    elif not mime.startswith("application/vnd.google-apps."):
                        version = f.get("md5Checksum") or f.get("version") or f.get("modifiedTime", "")
                        docs.append(RemoteDoc(f["id"], f"{prefix}{f['name']}", version,
                                              int(f["size"]) if f.get("size") else None))
                token = page.get("nextPageToken")
                if not token:
                    break
        return docs

    async def download(self, doc: RemoteDoc, dest: Path) -> None:
        if "export" in doc.extra:
            url, params = f"{DRIVE}/files/{doc.id}/export", {"mimeType": doc.extra["export"]}
        else:
            url, params = f"{DRIVE}/files/{doc.id}", {"alt": "media", "supportsAllDrives": "true"}
        await stream_to(self.client.stream("GET", url, params=params, headers=await self.auth.headers()), dest)

    async def aclose(self) -> None:
        await self.client.aclose()


class GCSConnector:
    def __init__(self, name: str, *, bucket: str, prefix: str = "", credentials_file: str | None = None,
                 client: httpx.AsyncClient | None = None, auth=None):
        self.name, self.bucket, self.prefix = name, bucket, prefix.lstrip("/")
        self.auth = auth or GoogleAuth(["https://www.googleapis.com/auth/devstorage.read_only"], credentials_file)
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT)

    async def list(self) -> list[RemoteDoc]:
        docs, token = [], None
        while True:
            params = {"prefix": self.prefix, "maxResults": 1000}
            if token:
                params["pageToken"] = token
            page = check(await self.client.get(f"{STORAGE}/b/{quote(self.bucket, safe='')}/o", params=params,
                                               headers=await self.auth.headers()), "Listing bucket").json()
            for obj in page.get("items", []):
                name = obj["name"]
                if name.endswith("/"):  # folder placeholder
                    continue
                path = name[len(self.prefix):].lstrip("/") if self.prefix and name.startswith(self.prefix) else name
                docs.append(RemoteDoc(name, path, str(obj.get("generation") or obj.get("md5Hash") or obj.get("updated")),
                                      int(obj["size"]) if obj.get("size") else None))
            token = page.get("nextPageToken")
            if not token:
                return docs

    async def download(self, doc: RemoteDoc, dest: Path) -> None:
        # The whole object name is one path segment: "/" must be encoded too.
        url = f"{STORAGE}/b/{quote(self.bucket, safe='')}/o/{quote(doc.id, safe='')}"
        await stream_to(self.client.stream("GET", url, params={"alt": "media"}, headers=await self.auth.headers()), dest)

    async def aclose(self) -> None:
        await self.client.aclose()
