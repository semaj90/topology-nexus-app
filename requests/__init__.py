"""Minimal requests compatibility layer for offline test environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError as UrlLibHTTPError
from urllib.parse import urlencode, urlparse


class HTTPError(RuntimeError):
    pass


@dataclass
class Response:
    status_code: int
    url: str
    _content: bytes

    @property
    def text(self) -> str:
        return self._content.decode("utf-8", errors="replace")

    @property
    def content(self) -> bytes:
        return self._content

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code} for {self.url}")


class Session:
    def __init__(self) -> None:
        self.headers: Dict[str, str] = {}

    def update_headers(self, headers: Dict[str, str]) -> None:
        self.headers.update(headers)

    def get(self, url: str, timeout: Optional[int] = None) -> Response:
        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            path = Path(parsed.path if parsed.scheme else url)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            data = path.read_bytes()
            return Response(status_code=200, url=str(path), _content=data)

        req = urlrequest.Request(url, headers=self.headers)
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
                data = resp.read()
                return Response(status_code=resp.getcode() or 200, url=url, _content=data)
        except UrlLibHTTPError as exc:
            raise HTTPError(str(exc)) from exc

    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Response:
        body: Optional[bytes] = None
        headers = dict(self.headers)
        if json is not None:
            body = json_dumps(json).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        elif data is not None:
            body = urlencode(data).encode("utf-8")
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

        req = urlrequest.Request(url, data=body, headers=headers)
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
                content = resp.read()
                return Response(status_code=resp.getcode() or 200, url=url, _content=content)
        except UrlLibHTTPError as exc:
            raise HTTPError(str(exc)) from exc


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def get(url: str, timeout: Optional[int] = None) -> Response:
    return Session().get(url, timeout=timeout)


def post(
    url: str,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
) -> Response:
    return Session().post(url, json=json, data=data, timeout=timeout)


__all__ = ["Session", "get", "post", "Response", "HTTPError"]
