"""Internal REST client for the host's OAuth protected Streamable HTTP endpoint.

Only the host constructs this object. It carries an OAuth access token to the
same host's loopback REST listener and preserves the original request's access
tier, so a loopback hop cannot turn a remote caller into a LAN operator.
"""
from __future__ import annotations

import re
import ssl
from http.cookiejar import CookieJar
import urllib.parse
import urllib.request

from .rest import Rest, RestError, _NoRedirect


_SITE_PAGE = re.compile(r"/site/[a-z0-9][a-z0-9-]{0,31}/(?:[A-Za-z0-9._-]{1,128}\.html)?\Z")


def _host_path_allowed(path, body, method):
    # The host's declared map pages are read-only MCP tools outside /api/.
    return path.startswith("/api/") or (
        body is None and method in (None, "GET") and _SITE_PAGE.fullmatch(path) is not None
    )


class BearerRest(Rest):
    def __init__(self, base: str, token: str, origin: str, tier: str, *, ca_file: str):
        parsed = urllib.parse.urlsplit(base)
        if parsed.scheme != "https" or parsed.hostname != "127.0.0.1":
            raise RestError("internal MCP REST requires loopback HTTPS")
        opener = urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca_file)),
            urllib.request.HTTPCookieProcessor(CookieJar()))
        super().__init__(base, "oauth", None, opener=opener)
        self.token = token
        self.origin = origin
        self.tier = tier
        self.signed_in = True

    def sign_in(self) -> None:
        raise RestError("OAuth token expired", 401)

    def _scrub(self, message: str, secret: str | None) -> str:
        return message.replace(self.token, "[credential]")

    def _http(self, path, body=None, timeout=40, secret=None, raw_limit=None, method=None, headers=None):
        if not _host_path_allowed(path, body, method):
            raise RestError("only the host API may be called", 400)
        sent = dict(headers or {})
        sent.update({"Authorization": "Bearer " + self.token,
                     "X-Speaktome-MCP-Origin": self.origin,
                     "X-Speaktome-MCP-Tier": self.tier})
        return super()._http(path, body, timeout, secret=self.token, raw_limit=raw_limit, method=method, headers=sent)


class PreviewRest(BearerRest):
    """A server-created seven-day preview; its token never reaches an MCP client."""

    def _http(self, path, body=None, timeout=40, secret=None, raw_limit=None, method=None, headers=None):
        if not _host_path_allowed(path, body, method):
            raise RestError("only the host API may be called", 400)
        sent = dict(headers or {})
        sent.update({"Authorization": "Preview " + self.token,
                     "X-Speaktome-MCP-Origin": self.origin,
                     "X-Speaktome-MCP-Tier": self.tier})
        return Rest._http(self, path, body, timeout, secret=self.token, raw_limit=raw_limit, method=method, headers=sent)
