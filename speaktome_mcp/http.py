"""Internal REST client for the host's OAuth protected Streamable HTTP endpoint.

Only the host constructs this object. It carries an OAuth access token to the
same host's loopback REST listener and preserves the original request's access
tier, so a loopback hop cannot turn a remote caller into a LAN operator.
"""
from __future__ import annotations

import urllib.request

from .rest import Rest, RestError, _NoRedirect


class BearerRest(Rest):
    def __init__(self, base: str, token: str, origin: str, tier: str):
        opener = urllib.request.build_opener(_NoRedirect())
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
        if not path.startswith("/api/"):
            raise RestError("only the host API may be called", 400)
        sent = dict(headers or {})
        sent.update({"Authorization": "Bearer " + self.token,
                     "X-Speaktome-MCP-Origin": self.origin,
                     "X-Speaktome-MCP-Tier": self.tier})
        return super()._http(path, body, timeout, secret=self.token, raw_limit=raw_limit, method=method, headers=sent)
