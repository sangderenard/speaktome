"""One signed-in HTTP session against the existing Meowdoku REST service (standard library only).

With no account named (SPEAKTOME_USER unset) it is a VISITOR's session instead: it never signs in, and the service answers
it as a guest, the foyer of ``man foyer`` (owner, 2026-11-13). Holds an in-memory cookie jar and nothing else. The credential is read from its file only at sign-in, is
never stored on the object, and is scrubbed from any text this module returns.
"""
from __future__ import annotations

import http.client
import http.cookiejar
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request


class RestError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message, self.status = message, status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RestError("API redirect refused; use the verified HTTPS origin directly", code)


def secret_path(user: str, env=None) -> str:
    env = os.environ if env is None else env
    if env.get("SPEAKTOME_SECRET_FILE"):
        return env["SPEAKTOME_SECRET_FILE"]
    state = env.get("SPEAKTOME_STATE_DIR")
    if not state:
        raise RestError("set SPEAKTOME_SECRET_FILE, or SPEAKTOME_STATE_DIR (secrets/operator_credentials/<user>.secret)")
    return os.path.join(state, "secrets", "operator_credentials", f"{user}.secret")


class Rest:
    """``opener`` is injectable (an object with ``open(request, timeout=)``); with one injected the https check is skipped."""

    def __init__(self, base: str, user: str | None, secret_file: str | None, ca_file: str | None = None, opener=None):
        self.base = base.rstrip("/")
        self.user, self.secret_file = user, secret_file
        self.visitor = user is None                # no account: the foyer; nothing is signed in, ever
        if opener is None:
            parsed = urllib.parse.urlsplit(self.base)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise RestError("SPEAKTOME_BASE must be an HTTPS origin without credentials, query or fragment")
            context = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
                                                 _NoRedirect(), urllib.request.HTTPSHandler(context=context))
        self.opener = opener
        self.signed_in = False
        self.logins = 0

    @classmethod
    def from_env(cls, env=None) -> "Rest":
        env = os.environ if env is None else env
        base, user = env.get("SPEAKTOME_BASE"), env.get("SPEAKTOME_USER")
        if not base:
            raise RestError("SPEAKTOME_BASE is required")
        if not user:                               # a visitor: no account, no secret
            return cls(base, None, None, env.get("SPEAKTOME_CA_FILE") or None)
        return cls(base, user, secret_path(user, env), env.get("SPEAKTOME_CA_FILE") or None)

    def _read_secret(self) -> str:
        try:
            with open(self.secret_file, encoding="utf-8") as handle:
                secret = handle.read().strip()
        except OSError as error:
            raise RestError(f"cannot read the secret file: {error.strerror}") from None
        if not secret:
            raise RestError("the secret file is empty")
        return secret

    def _http(self, path: str, body=None, timeout: float = 40, secret: str | None = None, raw_limit: int | None = None,
              method: str | None = None, headers: dict | None = None):
        if method == "POST" and body is None:
            body = {}                          # a POST always carries a JSON object, as every handler here expects
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        sent = dict(headers or {})
        if body is not None:
            sent["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST", headers=sent)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                if raw_limit is not None:      # a file, not JSON: the bytes and their declared type, bounded
                    raw = response.read(raw_limit + 1)
                    if len(raw) > raw_limit:
                        raise RestError(f"the file is larger than {raw_limit} bytes", 413)
                    return raw, response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0].strip()
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("error", "")
            except (ValueError, AttributeError, http.client.HTTPException, OSError):
                message = ""
            raise RestError(self._scrub(str(message) or f"HTTP {error.code}", secret), error.code) from None
        except RestError:
            raise
        except (ValueError, UnicodeError, http.client.HTTPException):
            raise RestError("invalid or incomplete response; the outcome may be uncertain") from None
        except (urllib.error.URLError, OSError) as error:
            raise RestError(f"cannot reach the server: {self._scrub(str(getattr(error, 'reason', error)), secret)}") from None
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeError):
            raise RestError("invalid JSON response; the outcome may be uncertain") from None
        return parsed

    @staticmethod
    def _scrub(text: str, secret: str | None) -> str:
        return text.replace(secret, "[credential]") if secret else text

    def sign_in(self) -> None:
        if self.visitor:
            raise RestError("a visitor has no account to sign in; an admin makes an intern account (man foyer)", 401)
        secret = self._read_secret()
        try:
            self._http("/api/account/login", {"username": self.user, "password": secret}, secret=secret)
        except RestError as error:
            raise RestError(f"sign-in failed: {error.message}", error.status) from None
        finally:
            secret = None
        self.signed_in = True
        self.logins += 1

    @staticmethod
    def _unsigned(error: RestError) -> bool:
        text = error.message.lower()
        return error.status == 401 or (error.status == 403 and ("sign in" in text or "signed in" in text))

    def call(self, path: str, body=None, timeout: float = 40, raw_limit: int | None = None, *,
             method: str | None = None, headers: dict | None = None):
        """One request as the signed-in account. With ``raw_limit`` the answer is ``(bytes, content type)``, not JSON.

        ``method="POST"`` sends an empty JSON object when there is no body; ``headers`` are fixed request headers a
        handler insists on. Neither can carry a credential: the session cookie is the only one this object holds.
        """
        if self.visitor:
            return self._http(path, body, timeout, raw_limit=raw_limit, method=method, headers=headers)
        if not self.signed_in:
            self.sign_in()
        try:
            return self._http(path, body, timeout, raw_limit=raw_limit, method=method, headers=headers)
        except RestError as error:
            if not self._unsigned(error):
                raise
        self.signed_in = False
        self.sign_in()
        return self._http(path, body, timeout, raw_limit=raw_limit, method=method, headers=headers)
