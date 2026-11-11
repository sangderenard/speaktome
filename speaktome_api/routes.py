#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Shared REST handlers extracted from Turing's existing Meowdoku service.

The host supplies its authenticated request handler and existing services.
There is no listener, account store, world registry or simulation in this module.
"""
from __future__ import annotations

from http import HTTPStatus
# --- END HEADER ---

API_VERSION = 3   # 3: bounded PNG messages and per-chat image consent; retains picture/site bindings from 2.


def messages_route(self, path: str, method: str, *, messaging, accounts_db) -> None:
    from urllib.parse import parse_qs
    admin = path.startswith("/api/messages/admin/")
    if not self._require("account_manage" if admin else "message"):
        return
    self._set_cookie = []
    me = self._principal().user_id
    query = {k: v[0] for k, v in parse_qs(self.path.partition("?")[2]).items()}
    body = {}
    if method == "POST":
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self.close_connection = True
            return self._json({"error": "JSON only"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        body = self._body_json(messaging.MAX_IMAGE_REQUEST if path == "/api/messages/send" else 16384)
        if body is None:
            return
    db = messaging.connect(accounts_db)
    try:
        try:
            if method == "GET" and path == "/api/messages/conversations":
                if not self._is_secure():
                    return self._json({"error": "HTTPS required"}, HTTPStatus.FORBIDDEN)
                return self._json(messaging.conversations(db, me))
            if method == "GET" and path == "/api/messages/contacts":
                return self._json(messaging.contacts(db, me))
            if path in ("/api/messages/images", "/api/messages/image"):
                if not self._is_secure():
                    return self._json({"error": "HTTPS required"}, HTTPStatus.FORBIDDEN)
                if method == "GET" and path == "/api/messages/images":
                    return self._json(messaging.image_status(db, me, query.get("peer")))
                if method == "POST" and path == "/api/messages/images":
                    if set(body) != {"conversation_handle", "enabled"}:
                        return self._json({"error": "provide conversation_handle and enabled only"}, HTTPStatus.BAD_REQUEST)
                    return self._json(messaging.image_consent(db, me, body["conversation_handle"], body["enabled"]))
                if method == "GET" and path == "/api/messages/image":
                    raw = messaging.image_bytes(db, me, query.get("id"))
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "private, no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Security-Policy", "default-src 'none'; sandbox")
                    self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
            if method == "GET" and path == "/api/messages/thread":
                before = query.get("before")
                return self._json(messaging.thread(db, me, query.get("peer"), int(before) if before else None,
                                                       subject_kind=query.get("subject_kind"), subject_ref=query.get("subject_ref")))
            if method == "GET" and path == "/api/messages/since":
                rows = messaging.long_poll(db, me, int(query.get("after") or 0), float(query.get("wait") or 0),
                                               subject_kind=query.get("subject_kind"), subject_ref=query.get("subject_ref"))
                return self._json({"messages": rows, "latest": messaging.latest_id(db, me)})
            if method == "GET" and path == "/api/messages/inbox":
                return self._json(messaging.inbox(db, me))
            if method == "GET" and path == "/api/messages/admin/thread":
                before = query.get("before")
                return self._json(messaging.admin_thread(db, me, query.get("a"), query.get("b"), int(before) if before else None))
            if method == "POST" and path == "/api/messages/contacts":
                action, name = body.get("action"), body.get("username")
                if action == "request":
                    return self._json(messaging.request_contact(db, me, name))
                if action in ("accept", "decline"):
                    return self._json(messaging.respond_contact(db, me, name, action == "accept"))
                if action == "block":
                    return self._json(messaging.block(db, me, name))
                if action == "remove":
                    return self._json(messaging.remove(db, me, name))
                return self._json({"error": "action must be request, accept, decline, block or remove"}, HTTPStatus.BAD_REQUEST)
            if method == "POST" and path == "/api/messages/send":
                if "conversation_handle" in body:
                    if not self._is_secure():
                        return self._json({"error": "HTTPS required"}, HTTPStatus.FORBIDDEN)
                    if body.get("conversation_handle") is None or "to" in body:
                        return self._json({"error": "provide a conversation_handle and no recipient name"}, HTTPStatus.BAD_REQUEST)
                return self._json(messaging.send(db, me, body.get("to"), body.get("body"), subject=body.get("subject"),
                                                    msg_type=body.get("type"), conversation_handle=body.get("conversation_handle"),
                                                    idempotency_key=body.get("idempotency_key"), image=body.get("image")))
            if method == "POST" and path == "/api/messages/ack":
                return self._json(messaging.ack(db, me, body.get("up_to")))
            if method == "POST" and path == "/api/messages/read":
                return self._json(messaging.mark_read(db, me, body.get("peer"), int(body.get("up_to") or 0)))
            if method == "POST" and path == "/api/messages/admin/remove":
                return self._json(messaging.admin_remove(db, me, int(body.get("id"))))
            if method == "POST" and path == "/api/messages/admin/suspend":
                return self._json(messaging.admin_suspend(db, me, body.get("username"), bool(body.get("suspended"))))
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except messaging.MessagingError as error:
            return self._json({"error": error.message}, error.status)
        except (ValueError, TypeError):
            return self._json({"error": "bad request"}, HTTPStatus.BAD_REQUEST)
    finally:
        db.close()


def living_data_dm_route(self, path, method, *, service, CursorError, QueryError, MAX_BODY):
    import re
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    if self._is_viewer():
        return self._refuse_viewer()
    if not self._require("dev_ui"):
        return
    account = self._account()
    if account is None:
        return self._json({"error": "sign in to use static DM commands"}, 401)
    try:
        base = "/api/living-data/dm/worlds"
        match = re.fullmatch(base + r"/([0-9a-f]{32})(?:/(commands|host|close)(?:/([A-Za-z0-9_-]{1,64}))?)?", path)
        if self.path.partition("?")[2]:
            raise CursorError("DM routes accept no query parameters")
        if method == "GET":
            if match and not match[2]:
                return self._json(service.status(account["id"], match[1]))
            if match and match[2] == "commands" and match[3]:
                return self._json(service.receipt(account["id"], match[1], match[3]))
            return self._json({"error": "not found"}, 404)
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self._json({"error": "application/json required"}, 415)
        body = self._body_json(MAX_BODY if path == base or (match and match[2] == "host") else 4096)
        if body is None:
            return
        if path == base:
            return self._json(service.register(account["id"], body), 201)
        if not match or match[3]:
            return self._json({"error": "not found"}, 404)
        if match[2] == "commands":
            result = service.command(account["id"], match[1], body)
            return self._json(result, 202 if result["state"] in ("queued", "rendering") else 200)
        if match[2] == "host":
            return self._json(service.host(account["id"], match[1], body))
        if match[2] == "close" and body == {}:
            return self._json(service.close(account["id"], match[1]))
        return self._json({"error": "not found"}, 404)
    except CursorError as error:
        return self._json({"error": str(error)}, error.status)
    except (QueryError, ValueError, RecursionError):
        return self._json({"error": "invalid or excessive static DM request"}, 400)


def living_data_cursor_route(self, path, method, *, service, CursorError, QueryError, MAX_BODY):
    from urllib.parse import parse_qs
    import re
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    if self._is_viewer():
        return self._refuse_viewer()
    if not self._require("dev_ui"):
        return
    account = self._account()
    if account is None:
        return self._json({"error": "sign in to navigate Living Data"}, 401)
    try:
        match = re.fullmatch(r"/api/living-data/sessions/([0-9a-f]{32})(?:/(move|close))?", path)
        if method == "GET":
            if not match or match[2]:
                return self._json({"error": "not found"}, 404)
            query = parse_qs(self.path.partition("?")[2], keep_blank_values=True)
            if set(query) - {"offset"} or any(len(v) != 1 for v in query.values()):
                raise CursorError("invalid status query")
            return self._json(service.status(account["id"], match[1], int(query.get("offset", ["0"])[0])))
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self._json({"error": "application/json required"}, 415)
        body = self._body_json(MAX_BODY if path == "/api/living-data/sessions" else 4096)
        if body is None:
            return
        if path == "/api/living-data/sessions":
            return self._json(service.create(account["id"], body), 201)
        if not match or match[2] not in ("move", "close"):
            return self._json({"error": "not found"}, 404)
        return self._json(getattr(service, match[2])(account["id"], match[1], body))
    except CursorError as error:
        return self._json({"error": str(error)}, error.status)
    except (QueryError, ValueError, RecursionError):
        return self._json({"error": "invalid or excessive graph navigation request"}, 400)


def living_data_query_route(self, *, living_data_query):
    self._set_cookie = []
    self.close_connection = True
    if self._is_viewer():
        return self._refuse_viewer()
    if not self._require("dev_ui"):
        return
    if self._account() is None:
        return self._json({"error": "sign in to query Living Data"}, 401)
    if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
        return self._json({"error": "application/json required"}, 415)
    try:
        body = self._body_json(living_data_query.MAX_BODY)
        if body is not None:
            return self._json(living_data_query.query(body))
    except (living_data_query.QueryError, RecursionError):
        return self._json({"error": "invalid or excessive Living Data query"}, 400)


def picture_route(self, path, method, *, service, owners, recheck, PictureError, MAX_BODY):
    """show, with a camera: ``POST /api/show/pictures`` asks for a picture, ``GET /api/show/pictures/{id}`` reads its receipt.

    The body is ``{request_id, settings?, photo?}``; the host turns it into its Studio request, carries it over its
    Nodus destination and births an account-owned photo job. The account, its owner keys and the authority recheck
    come from the authenticated host (``owners``, ``recheck``), never from the body. The receipt names the real job;
    the job's status, image and cancellation stay on the existing photo endpoints.
    """
    import re
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    if self._is_viewer():
        return self._refuse_viewer()
    if not self._require("photo_submit"):
        return
    account = self._account()
    if account is None:
        return self._json({"error": "sign in to ask for a picture"}, 401)
    if service is None:
        return self._json({"error": "no Studio destination is configured on this host"}, 503)
    if self.path.partition("?")[2]:
        return self._json({"error": "picture routes accept no query parameters"}, 400)
    base = "/api/show/pictures"
    try:
        if method == "GET":
            match = re.fullmatch(base + r"/([0-9a-f]{32})", path)
            if not match:
                return self._json({"error": "not found"}, 404)
            return self._json(service.status(account["id"], match[1]))
        if path != base:
            return self._json({"error": "not found"}, 404)
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self._json({"error": "application/json required"}, 415)
        body = self._body_json(MAX_BODY)
        if body is None:
            return
        receipt = service.submit(account["id"], owners, recheck, body)
        return self._json(receipt, 200 if receipt.get("replayed") else 202)
    except PictureError as error:
        return self._json({"error": str(error)}, error.status)
    except (ValueError, TypeError, RecursionError):
        return self._json({"error": "invalid picture request"}, 400)


def site_route(self, path, *, site, maps):
    """A map as a website: ``GET /site/<map>/`` is the map's root place, ``GET /site/<map>/<address>.html`` a place.

    ``maps`` is the host's declaration, name -> ``{"source": callable returning the map, "capability": str,
    "ttl": seconds}``. A page is rendered on first look and kept (``speaktome_api.site``); nothing is written for a
    place nobody looks at. Pages carry no script and take everything from the map as data.
    """
    import re
    self._set_cookie = []
    match = re.fullmatch(r"/site/([a-z0-9][a-z0-9-]{0,31})(?:/(?:([A-Za-z0-9._-]{1,128})\.html)?)?", path)
    declared = maps.get(match[1]) if match else None
    if declared is None:
        return self._json({"error": "not found"}, 404)
    if not self._require(declared["capability"]):
        return
    if not path.startswith(f"/site/{match[1]}/"):
        self.send_response(301)
        self.send_header("Location", f"/site/{match[1]}/")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return
    try:
        atlas = site.atlas(match[1], declared["source"], declared.get("ttl", 0.0))
    except Exception as error:   # the map's source is another service; its failure is not this page's
        return self._json({"error": f"the {match[1]} map is unavailable: {type(error).__name__}"}, 503)
    try:
        data = atlas.page(match[2] or "index")
    except KeyError:
        return self._json({"error": "no such place on this map"}, 404)
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    self.wfile.write(data)


def index_route(self, path, *, catalog):
    """``GET /api/index[?group=NAME]``: the declared catalog of this API (``speaktome_api.catalog``).

    Every endpoint carries ``allowed``: whether this request, with the capabilities the host gave it, may use it. The
    answer is a description, never a grant; each handler still decides for itself.
    """
    from urllib.parse import parse_qs
    self._set_cookie = []
    query = parse_qs(self.path.partition("?")[2], keep_blank_values=True)
    if set(query) - {"group"} or any(len(values) != 1 for values in query.values()):
        return self._json({"error": "unexpected index query"}, 400)
    group = query.get("group", [None])[0]
    if group is not None and group not in catalog.GROUPS:
        return self._json({"error": "unknown group"}, 404)
    return self._json(catalog.index(sorted(self._caps()), self._principal().user_id is not None, group))
