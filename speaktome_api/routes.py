#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Shared REST handlers extracted from Turing's existing Meowdoku service.

The host supplies its authenticated request handler and existing services.
There is no listener, account store, world registry or simulation in this module.
"""
from __future__ import annotations

from http import HTTPStatus
# --- END HEADER ---

API_VERSION = 1


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
        body = self._body_json(16384)
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
                                                    idempotency_key=body.get("idempotency_key")))
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
