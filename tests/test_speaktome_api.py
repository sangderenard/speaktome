#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Offline host-contract checks; no database, listener or compiler is started."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from speaktome_api import routes
# --- END HEADER ---


class RouteError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


class QueryError(Exception):
    pass


class Host:
    """Only the existing HTTP handler hooks consumed by shared routes."""

    def __init__(self, path, body=None, *, allowed=True, secure=True, viewer=False, account=7):
        self.path, self.body = path, body
        self.headers = {"Content-Type": "application/json"}
        self.close_connection = False
        self.required = []
        self.allowed, self.secure, self.viewer, self.account = allowed, secure, viewer, account
        self.body_limits = []
        self.response = None

    def _require(self, capability):
        self.required.append(capability)
        if not self.allowed:
            self._json({"error": capability}, 403)
        return self.allowed

    def _principal(self):
        return SimpleNamespace(user_id=self.account)

    def _account(self):
        return None if self.account is None else {"id": self.account}

    def _is_secure(self):
        return self.secure

    def _is_viewer(self):
        return self.viewer

    def _refuse_viewer(self):
        return self._json({"error": "viewer"}, 403)

    def _body_json(self, limit):
        self.body_limits.append(limit)
        return self.body

    def _json(self, body, status=200):
        self.response = (int(status), body)
        return self.response


class SharedRoutesTest(unittest.TestCase):
    def messaging(self):
        db = Mock()
        backend = SimpleNamespace(connect=Mock(return_value=db), MessagingError=RouteError,
                                  send=Mock(return_value={"ok": True}), contacts=Mock(return_value={"contacts": []}))
        return backend, db

    def message(self, host, backend, method="GET"):
        return routes.messages_route(host, host.path.partition("?")[0], method,
                                     messaging=backend, accounts_db="existing-accounts.db")

    def living(self, kind, host, service, method="POST"):
        return getattr(routes, f"living_data_{kind}_route")(
            host, host.path.partition("?")[0], method, service=service,
            CursorError=RouteError, QueryError=QueryError, MAX_BODY=99999)

    def test_message_capability_checked_before_storage(self):
        for path, capability in (("/api/messages/contacts", "message"),
                                 ("/api/messages/admin/thread", "account_manage")):
            with self.subTest(path=path):
                backend, db = self.messaging()
                host = Host(path, allowed=False)
                self.message(host, backend)
                self.assertEqual(host.response[0], 403)
                self.assertEqual(host.required, [capability])
                backend.connect.assert_not_called()

    def test_message_post_rejects_non_json_before_storage(self):
        backend, db = self.messaging()
        host = Host("/api/messages/send", {})
        host.headers = {"Content-Type": "text/plain"}
        self.assertEqual(self.message(host, backend, "POST")[0], 415)
        self.assertTrue(host.close_connection)
        backend.connect.assert_not_called()

    def test_message_uses_host_identity_and_same_backend(self):
        backend, db = self.messaging()
        host = Host("/api/messages/send", {"conversation_handle": "handle", "body": "hi",
                    "idempotency_key": "same-key", "sender": 999})
        self.assertEqual(self.message(host, backend, "POST"), (200, {"ok": True}))
        backend.connect.assert_called_once_with("existing-accounts.db")
        backend.send.assert_called_once_with(db, 7, None, "hi", subject=None, msg_type=None,
                                            conversation_handle="handle", idempotency_key="same-key")
        db.close.assert_called_once()
        self.assertEqual(host.body_limits, [16384])

    def test_secure_handle_routes_refuse_plaintext(self):
        for path, method, body in (("/api/messages/conversations", "GET", None),
                                   ("/api/messages/send", "POST", {"conversation_handle": "h"})):
            with self.subTest(path=path):
                backend, db = self.messaging()
                host = Host(path, body, secure=False)
                self.assertEqual(self.message(host, backend, method), (403, {"error": "HTTPS required"}))
                backend.send.assert_not_called()
                db.close.assert_called_once()

    def test_handle_and_name_cannot_be_mixed(self):
        backend, db = self.messaging()
        host = Host("/api/messages/send", {"conversation_handle": "h", "to": "peer"})
        self.assertEqual(self.message(host, backend, "POST")[0], 400)
        backend.send.assert_not_called()
        db.close.assert_called_once()

    def test_backend_error_status_and_connection_lifetime(self):
        backend, db = self.messaging()
        backend.contacts.side_effect = RouteError("limited", 429)
        self.assertEqual(self.message(Host("/api/messages/contacts"), backend), (429, {"error": "limited"}))
        db.close.assert_called_once()

    def test_unexpected_backend_failure_still_closes_connection(self):
        backend, db = self.messaging()
        backend.contacts.side_effect = RuntimeError("storage")
        with self.assertRaisesRegex(RuntimeError, "storage"):
            self.message(Host("/api/messages/contacts"), backend)
        db.close.assert_called_once()

    def test_world_and_cursor_preserve_viewer_capability_and_account_gates(self):
        for kind in ("dm", "cursor"):
            for options, status in (({"viewer": True}, 403), ({"allowed": False}, 403),
                                    ({"account": None}, 401)):
                with self.subTest(kind=kind, options=options):
                    service = Mock()
                    host = Host("/api/living-data/dm/worlds", {}, **options)
                    self.living(kind, host, service)
                    self.assertEqual(host.response[0], status)
                    self.assertEqual(service.mock_calls, [])

    def test_world_command_preserves_owner_receipt_and_status(self):
        world = "a" * 32
        command = {"request_id": "same-request", "operation": "render-phong"}
        for state, status in (("queued", 202), ("rendering", 202), ("complete", 200), ("failed", 200)):
            with self.subTest(state=state):
                service = Mock()
                service.command.return_value = {"state": state}
                host = Host(f"/api/living-data/dm/worlds/{world}/commands", command)
                self.assertEqual(self.living("dm", host, service), (status, {"state": state}))
                service.command.assert_called_once_with(7, world, command)
                self.assertEqual(host.body_limits, [4096])

    def test_world_query_rejected_and_backend_error_preserved(self):
        world = "a" * 32
        service = Mock()
        host = Host(f"/api/living-data/dm/worlds/{world}?owner=999")
        self.assertEqual(self.living("dm", host, service, "GET")[0], 400)
        self.assertEqual(service.mock_calls, [])
        service.status.side_effect = RouteError("static world not found", 404)
        host = Host(f"/api/living-data/dm/worlds/{world}")
        self.assertEqual(self.living("dm", host, service, "GET"), (404, {"error": "static world not found"}))

    def test_cursor_passes_existing_identity_and_rejects_duplicate_offset(self):
        identity = "b" * 32
        service = Mock()
        service.status.return_value = {"at": "existing-node"}
        host = Host(f"/api/living-data/sessions/{identity}?offset=2")
        self.assertEqual(self.living("cursor", host, service, "GET"), (200, {"at": "existing-node"}))
        service.status.assert_called_once_with(7, identity, 2)
        host = Host(f"/api/living-data/sessions/{identity}?offset=2&offset=3")
        self.assertEqual(self.living("cursor", host, service, "GET")[0], 400)
        self.assertEqual(service.status.call_count, 1)

    def test_graph_query_keeps_existing_projection_and_body_bound(self):
        backend = SimpleNamespace(MAX_BODY=12345, QueryError=QueryError, query=Mock(return_value={"identity": "source-node"}))
        host = Host("/api/living-data/query", {"map": "fixture"})
        self.assertEqual(routes.living_data_query_route(host, living_data_query=backend),
                         (200, {"identity": "source-node"}))
        backend.query.assert_called_once_with(host.body)
        self.assertEqual(host.body_limits, [12345])


if __name__ == "__main__":
    unittest.main()
