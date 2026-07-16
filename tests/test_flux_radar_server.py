"""Tests for speaktome.flux_radar_server -- HTTP routing/safety only, no real model load."""

import json
import threading
import urllib.request
import urllib.error

from http.server import ThreadingHTTPServer

from speaktome import flux_radar_server as server_mod


def _start_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def test_get_root_serves_index_html():
    httpd, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "FluxGraph radar" in body
    finally:
        httpd.shutdown()


def test_get_unknown_path_is_404():
    httpd, port = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()


def test_static_path_traversal_is_rejected():
    httpd, port = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/static/../flux_radar_server.py")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code in (403, 404)
    finally:
        httpd.shutdown()


def test_api_run_surfaces_errors_as_json_not_a_crash(monkeypatch):
    class ExplodingBundle:
        def run(self, params):
            raise ValueError("simulated failure")

    monkeypatch.setattr(server_mod, "get_bundle", lambda: ExplodingBundle())

    httpd, port = _start_server()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"seed": "x", "ticks": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 500
            payload = json.loads(e.read().decode("utf-8"))
            assert "simulated failure" in payload["error"]
    finally:
        httpd.shutdown()


def test_api_run_returns_bundle_result_as_json(monkeypatch):
    class FakeBundle:
        def run(self, params):
            return {"run_id": params.get("seed"), "history": [{"tick": 0, "nodes": []}]}

    monkeypatch.setattr(server_mod, "get_bundle", lambda: FakeBundle())

    httpd, port = _start_server()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"seed": "hello", "ticks": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["run_id"] == "hello"
            assert payload["history"][0]["tick"] == 0
    finally:
        httpd.shutdown()
