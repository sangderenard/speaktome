"""SPEAKTOME_TOOLS=by_place: the walking verbs plus the acts of the decision place, announced when they change."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speaktome_api import catalog  # noqa: E402
from speaktome_mcp import api  # noqa: E402
from speaktome_mcp.server import DECISIONS, SESSIONS, serve  # noqa: E402
import io  # noqa: E402

CAPS = ["dev_ui", "message", "view_site", "photo_read", "research_read", "research_write"]
PLACES = {"decisions:g/a": {"title": "Desk", "acts": ["messages_inbox"], "routes": [("busy", "decisions:g/b")]},
          "decisions:g/b": {"title": "Library", "acts": ["reports_list", "man"], "routes": []},
          "decisions:g/c": {"title": "Porch", "acts": ["reports_list"], "routes": []}}


def exits(at):
    return [{"to": to, "relationship": cond, "direction": "outgoing"} for cond, to in PLACES[at]["routes"]]


class Fake:
    def __init__(self):
        self.at, self.calls, self.revision = "decisions:g/a", [], 1

    def view(self):
        p = PLACES[self.at]
        return {"at": self.at, "title": p["title"], "body": "", "acts": [{"tool": t, "for": "x"} for t in p["acts"]], "routes": []}

    def cursor(self):
        return {"session_id": "s1", "revision": self.revision, "at": {"identity": self.at}, "exits": exits(self.at)}

    def call(self, path, body=None, timeout=40, raw_limit=None, *, method=None, headers=None):
        self.calls.append(path)
        if path == api.INDEX:
            return json.loads(json.dumps(catalog.index(CAPS, True)))
        if path == DECISIONS + "/map":
            return {"map": {"kind": "decisions", "identity": "decisions:g", "regions": [], "world": {"objects": []}}, "at": self.at}
        if path == SESSIONS:
            return self.cursor()
        if path == SESSIONS + "/s1":
            return self.cursor()
        if path == SESSIONS + "/s1/move":
            self.at, self.revision = body["to"], self.revision + 1
            return self.cursor()
        if path == SESSIONS + "/s1/close":
            return {"closed": True}
        if path == DECISIONS + "/at":
            if body:
                self.at = body["at"]
            return self.view()
        return {"ok": True}


def rpc(server, method, params=None):
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})["result"]


def names(server):
    return [t["name"] for t in rpc(server, "tools/list")["tools"]]


def call(server, name, **a):
    answer = rpc(server, "tools/call", {"name": name, "arguments": a})
    return json.loads(answer["content"][0]["text"]), answer["isError"]


def lines_for(server, message):
    """Run one request through serve() and return the JSON lines written."""
    out = io.BytesIO()
    serve(server, io.BytesIO((json.dumps(message) + "\n").encode()), out)
    return [json.loads(x) for x in out.getvalue().decode().splitlines()]


def test_by_place_lists_verbs_and_acts_only():
    server = api.ApiServer(Fake(), by_place=True)
    listed = names(server)
    assert set(listed) == set(api.WALKING_VERBS) | {"messages_inbox"}
    assert rpc(server, "initialize", {"protocolVersion": "2025-06-18"})["capabilities"] == {"tools": {"listChanged": True}}


def test_a_move_changes_the_list_and_emits_one_notification_after_the_response():
    fake = Fake()
    server = api.ApiServer(fake, by_place=True)
    names(server)
    call(server, "enter", map="decisions")
    out = lines_for(server, {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                             "params": {"name": "go", "arguments": {"relationship": "busy"}}})
    assert [("id" in m, m.get("method")) for m in out] == [(True, None), (False, "notifications/tools/list_changed")]
    listed = names(server)
    assert "reports_list" in listed and "messages_inbox" not in listed
    # a call that does not change the set emits nothing
    assert len(lines_for(server, {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                                  "params": {"name": "where", "arguments": {}}})) == 1


def test_out_of_place_call_is_refused_with_the_place_named():
    server = api.ApiServer(Fake(), by_place=True)
    names(server)
    body, is_error = call(server, "contacts_x") if False else call(server, "reports_list")
    assert is_error and body["place"] == "Desk" and body["acts"] == ["messages_inbox"]
    assert "Desk" in body["say"] and "map=decisions" in body["say"]


def test_default_mode_is_unchanged(monkeypatch):
    monkeypatch.delenv("SPEAKTOME_TOOLS", raising=False)
    fake = Fake()
    server = api.ApiServer(fake)
    assert "reports_list" in names(server) and len(names(server)) > len(api.WALKING_VERBS) + 5
    assert api.PLACE_ROUTE not in fake.calls
    assert rpc(server, "initialize", {"protocolVersion": "2025-06-18"})["capabilities"] == {"tools": {}}
    assert not lines_for(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                  "params": {"name": "whoami", "arguments": {}}})[1:]
