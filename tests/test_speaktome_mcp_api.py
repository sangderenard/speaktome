"""The generated MCP: its tools are the catalog, a call is exactly the declared request, and it degrades without the index."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speaktome_api import catalog  # noqa: E402
from speaktome_mcp import api  # noqa: E402
from speaktome_mcp.rest import RestError  # noqa: E402
from speaktome_mcp.server import Server  # noqa: E402

AGENT = ["dev_ui", "message", "view_site", "photo_read"]


class Fake:
    """Answers /api/index (or refuses it) and records every call."""

    def __init__(self, capabilities=AGENT, signed_in=True, index=True, reachable=True, answers=None, kiosk=False, guest=False):
        self.capabilities, self.signed_in, self.index, self.reachable = capabilities, signed_in, index, reachable
        self.kiosk, self.guest = kiosk, guest
        self.answers = answers or {}
        self.calls = []

    def call(self, path, body=None, timeout=40, raw_limit=None, *, method=None, headers=None):
        self.calls.append({"path": path, "body": body, "raw_limit": raw_limit, "method": method, "headers": headers})
        if not self.reachable:
            raise RestError("cannot reach the server")
        if path == api.INDEX:
            if not self.index:
                raise RestError("not found", 404)
            return json.loads(json.dumps(catalog.index(self.capabilities, self.signed_in, kiosk=self.kiosk, guest=self.guest)))
        if path == "/api/whoami":
            return {"capabilities": self.capabilities, "signed_in": self.signed_in, "kiosk": self.kiosk, "guest": self.guest}
        if path in self.answers:
            return self.answers[path]
        return (b"\x89PNG", "image/png") if raw_limit is not None else {"ok": True}


def rpc(server, method, params=None):
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})["result"]


def call(server, name, **arguments):
    answer = rpc(server, "tools/call", {"name": name, "arguments": arguments})
    return json.loads(answer["content"][0]["text"]), answer["isError"], answer["content"]


def names(server):
    return {tool["name"] for tool in rpc(server, "tools/list")["tools"]}


def test_every_offered_endpoint_is_a_tool_and_nothing_else_is():
    server = api.ApiServer(Fake())
    listed = names(server)
    verbs = set(Server(None).table)
    expected = {e["name"] for e in catalog.ENDPOINTS
                if e["tool"] and e["request"] == "json" and catalog.allowed(e, AGENT, True)}
    assert listed == verbs | {"api_index"} | expected
    assert not verbs & {e["name"] for e in catalog.ENDPOINTS}, "a verb and an endpoint never share a name"
    assert "messages_inbox" in listed and "control_instances" not in listed
    assert not {e["name"] for e in catalog.ENDPOINTS if not e["tool"]} & listed
    assert "photo_submit" not in listed, "a binary upload is not a tool"


def test_an_admin_sees_the_control_tools():
    assert "control_instances_request" in names(api.ApiServer(Fake(AGENT + ["account_manage"])))


def test_schema_and_description_come_from_the_entry():
    server = api.ApiServer(Fake())
    tools = {tool["name"]: tool for tool in rpc(server, "tools/list")["tools"]}
    ack = tools["messages_ack"]
    entry = catalog.by_name()["messages_ack"]
    assert ack["inputSchema"]["additionalProperties"] is False
    assert set(ack["inputSchema"]["properties"]) == set(entry["body"]["fields"])
    assert ack["description"].startswith(entry["summary"]) and "POST /api/messages/ack" in ack["description"]
    assert ack["annotations"] == {"readOnlyHint": False}
    assert tools["messages_inbox"]["annotations"] == {"readOnlyHint": True}
    assert tools["nodus_share"]["description"].count("dev_ui and message") == 1
    for tool in tools.values():
        json.dumps(tool)
        assert tool["inputSchema"]["type"] == "object"


def test_a_call_sends_exactly_the_declared_request():
    fake = Fake()
    server = api.ApiServer(fake)
    result, is_error, _ = call(server, "reports_get", id="ab" * 16)
    assert not is_error and result == {"endpoint": "reports_get", "data": {"ok": True}}
    assert fake.calls[-1] == {"path": "/api/reports/" + "ab" * 16, "body": None, "raw_limit": None, "method": "GET", "headers": {}}
    call(server, "messages_ack", up_to=7)
    assert fake.calls[-1]["path"] == "/api/messages/ack" and fake.calls[-1]["body"] == {"up_to": 7}
    assert fake.calls[-1]["method"] == "POST"
    call(server, "reports_issue_create", title="t", description="d")
    assert fake.calls[-1]["headers"] == {"X-Meowdoku-Issue": "1"}
    call(server, "nodus_graph", instance="a b&c")
    assert fake.calls[-1]["path"] == "/api/nodus/graph?instance=a+b%26c"


def test_a_path_variable_cannot_leave_its_segment():
    fake = Fake()
    server = api.ApiServer(fake)
    call(server, "reports_issue_get", id="../account/login?x=1")
    assert fake.calls[-1]["path"] == "/api/issues/..%2Faccount%2Flogin%3Fx%3D1"
    before = len(fake.calls)
    result, is_error, _ = call(server, "reports_issue_get", id="..")
    assert is_error and len(fake.calls) == before


def test_undeclared_arguments_are_refused_unless_the_body_is_open():
    fake = Fake()
    server = api.ApiServer(fake)
    names(server)
    before = len(fake.calls)
    result, is_error, _ = call(server, "messages_ack", up_to=1, extra=2)
    assert is_error and "not an accepted field" in result["error"] and len(fake.calls) == before
    result, is_error, _ = call(server, "messages_ack")
    assert is_error and "required" in result["error"]
    assert catalog.by_name()["photo_film_preview"]["body"] == {"fields": {}, "open": True}
    fake = Fake(AGENT + ["photo_submit"])
    call(api.ApiServer(fake), "photo_film_preview", anything={"deep": 1})
    assert fake.calls[-1]["body"] == {"anything": {"deep": 1}}


def test_an_image_is_attached_and_a_large_answer_is_refused():
    fake = Fake(answers={"/api/nodus/graph": {"blob": "x" * (api.MAX_ANSWER_BYTES + 1)}})
    server = api.ApiServer(fake)
    result, is_error, content = call(server, "messages_image", id="5")
    assert not is_error and result["content_type"] == "image/png" and content[1]["type"] == "image"
    assert fake.calls[-1]["raw_limit"] == api.MAX_IMAGE_BYTES
    result, is_error, _ = call(server, "nodus_graph")
    assert is_error and result["status"] == 413


def test_the_index_tool_explains_what_is_not_offered():
    server = api.ApiServer(Fake())
    result, is_error, _ = call(server, "api_index")
    assert not is_error and set(result["groups"]) == set(catalog.GROUPS) and result["source"] == "the service index"
    rows = {row["tool"]: row for group in result["groups"].values() for row in group["endpoints"]}
    assert len(rows) == len(catalog.ENDPOINTS)
    assert "not_a_tool" not in rows["messages_inbox"]
    assert "lacks" in rows["control_instances"]["not_a_tool"]
    assert rows["identity_login"]["not_a_tool"] == "not offered to tools"
    assert "binary" in rows["photo_submit"]["not_a_tool"]
    only, _, _ = call(server, "api_index", group="messages")
    assert set(only["groups"]) == {"messages"}
    result, is_error, _ = call(server, "api_index", group="nothing")
    assert is_error and result["status"] == 404


def test_without_the_service_index_the_packaged_catalog_is_used():
    older = api.ApiServer(Fake(index=False))
    assert "messages_inbox" in names(older) and "control_instances" not in names(older)
    assert "packaged catalog with this account" in call(older, "api_index")[0]["source"]
    unreachable = api.ApiServer(Fake(reachable=False))
    assert "control_instances" in names(unreachable), "unchecked: the server refuses what the account may not do"
    assert "unchecked" in call(unreachable, "api_index")[0]["source"]
    result, is_error, _ = call(unreachable, "messages_inbox")
    assert is_error


def test_initialize_names_the_api_server_and_a_verb_still_works():
    server = api.ApiServer(Fake())
    hello = rpc(server, "initialize", {"protocolVersion": "2025-06-18"})
    assert hello["serverInfo"]["name"] == "speaktome-api" and "api_index" in hello["instructions"]
    result, is_error, _ = call(server, "man", topic="go")
    assert not is_error and result["page"].startswith("GO(1)")


def test_the_catalog_gives_each_argument_one_meaning():
    for entry in catalog.ENDPOINTS:
        declared = list(entry["path_params"]) + list(entry["query"]) + list((entry["body"] or {"fields": {}})["fields"])
        assert len(declared) == len(set(declared)), entry["name"]
        assert entry["request"] in ("json", "binary")
        assert isinstance(entry["headers"], dict) and all(isinstance(v, str) for v in entry["headers"].values())
        if entry["capability_all"]:
            assert isinstance(entry["capability"], tuple)
    share = catalog.by_name()["nodus_share"]
    assert not catalog.allowed(share, ["message"], True) and catalog.allowed(share, ["message", "dev_ui"], True)


def test_report_sharing_and_groups_are_available_to_signed_in_members():
    fake=Fake(capabilities=[],signed_in=True)
    server=api.ApiServer(fake)
    offered=names(server)
    assert {'reports_make','reports_sharing','groups_create','groups_invite','groups_respond'} <= offered
    result,error,_=call(server,'reports_sharing',id='ab'*16,visibility='public',expected_revision=0,sign_off=True)
    assert not error
    assert fake.calls[-1]['path']=='/api/reports/'+'ab'*16+'/sharing'
    assert fake.calls[-1]['body']==dict(visibility='public',expected_revision=0,sign_off=True)
    call(server,'groups_member_role',id='ab'*16,username='member',expected_revision=2)
    assert fake.calls[-1]['body']==dict(username='member',expected_revision=2)
    assert {'reports_public_list','reports_public_get'} <= names(api.ApiServer(Fake(capabilities=[],signed_in=False)))
    assert 'reports_sharing' not in names(api.ApiServer(Fake(capabilities=[],signed_in=False)))


def test_a_kiosk_account_is_offered_only_the_kiosk_tools_and_reading_verbs():
    from speaktome_mcp import server as verbs
    intern = ["view_site", "photo_read"]
    kiosk = {e["name"] for e in catalog.ENDPOINTS if e["kiosk"] and e["tool"] and catalog.allowed(e, intern, True)}
    assert kiosk == {"identity_me", "identity_index", "identity_whoami", "photo_jobs", "photo_lenses", "photo_status", "photo_image",
                     "site_map_root", "site_place", "reports_public_list", "reports_public_get",
                     "foyer_status", "foyer_command", "foyer_receipt"}
    for fake in (Fake(intern, kiosk=True), Fake(intern, kiosk=True, index=False)):
        server = api.ApiServer(fake)
        assert names(server) == set(verbs.KIOSK_VERBS) | {"api_index"} | kiosk
        refused = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                 "params": {"name": "say", "arguments": {"to": "someone", "body": "hello"}}})
        assert "unknown tool say" in refused["error"]["message"]
    assert "make" in names(api.ApiServer(Fake(intern))), "the same capabilities without kiosk keep every verb"


def test_a_visitor_looks_through_the_window():
    from speaktome_mcp import server as verbs
    visitor = ["view_site", "photo_read", "play"]
    reads = {e["name"] for e in catalog.ENDPOINTS
             if e["tool"] and e["request"] == "json" and catalog.allowed(e, visitor, False, guest=True)}
    assert {"identity_whoami", "photo_image", "reports_public_list", "site_place", "play_radio"} <= reads
    assert {n for n in reads if catalog.by_name()[n]["changes"]} == {"foyer_command"}, "a visitor's tools only read, or ask for a picture"
    for fake in (Fake(visitor, signed_in=False, guest=True), Fake(visitor, signed_in=False, guest=True, index=False)):
        server = api.ApiServer(fake)
        assert names(server) == set(verbs.KIOSK_VERBS) | {"api_index"} | reads
        window = call(server, "api_index")[0]
        assert window["visitor"] is True and "foyer" in window["say"]
        rows = {row["tool"]: row for group in window["groups"].values() for row in group["endpoints"]}
        assert rows["messages_send"]["not_a_tool"] == api.VISITOR
    who = call(api.ApiServer(Fake(visitor, signed_in=False, guest=True,
                                  answers={"/api/whoami": {"signed_in": False, "guest": True}})), "whoami")[0]
    assert "foyer" in who["say"]


def test_without_an_account_name_the_session_is_a_visitor_that_never_signs_in():
    from speaktome_mcp.rest import Rest

    class Opener:
        def __init__(self):
            self.paths = []

        def open(self, request, timeout=None):
            import io
            self.paths.append(request.full_url)

            class Answer(io.BytesIO):
                headers = {}

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False
            return Answer(b'{"signed_in": false, "guest": true}')

    assert Rest.from_env({"SPEAKTOME_BASE": "https://cats.example.org"}).visitor is True
    opener = Opener()
    rest = Rest("https://cats.example.org", None, None, opener=opener)
    assert rest.call("/api/whoami") == {"signed_in": False, "guest": True}
    assert opener.paths == ["https://cats.example.org/api/whoami"] and rest.logins == 0
    with pytest.raises(RestError):
        rest.sign_in()


def test_the_turtle_walks_turns_and_takes_a_picture_in_the_foyer():
    import base64
    png = bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10]) + bytes(16)
    fake = Fake(["view_site", "photo_read", "play"], signed_in=False, guest=True, answers={
        "/api/foyer": {"greeting": "a demo", "start": {"position": [0, 1, 0], "facing": [0, 0, 1]}},
        "/api/foyer/commands": {"state": "complete", "result": {"png_base64": base64.b64encode(png).decode(), "png_sha256": "x"}}})
    server = api.ApiServer(fake)
    assert {"foyer", "walk", "turn", "picture"} <= names(server)
    entered = call(server, "foyer")[0]
    assert entered["demo"] and entered["untrusted"]["greeting"] == "a demo" and "Next: walk" in entered["say"]
    assert call(server, "walk", metres=2)[0]["position"] == [0, 1, 2]
    turned = call(server, "turn", direction="right", degrees=90)[0]       # right of +Z is -X: the world is right-handed, Y up
    assert [round(v, 9) for v in turned["facing"]] == [-1, 0, 0]
    assert [round(v, 9) for v in call(server, "walk", metres=1)[0]["position"]] == [-1, 1, 2]
    taken, is_error, content = call(server, "picture", width=320, height=240)
    assert not is_error and taken["demo"] and content[1]["type"] == "image" and base64.b64decode(content[1]["data"]) == png
    sent = next(c for c in fake.calls if c["path"] == "/api/foyer/commands")["body"]
    assert sent["operation"] == "render-phong" and sent["position"] == [-1, 1, 2] and sent["width"] == 320
    assert [round(v, 9) for v in sent["facing"]] == [-1, 0, 0]
    up = call(server, "turn", direction="up", degrees=180)[0]               # pitch stops at 80 degrees
    assert round(up["facing"][1], 6) == round(__import__("math").sin(__import__("math").radians(80)), 6)

