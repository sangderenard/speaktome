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

    def __init__(self, capabilities=AGENT, signed_in=True, index=True, reachable=True, answers=None):
        self.capabilities, self.signed_in, self.index, self.reachable = capabilities, signed_in, index, reachable
        self.answers = answers or {}
        self.calls = []

    def call(self, path, body=None, timeout=40, raw_limit=None, *, method=None, headers=None):
        self.calls.append({"path": path, "body": body, "raw_limit": raw_limit, "method": method, "headers": headers})
        if not self.reachable:
            raise RestError("cannot reach the server")
        if path == api.INDEX:
            if not self.index:
                raise RestError("not found", 404)
            return json.loads(json.dumps(catalog.index(self.capabilities, self.signed_in)))
        if path == "/api/whoami":
            return {"capabilities": self.capabilities, "signed_in": self.signed_in}
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


def test_inline_field_schema_is_preserved_without_mutating_catalogue():
    nested = {
        "type": "object", "required": ["voices"], "additionalProperties": False,
        "properties": {"voices": {"type": "array", "minItems": 1,
            "items": {"type": "object", "required": ["gain"],
                "properties": {"gain": {"type": "number", "minimum": 0, "maximum": 1},
                               "mode": {"enum": ["a", "b"]}},
                "additionalProperties": False}}},
        "allOf": [{"description": "Keep composition metadata intact"}],
    }
    endpoint = {"body": {"fields": {"settings": {
        "type": "object", "description": "Settings", "required": True, "schema": nested}}, "open": False}}
    result = api.schema(endpoint)
    assert result["required"] == ["settings"]
    assert result["properties"]["settings"] == nested
    result["properties"]["settings"]["properties"]["voices"]["minItems"] = 5
    assert nested["properties"]["voices"]["minItems"] == 1
    assert api._property({"type": "string", "schema": {"type": "string", "maxLength": 90000}}) == {
        "type": "string", "maxLength": 90000}


def test_legacy_field_schemas_are_unchanged_for_entire_catalogue():
    for entry in catalog.ENDPOINTS:
        fields = list(entry["path_params"].values()) + list(entry["query"].values())
        fields += list((entry["body"] or {"fields": {}})["fields"].values())
        for spec in fields:
            kind = spec.get("type") if spec.get("type") in api.TYPES else "string"
            expected = {"type": kind, "description": str(spec.get("description", ""))}
            if kind == "string":
                expected["maxLength"] = 65536
            elif kind == "object":
                expected["additionalProperties"] = True
            assert api._property(spec) == expected


def test_inline_schema_reaches_tools_list_and_preserves_request_routing():
    fake = Fake()
    document = catalog.index(AGENT, True)
    entry = next(e for e in document["endpoints"] if e["name"] == "messages_ack")
    entry["body"]["fields"]["up_to"]["schema"] = {"type": "integer", "enum": [7, 8]}
    server = api.ApiServer(fake)
    original = fake.call
    fake.call = lambda path, *args, **kwargs: document if path == api.INDEX else original(path, *args, **kwargs)
    tool = next(t for t in rpc(server, "tools/list")["tools"] if t["name"] == "messages_ack")
    assert tool["inputSchema"]["properties"]["up_to"] == {"type": "integer", "enum": [7, 8]}
    result, error, _ = call(server, "messages_ack", up_to=7)
    assert not error and fake.calls[-1]["body"] == {"up_to": 7}
    count = len(fake.calls)
    result, error, _ = call(server, "messages_ack", up_to=9)
    assert error and "7, 8" in result["error"] and len(fake.calls) == count


def test_boolean_schemas_are_handled_by_lightweight_argument_checks():
    from speaktome_mcp.server import _validate, ToolFail
    assert api._property({"schema": True}) is True
    _validate({"anything": 1}, True)
    with pytest.raises(ToolFail, match="not permitted"):
        _validate(1, False)
    with pytest.raises(ToolFail, match="not permitted"):
        _validate([1], {"type": "array", "items": False})
    with pytest.raises(ToolFail, match="object or boolean"):
        api._property({"schema": "not a schema"})
