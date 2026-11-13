import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speaktome_api import catalog  # noqa: E402

ENDPOINTS = catalog.ENDPOINTS
NAME = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
TYPES = {"string", "integer", "number", "boolean", "object", "array"}
RETURNS = {"json", "png", "binary", "html", "text"}
PARAM = re.compile(r"\{([^}]+)\}")


def test_names_unique_and_well_formed():
    names = [e["name"] for e in ENDPOINTS]
    assert len(names) == len(set(names))
    for name in names:
        assert NAME.match(name), name


def test_method_path_group_summary_returns():
    seen = set()
    for e in ENDPOINTS:
        assert e["group"] in catalog.GROUPS, e["name"]
        assert e["method"] in ("GET", "POST"), e["name"]
        assert e["path"].startswith("/"), e["name"]
        key = (e["method"], e["path"])
        assert key not in seen, key
        seen.add(key)
        s = e["summary"]
        assert 0 < len(s) <= 140, e["name"]
        assert s.endswith(".") and not s.endswith(".."), e["name"]
        assert e["returns"] in RETURNS, e["name"]
        assert isinstance(e["tool"], bool) and isinstance(e["account"], bool) and isinstance(e["changes"], bool)
        cap = e["capability"]
        assert cap is None or isinstance(cap, str) or (isinstance(cap, tuple) and cap and all(isinstance(c, str) for c in cap))
    for group in catalog.GROUPS:
        assert any(e["group"] == group for e in ENDPOINTS), group


def test_path_params_match_path():
    for e in ENDPOINTS:
        assert set(PARAM.findall(e["path"])) == set(e["path_params"]), e["name"]
        for spec in e["path_params"].values():
            assert spec["type"] in TYPES and spec["description"]


def test_field_specs_and_get_has_no_body():
    for e in ENDPOINTS:
        for spec in e["query"].values():
            assert spec["type"] in ("string", "integer", "number", "boolean"), e["name"]
            assert spec["description"]
        if e["method"] == "GET":
            assert e["body"] is None, e["name"]
        body = e["body"]
        if body is not None:
            assert set(body) == {"fields", "open"}, e["name"]
            assert isinstance(body["open"], bool)
            for spec in body["fields"].values():
                assert spec["type"] in TYPES, e["name"]
                assert spec["description"]
                assert isinstance(spec.get("required", False), bool)


def test_tool_rules():
    no_prefix = ("/api/account/login", "/api/account/logout", "/api/account/signup", "/api/admin/", "/api/lifecycle",
                 "/api/account/admin/", "/api/spectate", "/api/share", "/api/clientlog", "/api/ready",
                 "/api/loading_", "/api/client_rays/")
    no_exact = ("/api/state", "/api/action", "/api/action/batch", "/api/new", "/api/catalog", "/api/profile")
    for e in ENDPOINTS:
        path = e["path"]
        must_hide = path.startswith(no_prefix) or path in no_exact
        assert e["tool"] is (not must_hide), (e["name"], path)
        assert not path.startswith("/ws"), path
    by_path = {(e["method"], e["path"]): e for e in ENDPOINTS}
    assert by_path[("GET", "/api/profile/me")]["tool"] is True
    assert by_path[("GET", "/api/profile")]["tool"] is False


def test_allowed_truth_table():
    none = {"capability": None, "account": False}
    single = {"capability": "message", "account": False}
    pair = {"capability": ("account_manage", "lifecycle_admin"), "account": False}
    acct = {"capability": None, "account": True}
    assert catalog.allowed(none, [], False)
    assert not catalog.allowed(single, [], False)
    assert catalog.allowed(single, ["message"], False)
    assert not catalog.allowed(single, ["dev_ui"], True)
    assert catalog.allowed(pair, {"lifecycle_admin"}, False)
    assert catalog.allowed(pair, ["account_manage", "x"], False)
    assert not catalog.allowed(pair, ["message"], True)
    assert not catalog.allowed(acct, [], False)
    assert catalog.allowed(acct, [], True)
    both = {"capability": "message", "account": True}
    assert not catalog.allowed(both, ["message"], False)
    assert catalog.allowed(both, ["message"], True)


def test_public_is_json_and_allowed_only_on_request():
    for e in ENDPOINTS:
        plain = catalog.public(e)
        assert "allowed" not in plain
        json.dumps(plain)
        asked = catalog.public(e, ["message"], True)
        assert asked["allowed"] is catalog.allowed(e, ["message"], True)
        json.dumps(asked)
        assert not isinstance(asked["capability"], tuple)
    pair = catalog.by_name()["control_instances"]
    assert catalog.public(pair)["capability"] == ["account_manage", "lifecycle_admin"]
    assert catalog.by_name()["messages_inbox"]["path"] == "/api/messages/inbox"


def _sources():
    base = Path(os.environ.get("TURING_REPO", str(ROOT.parent / "turing"))) / "examples" / "meowdoku" / "server.py"
    if not base.is_file():
        return None
    texts = [base.read_text(encoding="utf-8", errors="replace")]
    for delegate in ("branch_review.py", "file_index.py", "relay.py", "core_git.py"):       # route functions the server hands whole paths to
        module = base.with_name(delegate)
        if module.is_file():
            texts.append(module.read_text(encoding="utf-8", errors="replace"))
    for name in ("routes.py", "reports.py", "groups.py"):
        texts.append((ROOT / "speaktome_api" / name).read_text(encoding="utf-8", errors="replace"))
    return texts


def test_paths_occur_in_source():
    texts = _sources()
    if texts is None:
        pytest.skip("the turing checkout is absent")
    joined = "\n".join(texts)
    missing = []
    for e in ENDPOINTS:
        prefix = e["path"].split("{", 1)[0]
        if prefix in joined:
            continue
        # The handlers build some paths from a base and a name list (for example /api/nodus/ plus "digest"):
        # then the parent directory and the last segment must each occur.
        parent, _, last = prefix.rstrip("/").rpartition("/")
        if parent + "/" in joined and last in joined:
            continue
        missing.append((e["name"], prefix))
    assert not missing, missing


def test_index_route_marks_each_endpoint_for_the_caller():
    from types import SimpleNamespace
    from speaktome_api import routes

    class Handler:
        def __init__(self, path, capabilities, user_id):
            self.path, self._capabilities, self._user = path, capabilities, user_id
            self.sent = None

        def _caps(self):
            return set(self._capabilities)

        def _principal(self):
            return SimpleNamespace(user_id=self._user)

        def _json(self, document, status=200):
            self.sent = (status, json.loads(json.dumps(document)))

    handler = Handler("/api/index", ["message"], 7)
    routes.index_route(handler, "/api/index", catalog=catalog)
    status, document = handler.sent
    rows = {e["name"]: e for e in document["endpoints"]}
    assert status == 200 and len(rows) == len(ENDPOINTS) and document["groups"] == catalog.GROUPS
    assert rows["messages_inbox"]["allowed"] is True and rows["control_instances"]["allowed"] is False
    guest = Handler("/api/index?group=messages", ["message"], None)
    routes.index_route(guest, "/api/index", catalog=catalog)
    assert {e["group"] for e in guest.sent[1]["endpoints"]} == {"messages"}
    assert not any(e["allowed"] for e in guest.sent[1]["endpoints"]), "messaging needs a signed-in account"
    for bad, code in (("/api/index?group=nothing", 404), ("/api/index?x=1", 400), ("/api/index?group=a&group=b", 400)):
        refused = Handler(bad, [], None)
        routes.index_route(refused, "/api/index", catalog=catalog)
        assert refused.sent[0] == code


def test_kiosk_entries_and_their_paths():
    kiosk = {e["name"] for e in ENDPOINTS if e["kiosk"]}
    assert kiosk == {"identity_login", "identity_logout", "identity_me", "identity_index", "identity_whoami", "photo_jobs",
                     "photo_lenses", "photo_status", "photo_image", "site_map_root", "site_place", "reports_public_list",
                     "reports_public_get", "foyer_status", "foyer_command", "foyer_receipt"}
    # only the session boundary and a queued foyer picture change anything
    assert {e["name"] for e in ENDPOINTS if e["kiosk"] and e["changes"]} == {"identity_login", "identity_logout", "foyer_command"}
    found = catalog.kiosk_endpoint
    assert found("GET", "/api/whoami")["name"] == "identity_whoami"
    assert found("GET", "/site/manual/")["name"] == "site_map_root"
    assert found("GET", "/site/manual/go.html")["name"] == "site_place"
    assert found("GET", "/api/reports/public/" + "a" * 32)["name"] == "reports_public_get"
    for method, path in (("POST", "/api/whoami"), ("GET", "/site/manual/a/b.html"), ("GET", "/site/manual"),
                         ("GET", "/api/reports/" + "a" * 32), ("POST", "/api/reports"), ("POST", "/api/relay/send"),
                         ("GET", "/api/account/groups"), ("GET", "/api/whoami/extra")):
        assert found(method, path) is None, (method, path)


def test_a_kiosk_caller_is_allowed_only_kiosk_entries():
    by = catalog.by_name()
    assert catalog.allowed(by["relay_send"], [], True) and not catalog.allowed(by["relay_send"], [], True, kiosk=True)
    assert catalog.allowed(by["photo_image"], ["photo_read"], True, kiosk=True)
    assert not catalog.allowed(by["photo_image"], [], True, kiosk=True), "kiosk never replaces the capability"
    document = catalog.index(["view_site", "photo_read"], True, kiosk=True)
    assert document["kiosk"] is True and catalog.index(["view_site"], True)["kiosk"] is False
    assert {e["name"] for e in document["endpoints"] if e["allowed"]} == {e["name"] for e in ENDPOINTS if e["kiosk"]}



def test_guest_reachable_entries_and_credentials():
    guest = {e["name"] for e in ENDPOINTS if e["guest"]}
    assert {"play_state", "play_action", "play_new", "identity_signup", "control_route", "control_lifecycle", "play_radio"} <= guest
    assert not any(e["guest"] and e["kiosk"] for e in ENDPOINTS), "kiosk entries are reachable by guests already"
    assert {e["name"]: e["credential"] for e in ENDPOINTS if e["credential"]} == {
        "identity_operator_grant": "admin_token", "control_lifecycle_admin": "admin_token", "play_client_rays_engine": "engine_secret"}
    found = catalog.guest_endpoint
    assert found("POST", "/api/action")["name"] == "play_action"
    assert found("POST", "/api/admin/lifecycle")["name"] == "control_lifecycle_admin"
    assert found("GET", "/api/whoami")["name"] == "identity_whoami"                     # kiosk entries too
    for method, path in (("POST", "/api/reports"), ("POST", "/api/tickets"), ("POST", "/api/obj_assets"), ("GET", "/api/log"),
                         ("POST", "/api/relay/send"), ("GET", "/api/user-files")):
        assert found(method, path) is None, (method, path)
    assert catalog.endpoint_for("GET", "/api/reports/public")["name"] == "reports_public_list"   # the literal template wins
    assert catalog.endpoint_for("POST", "/api/messages/send")["capability"] == "message"
    document = catalog.index(["view_site", "photo_read", "play"], False, guest=True)
    assert document["guest"] is True
    assert {e["name"] for e in document["endpoints"] if e["allowed"]} <= {e["name"] for e in ENDPOINTS if catalog.reachable_by_guest(e)}
