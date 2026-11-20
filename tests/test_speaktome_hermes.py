"""speaktome_hermes against a fake REST layer: no network, no Hermes run."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HERMES = Path(r"C:\dev\Powershell\srv\tools\hermes-agent\0.19.0\venv\Lib\site-packages")
try:
    import agent.memory_provider  # noqa: F401
except ImportError:
    sys.path.append(str(HERMES))

from speaktome_hermes import SpeakToMeMemoryProvider, HEADER  # noqa: E402
from speaktome_mcp.rest import RestError  # noqa: E402

ID = "a" * 32


class FakeRest:
    def __init__(self, files=None):
        self.files = dict(files or {})           # id -> (title, body, updated)
        self.reports = {}
        self.calls = []

    def call(self, path, body=None, timeout=40, raw_limit=None, *, method=None, headers=None):
        self.calls.append((path, method, headers))
        if path == "/api/memory" and body is None:
            return {"files": [{"id": i, "title": t, "updated": u} for i, (t, _, u) in self.files.items()]}
        if path.startswith("/api/memory/file/"):
            return {"body": self.files[path.rsplit("/", 1)[1]][1]}
        if path == "/api/memory":
            assert headers == {"X-Meowdoku-Workspace": "1"} and body["action"] == "write"
            self.files[body["id"]] = (body["title"], body["body"], 1e9)
            return {}
        if path == "/api/reports":
            assert set(body) == {"request_id", "category", "title", "body", "prompt_history", "next_steps"}
            if body["request_id"] in self.reports and self.reports[body["request_id"]] != body:
                raise RestError("request_id reused with different report", 409)
            self.reports[body["request_id"]] = body
            return {"id": "r"}
        raise AssertionError(path)


def provider(rest):
    p = SpeakToMeMemoryProvider(rest=rest)
    p.initialize("sess-1", hermes_home="x", platform="cli")
    return p


def test_reads_account_memory_and_fits_budget():
    lines = "".join(f"line {i:04d}\n" for i in range(2000))
    rest = FakeRest({ID: ("m", HEADER + lines, 5), "b" * 32: ("old", "stale", 1)})
    p = provider(rest)
    p.budget = 300
    block = p.system_prompt_block()
    assert "line 1999" in block and "line 0000" not in block and "omitted" in block
    assert len(block.encode()) < 700
    assert p.prefetch("hi") == ""                      # unchanged since the prompt showed it


def test_writes_go_to_memory_change_and_create_when_empty():
    rest = FakeRest()
    p = provider(rest)
    assert p.is_available()
    out = json.loads(p.handle_tool_call("speaktome_memory_append", {"text": "owner likes tea"}))
    assert out == {"saved": True}
    (mid, (title, body, _)), = rest.files.items()
    assert len(mid) == 32 and body.startswith(HEADER) and body.rstrip().endswith("owner likes tea")
    p.on_memory_write("add", "memory", "second fact")
    assert rest.files[mid][1].count("\n") == 3
    assert "second fact" in p.prefetch("q")            # changed since the prompt: returned again
    p.handle_tool_call("speaktome_memory_rewrite", {"body": "only this\n"})
    assert rest.files[mid][1] == HEADER + "only this\n"


def test_subagent_is_read_only():
    rest = FakeRest()
    p = SpeakToMeMemoryProvider(rest=rest)
    p.initialize("s", agent_context="subagent")
    assert p.get_tool_schemas() == []
    p.on_session_end([{"role": "user", "content": "x"}])
    assert not rest.reports


def test_session_end_files_one_stable_report():
    rest = FakeRest()
    p = provider(rest)
    msgs = [{"role": "user", "content": "build the thing"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "terminal"}}]},
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "and test it"}]},
            {"role": "assistant", "content": "Done and tested.\n\nNext steps:\n- ship it\n- document it\n"}]
    p.on_session_end(msgs)
    p.reported = False                                  # a retry of the same session
    p.on_session_end(msgs)
    (rid, rep), = rest.reports.items()
    assert rid == p.request_id() and len(rid) <= 64
    assert rep["category"] == "LOG" and rep["prompt_history"] == ["build the thing", "and test it"]
    assert rep["next_steps"] == ["ship it", "document it"] and "terminal x1" in rep["body"]
    p.on_session_end([])                                # nothing said: nothing filed
    assert len(rest.reports) == 1
