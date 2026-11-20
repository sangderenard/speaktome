"""Hermes Agent memory-provider plugin: the signed-in SpeakToMe account's memory file IS Hermes' memory.

Hermes keeps no memory of its own. The provider (a MemoryProvider, agent/memory_provider.py) reads the account's
newest memory file through the existing REST client (speaktome_mcp.rest.Rest: SPEAKTOME_BASE / SPEAKTOME_USER /
SPEAKTOME_SECRET_FILE), writes through POST /api/memory (the memory_change verb), and on session end files one LOG
experience report (POST /api/reports, the reports_make verb) with a request_id stable for the Hermes session.

Install: HERMES_HOME/plugins/speaktome -> this directory; config.yaml: memory: {provider: speaktome}.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:
    from speaktome_mcp.rest import Rest, RestError
except ImportError:                      # the plugin is a copy or link under HERMES_HOME; find the repository root
    for _root in (os.environ.get("SPEAKTOME_ROOT"), str(Path(__file__).resolve().parent.parent)):
        if _root and _root not in sys.path and (Path(_root) / "speaktome_mcp").is_dir():
            sys.path.insert(0, _root)
    from speaktome_mcp.rest import Rest, RestError

logger = logging.getLogger(__name__)

WORKSPACE = {"X-Meowdoku-Workspace": "1"}          # the header /api/memory writes insist on (catalog.py memory_change)
FILE_BYTES = 131072                                 # server limit per memory file (MAX_SCRATCH_FILE_BYTES)
STORED_BYTES = 32768                                # what we keep in the file: oldest lines drop first
BUDGET_BYTES = 6000                                 # what one prompt carries (SPEAKTOME_MEMORY_BUDGET overrides)
MEMORY_TITLE = "Hermes memory"
HEADER = "# Memory (SpeakToMe account file; newest lines last)\n"
REPORT_BODY_CHARS = 12000
PROMPT_CHARS = 4000


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") in (None, "text", "input_text", "output_text"))
    return ""


def _fit(body: str, budget: int) -> str:
    """The newest whole lines of ``body`` that fit ``budget`` UTF-8 bytes (the file is a log: newest last)."""
    if len(body.encode("utf-8")) <= budget:
        return body
    kept: List[str] = []
    used = 0
    for line in reversed(body.splitlines()):
        size = len(line.encode("utf-8")) + 1
        if used + size > budget:
            break
        kept.append(line)
        used += size
    return "[older memory lines omitted to fit the budget]\n" + "\n".join(reversed(kept))


def _bounded(body: str, limit: int) -> str:
    """Drop the oldest lines until the file fits ``limit`` bytes, keeping the header."""
    text = body[len(HEADER):] if body.startswith(HEADER) else body
    lines = [line for line in text.splitlines() if line.strip()]
    while lines and len((HEADER + "\n".join(lines) + "\n").encode("utf-8")) > limit:
        lines.pop(0)
    return HEADER + "\n".join(lines) + "\n"


class SpeakToMeMemoryProvider(MemoryProvider):
    def __init__(self, rest=None, clock=time.time):
        self._rest, self._clock = rest, clock
        self.session_id = ""
        self.writable = True
        self.memory_id: Optional[str] = None
        self.body = ""
        self.loaded = False
        self.shown = ""                      # what system_prompt_block last carried
        self.reported = False
        self.budget = int(os.environ.get("SPEAKTOME_MEMORY_BUDGET") or BUDGET_BYTES)

    # -- required -----------------------------------------------------------
    @property
    def name(self) -> str:
        return "speaktome"

    def is_available(self) -> bool:                       # no network: only that the account is named
        return self._rest is not None or bool(os.environ.get("SPEAKTOME_BASE"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or ""
        self.writable = kwargs.get("agent_context", "primary") == "primary"
        self.reported = False
        self.memory_id = os.environ.get("SPEAKTOME_MEMORY_ID") or None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self.writable:
            return []
        return [
            {"name": "speaktome_memory_append",
             "description": "Remember one fact or note: appends a line to the signed-in SpeakToMe account's memory file.",
             "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The note, one or a few lines."}}, "required": ["text"]}},
            {"name": "speaktome_memory_rewrite",
             "description": "Replace the whole account memory file (use to tidy or correct it). Keep the lines that still matter.",
             "parameters": {"type": "object", "properties": {"body": {"type": "string", "description": "The complete new memory text."}}, "required": ["body"]}},
        ]

    # -- REST ---------------------------------------------------------------
    def _client(self):
        if self._rest is None:
            self._rest = Rest.from_env()
        return self._rest

    def _load(self, force: bool = False) -> str:
        if self.loaded and not force:
            return self.body
        try:
            rest = self._client()
            if self.memory_id is None:
                files = (rest.call("/api/memory") or {}).get("files") or []
                newest = max(files, key=lambda f: f.get("updated") or 0, default=None)
                self.memory_id = newest.get("id") if newest else None
            if self.memory_id:
                got = rest.call(f"/api/memory/file/{self.memory_id}") or {}
                if isinstance(got.get("body"), str):
                    self.body = got["body"]
            self.loaded = True
        except (RestError, OSError, ValueError, AttributeError) as error:
            logger.warning("speaktome memory not read: %s", error)
        return self.body

    def _write(self, body: str) -> bool:
        if not self.writable:
            return False
        self._load()
        created = self.memory_id is None
        memory_id = self.memory_id or os.urandom(16).hex()
        body = _bounded(body, STORED_BYTES)
        try:
            self._client().call("/api/memory", {"action": "write", "id": memory_id, "create": created, "title": MEMORY_TITLE,
                                                "body": body}, method="POST", headers=WORKSPACE)
        except (RestError, OSError, ValueError) as error:
            logger.warning("speaktome memory not saved: %s", error)
            return False
        self.memory_id, self.body, self.loaded = memory_id, body, True
        return True

    def _append(self, text: str) -> bool:
        line = " ".join(text.split())
        if not line:
            return False
        stamp = time.strftime("%Y-%m-%d %H:%M", time.gmtime(self._clock()))
        self._load(force=True)                            # fold in edits made elsewhere before adding to them
        return self._write((self.body if self.body.startswith(HEADER) else HEADER + self.body.lstrip("\n")).rstrip("\n") + f"\n{stamp} {line}\n")

    # -- recall -------------------------------------------------------------
    def system_prompt_block(self) -> str:
        body = _fit(self._load(), self.budget).strip()
        self.shown = self.body
        return ("Your memory is the signed-in SpeakToMe account's memory file; tools speaktome_memory_append and "
                "speaktome_memory_rewrite change it.\n" + (body or "(empty)"))

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """The memory file when it changed since the system prompt showed it (the prompt is cached for the session)."""
        body = self._load(force=True)
        if body == self.shown:
            return ""
        return "Updated account memory:\n" + _fit(body, self.budget).strip()

    # -- writes -------------------------------------------------------------
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "speaktome_memory_append":
            ok = self._append(str(args.get("text", "")))
        elif tool_name == "speaktome_memory_rewrite":
            ok = bool(str(args.get("body", "")).strip()) and self._write(str(args["body"]))
        else:
            return json.dumps({"error": f"unknown tool {tool_name}"})
        return json.dumps({"saved": ok})

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if action == "add" and content:                   # Hermes' built-in memory tool: mirror into the account file
            self._append(content)

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs) -> None:
        self.session_id = new_session_id or self.session_id
        if reset:
            self.reported = False

    # -- experience report --------------------------------------------------
    def request_id(self) -> str:
        return "hermes-" + hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:32]

    def build_report(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        users = [_text(m.get("content")).strip() for m in messages if m.get("role") == "user"]
        users = [u for u in users if u]
        if not users:
            return None
        assistants = [m for m in messages if m.get("role") == "assistant"]
        tools: Dict[str, int] = {}
        for m in assistants:
            for call in m.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name") or call.get("name") or "?"
                tools[name] = tools.get(name, 0) + 1
        final = next((t for t in (_text(m.get("content")).strip() for m in reversed(assistants)) if t), "")
        steps: List[str] = []
        grab = False
        for line in final.splitlines():
            plain = line.strip().lstrip("#*- ").strip()
            if "next step" in plain.lower() or plain.lower().startswith("next:"):
                grab = True
                tail = plain.split(":", 1)[1].strip() if ":" in plain else ""
                if tail:
                    steps.append(tail[:2000])
            elif grab and line.strip()[:1] in "-*0123456789":
                steps.append(line.strip().lstrip("-*0123456789.) ").strip()[:2000])
            elif grab and line.strip():
                grab = False
        first = " ".join(users[0].split())
        history = [u[:PROMPT_CHARS] for u in (users if len(users) <= 32 else users[:8] + users[-24:])]
        done = "; ".join(f"{n} x{c}" for n, c in sorted(tools.items())) or "none"
        body = (f"Hermes session {self.session_id}: {len(users)} user message(s), {len(assistants)} assistant message(s).\n"
                f"Tools used: {done}.\n\nWhat was asked first:\n{first[:1500]}\n\nState at the end (last assistant message):\n"
                f"{final[:REPORT_BODY_CHARS] or '(none)'}")
        return {"request_id": self.request_id(), "category": "LOG", "title": ("Hermes session: " + first)[:200],
                "body": body[:REPORT_BODY_CHARS + 3000], "prompt_history": history, "next_steps": steps[:32]}

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self.writable or self.reported:
            return
        report = self.build_report(messages or [])
        if report is None:
            return
        try:
            self._client().call("/api/reports", report, method="POST")
            self.reported = True
        except RestError as error:
            if error.status == 409:                       # same request_id already filed with other text: leave it
                self.reported = True
            logger.warning("speaktome report not filed: %s", error)
        except (OSError, ValueError) as error:
            logger.warning("speaktome report not filed: %s", error)


def register(ctx) -> None:
    ctx.register_memory_provider(SpeakToMeMemoryProvider())
