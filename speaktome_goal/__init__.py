"""The Goal document: what a Dream document is processed into.

Owner, 2026-11-11: "dream document gets processed into a Goal document, which can run in a language or as nodus
nodes, the piece by piece compiler will have resolved all the api in general, then if the goal is achieved you have an
actionable system diagram", and of its form: "A new format with linking syntax and abi/api comment tables, ready to be
a graph representation of itself because then anything can work with it".

A Goal document is bytes framed by comment sentinels, so a block's text stays valid inside a comment-tolerant reader:

    /*@goal.document.v1
    id = example
    source = job 02172e9cfcf7
    target | target     | units | lowered | achieved
    target | c          | 3     | 2       | no
    @end*/

    /*@goal.unit.v1
    id = piece-1
    language = python
    state = lowered
    api   | entry | name   | role      | type
    api   | step  | x      | parameter | float
    api   | step  | return | result    | float
    link  | name   | unit    | entry  | how
    link  | double | piece-0 | double | reference
    build | target | entry | symbol     | product | sha256 | record         | note
    build | c      | step  | step__step | step.c  | 9f..   | piece-1.c.step | emitted-not-built
    abi   | target     | entry | name | role  | dtype   | passing | source
    abi   | javascript | step  | t0   | input | float64 | value   | x
    @end*/
    def step(x: float) -> float:
        y = double(x) + 1.0
        return y
    /*@goal.end*/

A header line is either ``key = value`` or a table row ``table | cell | cell ...``. The first row of each table names
its columns, so a reader that knows nothing of this module can still read every table. Cells are trimmed; an empty
cell is written ``-``; a pipe, a backslash or a line break inside a cell is escaped. The tables:

  api     what the unit offers: one row per parameter and result of each entry, as authored.
  link    the linking syntax: a name this unit's text uses, the unit and entry that provide it, and how it is bound
          (``reference``: the text only names it; ``carried``: the provider's definition is still inside this text).
  build   each lowering of an entry that exists: its target language, exported symbol, product and hash.
  abi     how a build is called: one row per parameter of a build, as its compiler described it.
  target  (document block) per target language, how many units have a build for it and whether all do.

A unit's text holds its own definitions. What it needs from another unit is a ``link`` row, not a copy.
A unit's ``state`` is ``lowered`` when at least one build of it exists, otherwise ``source``: it runs as written.
The goal is ACHIEVED for a target when every unit has a build for it. ``as_graph`` and ``as_map`` give the document
as a graph of itself: units are nodes, links are edges, builds and targets are nodes too.

Standard library only. Nothing here compiles, links or runs anything.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

SCHEMA = "speaktome.goal-document.v1"
TABLES = {                                    # table -> its columns, in order
    "api": ("entry", "name", "role", "type"),
    "link": ("name", "unit", "entry", "how"),
    "build": ("target", "entry", "symbol", "product", "sha256", "record", "note"),
    "abi": ("target", "entry", "name", "role", "dtype", "passing", "source"),
    "target": ("target", "units", "lowered", "achieved"),
}
UNIT_TABLES = ("api", "link", "build", "abi")
STATES = ("lowered", "source")
HOWS = ("reference", "carried")
_OPEN = re.compile(rb"/\*@goal\.(document|unit)\.v1[ \t]*\r?\n(.*?)@end\*/", re.DOTALL)
_CLOSE = b"/*@goal.end*/"
_RESERVED = frozenset({"id", "language", "encoding", "sha256", "state"})
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:<>-]{0,127}$")


class GoalError(ValueError):
    pass


def _rows(rows) -> tuple:
    return tuple(MappingProxyType(dict(row)) for row in rows)


@dataclass(frozen=True)
class Unit:
    identity: str
    language: str
    text: str
    state: str = "source"
    encoding: str = "utf-8"
    facts: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    api: tuple = ()
    link: tuple = ()
    build: tuple = ()
    abi: tuple = ()

    def entries(self) -> tuple:
        seen = []
        for row in self.api:
            if row["entry"] not in seen:
                seen.append(row["entry"])
        return tuple(seen)

    def targets(self) -> tuple:
        return tuple(sorted({row["target"] for row in self.build}))


@dataclass(frozen=True)
class GoalDocument:
    identity: str
    units: tuple
    facts: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not _IDENTITY.match(self.identity or ""):
            raise GoalError(f"bad document identity {self.identity!r}")
        known = {}
        for unit in self.units:
            if not _IDENTITY.match(unit.identity or ""):
                raise GoalError(f"bad unit identity {unit.identity!r}")
            if unit.identity in known:
                raise GoalError(f"unit identities must be unique: {unit.identity!r}")
            if unit.state not in STATES:
                raise GoalError(f"unit {unit.identity!r}: state must be one of {', '.join(STATES)}")
            if not unit.language:
                raise GoalError(f"unit {unit.identity!r} requires a language")
            for table in UNIT_TABLES:
                for row in getattr(unit, table):
                    if tuple(row) != TABLES[table]:
                        raise GoalError(f"unit {unit.identity!r}: a {table} row must have the columns {TABLES[table]}")
            known[unit.identity] = unit
        for unit in self.units:
            entries = set(unit.entries())
            for row in unit.link:
                provider = known.get(row["unit"])
                if provider is None:
                    raise GoalError(f"unit {unit.identity!r} links {row['name']!r} to unknown unit {row['unit']!r}")
                if row["entry"] not in provider.entries():
                    raise GoalError(f"unit {unit.identity!r} links {row['name']!r} to {row['unit']}:{row['entry']}, "
                                    "which that unit does not offer")
                if row["how"] not in HOWS:
                    raise GoalError(f"unit {unit.identity!r}: a link is bound by one of {', '.join(HOWS)}")
            built = set()
            for row in unit.build:
                if row["entry"] not in entries:
                    raise GoalError(f"unit {unit.identity!r}: a build names entry {row['entry']!r} it does not offer")
                built.add((row["target"], row["entry"]))
            for row in unit.abi:
                if (row["target"], row["entry"]) not in built:
                    raise GoalError(f"unit {unit.identity!r}: an abi row describes a build that is not listed")
            if (unit.state == "lowered") != bool(unit.build):
                raise GoalError(f"unit {unit.identity!r}: state is lowered exactly when it has a build")

    def unit(self, identity: str) -> Unit:
        for unit in self.units:
            if unit.identity == identity:
                return unit
        raise KeyError(identity)

    def targets(self) -> tuple:
        """Per target language: how many units there are, how many have a build for it, and whether all do."""
        names = sorted({target for unit in self.units for target in unit.targets()})
        total = len(self.units)
        rows = []
        for name in names:
            lowered = sum(1 for unit in self.units if name in unit.targets())
            rows.append({"target": name, "units": str(total), "lowered": str(lowered),
                         "achieved": "yes" if total and lowered == total else "no"})
        return _rows(rows)

    def achieved(self) -> tuple:
        """The targets the goal is achieved for: every unit has a build in that language."""
        return tuple(row["target"] for row in self.targets() if row["achieved"] == "yes")


# ---------------------------------------------------------------------------------------------------------- writing
def _cell(text) -> str:
    out = []
    for ch in str(text):
        if ch == "\\":
            out.append("\\\\")
        elif ch == "|":
            out.append("\\|")
        elif ch == "\n":
            out.append("\\n")
        elif ord(ch) < 0x20 or ord(ch) >= 0x7F:
            out.append("\\u{%x}" % ord(ch))
        else:
            out.append(ch)
    cell = "".join(out).strip()
    if "@end*/" in cell:
        raise GoalError("a header value cannot contain the header's closing mark")
    if cell == "-":
        return "\\-"                               # a real dash; a bare dash is the empty cell
    return cell if cell else "-"


def _uncell(cell: str) -> str:
    cell = cell.strip()
    if cell == "-":
        return ""
    out, i = [], 0
    while i < len(cell):
        ch = cell[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = cell[i + 1:i + 2]
        if nxt in ("\\", "|", "-"):
            out.append(nxt)
            i += 2
        elif nxt == "n":
            out.append("\n")
            i += 2
        elif nxt == "u" and cell[i + 2:i + 3] == "{" and "}" in cell[i + 3:]:
            end = cell.index("}", i + 3)
            out.append(chr(int(cell[i + 3:end], 16)))
            i = end + 1
        else:
            raise GoalError(f"bad escape in header cell {cell!r}")
    return "".join(out)


def _split(line: str) -> list:
    """Cells of a table row: split on every pipe that is not escaped."""
    cells, current, i = [], [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            current.append(line[i:i + 2])
            i += 2
        elif line[i] == "|":
            cells.append("".join(current))
            current = []
            i += 1
        else:
            current.append(line[i])
            i += 1
    cells.append("".join(current))
    return cells


def _fact(key: str, value) -> bytes:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,63}", key) or key in TABLES:
        raise GoalError(f"bad header key {key!r}")
    return f"{key} = {_cell(value)}\n".encode("ascii")


def _table(name: str, rows) -> bytes:
    if not rows:
        return b""
    columns = TABLES[name]
    grid = [list(columns)] + [[_cell(row[column]) for column in columns] for row in rows]
    widths = [max(len(line[i]) for line in grid) for i in range(len(columns))]
    out = []
    for line in grid:
        cells = " | ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
        out.append(f"{name:<6} | {cells}\n")
    return "".join(out).encode("ascii")


def render(document: GoalDocument) -> bytes:
    """The bytes of a Goal document. ``parse(render(d))`` gives back an equal document."""
    out = bytearray(b"/*@goal.document.v1\n")
    out += _fact("id", document.identity) + _fact("schema", document.schema)
    for key in sorted(document.facts):
        if key in ("id", "schema"):
            raise GoalError(f"document facts use the reserved key {key!r}")
        out += _fact(key, document.facts[key])
    out += _table("target", document.targets()) + b"@end*/\n"
    for unit in document.units:
        try:
            payload = unit.text.encode(unit.encoding)
        except (LookupError, UnicodeEncodeError) as error:
            raise GoalError(f"unit {unit.identity!r} cannot be encoded as {unit.encoding!r}") from error
        if _CLOSE in payload:
            raise GoalError(f"unit {unit.identity!r}: the text contains the closing sentinel")
        reserved = _RESERVED & set(unit.facts)
        if reserved:
            raise GoalError(f"unit {unit.identity!r}: facts use reserved keys {sorted(reserved)!r}")
        out += b"\n/*@goal.unit.v1\n"
        out += _fact("id", unit.identity) + _fact("language", unit.language) + _fact("encoding", unit.encoding)
        out += _fact("sha256", hashlib.sha256(payload).hexdigest()) + _fact("state", unit.state)
        for key in sorted(unit.facts):
            out += _fact(key, unit.facts[key])
        for table in UNIT_TABLES:
            out += _table(table, getattr(unit, table))
        out += b"@end*/\n" + payload + (b"\r\n" if payload.endswith(b"\r") else b"\n") + _CLOSE + b"\n"
    return bytes(out)


# ---------------------------------------------------------------------------------------------------------- reading
def _header(raw: bytes, *, where: str):
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise GoalError(f"{where}: headers must be restricted ASCII") from error
    facts, tables = {}, {name: [] for name in TABLES}
    columns = {}
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split("|", 1)[0].strip()
        if "|" in stripped and first in TABLES:
            cells = [_uncell(cell) for cell in _split(stripped)[1:]]
            if first not in columns:
                if tuple(cells) != TABLES[first]:
                    raise GoalError(f"{where}: line {number}: the {first} table's columns must be {TABLES[first]}")
                columns[first] = tuple(cells)
                continue
            if len(cells) != len(columns[first]):
                raise GoalError(f"{where}: line {number}: a {first} row has {len(cells)} cells, not {len(columns[first])}")
            tables[first].append(dict(zip(columns[first], cells)))
            continue
        if "=" not in stripped:
            raise GoalError(f"{where}: line {number} is neither 'key = value' nor a table row")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or key in facts:
            raise GoalError(f"{where}: invalid or duplicate key {key!r}")
        facts[key] = _uncell(value)
    return facts, tables


def parse(source) -> GoalDocument:
    raw = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    position, identity, doc_facts, stated_targets, units = 0, None, {}, None, []
    while position < len(raw):
        match = _OPEN.search(raw, position)
        if match is None:
            if raw[position:].strip():
                raise GoalError("unframed bytes remain after the last block")
            break
        if raw[position:match.start()].strip():
            raise GoalError("every non-whitespace byte must belong to a block")
        kind = match.group(1).decode("ascii")
        if kind == "document":
            if identity is not None:
                raise GoalError("a Goal document has one document block")
            facts, tables = _header(match.group(2), where="document block")
            identity = facts.pop("id", "")
            if facts.pop("schema", SCHEMA) != SCHEMA:
                raise GoalError(f"this reader reads {SCHEMA}")
            doc_facts, stated_targets = facts, tables["target"]
            position = match.end()
            continue
        if identity is None:
            raise GoalError("the document block comes first")
        facts, tables = _header(match.group(2), where="unit block")
        name = facts.pop("id", "")
        close = raw.find(_CLOSE, match.end())
        if close < 0:
            raise GoalError(f"unit {name!r} has no closing sentinel")
        payload = raw[match.end():close]
        payload = payload[2:] if payload.startswith(b"\r\n") else payload[1:] if payload.startswith(b"\n") else payload
        payload = payload[:-2] if payload.endswith(b"\r\n") else payload[:-1] if payload.endswith(b"\n") else payload
        encoding = facts.pop("encoding", "utf-8")
        expected = facts.pop("sha256", None)
        if expected and expected.lower() != hashlib.sha256(payload).hexdigest():
            raise GoalError(f"unit {name!r} failed its sha256")
        try:
            text = payload.decode(encoding)
        except (LookupError, UnicodeDecodeError) as error:
            raise GoalError(f"unit {name!r} cannot be decoded as {encoding!r}") from error
        units.append(Unit(identity=name, language=facts.pop("language", ""), text=text, state=facts.pop("state", "source"),
                          encoding=encoding, facts=MappingProxyType(facts),
                          **{table: _rows(tables[table]) for table in UNIT_TABLES}))
        position = close + len(_CLOSE)
    if identity is None:
        raise GoalError("no document block")
    document = GoalDocument(identity, tuple(units), MappingProxyType(doc_facts))
    if stated_targets is not None and [dict(r) for r in document.targets()] != stated_targets:
        raise GoalError("the target table does not match the units of the document")
    return document


def load(path) -> GoalDocument:
    return parse(Path(path).read_bytes())


# -------------------------------------------------------------------------------------- the document as a graph of itself
def as_graph(document: GoalDocument) -> list:
    """The document as a generic graph: a list of nodes ``{identity, kind, name, dependencies: [{target, relationship}]}``.

    Kinds: goal, target, unit, build. Relationships: ``contained-by`` (unit -> goal, build -> unit), ``links-to``
    (unit -> the unit that provides a name it uses), ``built-for`` (build -> target), ``achieves`` (goal -> each target
    every unit is built for)."""
    root = f"goal:{document.identity}"
    nodes = [{"identity": root, "kind": "goal", "name": document.identity,
              "dependencies": [{"target": f"{root}/target/{name}", "relationship": "achieves"} for name in document.achieved()]}]
    for row in document.targets():
        nodes.append({"identity": f"{root}/target/{row['target']}", "kind": "target", "name": row["target"],
                      "dependencies": [{"target": root, "relationship": "contained-by"}]})
    for unit in document.units:
        here = f"{root}/unit/{unit.identity}"
        links = []
        for row in unit.link:
            edge = {"target": f"{root}/unit/{row['unit']}", "relationship": "links-to"}
            if edge not in links:
                links.append(edge)
        nodes.append({"identity": here, "kind": "unit", "name": ", ".join(unit.entries()) or unit.identity,
                      "dependencies": [{"target": root, "relationship": "contained-by"}] + links})
        for row in unit.build:
            nodes.append({"identity": f"{here}/build/{row['target']}/{row['entry']}", "kind": "build",
                          "name": f"{row['entry']} in {row['target']}",
                          "dependencies": [{"target": here, "relationship": "contained-by"},
                                           {"target": f"{root}/target/{row['target']}", "relationship": "built-for"}]})
    return nodes


def as_map(document: GoalDocument) -> dict:
    """The same graph as an ``abstract-ui-div-map-v0`` map, the form the living data endpoints and the site's pages
    accept: the goal is the root, every other node a flat region, every relationship a track, and each unit is also
    an object of the world whose record holds its text and tables. So the diagram can be walked, examined and baked."""
    nodes = as_graph(document)
    root = nodes[0]["identity"]
    achieved = document.achieved()
    regions, tracks, objects = [], [], []
    for node in nodes[1:]:
        regions.append({"identity": node["identity"], "kind": node["kind"], "name": node["name"], "buildings": []})
    for node in nodes:
        for dependency in node["dependencies"]:
            tracks.append({"source": node["identity"], "target": dependency["target"],
                           "relationship": dependency["relationship"]})
    for unit in document.units:
        header = render(GoalDocument("unit", (Unit(**{**_unit_fields(unit), "link": (), "facts": MappingProxyType({})}),)))
        table_text = header.decode("ascii", "replace").split("/*@goal.unit.v1\n", 1)[1].split("@end*/", 1)[0]
        links = "".join(f"link  | {row['name']} -> {row['unit']} : {row['entry']} ({row['how']})\n" for row in unit.link)
        objects.append({"identity": f"{root}/unit/{unit.identity}", "kind": "unit", "name": ", ".join(unit.entries()) or unit.identity,
                        "summary": f"{unit.language} unit, {unit.state}; built for: {', '.join(unit.targets()) or 'nothing'}",
                        "state": unit.state, "language": unit.language,
                        "text": table_text + links + "\n" + unit.text})
    return {"schema": "abstract-ui-div-map-v0", "identity": root, "kind": "goal",
            "name": f"Goal {document.identity}: " + (f"achieved for {', '.join(achieved)}" if achieved else "not achieved"),
            "regions": regions, "tracks": tracks, "world": {"objects": objects}}


def _unit_fields(unit: Unit) -> dict:
    return {"identity": unit.identity, "language": unit.language, "text": "", "state": unit.state, "encoding": unit.encoding,
            "facts": unit.facts, "api": unit.api, "link": unit.link, "build": unit.build, "abi": unit.abi}
