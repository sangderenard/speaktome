"""The SpeakToMe manual: one plain-text page per verb, format and concept, in ``pages/<file>.<section>.txt``.

A page is written as ``man`` shows it: headings at the left margin, text indented. Its NAME paragraph is
``name[, name ...] - one-line summary``; every name listed there finds the page. Nothing here asks the server anything.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

PAGES = Path(__file__).resolve().parent / "pages"
SECTIONS = {"1": "Verbs", "5": "Formats", "7": "Concepts"}
TITLE = "SpeakToMe Manual"
WIDTH = 78
_REFERENCE = re.compile(r"([a-z_]+)\(([0-9])\)")


class ManError(Exception):
    def __init__(self, message: str, nearest: tuple = ()):
        super().__init__(message)
        self.message, self.nearest = message, tuple(nearest)


@dataclass(frozen=True)
class Page:
    file: str            # file stem without the section, e.g. "system"
    section: str
    names: tuple         # every name the NAME paragraph lists
    summary: str
    body: str

    def headings(self) -> list:
        return [line for line in self.body.splitlines() if line and not line[0].isspace()]

    def part(self, heading: str) -> str:
        """The text under one left-margin heading, or '' when the page has none."""
        out, inside = [], False
        for line in self.body.splitlines():
            if line and not line[0].isspace():
                inside = line == heading
                continue
            if inside:
                out.append(line)
        return "\n".join(out).strip("\n")

    def references(self) -> list:
        return _REFERENCE.findall(self.part("SEE ALSO"))

    def render(self) -> str:
        label = f"{self.names[0].upper()}({self.section})"
        gap = max(2, WIDTH - 2 * len(label) - len(TITLE))
        head = label + " " * (gap // 2) + TITLE + " " * (gap - gap // 2) + label
        return head + "\n\n" + self.body.rstrip("\n") + "\n"


def _read(path: Path) -> Page:
    file, section = path.name[:-len(".txt")].rsplit(".", 1)
    body = path.read_text(encoding="utf-8")
    lines = body.splitlines()
    if not lines or lines[0] != "NAME":
        raise ManError(f"{path.name}: a page begins with NAME")
    paragraph = []
    for line in lines[1:]:
        if not line.strip():
            break
        paragraph.append(line.strip())
    names, dash, summary = " ".join(paragraph).partition(" - ")
    if not dash or not summary:
        raise ManError(f"{path.name}: NAME must read 'name - summary'")
    return Page(file, section, tuple(n.strip() for n in names.split(",")), summary, body)


def pages() -> list:
    """Every page, by section and then by first name."""
    found = [_read(path) for path in PAGES.glob("*.[0-9].txt")]
    return sorted(found, key=lambda p: (p.section, p.names[0]))


def page(topic: str, section: str | None = None) -> Page:
    """The page a name finds; the lowest section wins unless one is given."""
    wanted = topic.strip().lower()
    everything = pages()
    for candidate in everything:
        if wanted in candidate.names and section in (None, candidate.section):
            return candidate
    known = sorted({name for p in everything for name in p.names})
    where = f" in section {section}" if section else ""
    raise ManError(f"no such page: {topic}{where}", difflib.get_close_matches(wanted, known, n=3, cutoff=0.5))


def index(keyword: str | None = None) -> list:
    """One row per name: {name, section, summary}; with a keyword, only rows whose name or summary contains it."""
    word = keyword.strip().lower() if keyword else None
    rows = sorted(({"name": name, "section": p.section, "summary": p.summary} for p in pages() for name in p.names),
                  key=lambda row: (row["section"], row["name"]))
    if word:
        rows = [row for row in rows if word in row["name"].lower() or word in row["summary"].lower()]
    return rows


ROOT = "manual:speaktome"
KINDS = {"1": "verb", "5": "format", "7": "concept"}


def place(found: Page) -> str:
    """The identity of a page's place on the manual's map."""
    return f"{ROOT}/{found.section}/{found.file}"


def as_map() -> dict:
    """The manual as a map of the form the living data endpoints accept (``abstract-ui-div-map-v0``).

    The same form the service gives a Nodus canvas: the map's identity is the root, every other place is a flat
    entry of ``regions``, and every relationship is a ``tracks`` record. A section is a place contained by the
    manual; a page is a place contained by its section; each SEE ALSO reference is a ``see-also`` track from page to
    page. Each page is also an object of ``world.objects`` under the same identity, and that object is the page's
    whole record, so the narrator's ``inspect`` returns the text.
    """
    everything = pages()
    regions, tracks, objects = [], [], []
    for section, name in SECTIONS.items():
        identity = f"{ROOT}/{section}"
        regions.append({"identity": identity, "kind": "section", "name": name, "buildings": []})
        tracks.append({"source": identity, "target": ROOT, "relationship": "contained-by"})
    for found in everything:
        identity, kind = place(found), KINDS[found.section]
        targets = [place(page(name, section)) for name, section in found.references()]
        regions.append({"identity": identity, "kind": kind, "name": found.names[0], "buildings": []})
        tracks.append({"source": identity, "target": f"{ROOT}/{found.section}", "relationship": "contained-by"})
        tracks.extend({"source": identity, "target": target, "relationship": "see-also"} for target in targets)
        objects.append({"identity": identity, "kind": kind, "name": found.names[0], "names": list(found.names),
                        "section": found.section, "summary": found.summary, "status": found.part("STATUS"),
                        "text": found.render(),
                        "dependencies": [{"target": target, "relationship": "see-also"} for target in targets]})
    return {"schema": "abstract-ui-div-map-v0", "identity": ROOT, "kind": "manual", "name": TITLE,
            "regions": regions, "tracks": tracks, "world": {"objects": objects}}


def listing(rows: list) -> str:
    return "\n".join(f"{row['name'] + '(' + row['section'] + ')':<24}- {row['summary']}" for row in rows)
