"""Because we have a graph, we can make a website.

Every place of a map becomes one HTML page and every declared edge becomes one link. A page is rendered the first
time someone looks at it and then kept on disk; the whole map can also be baked ahead of time into a folder of
static files. Both use the same renderer, ``Atlas.render``.

The host injects its graph indexer (``graph_view(map_document, layer)`` with ``.nodes``, ``.adj`` and ``.node``);
this module never parses regions or tracks itself. Everything taken from the map is data and is escaped.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_ADDRESS = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_SLUG = 80
_STEPS = 32
_MEMO = 4
INDEX = "index"
SENTENCE = "Generated from a graph: each place is a page, each edge is a link."
_SKIP = {"identity", "name", "kind", "summary", "text"}


def address(identity: str) -> str:
    """A stable file-safe name for a place: a slug of the identity plus a short hash of the whole identity."""
    slug = _UNSAFE.sub("_", identity)[:_SLUG]
    if slug.startswith("."):
        slug = "_" + slug
    return slug + "-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]


def digest(map_document: dict) -> str:
    text = json.dumps(map_document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _e(value) -> str:
    return html.escape(str(value))


_STYLE = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--dim:#6b6a63;--link:#1a52a8;--box:#efede6;--rule:#d9d6cb}
@media (prefers-color-scheme: dark){:root{--bg:#161614;--fg:#e8e6df;--dim:#a09e94;--link:#8db4f0;--box:#23231f;--rule:#3a3934}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:46rem;margin:0 auto;padding:16px}
a{color:var(--link)}
nav{color:var(--dim);font-size:.9rem;overflow-wrap:anywhere}
h1{margin:.6em 0 .1em;overflow-wrap:anywhere}
h2{font-size:1rem;margin:1.4em 0 .3em;border-bottom:1px solid var(--rule)}
small,.id{color:var(--dim);font-size:.8rem;overflow-wrap:anywhere}
pre{background:var(--box);padding:12px;overflow-x:auto;max-width:100%;font-size:.85rem}
dt{font-weight:600;margin-top:.5em}
dd{margin:0;overflow-wrap:anywhere}
ul{padding-left:1.2rem}
li{margin:.3em 0;overflow-wrap:anywhere}
footer{margin-top:2.5em;padding-top:.8em;border-top:1px solid var(--rule);color:var(--dim);font-size:.8rem}
""".strip()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


class Atlas:
    def __init__(self, site: "Site", name: str, map_document: dict):
        self._site = site
        self.name = name
        self.digest = digest(map_document)
        self._map = map_document
        self._layout = site._graph_view(map_document, "layout")
        self._world = site._graph_view(map_document, "world")
        root = map_document.get("identity") if isinstance(map_document, dict) else None
        self.root = root if isinstance(root, str) else ""
        self._addresses = {address(identity): identity for identity in self._layout.nodes}
        self._folder = Path(site.cache_dir) / name / self.digest
        self._lock = threading.Lock()

    def addresses(self) -> dict:
        return dict(self._addresses)

    # -- rendering -------------------------------------------------------------------------------------------
    def _label(self, identity: str) -> str:
        record = self._layout.nodes.get(identity)
        if isinstance(record, dict) and isinstance(record.get("name"), str) and record["name"]:
            return record["name"]
        return identity

    def _link(self, identity: str) -> str:
        return f'<a href="{_e(address(identity))}.html">{_e(self._label(identity))}</a>'

    def _crumbs(self, identity: str) -> list:
        chain, seen, here = [], {identity}, identity
        for _ in range(_STEPS):
            parent = None
            for edge in self._layout.adj.get(here, []):
                if edge["source"] == here and edge["relationship"] == "contained-by":
                    parent = edge["target"]
                    break
            if parent is None or parent in seen:
                break
            seen.add(parent)
            chain.append(parent)
            here = parent
        return list(reversed(chain))

    def render(self, identity: str) -> str:
        record = self._layout.nodes.get(identity)
        if not isinstance(record, dict):
            raise KeyError(identity)
        label = self._label(identity)
        kind = record.get("kind") if isinstance(record.get("kind"), str) else ""
        out = ["<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
               '<meta name="viewport" content="width=device-width, initial-scale=1">',
               f"<title>{_e(label)} - {_e(self._site.title)}</title>", f"<style>\n{_STYLE}\n</style>",
               "</head>", "<body>", "<main>"]
        crumbs = self._crumbs(identity)
        if crumbs:
            parts = [self._link(c) if c in self._layout.nodes else _e(c) + " (not on this map)" for c in crumbs]
            out.append('<nav aria-label="Breadcrumb">' + " / ".join(parts) + "</nav>")
        out.append(f"<h1>{_e(label)}</h1>")
        small = (_e(kind) + " &middot; " if kind else "") + _e(identity)
        out.append(f'<p class="id">{small}</p>')
        world = self._world.nodes.get(identity)
        if isinstance(world, dict):
            if isinstance(world.get("summary"), str):
                out.append(f"<p>{_e(world['summary'])}</p>")
            if isinstance(world.get("text"), str):
                out.append(f"<pre>{_e(world['text'])}</pre>")
            fields = [(k, v) for k, v in world.items()
                      if k not in _SKIP and isinstance(k, str) and isinstance(v, (str, int, float, bool))]
            if fields:
                out.append("<dl>")
                for key, value in fields:
                    out.append(f"<dt>{_e(key)}</dt><dd>{_e(value)}</dd>")
                out.append("</dl>")
        groups = OrderedDict()
        for edge in self._layout.adj.get(identity, []):
            direction = "outgoing" if edge["source"] == identity else "incoming"
            groups.setdefault((edge["relationship"], direction), []).append(edge)
        if groups:
            out.append("<h2>Exits</h2>")
        for (relationship, direction), edges in groups.items():
            out.append(f"<h3>{_e(relationship)} ({direction})</h3>")
            out.append("<ul>")
            for edge in edges:
                other = edge["target"] if edge["source"] == identity else edge["source"]
                if other in self._layout.nodes:
                    out.append(f'<li>{self._link(other)} <small>{_e(other)}</small></li>')
                else:
                    out.append(f"<li>{_e(other)} <small>not on this map</small></li>")
            out.append("</ul>")
        map_name = self._map.get("name") if isinstance(self._map, dict) else None
        if not isinstance(map_name, str) or not map_name:
            map_name = self.root
        out.append(f"<footer>{_e(map_name)} &middot; {_e(self.digest[:12])}<br>{_e(SENTENCE)}</footer>")
        out.extend(["</main>", "</body>", "</html>"])
        return "\n".join(out) + "\n"

    # -- cache-through ---------------------------------------------------------------------------------------
    def _identity_of(self, addr: str) -> str:
        if not isinstance(addr, str) or not _ADDRESS.match(addr):
            raise KeyError(addr)
        if addr == INDEX:
            return self.root
        if addr not in self._addresses:
            raise KeyError(addr)
        return self._addresses[addr]

    def page(self, addr: str) -> bytes:
        identity = self._identity_of(addr)
        path = self._folder / (addr + ".html")
        with self._lock:
            try:
                return path.read_bytes()
            except OSError:
                pass
            data = self.render(identity).encode("utf-8")
            _write_atomic(path, data)
            return data

    def bake(self, out_dir=None) -> int:
        folder = Path(out_dir) if out_dir is not None else self._folder
        folder.mkdir(parents=True, exist_ok=True)
        count = 0
        for addr, identity in self._addresses.items():
            _write_atomic(folder / (addr + ".html"), self.render(identity).encode("utf-8"))
            count += 1
        _write_atomic(folder / "index.html", self.render(self.root).encode("utf-8"))
        return count + 1


class Site:
    def __init__(self, cache_dir, graph_view, *, keep: int = 2, title: str = "SpeakToMe"):
        self.cache_dir = Path(cache_dir)
        self._graph_view = graph_view
        self.keep = max(1, int(keep))
        self.title = title
        self._atlases = OrderedDict()
        self._current = {}
        self._lock = threading.Lock()

    def open(self, name: str, map_document: dict) -> Atlas:
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError("name must match ^[a-z0-9][a-z0-9-]{0,31}$")
        key = (name, digest(map_document))
        with self._lock:
            found = self._atlases.get(key)
            if found is not None:
                self._atlases.move_to_end(key)
                return found
            atlas = Atlas(self, name, map_document)
            atlas._folder.mkdir(parents=True, exist_ok=True)
            self._prune(name, atlas.digest)
            self._atlases[key] = atlas
            while len(self._atlases) > _MEMO:
                self._atlases.popitem(last=False)
            return atlas

    def atlas(self, name: str, source, ttl: float = 0.0) -> Atlas:
        """The current Atlas of a named map. ``source()`` gives the map; it is asked again once ``ttl`` seconds pass."""
        now = time.monotonic()
        with self._lock:
            held = self._current.get(name)
        if held is not None and now < held[0]:
            return held[1]
        opened = self.open(name, source())
        with self._lock:
            self._current[name] = (now + max(0.0, float(ttl)), opened)
        return opened

    def _prune(self, name: str, current: str) -> None:
        base = self.cache_dir / name
        found = []
        for entry in base.iterdir():
            if _DIGEST.match(entry.name) and entry.is_dir() and not entry.is_symlink():
                found.append((entry.stat().st_mtime, entry.name))
        found.sort(reverse=True)
        # the current folder always stays; fill the remaining slots with the newest others
        others = [n for _, n in found if n != current][:max(0, self.keep - 1)]
        kept = {current, *others}
        for _, entry_name in found:
            if entry_name not in kept:
                target = base / entry_name
                if target.parent == base:
                    shutil.rmtree(target, ignore_errors=True)
