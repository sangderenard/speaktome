import html
import importlib.util
import os
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import speaktome_man  # noqa: E402
from speaktome_api import site as S  # noqa: E402

_REAL = ROOT.parent / "turing" / "examples" / "meowdoku" / "living_data_query.py"


def _load_real():
    spec = importlib.util.spec_from_file_location("living_data_query_for_site_tests", _REAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphView


class DoubleView:
    """Same interface as the host's GraphView, for when the other repository is absent."""

    def __init__(self, model, layer="layout"):
        if not isinstance(model, dict) or model.get("schema") != "abstract-ui-div-map-v0":
            raise ValueError("bad map")
        self.nodes, self.adj = {}, {}
        todo = [(model, "regions" if layer == "layout" else None)]
        order = {"regions": "buildings", "buildings": "rooms", "rooms": None}
        while todo:
            node, kids = todo.pop(0)
            self.nodes[node["identity"]] = node
            self.adj[node["identity"]] = []
            if kids:
                todo.extend((k, order[kids]) for k in node.get(kids, []))
        if layer == "world":
            for obj in model.get("world", {}).get("objects", []):
                self.nodes[obj["identity"]] = obj
                self.adj[obj["identity"]] = []
        seen = set()

        def edge(s, t, r):
            if (s, t, r) in seen:
                return
            seen.add((s, t, r))
            rec = {"source": s, "target": t, "relationship": r}
            for i in {s, t}:
                self.adj.setdefault(i, []).append(rec)
        for identity, node in list(self.nodes.items()):
            for dep in node.get("dependencies", []):
                edge(identity, dep["target"], dep["relationship"])
        if layer == "layout":
            for t in model.get("tracks", []):
                edge(t["source"], t["target"], t["relationship"])

    def node(self, identity):
        return {"identity": identity, "known": identity in self.nodes}


GraphView = _load_real() if _REAL.exists() else DoubleView


def make_site(tmp_path, **kw):
    return S.Site(tmp_path / "cache", GraphView, **kw)


def hostile_map():
    return {"schema": "abstract-ui-div-map-v0", "identity": "root", "name": "Root",
            "regions": [{"identity": "r/<b>\"x\"", "kind": "room", "name": "<script>alert(\"hi\")</script>"}],
            "tracks": [{"source": "r/<b>\"x\"", "target": "root", "relationship": "contained-by"},
                       {"source": "r/<b>\"x\"", "target": "gone<i>", "relationship": "see-also"}],
            "world": {"objects": [{"identity": "r/<b>\"x\"", "name": "n", "summary": "<script>s</script>",
                                   "text": "<script>alert('t')</script> \"q\"", "note": "a\"b<c"}]}}


def small_map(extra=None):
    doc = {"schema": "abstract-ui-div-map-v0", "identity": "house", "name": "The House",
           "regions": [{"identity": "house/hall", "name": "Hall",
                        "dependencies": [{"target": "house/stair", "relationship": "leads-to"}]},
                       {"identity": "house/stair", "name": "Stair"}]}
    if extra:
        doc["regions"].append({"identity": extra, "name": extra})
    return doc


# -- address -------------------------------------------------------------------------------------------------
def test_address_stable_safe_distinct():
    assert S.address("a/b") == S.address("a/b")
    assert S.address("a/b") != S.address("a:b")
    assert re.fullmatch(r"[A-Za-z0-9._-]+-[0-9a-f]{10}", S.address("we ird/é<>\"id"))
    assert not S.address(".hidden").startswith(".")
    assert not S.address("..").startswith(".")
    long = S.address("x" * 500)
    assert len(long) == 80 + 11


def test_digest_is_sha256_hex():
    d = S.digest(small_map())
    assert re.fullmatch(r"[0-9a-f]{64}", d)
    assert d != S.digest(small_map("extra"))


# -- the manual ----------------------------------------------------------------------------------------------
def test_manual_pages(tmp_path):
    doc = speaktome_man.as_map()
    atlas = make_site(tmp_path).open("manual", doc)
    assert atlas.root == speaktome_man.ROOT
    addrs = atlas.addresses()
    assert len(addrs) == len(atlas._layout.nodes)
    for addr in addrs:
        assert atlas.page(addr).startswith(b"<!DOCTYPE html>")
    assert atlas.page("index") == atlas.page(S.address(atlas.root))

    go = next(o for o in doc["world"]["objects"] if o["identity"].endswith("/1/go"))
    text = atlas.page(S.address(go["identity"])).decode("utf-8")
    assert html.escape(go["text"]) in text
    targets = [t["target"] for t in doc["tracks"]
               if t["source"] == go["identity"] and t["relationship"] == "see-also"]
    assert targets
    for target in targets:
        assert f'href="{S.address(target)}.html"' in text
    assert "see-also (outgoing)" in text
    # breadcrumbs: section then manual root
    assert f'href="{S.address(speaktome_man.ROOT + "/1")}.html"' in text
    assert f'href="{S.address(speaktome_man.ROOT)}.html"' in text
    assert "<script" not in text


# -- cache ---------------------------------------------------------------------------------------------------
def test_cache_through(tmp_path):
    atlas = make_site(tmp_path).open("house", small_map())
    calls = []
    original = atlas.render
    atlas.render = lambda identity: (calls.append(identity), original(identity))[1]
    addr = S.address("house/hall")
    first = atlas.page(addr)
    path = tmp_path / "cache" / "house" / atlas.digest / (addr + ".html")
    assert path.read_bytes() == first
    assert atlas.page(addr) == first
    assert len(calls) == 1
    path.write_bytes(b"changed")
    assert atlas.page(addr) == b"changed"
    assert len(calls) == 1
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".tmp-")]


def test_open_memoizes(tmp_path):
    site = make_site(tmp_path)
    assert site.open("house", small_map()) is site.open("house", small_map())


# -- bake ----------------------------------------------------------------------------------------------------
def test_bake_links_resolve(tmp_path):
    atlas = make_site(tmp_path).open("manual", speaktome_man.as_map())
    out = tmp_path / "baked"
    written = atlas.bake(out)
    files = {p.name for p in out.iterdir()}
    assert "index.html" in files
    assert written == len(atlas.addresses()) + 1
    for addr in atlas.addresses():
        assert addr + ".html" in files
    assert len(files) == len(atlas.addresses()) + 1
    hrefs = 0
    for p in out.iterdir():
        for href in re.findall(r'href="([^"]*)"', p.read_text(encoding="utf-8")):
            hrefs += 1
            assert href in files, (p.name, href)
    assert hrefs > 0
    assert (out / "index.html").read_bytes() == atlas.page("index")


def test_bake_default_fills_cache(tmp_path):
    atlas = make_site(tmp_path).open("house", small_map())
    atlas.bake()
    folder = tmp_path / "cache" / "house" / atlas.digest
    assert (folder / "index.html").exists()
    assert (folder / (S.address("house/stair") + ".html")).exists()


# -- pruning -------------------------------------------------------------------------------------------------
def test_prune_old_digests(tmp_path):
    site = make_site(tmp_path, keep=2)
    outside = tmp_path / "cache" / "other" / ("0" * 64)
    outside.mkdir(parents=True)
    now = time.time()
    folders = []
    for age, extra in ((300, "one"), (200, "two")):
        atlas = site.open("house", small_map(extra))
        folder = tmp_path / "cache" / "house" / atlas.digest
        os.utime(folder, (now - age, now - age))
        folders.append(folder)
    assert all(f.exists() for f in folders)
    third = site.open("house", small_map("three"))
    assert (tmp_path / "cache" / "house" / third.digest).exists()
    assert not folders[0].exists()
    assert folders[1].exists()
    assert outside.exists()


# -- hostile data --------------------------------------------------------------------------------------------
def test_hostile_escaped(tmp_path):
    atlas = make_site(tmp_path).open("evil", hostile_map())
    identity = 'r/<b>"x"'
    text = atlas.page(S.address(identity)).decode("utf-8")
    assert "<script" not in text
    assert "<b>" not in text
    assert "&lt;script&gt;alert(&quot;hi&quot;)&lt;/script&gt;" in text
    assert html.escape("<script>alert('t')</script> \"q\"") in text
    assert html.escape('a"b<c') in text
    assert "not on this map" in text
    assert "gone&lt;i&gt;" in text
    assert "gone<i>" not in text


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "a\\b", "unknown", "", "x" * 300, "house-0000000000", None])
def test_bad_address_raises_and_touches_nothing(tmp_path, bad):
    atlas = make_site(tmp_path).open("house", small_map())
    with pytest.raises(KeyError):
        atlas.page(bad)
    folder = tmp_path / "cache" / "house" / atlas.digest
    assert list(folder.iterdir()) == []
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_render_unknown_raises(tmp_path):
    atlas = make_site(tmp_path).open("house", small_map())
    with pytest.raises(KeyError):
        atlas.render("nope")


@pytest.mark.parametrize("name", ["", "Upper", "-x", "a b", "a/b", "..", "x" * 33])
def test_bad_name(tmp_path, name):
    with pytest.raises(ValueError):
        make_site(tmp_path).open(name, small_map())
