"""The manual stays whole: every page is well formed, every reference resolves, and the verbs and the tools agree."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import speaktome_man as manual  # noqa: E402
from speaktome_man.__main__ import main as man_main  # noqa: E402
from speaktome_mcp.server import Server  # noqa: E402

PAGES = manual.pages()
VERB_PARTS = ("NAME", "SYNOPSIS", "DESCRIPTION", "IN A STORY", "NEEDS", "STATUS", "SEE ALSO")
EVERY_PARTS = ("NAME", "DESCRIPTION", "SEE ALSO")


def _tools() -> set:
    return set(Server(None).table)


def _call(arguments: dict) -> tuple:
    answer = Server(None).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "man", "arguments": arguments}})["result"]
    return json.loads(answer["content"][0]["text"]), answer["isError"]


def test_the_manual_has_pages_in_every_section():
    assert {p.section for p in PAGES} == set(manual.SECTIONS)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.file}.{p.section}")
def test_page_is_well_formed(page):
    headings = page.headings()
    for part in VERB_PARTS if page.section == "1" else EVERY_PARTS:
        assert part in headings, f"{part} is missing"
    assert headings[0] == "NAME" and headings[-1] == "SEE ALSO"
    assert page.file in page.names or all(name.startswith(page.file + "_") for name in page.names), (
        "the file is named for one of the page's names, or for the prefix they share")
    assert page.summary[0].islower() and not page.summary.endswith(".")
    for number, line in enumerate(page.body.splitlines(), 1):
        assert line.isascii(), f"line {number} is not ASCII"
        assert len(line) <= manual.WIDTH, f"line {number} is wider than {manual.WIDTH}"
        assert "\t" not in line and line == line.rstrip(), f"line {number} has a tab or trailing space"
    assert len(page.render().splitlines()[0]) <= manual.WIDTH


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.file}.{p.section}")
def test_every_reference_resolves(page):
    references = page.references()
    assert references, "SEE ALSO names at least one page"
    for name, section in references:
        assert manual.page(name, section).section == section
        assert name not in page.names or section != page.section, "a page does not refer to itself"


def test_names_are_unique_within_a_section():
    seen = [(name, p.section) for p in PAGES for name in p.names]
    assert len(seen) == len(set(seen))


def test_every_tool_has_a_verb_page():
    for tool in _tools():
        assert manual.page(tool, "1").section == "1"


def test_a_verb_page_states_a_tool_form_exactly_when_a_tool_exists():
    tools = _tools()
    for page in (p for p in PAGES if p.section == "1"):
        states_tool = any(line.strip().startswith("Tool:") for line in page.part("SYNOPSIS").splitlines())
        offered = [name in tools for name in page.names]
        assert all(offered) or not any(offered), f"{page.file}: some of its names are tools and some are not"
        assert states_tool == all(offered), f"{page.file}: the SYNOPSIS and the tool table disagree"


def test_planned_verbs_are_not_tools_and_say_so():
    tools = _tools()
    for verb in ("take", "use"):
        page = manual.page(verb, "1")
        assert page.part("STATUS").strip().startswith("Planned.")
        assert verb not in tools


def test_make_is_bounded_to_experience_reports():
    assert "make" in _tools()
    assert "experience reports" in manual.page("make", "1").part("STATUS")
    assert "General graph mutation remains planned" in manual.page("make", "1").part("STATUS")


def test_lowest_section_wins_and_a_section_can_be_asked_for():
    assert manual.page("map").section == "5"
    assert manual.page("cursor").file == "entity"
    with pytest.raises(manual.ManError):
        manual.page("go", "7")


def test_unknown_page_names_the_nearest():
    with pytest.raises(manual.ManError) as failure:
        manual.page("entr")
    assert "enter" in failure.value.nearest


def test_the_manual_is_a_map():
    document = manual.as_map()
    assert document["schema"] == "abstract-ui-div-map-v0" and document["identity"] == manual.ROOT
    places = [region["identity"] for region in document["regions"]]
    assert len(places) == len(set(places)) == len(manual.SECTIONS) + len(PAGES)
    known = set(places) | {manual.ROOT}
    for track in document["tracks"]:
        assert track["source"] in known and track["target"] in known
        assert track["relationship"] in ("contained-by", "see-also")
    contained = {t["source"]: t["target"] for t in document["tracks"] if t["relationship"] == "contained-by"}
    assert set(contained) == set(places), "every place is contained by exactly one place"
    objects = {o["identity"]: o for o in document["world"]["objects"]}
    assert len(objects) == len(PAGES)
    for page in PAGES:
        record = objects[manual.place(page)]
        assert contained[manual.place(page)] == f"{manual.ROOT}/{page.section}"
        assert record["text"] == page.render() and record["kind"] == manual.KINDS[page.section]
        assert len(record["dependencies"]) == len(page.references())
    json.dumps(document)


def test_tool_returns_a_page():
    result, is_error = _call({"topic": "go"})
    assert not is_error and result["section"] == "1" and result["at"] == manual.place(manual.page("go"))
    assert result["page"].startswith("GO(1)") and "SEE ALSO" in result["page"]


def test_tool_lists_and_searches():
    everything, _ = _call({})
    assert {"name": "go", "section": "1", "summary": manual.page("go").summary} in everything["pages"]
    concepts, _ = _call({"section": "7"})
    assert concepts["pages"] and all(row["section"] == "7" for row in concepts["pages"])
    found, _ = _call({"keyword": "cursor"})
    assert {row["name"] for row in found["pages"]} >= {"enter", "leave", "cursor"}


def test_tool_refuses_an_unknown_page_and_a_mixed_request():
    result, is_error = _call({"topic": "entr"})
    assert is_error and result["status"] == 404 and "enter" in result["say"]
    result, is_error = _call({"topic": "go", "keyword": "edge"})
    assert is_error and result["status"] == 400


def test_command_line(capsys):
    assert man_main(["go"]) == 0
    assert capsys.readouterr().out.startswith("GO(1)")
    assert man_main(["7", "entity"]) == 0
    assert man_main(["-k", "message"]) == 0
    assert "say(1)" in capsys.readouterr().out
    assert man_main(["nothing-like-this"]) == 1
    assert man_main([]) == 0
