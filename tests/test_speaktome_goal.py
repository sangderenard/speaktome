"""The Goal document format: tables in comment headers, links as rows, and the document as a graph of itself."""
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import speaktome_goal as goal  # noqa: E402
from speaktome_api import site  # noqa: E402
from test_speaktome_site import GraphView  # noqa: E402


def row(table, *cells):
    return dict(zip(goal.TABLES[table], cells))


def example(*, built_step=True):
    double = goal.Unit(
        identity="piece-0", language="python", state="lowered", text="RATE = 2.0\n\ndef double(x: float) -> float:\n    return x * RATE",
        facts=MappingProxyType({"file": "example.py"}),
        api=goal._rows([row("api", "double", "x", "parameter", "float"), row("api", "double", "return", "result", "float")]),
        build=goal._rows([row("build", "c", "double", "double__double", "double.c", "a" * 64, "piece-0.c.double", "emitted-not-built"),
                          row("build", "javascript", "double", "double__double", "double.mjs", "b" * 64, "piece-0.js.double", "")]),
        abi=goal._rows([row("abi", "javascript", "double", "t0", "input", "float64", "value", "x")]))
    step_builds = [row("build", "c", "step", "step__step", "step.c", "c" * 64, "piece-1.c.step", "emitted-not-built")] if built_step else []
    step = goal.Unit(
        identity="piece-1", language="python", state="lowered" if built_step else "source",
        text="def step(x: float) -> float:\n    y = double(x) + 1.0\n    return y",
        api=goal._rows([row("api", "step", "x", "parameter", "float"), row("api", "step", "return", "result", "float")]),
        link=goal._rows([row("link", "double", "piece-0", "double", "reference")]),
        build=goal._rows(step_builds))
    return goal.GoalDocument("example", (double, step), MappingProxyType({"source": "job 02172e9cfcf7"}))


def facts(document):
    return (document.identity, dict(document.facts),
            [(u.identity, u.language, u.text, u.state, u.encoding, dict(u.facts),
              [dict(r) for r in u.api], [dict(r) for r in u.link], [dict(r) for r in u.build], [dict(r) for r in u.abi])
             for u in document.units])


def test_a_document_reads_back_exactly_and_its_tables_are_plain_text():
    document = example()
    rendered = goal.render(document)
    assert facts(goal.parse(rendered)) == facts(document)
    text = rendered.decode("ascii")
    assert "link   | name   | unit    | entry  | how" in text
    assert "link   | double | piece-0 | double | reference" in text
    assert "api    | step  | x      | parameter | float" in text
    assert "target | c          | 2     | 2       | yes" in text
    assert "target | javascript | 2     | 1       | no" in text
    assert goal.render(goal.parse(rendered)) == rendered, "rendering is stable"


def test_the_goal_is_achieved_for_a_target_when_every_unit_is_built_for_it():
    assert example().achieved() == ("c",)
    partial = example(built_step=False)
    assert partial.achieved() == () and partial.unit("piece-1").state == "source"
    assert [dict(r) for r in partial.targets()] == [
        {"target": "c", "units": "2", "lowered": "1", "achieved": "no"},
        {"target": "javascript", "units": "2", "lowered": "1", "achieved": "no"}]


def test_cells_keep_pipes_dashes_empties_and_other_characters():
    unit = goal.Unit(identity="u", language="python", text="x = 1", api=goal._rows([
        row("api", "f", "x", "parameter", "int | None"), row("api", "f", "y", "parameter", "-"),
        row("api", "f", "z", "parameter", ""), row("api", "f", "w", "parameter", "café\\path")]))
    read = goal.parse(goal.render(goal.GoalDocument("d", (unit,))))
    assert [r["type"] for r in read.unit("u").api] == ["int | None", "-", "", "café\\path"]


@pytest.mark.parametrize("payload", ["ends with a return\r", "\n\nblank\n\n", "", "crlf\r\ninside\r\n"])
def test_a_units_text_survives_at_its_edges(payload):
    document = goal.GoalDocument("d", (goal.Unit(identity="u", language="text", text=payload),))
    assert goal.parse(goal.render(document)).unit("u").text == payload


def test_what_does_not_hold_together_is_refused():
    base = example()
    double, step = base.units
    def with_step(**changes):
        fields = goal._unit_fields(step)
        fields["text"] = step.text
        fields.update(changes)
        return goal.GoalDocument("d", (double, goal.Unit(**fields)))
    with pytest.raises(goal.GoalError, match="unknown unit"):
        with_step(link=goal._rows([row("link", "double", "piece-9", "double", "reference")]))
    with pytest.raises(goal.GoalError, match="does not offer"):
        with_step(link=goal._rows([row("link", "triple", "piece-0", "triple", "reference")]))
    with pytest.raises(goal.GoalError, match="bound by"):
        with_step(link=goal._rows([row("link", "double", "piece-0", "double", "guessed")]))
    with pytest.raises(goal.GoalError, match="does not offer"):
        with_step(build=goal._rows([row("build", "c", "other", "s", "p", "0" * 64, "r", "")]))
    with pytest.raises(goal.GoalError, match="lowered exactly when"):
        with_step(state="source")
    with pytest.raises(goal.GoalError, match="not listed"):
        with_step(abi=goal._rows([row("abi", "wasm", "step", "t0", "input", "f64", "value", "x")]))
    with pytest.raises(goal.GoalError, match="unique"):
        goal.GoalDocument("d", (double, double))
    with pytest.raises(goal.GoalError, match="closing sentinel"):
        goal.render(goal.GoalDocument("d", (goal.Unit(identity="u", language="text", text="a /*@goal.end*/ b"),)))


def test_a_reader_refuses_a_changed_text_a_wrong_target_table_and_stray_bytes():
    rendered = goal.render(example())
    with pytest.raises(goal.GoalError, match="sha256"):
        goal.parse(rendered.replace(b"y = double(x) + 1.0", b"y = double(x) + 2.0"))
    with pytest.raises(goal.GoalError, match="target table"):
        goal.parse(rendered.replace(b"| 2       | yes", b"| 1       | no "))
    with pytest.raises(goal.GoalError, match="belong to a block"):
        goal.parse(b"stray\n" + rendered)
    with pytest.raises(goal.GoalError, match="document block"):
        goal.parse(rendered.split(b"@end*/\n", 1)[1])


def test_the_document_is_a_graph_of_itself():
    nodes = {n["identity"]: n for n in goal.as_graph(example())}
    root = "goal:example"
    assert nodes[root]["kind"] == "goal"
    assert nodes[root]["dependencies"] == [{"target": f"{root}/target/c", "relationship": "achieves"}]
    step = nodes[f"{root}/unit/piece-1"]
    assert {"target": f"{root}/unit/piece-0", "relationship": "links-to"} in step["dependencies"]
    build = nodes[f"{root}/unit/piece-1/build/c/step"]
    assert {"target": f"{root}/target/c", "relationship": "built-for"} in build["dependencies"]
    kinds = sorted(n["kind"] for n in nodes.values())
    assert kinds == ["build", "build", "build", "goal", "target", "target", "unit", "unit"]
    known = set(nodes)
    assert all(d["target"] in known for n in nodes.values() for d in n["dependencies"]), "every edge ends at a node"


def test_the_graph_becomes_a_map_that_the_site_can_bake(tmp_path):
    document = example()
    atlas = site.Site(tmp_path / "cache", GraphView).open("goal", goal.as_map(document))
    assert atlas.bake(tmp_path / "out") == len(goal.as_graph(document)) + 1
    index = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert "achieved for c" in index
    step_page = (tmp_path / "out" / (site.address("goal:example/unit/piece-1") + ".html")).read_text(encoding="utf-8")
    assert "links-to (outgoing)" in step_page and "def step" in step_page
    assert "link  | double -&gt; piece-0 : double (reference)" in step_page
