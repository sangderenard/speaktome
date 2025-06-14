from AGENTS.tools.headers.dynamic_header_recognition import load_template_tree


def test_load_template_tree_has_docstring_and_import():
    tree = load_template_tree()
    names = [child.name for child in tree.children]
    assert "docstring" in names
    assert "import" in names
