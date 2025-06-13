import logging

from AGENTS.tools.pretty_logger import AsciiTreeRenderer, TreeNode

logger = logging.getLogger(__name__)


def test_ascii_tree_renderer_basic():
    root = TreeNode("root", [TreeNode("child1"), TreeNode("child2", [TreeNode("leaf")])])
    renderer = AsciiTreeRenderer(use_color=False)
    output = renderer.render(root)
    logger.info("\n" + output)
    assert "root" in output
    assert "child1" in output
    assert "leaf" in output
