"""Smoke tests for miscellaneous modules.

These checks ensure that a few lightweight utilities instantiate correctly and
that the basic command-line interface functions as expected.
"""

import importlib
import logging
import sys
import types
import pytest

from speaktome.util.cli_permutations import CLIArgumentMatrix
from speaktome.util.token_vocab import TokenVocabulary
from speaktome.faculty import DEFAULT_FACULTY, Faculty
# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_token_vocabulary_round_trip():
    """Round trip a short string through :class:`TokenVocabulary`."""
    logger.info("test_token_vocabulary_round_trip start")
    vocab = TokenVocabulary("ab ")
    ids = vocab.encode("ba")
    assert vocab.decode(ids) == "ba"
    TokenVocabulary.test()
    logger.info("test_token_vocabulary_round_trip end")


def test_cli_argument_matrix_basic():
    """Ensure the argument matrix expands options as expected."""
    logger.info("test_cli_argument_matrix_basic start")
    matrix = CLIArgumentMatrix()
    matrix.add_option("--flag", [1, 2])
    combos = matrix.generate()
    assert ["--flag", "1"] in combos and ["--flag", "2"] in combos
    logger.info("test_cli_argument_matrix_basic end")


# ----- Stub placeholders for complex classes -----
STUB_MODULES = [
    ("speaktome.core.beam_graph_operator", "BeamGraphOperator"),
    ("speaktome.core.beam_retirement_manager", "BeamRetirementManager"),
    ("speaktome.core.beam_search", "BeamSearch"),
    ("speaktome.core.beam_search_instruction", "BeamSearchInstruction"),
    ("speaktome.core.beam_tree_node", "BeamTreeNode"),
    ("speaktome.core.beam_tree_visualizer", "BeamTreeVisualizer"),
    ("speaktome.core.compressed_beam_tree", "CompressedBeamTree"),
    ("speaktome.cpu_demo", "RandomModel"),
    ("speaktome.core.human_pilot_controller", "HumanPilotController"),
    ("speaktome.core.human_scorer_policy_manager", "HumanScorerPolicyManager"),
    ("speaktome.core.model_abstraction", "PyTorchModelWrapper"),
    ("speaktome.domains.geo.pyg_graph_controller", "PyGGraphController"),
    ("speaktome.domains.geo.pygeo_mind", "PyGeoMind"),
    ("speaktome.core.scorer", "Scorer"),
    ("speaktome.tensors.torch_backend", "PyTorchTensorOperations"),
    ("speaktome.tensors.pure_backend", "PurePythonTensorOperations"),
]


@pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
def test_class_instantiation(mod_name: str, cls_name: str) -> None:
    """Import each listed module and instantiate the referenced class."""
    if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
        pytest.skip("Optional libraries not installed; skipping class instantiation tests")

    logger.info(f"test_class_instantiation start for {cls_name}")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        logger.error('import failed for %s: %s', mod_name, exc)
        pytest.fail(f"import failed for {mod_name}: {exc}")

    cls = getattr(mod, cls_name, None)
    if cls is None:
        logger.error('class %s not found in %s', cls_name, mod_name)
        pytest.fail(f"class {cls_name} not found in {mod_name}")

    logger.info('instantiating %s', cls_name)
    try:
        if cls_name == "PyTorchModelWrapper":
            dummy = types.SimpleNamespace()
            setattr(dummy, 'parameters', lambda: iter([0]))
            cls(dummy)
        else:
            cls()
    except Exception as exc:
        logger.error('failed to instantiate %s: %s', cls_name, exc)
        pytest.fail(f"failed to instantiate {cls_name}: {exc}")
    logger.info('test_class_instantiation end for %s', cls_name)
