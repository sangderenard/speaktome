import importlib
import logging
import pytest

from speaktome.util.cli_permutations import CLIArgumentMatrix
from speaktome.util.token_vocab import TokenVocabulary

logger = logging.getLogger(__name__)


def test_token_vocabulary_round_trip():
    logger.info('test_token_vocabulary_round_trip start')
    vocab = TokenVocabulary("ab ")
    ids = vocab.encode("ba")
    assert vocab.decode(ids) == "ba"
    logger.info('test_token_vocabulary_round_trip end')


def test_cli_argument_matrix_basic():
    logger.info('test_cli_argument_matrix_basic start')
    matrix = CLIArgumentMatrix()
    matrix.add_option("--flag", [1, 2])
    combos = matrix.generate()
    assert ["--flag", "1"] in combos and ["--flag", "2"] in combos
    logger.info('test_cli_argument_matrix_basic end')


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
    ("speaktome.core.tensor_abstraction", "PyTorchTensorOperations"),
    ("speaktome.core.tensor_abstraction", "PurePythonTensorOperations"),
]


@pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
def test_class_instantiation(mod_name: str, cls_name: str):
    logger.info(f'test_class_instantiation start for {cls_name}')
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    logger.info('instantiating %s', cls_name)
    cls()
    logger.info('test_class_instantiation end for %s', cls_name)
