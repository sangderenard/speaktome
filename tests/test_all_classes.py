import importlib
import pytest

from speaktome.cli_permutations import CLIArgumentMatrix
from speaktome.token_vocab import TokenVocabulary


def test_token_vocabulary_round_trip():
    vocab = TokenVocabulary("ab ")
    ids = vocab.encode("ba")
    assert vocab.decode(ids) == "ba"


def test_cli_argument_matrix_basic():
    matrix = CLIArgumentMatrix()
    matrix.add_option("--flag", [1, 2])
    combos = matrix.generate()
    assert ["--flag", "1"] in combos and ["--flag", "2"] in combos


# ----- Stub placeholders for complex classes -----
STUB_MODULES = [
    ("speaktome.beam_graph_operator", "BeamGraphOperator"),
    ("speaktome.beam_retirement_manager", "BeamRetirementManager"),
    ("speaktome.beam_search", "BeamSearch"),
    ("speaktome.beam_search_instruction", "BeamSearchInstruction"),
    ("speaktome.beam_tree_node", "BeamTreeNode"),
    ("speaktome.beam_tree_visualizer", "BeamTreeVisualizer"),
    ("speaktome.compressed_beam_tree", "CompressedBeamTree"),
    ("speaktome.cpu_demo", "RandomModel"),
    ("speaktome.human_pilot_controller", "HumanPilotController"),
    ("speaktome.human_scorer_policy_manager", "HumanScorerPolicyManager"),
    ("speaktome.model_abstraction", "PyTorchModelWrapper"),
    ("speaktome.pyg_graph_controller", "PyGGraphController"),
    ("speaktome.pygeo_mind", "PyGeoMind"),
    ("speaktome.scorer", "Scorer"),
    ("speaktome.tensor_abstraction", "PyTorchTensorOperations"),
]


@pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
@pytest.mark.stub
def test_class_stub(mod_name: str, cls_name: str):
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, cls_name)
    except ModuleNotFoundError:
        pytest.skip(f"Optional dependency missing for {cls_name}")
