# Standard library imports
from typing import List, Tuple, Dict, Optional, Any, Callable, TYPE_CHECKING
import argparse

# Third-party imports
from . import Faculty, DEFAULT_FACULTY
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - runtime message only
    torch = None

TORCH_AVAILABLE = DEFAULT_FACULTY in (Faculty.TORCH, Faculty.PYGEO)
if not TORCH_AVAILABLE:
    print("PyTorch is not installed. Running CPU-only demo mode.")
try:
    from transformers import PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - runtime message only
    PreTrainedTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False
    print("Transformers is not installed. Running CPU-only demo mode.")

from .util.lazy_loader import lazy_import
if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData
    import torch_geometric.nn as pyg_nn
    from sentence_transformers import SentenceTransformer

# Local application/library specific imports
# Please adjust these import paths if your project structure differs.
# Import configuration dynamically to allow device changes at runtime
from . import config
from .config import get_sentence_transformer_model, GPU_LIMIT, LENGTH_LIMIT
# --- END HEADER ---


def main(raw_args=None, allow_retry=True):
    parser = argparse.ArgumentParser(description="Run Beam Search with optional GNN control.")
    parser.add_argument('-s', '--seed', type=str, default='The signet read: "',
                        help="Seed phrase for beam search.")
    parser.add_argument('-m', '--max_steps', type=int, default=5,
                        help="Maximum steps (depth) for the search.")
    parser.add_argument('-c', '--human_control', action='store_true', default=False,
                        help="Enable human-in-the-loop control (default: autonomous).")
    parser.add_argument('-l', '--lookahead', type=int, default=1024,
                        help="Number of lookahead steps for beam search.")
    parser.add_argument('-e', '--easy_mode', action='store_true', default=True,
                        help="Disable retirement and use maximum lookahead by default.")
    parser.add_argument('-f', '--full_summary', action='store_true', default=False,
                        help="Print a full summary of all beams after the run.")
    parser.add_argument('-w', '--beam_width', type=int, default=5,
                        help="Beam width for the search.")
    parser.add_argument('-p', '--preferred_scorer', type=str, default=None,
                        help="Preferred scoring function for default instructions.")
    parser.add_argument('-a', '--auto_expand', type=int, default=1,
                        help="Run 'expand_any' automatically for N rounds before normal control.")
    parser.add_argument('-g', '--gnn_rounds', type=int, default=0,
                        help="Let PyGeoMind control for N rounds before returning to human/auto mode.")
    parser.add_argument('--with_gnn', action='store_true', default=False,
                        help="Instantiate the optional PyGeoMind GNN controller (requires torch_geometric).")
    parser.add_argument('-x', '--safe_mode', action='store_true', default=False,
                        help="Force CPU mode and disable GPU usage.")
    parser.add_argument('--final_viz', action='store_true', default=False,
                        help="Visualize the final beam tree after completion.")
    parser.add_argument('--final_pca', action='store_true', default=False,
                        help="Visualize PCA of sentence embeddings after completion.")
    parser.add_argument('--preload_models', action='store_true', default=False,
                        help="Load all models up front to avoid lazy initialization delays.")
    parser.add_argument('seed_text', nargs='*', help='Seed text if not provided via -s')
    args = parser.parse_args(raw_args)

    print(f"Faculty level: {DEFAULT_FACULTY.name}")

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        from .cpu_demo import main as cpu_main
        cpu_main(raw_args)
        return

    if args.safe_mode:
        config.DEVICE = torch.device("cpu")

    def run_once():
        from .core.beam_search import BeamSearch
        from .core.scorer import Scorer
        from .domains.geo.pygeo_mind import PyGeoMind
        from .domains.geo.pyg_graph_controller import PyGGraphController
        from .core.compressed_beam_tree import CompressedBeamTree
        from .core.beam_tree_visualizer import BeamTreeVisualizer
        seed_from_flag = args.seed != parser.get_default('seed')
        final_seed = args.seed if seed_from_flag else ' '.join(args.seed_text)

        # 2.1 Instantiate the GPT-based Scorer
        scorer = Scorer()
        if args.preload_models:
            scorer.preload_models()
            config.get_sentence_transformer_model()

        # Determine lookahead steps and retirement status based on args
        effective_lookahead_steps = args.lookahead
        initial_retirement_enabled = not args.easy_mode

        if args.easy_mode:
            effective_lookahead_steps = LENGTH_LIMIT # Full lookahead
            print("Easy Mode Activated: Full lookahead, no retirement.")

        # Modify scorer's default policy if a preferred scorer is specified
        if args.preferred_scorer:
            available_scorers = Scorer.get_available_scoring_functions()
            preferred_fn = available_scorers.get(args.preferred_scorer)
            if preferred_fn:
                scorer.default_score_policy["score_bins"] = {
                    args.preferred_scorer: (preferred_fn, scorer.default_k, scorer.default_temp),
                }
                print(f"Preferred scorer set to: {args.preferred_scorer}")
            else:
                print(f"Warning: Preferred scorer '{args.preferred_scorer}' not found. Using default scorer policy.")

        # 2.2 Instantiate BeamSearch
        beam_search = BeamSearch(
            scorer,
            lookahead_steps=effective_lookahead_steps,
            beam_width=args.beam_width,
            initial_retirement_enabled=initial_retirement_enabled,
            max_len=LENGTH_LIMIT,
        )

        pygeomind_model = None
        if args.with_gnn:
            try:
                pygeomind_model = PyGeoMind(scorer=scorer, input_dim=3, hidden_dim=128).to(config.DEVICE)
            except ModuleNotFoundError:
                print("torch_geometric is required for the GNN brain. Install it to enable PyGeoMind.")
                print("Continuing without the GNN controller.")

        # 2.4 Wrap everything into the PyGGraphController
        controller = PyGGraphController(
            beam_search,
            pygeomind_model=pygeomind_model,
            human_in_control=args.human_control,
        )

        # Apply auto-run settings from CLI arguments
        if args.gnn_rounds > 0 and pygeomind_model:
            controller.auto_run_mode = "pygeomind_decide"
            controller.auto_run_counter = args.gnn_rounds
            controller.human_in_control = False
        elif args.auto_expand > 0:
            controller.auto_run_mode = "expand_any"
            controller.auto_run_counter = args.auto_expand

        print(f"\n⏳ Starting autonomous NXM search:")
        print(f"    • Seed              = '{final_seed}'")
        print(f"    • max_steps (depth) = {args.max_steps}")
        print(f"    • Human Control     = {args.human_control}")
        print(f"    • Lookahead Steps   = {beam_search.lookahead_steps}")
        print(f"    • Beam Width        = {beam_search.beam_width}")
        print(f"    • Retirement        = {'Enabled' if beam_search.initial_retirement_enabled else 'Disabled'}")
        if args.easy_mode:
            print(f"    • Easy Mode         = Active")

        # 2.6 Run the controller-loop exactly once
        final_tree = controller.run_loop(seed_text=final_seed, max_steps=args.max_steps)

        # 2.7 After run_loop returns, visualize tree, print summary & shut down retirement thread
        final_beam_tree_obj = beam_search.tree

        if args.final_viz and final_beam_tree_obj and final_beam_tree_obj.nodes:
            pyg_data = final_beam_tree_obj.get_pyg_data()
            if pyg_data and pyg_data.num_nodes > 0:
                print("\nVisualizing final beam tree...")
                visualizer = BeamTreeVisualizer()
                visualizer.visualize_subtree(
                    pyg_data,
                    scorer.tokenizer,
                    beam_search,
                    root_pyg_node_id=0,
                    title="Final Beam Tree",
                )
            else:
                print("Final beam tree is empty or could not be converted to PyG data. Skipping visualization.")

        if hasattr(beam_search, "active_leaf_indices"):
            print(f"    Active leaves:  {len(beam_search.active_leaf_indices)}")
        if hasattr(beam_search, "dead_end_indices"):
            print(f"    Dead-end beams: {len(beam_search.dead_end_indices)}")

        if args.full_summary:
            print("\n--- Full Beam Summary (All Beams) ---")
            full_summary_data = beam_search.get_all_beams_full_summary()
            if full_summary_data:
                for beam_idx, text, score, status in full_summary_data:
                    print(f"[{beam_idx:03d}] Score: {score:7.3f} | Status: {status:10} | {text}")
            else:
                print("No beams to summarize.")

        if args.final_pca and final_beam_tree_obj and final_beam_tree_obj.nodes:
            print("\nVisualizing node sentence embeddings (PCA)...")
            BeamTreeVisualizer().visualize_sentence_embeddings(
                final_beam_tree_obj,
                get_sentence_transformer_model(),
                scorer.tokenizer,
            )

        beam_search.shutdown_retirement_manager()

    try:
        run_once()
    except RuntimeError as exc:
        if allow_retry and not args.human_control and config.DEVICE.type == "cuda":
            print("RuntimeError detected on GPU. Switching to CPU and retrying once...")
            config.DEVICE = torch.device("cpu")
            config.sentence_transformer_model = None
            torch.cuda.empty_cache()
            run_once()
        else:
            raise


if __name__ == "__main__":
    main()
