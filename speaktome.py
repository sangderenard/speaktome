# Standard library imports
from typing import List, Tuple, Dict, Optional, Any, Callable
import argparse

# Third-party imports
import matplotlib.pyplot as plt # type: ignore # Used by main, and indirectly by visualizers
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
import torch_geometric.nn as pyg_nn # For PyGeoMind
from sentence_transformers import SentenceTransformer # For type hinting
from transformers import PreTrainedTokenizer

# Local application/library specific imports
# Please adjust these import paths if your project structure differs.
from .beam_search import BeamSearch
from .scorer import Scorer # Assuming Scorer is in scorer.py
from .pygeo_mind import PyGeoMind # Assuming PyGeoMind is in pygeomind.py
from .pyg_graph_controller import PyGGraphController # Assuming PyGGraphController is in pyg_graph_controller.py
from .compressed_beam_tree import CompressedBeamTree # CompressedBeamTree is used by SentenceEmbeddingPCAVisualizer
# Import both visualizers from beam_tree_visualizer.py
from .beam_tree_visualizer import BeamTreeVisualizer, SentenceEmbeddingPCAVisualizer
# Import DEVICE and lazy SentenceTransformer accessor from config
from .config import DEVICE, get_sentence_transformer_model, GPU_LIMIT, LENGTH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Run Beam Search with optional GNN control.")
    parser.add_argument('-s', '--seed', type=str, default='The signet read: "',
                        help="Seed phrase for beam search.")
    parser.add_argument('-m', '--max_steps', type=int, default=5,
                        help="Maximum steps (depth) for the search.")
    parser.add_argument('-c', '--human_control', action='store_true', default=False,
                        help="Enable human-in-the-loop control (default: autonomous).")
    parser.add_argument('-l', '--lookahead', type=int, default=3,
                        help="Number of lookahead steps for beam search.")
    parser.add_argument('-e', '--easy_mode', action='store_true', default=False,
                        help="Easy mode: full lookahead, no retirement.")
    parser.add_argument('-f', '--full_summary', action='store_true', default=False,
                        help="Print a full summary of all beams after the run.")
    parser.add_argument('-w', '--beam_width', type=int, default=5,
                        help="Beam width for the search.")
    parser.add_argument('-p', '--preferred_scorer', type=str, default=None,
                        choices=list(Scorer.get_available_scoring_functions().keys()),
                        help="Preferred scoring function for default instructions.")
    parser.add_argument('-a', '--auto_expand', type=int, default=0,
                        help="Run 'expand_any' automatically for N rounds before normal control.")
    parser.add_argument('-g', '--gnn_rounds', type=int, default=0,
                        help="Let PyGeoMind control for N rounds before returning to human/auto mode.")
    args = parser.parse_args()

    # 2.1 Instantiate the GPT-based Scorer
    scorer = Scorer()

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
                args.preferred_scorer: (preferred_fn, scorer.default_k, scorer.default_temp)
            }
            print(f"Preferred scorer set to: {args.preferred_scorer}")
        else:
            print(f"Warning: Preferred scorer '{args.preferred_scorer}' not found. Using default scorer policy.")


    # 2.2 Instantiate BeamSearch
    beam_search = BeamSearch(scorer,
                             lookahead_steps=effective_lookahead_steps,
                             beam_width=args.beam_width,
                             initial_retirement_enabled=initial_retirement_enabled,
                             max_len=LENGTH_LIMIT) # Pass LENGTH_LIMIT for max_len

    # 2.3 Instantiate your GNN policy network (PyGeoMind)
    pygeomind_model = PyGeoMind(scorer=scorer, input_dim=3, hidden_dim=128).to(DEVICE)

    # 2.4 Wrap everything into the PyGGraphController
    controller = PyGGraphController(
        beam_search,
        pygeomind_model=pygeomind_model,
        human_in_control=args.human_control
    )

    # Apply auto-run settings from CLI arguments
    if args.gnn_rounds > 0:
        controller.auto_run_mode = "pygeomind_decide"
        controller.auto_run_counter = args.gnn_rounds
        controller.human_in_control = False
    elif args.auto_expand > 0:
        controller.auto_run_mode = "expand_any"
        controller.auto_run_counter = args.auto_expand

    # 2.5 Choose fixed hyperparameters for this run:
    # seed_phrase = "The signet read: \"" # From args
    # max_steps = 5 # From args

    print(f"\n⏳ Starting autonomous NXM search:")
    print(f"    • Seed              = '{args.seed}'")
    print(f"    • max_steps (depth) = {args.max_steps}")
    print(f"    • Human Control     = {args.human_control}")
    print(f"    • Lookahead Steps   = {beam_search.lookahead_steps}") # Use actual value from beam_search
    print(f"    • Beam Width        = {beam_search.beam_width}")
    print(f"    • Retirement        = {'Enabled' if beam_search.initial_retirement_enabled else 'Disabled'}")
    if args.easy_mode:
        print(f"    • Easy Mode         = Active")

    # 2.6 Run the controller-loop exactly once
    final_tree = controller.run_loop(
        seed_text=args.seed,
        max_steps=args.max_steps
    )

    # 2.7 After run_loop returns, visualize tree, print summary & shut down retirement thread
    final_beam_tree_obj = beam_search.tree # Get the tree from beam_search instance

    # Standard visualization

    if final_beam_tree_obj and final_beam_tree_obj.nodes: # Check if the tree is not empty
        pyg_data = final_beam_tree_obj.get_pyg_data()
        if pyg_data and pyg_data.num_nodes > 0:
            print("\nVisualizing final beam tree...")
            visualizer = BeamTreeVisualizer()
            visualizer.visualize_subtree(
                pyg_data,
                scorer.tokenizer,
                beam_search, # Pass the beam_search object
                root_pyg_node_id=0, # Visualize from the root (assuming PyG node 0 is the root)
                title="Final Beam Tree"
            )
        else:
            print("Final beam tree is empty or could not be converted to PyG data. Skipping visualization.")
    else:
        print("No final beam tree generated. Skipping visualization.")

    
    if hasattr(beam_search, "active_leaf_indices"):
        print(f"    Active leaves:  {len(beam_search.active_leaf_indices)}")
    if hasattr(beam_search, "dead_end_indices"):
        print(f"    Dead-end beams: {len(beam_search.dead_end_indices)}")

    # Full summary if requested
    if args.full_summary:
        print("\n--- Full Beam Summary (All Beams) ---")
        full_summary_data = beam_search.get_all_beams_full_summary()
        if full_summary_data:
            for beam_idx, text, score, status in full_summary_data:
                print(f"[{beam_idx:03d}] Score: {score:7.3f} | Status: {status:10} | {text}")
        else:
            print("No beams to summarize.")
    # Add new visualizer call
    if final_beam_tree_obj and final_beam_tree_obj.nodes:
        print("\nVisualizing node sentence embeddings (PCA)...")
        pca_visualizer = SentenceEmbeddingPCAVisualizer(
            final_beam_tree_obj, get_sentence_transformer_model(), scorer.tokenizer
        )
        pca_visualizer.visualize()
    else:
        print("Final beam tree is empty. Skipping PCA sentence embedding visualization.")

    beam_search.shutdown_retirement_manager()

if __name__ == "__main__":
    main()
