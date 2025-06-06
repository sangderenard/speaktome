# Standard library imports
import queue as py_queue
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

# Third-party imports
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer  # For type hinting

# --- MetaBeamManager ---
class MetaBeamManager:
    def __init__(self, bins_config, max_len, device=None, cull_after=1):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.top_beams = {}  # text -> (score, age, vector)
        self.delivery_queue = py_queue.Queue(maxsize=8)
        self.cull_after = cull_after  # Number of rounds to keep before culling
        self.bins = {}
        self.survival_age = {}
        for name, cfg in bins_config.items():
            N = cfg['width']
            self.bins[name] = {
                'fn': cfg['fn'],
                'params': cfg.get('params', {}),
                'beams': torch.full((N, max_len), -1, dtype=torch.long, device=self.device),
                'scores': torch.full((N,), float('-inf'), device=self.device),
                'lengths': torch.zeros((N,), dtype=torch.long, device=self.device),
                'age': torch.zeros((N,), dtype=torch.long, device=self.device),
                'width': N # Store the width of the bin
            }

    def update(self, beams, scores, lengths, tokenizer, round_idx=0):
        # For each bin, score, pick winners, manage age
        for name, bin in self.bins.items():
            N = bin['beams'].shape[0]
            # Calculate score
            bin_scores = self.call_score_fn(bin['fn'], beams, scores, lengths, tokenizer, **bin['params'])
            # Stack with existing beams for the "M rounds before cull" (age increments)
            all_scores = torch.cat([bin['scores'], bin_scores])
            all_beams = torch.cat([bin['beams'], F.pad(beams, (0, bin['beams'].shape[1] - beams.shape[1]))])
            all_lengths = torch.cat([bin['lengths'], lengths])
            all_age = torch.cat([bin['age'] + 1, torch.zeros_like(bin_scores, dtype=torch.long)])

            # Select top N, only one spot per candidate (no dups)
            # Option: for pure uniqueness, use set of decoded strings
            _, top_idx = torch.topk(all_scores, k=N)
            bin['scores'] = all_scores[top_idx]
            bin['beams'] = all_beams[top_idx]
            bin['lengths'] = all_lengths[top_idx]
            bin['age'] = all_age[top_idx]

            # Cull: If cull_after is set, zero out entries exceeding cull_after
            if self.cull_after > 0:
                keep = bin['age'] <= self.cull_after
                for k in ['scores', 'beams', 'lengths', 'age']:
                    bin[k] = bin[k][keep]
                # Pad if needed
                pad_n = N - bin['scores'].shape[0]
                if pad_n > 0:
                    bin['scores'] = torch.cat([bin['scores'], torch.full((pad_n,), float('-inf'), device=self.device)])
                    bin['beams'] = torch.cat([bin['beams'], torch.full((pad_n, self.max_len), -1, dtype=torch.long, device=self.device)])
                    bin['lengths'] = torch.cat([bin['lengths'], torch.zeros((pad_n,), dtype=torch.long, device=self.device)])
                    bin['age'] = torch.cat([bin['age'], torch.zeros((pad_n,), dtype=torch.long, device=self.device)])

    def print_bins(self, tokenizer, max_chars=100):
        for name, bin in self.bins.items():
            print(f"\n== {name} ==")
            for i in range(min(bin['beams'].shape[0], 10)):
                l = bin['lengths'][i].item()
                score = bin['scores'][i].item()
                if l == 0: continue
                tokens = bin['beams'][i, :l].tolist()
                text = tokenizer.decode(tokens)
                print(f"[{i}] Score: {score:.2f} | {text    }")

    # --- Universal interface to scoring ---
    def call_score_fn(self, fn, beams, scores, lengths, tokenizer, **kwargs):
        # kwargs are the 'params' from the bin's configuration, e.g., {'prompt_len': 10}

        # Arguments that are always available for scoring functions.
        # These are the outputs from the main model's expansion step or current beam state.
        core_args = {
            'beams': beams,      # Current candidate token sequences
            'scores': scores,    # Log probabilities for these sequences (token-wise)
            'lengths': lengths,  # Lengths of these sequences
            'tokenizer': tokenizer
        }
        
        # Combine core_args with specific parameters from the bin's configuration.
        # bin_params (passed as **kwargs here) can override core_args if there's a name clash,
        # though unlikely for these core names. More importantly, bin_params adds extra
        # arguments needed by specific scoring functions.
        final_call_args = core_args.copy() # Start with core args
        final_call_args.update(kwargs)     # Add/override with bin-specific params
        
        # Call the scoring function with all prepared arguments.
        # Python's argument unpacking will match named parameters and pass the rest via **kwargs if defined.
        return fn(**final_call_args)