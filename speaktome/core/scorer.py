"""Scoring utilities for beam search and research experimentation.

This module lazily loads GPT‑2 resources and provides a suite of batched
scoring functions.  The design favours vectorised tensor operations so models
and heuristics can scale gracefully.  Future versions will introduce a queue
and mailbox mechanism so that tokenisation, model inference, and scoring can be
scheduled in worker threads.  This will allow the scoring pipeline to operate
on arbitrary array types while delivering results to dynamic mailboxes for
maximum throughput.
"""

from typing import Callable, Dict

import os
import torch
import torch.nn.functional as F
import queue

from ..tensors import (
    AbstractTensorOperations,
    get_tensor_operations,
)

from ..util.lazy_loader import lazy_import, optional_import
from .. import config
# --- END HEADER ---

class Scorer:
    """Lazy GPT-2 scorer with pluggable vectorised heuristics and bin management."""

    def __init__(self, tensor_ops: AbstractTensorOperations | None = None) -> None:
        """Initialise defaults, tensor operations and resolve the GPT-2 model path."""

        self._model = None
        self._tokenizer = None

        model_path = os.environ.get("GPT2_MODEL_PATH")
        if not model_path:
            root_dir = os.path.dirname(os.path.dirname(__file__))
            local_path = os.path.join(root_dir, "models", "gpt2")
            model_path = local_path if os.path.isdir(local_path) else "gpt2"
        self.model_path = model_path

        self.tensor_ops = tensor_ops or get_tensor_operations()

        self.default_scorer = Scorer.mean_logprob_score
        self.default_k = 5
        self.default_temp = 1.5
        self.default_pre_top_k = 50
        self.default_pre_temp = 1.5
        self.default_cull_after = 3

        # Default policy is intentionally simple and may be customised.
        self.default_score_bins = {
            "mean_logprob": (Scorer.mean_logprob_score, self.default_k, self.default_temp)
        }
        self.default_score_policy = {
            "pre_temp": self.default_pre_temp,
            "pre_top_k": self.default_pre_top_k,
            "cull_after": self.default_cull_after,
            "score_bins": self.default_score_bins,
            "lookahead_steps": 1,
        }

        # Meta-beam manager state (initialised when bins are configured)
        self.bins = {}
        self.survival_age = {}
        self.top_beams = {}
        self.delivery_queue = queue.Queue(maxsize=8)
        self.max_len = 0
        self.cull_after = 0
        self.device = config.DEVICE

    def _ensure_model(self) -> None:
        """Load the GPT-2 model and tokenizer on first use."""

        if self._model is None or self._tokenizer is None:
            GPT2LMHeadModel = optional_import("transformers.GPT2LMHeadModel")
            GPT2Tokenizer = optional_import("transformers.GPT2Tokenizer")
            if GPT2LMHeadModel is None or GPT2Tokenizer is None:
                raise RuntimeError(
                    "Transformers is required for the full beam search demo."
                )
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = (
                GPT2LMHeadModel.from_pretrained(self.model_path)
                .to(config.DEVICE)
                .eval()
            )

    @property
    def tokenizer(self):
        """Return the lazily loaded tokenizer."""

        self._ensure_model()
        return self._tokenizer

    @property
    def model(self):
        """Return the lazily loaded language model."""

        self._ensure_model()
        return self._model

    def preload_models(self) -> None:
        """Eagerly load both tokenizer and model."""

        self._ensure_model()
    @staticmethod
    def cosine_similarity_score(
        beams,
        scores,
        lengths,
        tokenizer,
        existing_embeddings=None,
        threshold: float = 0.92,
        **kwargs,
    ):
        """Reward beams that diverge in embedding space.

        Parameters are batched tensors.  If ``existing_embeddings`` are
        provided, the method computes similarity against this reference set;
        otherwise pairwise similarities within the batch are used.  The score is
        negative cosine similarity so that unique beams receive higher values.
        """

        texts = [tokenizer.decode(b[:l], skip_special_tokens=True) for b, l in zip(beams, lengths)]
        model = config.get_sentence_transformer_model()
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            device=str(config.DEVICE),
            batch_size=128,
        )

        if existing_embeddings is not None:
            sims = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1), existing_embeddings.unsqueeze(0), dim=-1
            )
            max_sims, _ = sims.max(dim=1)
        else:
            sims = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
            )
            sims.fill_diagonal_(-float("inf"))
            max_sims, _ = sims.max(dim=1)

        return -max_sims.float().to(beams.device)


    @staticmethod
    def get_available_scoring_functions() -> Dict[str, Callable]:
        """Map string names to scoring callables."""

        return {
            "mean_logprob": Scorer.mean_logprob_score,
            "next_token_logprob": Scorer.next_token_logprob_score,
            "sum_logprob": Scorer.sum_logprob_score,
            "sweet_spot": Scorer.sweet_spot_score,
            "ngram_diversity": Scorer.ngram_diversity_score,
            "pairwise_diversity": Scorer.pairwise_diversity_score,
            "cos_sim": Scorer.cosine_similarity_score
        }

    @staticmethod
    def next_token_logprob_score(beams=None, scores=None, lengths=None, tokenizer=None, **kwargs):
        """Score using the log probability of the most recent token."""

        idx = (lengths - 1).clamp(min=0)
        return scores[torch.arange(beams.shape[0]), idx]

    @staticmethod
    def mean_logprob_score(beams=None, scores=None, lengths=None, tokenizer=None, **kwargs):
        """Average log probability across all non-pad tokens."""

        mask = beams != tokenizer.pad_token_id
        total = (scores * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return total / count

    @staticmethod
    def sum_logprob_score(beams=None, scores=None, lengths=None, tokenizer=None, **kwargs):
        """Total log probability over the entire sequence."""

        if scores is not None and beams is not None:
            mask = beams != tokenizer.pad_token_id
            return (scores * mask).sum(dim=1)
        raise ValueError("sum_logprob_score requires tokens and logprobs")
    @staticmethod
    def sweet_spot_score(beams=None, scores=None, lengths=None, tokenizer=None, prompt_len=None, **kwargs):
        """Reward lengths near twice the prompt length."""

        if lengths is not None and prompt_len is not None:
            sweet_spot = 2 * prompt_len
            return -((lengths - sweet_spot).abs().float())
        raise ValueError("sweet_spot_score requires lengths and prompt_len")
    @staticmethod
    def ngram_diversity_score(beams=None, scores=None, lengths=None, tokenizer=None, n: int = 2, penalty: float = -1.0, **kwargs):
        """Penalise repeated n-grams within each beam."""
        
        batch, seq_len = beams.shape
        device = beams.device

        if seq_len < n:
            return torch.zeros(batch, device=device)

        base = getattr(tokenizer, "vocab_size", None)
        if base is None:
            base = int(beams.max().item()) + 1

        windows = beams.unfold(1, n, 1)
        valid_counts = (lengths - n + 1).clamp(min=0)
        max_windows = windows.shape[1]

        mask = torch.arange(max_windows, device=device).unsqueeze(0) < valid_counts.unsqueeze(1)
        multipliers = (base ** torch.arange(n, device=device)).view(1, 1, -1)
        hashed = (windows * multipliers).sum(dim=2)
        rows = torch.arange(batch, device=device).unsqueeze(1).expand(batch, max_windows)[mask]
        hashed_flat = hashed[mask]

        pairs = torch.stack([rows, hashed_flat], dim=1)
        unique_pairs, counts = torch.unique(pairs, return_counts=True, dim=0)
        duplicate_counts = counts - 1
        penalties = torch.zeros(batch, device=device, dtype=torch.float)
        penalties.index_add_(0, unique_pairs[:, 0], duplicate_counts.float() * penalty)

        return -penalties
    @staticmethod
    def pairwise_diversity_score(beams=None, lengths=None, tokenizer=None, **kwargs):
        """Discourage token overlap across beams using Jaccard distance."""
        batch = beams.shape[0]
        device = beams.device
        if tokenizer is not None and hasattr(tokenizer, "vocab_size"):
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = int(beams.max().item()) + 1

        # Build multi-hot representation for each beam without Python loops
        row_idx = torch.arange(batch, device=device).unsqueeze(1).expand_as(beams)
        col_mask = torch.arange(beams.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        rows = row_idx[col_mask]
        cols = beams[col_mask]
        one_hot = torch.zeros(batch, vocab_size, device=device, dtype=torch.float32)
        one_hot.index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float32), accumulate=True)
        one_hot_bool = one_hot.bool()

        inter = torch.matmul(one_hot_bool.float(), one_hot_bool.t().float())
        union = (
            one_hot_bool.sum(dim=1, keepdim=True) +
            one_hot_bool.sum(dim=1, keepdim=True).t() - inter
        )
        sim_matrix = torch.zeros_like(inter)
        mask_nonzero = union > 0
        sim_matrix[mask_nonzero] = inter[mask_nonzero] / union[mask_nonzero]
        sim_matrix.fill_diagonal_(0)

        diversity_scores = -sim_matrix.sum(dim=1) / (batch - 1)
        return diversity_scores



    # ------------------------------------------------------------------
    # MetaBeamManager functionality has been folded directly into Scorer
    # ------------------------------------------------------------------

    def init_bins(self, bins_config, max_len, device=None, cull_after=1):
        """Configure scoring bins using the tensor abstraction layer."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.cull_after = cull_after
        self.top_beams = {}
        self.delivery_queue = queue.Queue(maxsize=8)
        self.bins = {}
        self.survival_age = {}
        for name, cfg in bins_config.items():
            N = cfg['width']
            self.bins[name] = {
                'fn': cfg['fn'],
                'params': cfg.get('params', {}),
                'beams': self.tensor_ops.full((N, max_len), -1, dtype=torch.long, device=self.device),
                'scores': self.tensor_ops.full((N,), float('-inf'), dtype=torch.float32, device=self.device),
                'lengths': self.tensor_ops.zeros((N,), dtype=torch.long, device=self.device),
                'age': self.tensor_ops.zeros((N,), dtype=torch.long, device=self.device),
                'width': N
            }

    def update_bins(self, beams, scores, lengths, tokenizer, round_idx=0):
        """Update all bins with new candidate beams."""
        for name, bin in self.bins.items():
            N = bin['beams'].shape[0]
            bin_scores = self.call_score_fn(bin['fn'], beams, scores, lengths, tokenizer, **bin['params'])
            pad_n = bin['beams'].shape[1] - beams.shape[1]
            padded_beams = self.tensor_ops.pad(beams, (0, pad_n)) if pad_n > 0 else beams
            all_scores = self.tensor_ops.cat([bin['scores'], bin_scores], dim=0)
            all_beams = self.tensor_ops.cat([bin['beams'], padded_beams], dim=0)
            all_lengths = self.tensor_ops.cat([bin['lengths'], lengths], dim=0)
            all_age = self.tensor_ops.cat([bin['age'] + 1, self.tensor_ops.zeros(bin_scores.shape, dtype=torch.long, device=self.device)], dim=0)

            _, top_idx = self.tensor_ops.topk(all_scores, k=N, dim=0)
            bin['scores'] = all_scores[top_idx]
            bin['beams'] = all_beams[top_idx]
            bin['lengths'] = all_lengths[top_idx]
            bin['age'] = all_age[top_idx]

            if self.cull_after > 0:
                keep = bin['age'] <= self.cull_after
                for k in ['scores', 'beams', 'lengths', 'age']:
                    bin[k] = bin[k][keep]
                pad_n = N - bin['scores'].shape[0]
                if pad_n > 0:
                    bin['scores'] = self.tensor_ops.cat([
                        bin['scores'],
                        self.tensor_ops.full((pad_n,), float('-inf'), dtype=torch.float32, device=self.device)
                    ], dim=0)
                    bin['beams'] = self.tensor_ops.cat([
                        bin['beams'],
                        self.tensor_ops.full((pad_n, self.max_len), -1, dtype=torch.long, device=self.device)
                    ], dim=0)
                    bin['lengths'] = self.tensor_ops.cat([
                        bin['lengths'],
                        self.tensor_ops.zeros((pad_n,), dtype=torch.long, device=self.device)
                    ], dim=0)
                    bin['age'] = self.tensor_ops.cat([
                        bin['age'],
                        self.tensor_ops.zeros((pad_n,), dtype=torch.long, device=self.device)
                    ], dim=0)

    def print_bins(self, tokenizer, max_chars=100):
        for name, bin in self.bins.items():
            print(f"\n== {name} ==")
            for i in range(min(bin['beams'].shape[0], 10)):
                l = int(bin['lengths'][i].item())
                score = float(bin['scores'][i].item())
                if l == 0:
                    continue
                tokens = bin['beams'][i, :l].tolist()
                text = tokenizer.decode(tokens)
                print(f"[{i}] Score: {score:.2f} | {text}")

    def call_score_fn(self, fn, beams, scores, lengths, tokenizer, **kwargs):
        core_args = {
            'beams': beams,
            'scores': scores,
            'lengths': lengths,
            'tokenizer': tokenizer
        }
        final_call_args = core_args.copy()
        final_call_args.update(kwargs)
        return fn(**final_call_args)

