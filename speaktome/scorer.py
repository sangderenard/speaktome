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

from .lazy_loader import lazy_import, optional_import
from . import config

class Scorer:
    """Lazy GPT-2 scorer with pluggable vectorised heuristics."""

    def __init__(self) -> None:
        """Initialise defaults and resolve the GPT-2 model path."""

        self._model = None
        self._tokenizer = None

        model_path = os.environ.get("GPT2_MODEL_PATH")
        if not model_path:
            root_dir = os.path.dirname(os.path.dirname(__file__))
            local_path = os.path.join(root_dir, "models", "gpt2")
            model_path = local_path if os.path.isdir(local_path) else "gpt2"
        self.model_path = model_path

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

        batch = beams.shape[0]
        penalties = torch.zeros(batch, device=beams.device)
        for i in range(batch):
            l = lengths[i].item()
            tokens = beams[i, :l].tolist()
            ngrams = set()
            count = 0
            for j in range(l - n + 1):
                ng = tuple(tokens[j:j+n])
                if ng in ngrams:
                    count += 1
                else:
                    ngrams.add(ng)
            penalties[i] = penalty * count
        return -penalties
    @staticmethod
    def pairwise_diversity_score(beams=None, lengths=None, tokenizer=None, **kwargs):
        """Discourage token overlap across beams using Jaccard distance."""

        batch = beams.shape[0]
        diversity_scores = torch.zeros(batch, device=beams.device)

        for i in range(batch):
            s1 = set(beams[i, :lengths[i]].tolist())
            sim = 0
            for j in range(batch):
                if i == j:
                    continue
                s2 = set(beams[j, :lengths[j]].tolist())
                inter = len(s1 & s2)
                union = len(s1 | s2)
                if union > 0:
                    sim += inter / union
            diversity_scores[i] = -sim / (batch - 1)

        return diversity_scores
