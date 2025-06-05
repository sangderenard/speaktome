# Standard library imports
from typing import Callable, Dict

# Third-party imports
import torch
import os

# Defer heavy imports until needed
from .lazy_loader import lazy_import

# Local application/library specific imports
# Access configuration dynamically to allow device changes at runtime
from . import config

class Scorer:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        model_path = os.environ.get("GPT2_MODEL_PATH")
        if not model_path:
            local_path = os.path.join(os.path.dirname(__file__), "models", "gpt2")
            model_path = local_path if os.path.isdir(local_path) else "gpt2"
        self.model_path = model_path
        self.default_scorer = Scorer.mean_logprob_score
        self.default_k = 5
        self.default_temp = 1.5
        self.default_pre_top_k = 50
        self.default_pre_temp = 1.5
        self.default_cull_after = 3
        self.default_score_bins = {
                # "mean_logprob_score" is a good default.
                "mean_logprob": (Scorer.mean_logprob_score, self.default_k, self.default_temp),
#                "next_token_logprob": (Scorer.next_token_logprob_score, self.default_k, self.default_temp),
#                "sum_logprob": (Scorer.sum_logprob_score, self.default_k, self.default_temp),
#                "cosine_similarity_score": (Scorer.cosine_similarity_score, self.default_k, self.default_temp),
            }
        self.default_score_policy = {
            # These are parameters for the BeamSearchInstruction
            "pre_temp": self.default_pre_temp,
            "pre_top_k": self.default_pre_top_k,
            "cull_after": self.default_cull_after,
            "score_bins": self.default_score_bins,
            "lookahead_steps": 1 # Default lookahead steps
        }

    def _ensure_model(self):
        if self._model is None or self._tokenizer is None:
            GPT2LMHeadModel = lazy_import('transformers.GPT2LMHeadModel')
            GPT2Tokenizer = lazy_import('transformers.GPT2Tokenizer')
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = (
                GPT2LMHeadModel.from_pretrained(self.model_path)
                .to(config.DEVICE)
                .eval()
            )

    @property
    def tokenizer(self):
        self._ensure_model()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_model()
        return self._model

    def preload_models(self):
        """Load tokenizer and model immediately."""
        self._ensure_model()
    @staticmethod
    def cosine_similarity_score(beams, scores, lengths, tokenizer, existing_embeddings=None, threshold=0.92, **kwargs):
        # beams: [N, L] token tensors
        texts = [tokenizer.decode(b[:l], skip_special_tokens=True)
                for b, l in zip(beams, lengths)]
        model = config.get_sentence_transformer_model()
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            device=str(config.DEVICE),
            batch_size=128,
        )
        # Self-similarity or against external set
        if existing_embeddings is not None:
            # Similarity to previous (could be active beams, etc.)
            sims = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1), existing_embeddings.unsqueeze(0), dim=-1
            )  # [N, M]
            # For each beam, penalty = max similarity to any reference
            max_sims, _ = sims.max(dim=1)
        else:
            sims = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
            )  # [N, N]
            # Set diagonal to -inf so beam doesn't penalize itself
            sims.fill_diagonal_(-float("inf"))
            max_sims, _ = sims.max(dim=1)
        # Return a *negative* similarity, so beams that are most unique get highest score
        return -max_sims.float().to(beams.device)


    @staticmethod
    def get_available_scoring_functions() -> Dict[str, Callable]:
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
        idx = (lengths - 1).clamp(min=0)
        # Assume scores shape [N, seq_len]
        return scores[torch.arange(beams.shape[0]), idx]

    @staticmethod
    def mean_logprob_score(beams=None, scores=None, lengths=None, tokenizer=None, **kwargs):
        mask = (beams != tokenizer.pad_token_id)
        total = (scores * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return total / count

    # --- Universal scoring example ---
    @staticmethod
    def sum_logprob_score(beams=None, scores=None, lengths=None, tokenizer=None, **kwargs):
        # Accepts: beams (tokens), scores (logprobs), lengths, tokenizer
        if scores is not None and beams is not None:
            mask = (beams != tokenizer.pad_token_id)
            return (scores * mask).sum(dim=1)
        raise ValueError("sum_logprob_score requires tokens and logprobs")
    @staticmethod
    def sweet_spot_score(beams=None, scores=None, lengths=None, tokenizer=None, prompt_len=None, **kwargs):
        # Accepts: beams (tokens), lengths, prompt_len
        if lengths is not None and prompt_len is not None:
            sweet_spot = 2 * prompt_len
            return -((lengths - sweet_spot).abs().float())
        raise ValueError("sweet_spot_score requires lengths and prompt_len")
    @staticmethod
    def ngram_diversity_score(beams=None, scores=None, lengths=None, tokenizer=None, n=2, penalty=-1.0, **kwargs):
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
        # Penalize beams that are too similar to each other (measured by token overlap)
        batch = beams.shape[0]
        diversity_scores = torch.zeros(batch, device=beams.device)
        # Jaccard distance for all beam pairs
        for i in range(batch):
            s1 = set(beams[i, :lengths[i]].tolist())
            sim = 0
            for j in range(batch):
                if i == j: continue
                s2 = set(beams[j, :lengths[j]].tolist())
                inter = len(s1 & s2)
                union = len(s1 | s2)
                if union > 0:
                    sim += inter / union
            # High sim = low diversity, so negate
            diversity_scores[i] = -sim / (batch-1)
        return diversity_scores
