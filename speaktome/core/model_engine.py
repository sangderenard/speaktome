#!/usr/bin/env python3
"""Pluggable causal-LM loading: pick the backing model by name, swap engines freely.

A single small registry (``ENGINES``) maps a short name to where its
weights live -- everything downstream (``PyTorchModelWrapper``,
``ImplicitBackpathScorer``, ``FluxGraph``, ...) only ever asks a
``ModelEngine`` for ``.tokenizer``/``.model``/``.device`` and never cares
which family it actually is. Both registered engines load through the
same generic ``transformers.AutoModelForCausalLM``/``AutoTokenizer``
path, so adding another engine is just another ``EngineSpec`` entry, not
new loading code.

Local caching follows the same convention as ``Scorer``/``fetch_models.*``:
an env var override, else ``speaktome/models/<local_dirname>`` if present,
else the bare HuggingFace hub id (downloaded and cached by transformers
itself on first use).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .. import config
from ..util.lazy_loader import optional_import
# --- END HEADER ---


@dataclass(frozen=True)
class EngineSpec:
    name: str
    hf_id: str
    local_dirname: str
    env_var: str
    trust_remote_code: bool = False


ENGINES: Dict[str, EngineSpec] = {
    "gpt2": EngineSpec(
        name="gpt2",
        hf_id="gpt2",
        local_dirname="gpt2",
        env_var="GPT2_MODEL_PATH",
    ),
    "qwen3-1.7b-base": EngineSpec(
        name="qwen3-1.7b-base",
        hf_id="Qwen/Qwen3-1.7B-Base",
        local_dirname="qwen3-1.7b-base",
        env_var="QWEN3_1_7B_BASE_MODEL_PATH",
    ),
}

DEFAULT_ENGINE = "qwen3-1.7b-base"


def _resolve_model_path(spec: EngineSpec) -> str:
    override = os.environ.get(spec.env_var)
    if override:
        return override
    root_dir = os.path.dirname(os.path.dirname(__file__))  # speaktome/
    local_path = os.path.join(root_dir, "models", spec.local_dirname)
    return local_path if os.path.isdir(local_path) else spec.hf_id


class ModelEngine:
    """Lazily loads one named engine's tokenizer + causal LM, cached after first use.

    Mirrors ``Scorer``'s lazy ``.tokenizer``/``.model`` properties so it
    drops into existing call sites (``PyTorchModelWrapper(engine.model)``,
    ``ImplicitBackpathScorer(wrapper, engine.tokenizer, ...)``) without
    those sites needing to know which engine is actually loaded.
    """

    def __init__(self, spec: EngineSpec, device: Any = None):
        self.spec = spec
        self.device = device if device is not None else config.DEVICE
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        transformers_mod = optional_import("transformers")
        auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM", None)
        auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer", None)
        if auto_model_cls is None or auto_tokenizer_cls is None:
            raise RuntimeError(f"transformers is required to load engine {self.spec.name!r}")

        model_path = _resolve_model_path(self.spec)
        tokenizer = auto_tokenizer_cls.from_pretrained(
            model_path, trust_remote_code=self.spec.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = (
            auto_model_cls.from_pretrained(model_path, trust_remote_code=self.spec.trust_remote_code)
            .to(self.device)
            .eval()
        )
        self._tokenizer = tokenizer
        self._model = model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def preload(self) -> None:
        """Eagerly load both tokenizer and model instead of waiting for first use."""
        self._ensure_loaded()


def load_engine(name: str = DEFAULT_ENGINE, device: Any = None) -> ModelEngine:
    """Look up a registered engine by name; raises with the valid names if it's not one."""
    try:
        spec = ENGINES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown engine {name!r}. Available engines: {sorted(ENGINES)}") from exc
    return ModelEngine(spec, device=device)
