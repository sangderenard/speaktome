"""Tests for speaktome.core.model_engine -- registry/lazy-loading only, no real model download."""

import types

import pytest

from speaktome.core import model_engine


def test_registry_has_both_engines_and_a_default():
    assert "gpt2" in model_engine.ENGINES
    assert "qwen3-1.7b-base" in model_engine.ENGINES
    assert model_engine.DEFAULT_ENGINE in model_engine.ENGINES


def test_load_engine_unknown_name_raises_with_available_names():
    with pytest.raises(ValueError, match="gpt2"):
        model_engine.load_engine("not-a-real-engine")


def test_load_engine_is_lazy_until_first_property_access():
    engine = model_engine.load_engine("gpt2")
    assert engine._model is None
    assert engine._tokenizer is None


def test_resolve_model_path_prefers_env_var_override(monkeypatch):
    spec = model_engine.ENGINES["gpt2"]
    monkeypatch.setenv(spec.env_var, "/some/local/override")
    assert model_engine._resolve_model_path(spec) == "/some/local/override"


def test_resolve_model_path_falls_back_to_hf_id_when_nothing_local(monkeypatch):
    spec = model_engine.ENGINES["qwen3-1.7b-base"]
    monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.setattr(model_engine.os.path, "isdir", lambda p: False)
    assert model_engine._resolve_model_path(spec) == spec.hf_id


def test_ensure_loaded_uses_the_engines_own_hf_id_and_trust_flag(monkeypatch):
    calls = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeModel:
        def to(self, device):
            calls["to_device"] = device
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    fake_tokenizer_cls = types.SimpleNamespace(
        from_pretrained=lambda path, trust_remote_code=False: (
            calls.setdefault("tokenizer_path", path),
            calls.setdefault("tokenizer_trust", trust_remote_code),
            FakeTokenizer(),
        )[-1]
    )
    fake_model_cls = types.SimpleNamespace(
        from_pretrained=lambda path, trust_remote_code=False: (
            calls.setdefault("model_path", path),
            calls.setdefault("model_trust", trust_remote_code),
            FakeModel(),
        )[-1]
    )
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=fake_tokenizer_cls, AutoModelForCausalLM=fake_model_cls
    )
    monkeypatch.setattr(model_engine, "optional_import", lambda name: fake_transformers)

    engine = model_engine.load_engine("qwen3-1.7b-base", device="cpu")
    tokenizer = engine.tokenizer
    model = engine.model

    assert calls["tokenizer_path"] == "Qwen/Qwen3-1.7B-Base"
    assert calls["model_path"] == "Qwen/Qwen3-1.7B-Base"
    assert calls["to_device"] == "cpu"
    assert calls["eval_called"] is True
    assert tokenizer.pad_token == "<eos>"  # filled in from eos_token since it started None
    assert model is engine.model  # cached, not reloaded on second access


def test_ensure_loaded_raises_clearly_when_transformers_is_missing(monkeypatch):
    monkeypatch.setattr(model_engine, "optional_import", lambda name: None)
    engine = model_engine.load_engine("gpt2")
    with pytest.raises(RuntimeError, match="transformers"):
        engine.tokenizer
