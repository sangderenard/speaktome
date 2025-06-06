# Third-party imports
import os
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - runtime path
    torch = None
    TORCH_AVAILABLE = False
from sentence_transformers import SentenceTransformer

# Configuration constants
if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    class _DummyDevice:
        type = "cpu"

        def __str__(self):
            return "cpu"

    DEVICE = _DummyDevice()
GPU_LIMIT = 2500
LENGTH_LIMIT = 1023

# Placeholder for lazy-loaded SentenceTransformer model
sentence_transformer_model = None


def get_sentence_transformer_model() -> SentenceTransformer:
    """Return the SentenceTransformer model, loading it on first use.

    If the environment variable ``SENTENCE_TRANSFORMER_MODEL_PATH`` is set,
    its value will be used as the model path to avoid network downloads.
    """
    global sentence_transformer_model
    if sentence_transformer_model is None:
        model_name_or_path = os.environ.get("SENTENCE_TRANSFORMER_MODEL_PATH")
        if not model_name_or_path:
            local_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                "paraphrase-MiniLM-L6-v2",
            )
            if os.path.isdir(local_path):
                model_name_or_path = local_path
            else:
                model_name_or_path = "paraphrase-MiniLM-L6-v2"
        sentence_transformer_model = SentenceTransformer(
            model_name_or_path, device=DEVICE
        )
    return sentence_transformer_model

