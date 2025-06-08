# Third-party imports
import os

from .faculty import Faculty, DEFAULT_FACULTY
# --- END HEADER ---

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime path
    torch = None

TORCH_AVAILABLE = DEFAULT_FACULTY in (Faculty.TORCH, Faculty.PYGEO)
PYGEO_AVAILABLE = DEFAULT_FACULTY is Faculty.PYGEO

SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SentenceTransformer = _SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

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
    if SentenceTransformer is None:
        raise RuntimeError(
            "SentenceTransformer is required for embedding features."
        )
    if sentence_transformer_model is None:
        model_name_or_path = os.environ.get("SENTENCE_TRANSFORMER_MODEL_PATH")
        if not model_name_or_path:
            root_dir = os.path.dirname(os.path.dirname(__file__))
            local_path = os.path.join(
                root_dir,
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

