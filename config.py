# Third-party imports
import torch
from sentence_transformers import SentenceTransformer

# Configuration constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_LIMIT = 2500
LENGTH_LIMIT = 1023

# Initialize the SentenceTransformer model here
sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=DEVICE)