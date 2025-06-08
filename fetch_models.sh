#!/usr/bin/env bash
# Download GPT-2 and SentenceTransformer models for offline use
set -euo pipefail

MODELS_DIR="models"
mkdir -p "${MODELS_DIR}"

python <<'PY'
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

models_dir = "models"
gpt2_dir = os.path.join(models_dir, "gpt2")
if not os.path.exists(gpt2_dir):
    GPT2LMHeadModel.from_pretrained("gpt2").save_pretrained(gpt2_dir)
    GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(gpt2_dir)

st_dir = os.path.join(models_dir, "paraphrase-MiniLM-L6-v2")
if not os.path.exists(st_dir):
    SentenceTransformer("paraphrase-MiniLM-L6-v2").save(st_dir)
PY

echo "Models downloaded to ${MODELS_DIR}"
echo "Set GPT2_MODEL_PATH=${MODELS_DIR}/gpt2"
echo "Set SENTENCE_TRANSFORMER_MODEL_PATH=${MODELS_DIR}/paraphrase-MiniLM-L6-v2"

