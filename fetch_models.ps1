# Windows PowerShell script to download GPT-2 and SentenceTransformer models

$ErrorActionPreference = 'Continue'

function Safe-Run([ScriptBlock]$cmd) {
    try { & $cmd }
    catch {
        Write-Host "Warning: $($_.Exception.Message)"
    }
}

$modelsDir = 'models'
if (-not (Test-Path $modelsDir)) {
    Safe-Run { New-Item -ItemType Directory -Path $modelsDir | Out-Null }
}

@'
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

models_dir = 'models'
gpt2_dir = os.path.join(models_dir, 'gpt2')
if not os.path.exists(gpt2_dir):
    GPT2LMHeadModel.from_pretrained('gpt2').save_pretrained(gpt2_dir)
    GPT2Tokenizer.from_pretrained('gpt2').save_pretrained(gpt2_dir)

st_dir = os.path.join(models_dir, 'paraphrase-MiniLM-L6-v2')
if not os.path.exists(st_dir):
    SentenceTransformer('paraphrase-MiniLM-L6-v2').save(st_dir)
'@ | Safe-Run { .\.venv\Scripts\python.exe - }

Write-Host "Models downloaded to $modelsDir"
Write-Host "Set GPT2_MODEL_PATH=$modelsDir\gpt2"
Write-Host "Set SENTENCE_TRANSFORMER_MODEL_PATH=$modelsDir\paraphrase-MiniLM-L6-v2"
