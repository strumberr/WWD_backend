from transformers import AutoTokenizer, GPT2Model
import torch
import os

# Auto-download the model + tokenizer from Hugging Face Hub
model_dir = "./model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print("Downloading model from Hugging Face...")

# Download only if missing (Render rebuilds often)
if not os.path.exists(f"{model_dir}/model.pt"):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="strumber/magnusTransformer",
        repo_type="model",
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )

print("Model download complete.")
