import os
import torch
import torch.nn as nn
import chess
from transformers import GPT2Model, GPT2Config, PreTrainedTokenizerFast
from huggingface_hub import snapshot_download
import torch.nn.functional as F


# === Auto-download model from Hugging Face if not present ===
model_dir = "model"

if not os.path.exists(f"{model_dir}/model.pt"):
    print("Downloading model from Hugging Face...")
    snapshot_download(
        repo_id="strumber/magnusTransformer",
        repo_type="model",
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete.")

# === GPT2Config (hardcoded, matches original config.json) ===
config = GPT2Config(
    activation_function="gelu_new",
    attn_pdrop=0.1,
    bos_token_id=1,
    embd_pdrop=0.1,
    eos_token_id=2,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    model_type="gpt2",
    n_ctx=1024,
    n_embd=768,
    n_head=12,
    n_inner=None,
    n_layer=12,
    n_positions=1024,
    reorder_and_upcast_attn=False,
    resid_pdrop=0.1,
    scale_attn_by_inverse_layer_idx=False,
    scale_attn_weights=True,
    summary_activation=None,
    summary_first_dropout=0.1,
    summary_proj_to_labels=True,
    summary_type="cls_index",
    summary_use_proj=True,
    torch_dtype="float32",
    use_cache=True,
    vocab_size=72
)

# === Model Definition ===
class ChessMoveClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.n_embd, 4096)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(self.dropout(hidden_state))
        return {"logits": logits}

# === Load model and tokenizer ===
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{model_dir}/tokenizer.json")
model = ChessMoveClassifier(config=config)
model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Prediction function ===
def predict_best_move(fen: str) -> tuple[str, float]:
    board = chess.Board(fen)
    if board.is_game_over():
        return "[game over]", 0.0

    inputs = tokenizer(fen, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)["logits"][0]
        legal_moves = list(board.legal_moves)

        move_confidences = []

        for move in legal_moves:
            idx = move.from_square * 64 + move.to_square
            logit = logits[idx]
            move_confidences.append((move, logit))

        # Softmax over just the legal move logits
        move_logits = torch.tensor([logit for _, logit in move_confidences])
        probs = F.softmax(move_logits, dim=0)

        best_idx = torch.argmax(probs).item()
        best_move = move_confidences[best_idx][0]
        best_conf = probs[best_idx].item()

        return best_move.uci(), best_conf


    return "[no valid move]", 0.0



# === Example run ===
# if __name__ == "__main__":
#     example_fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
#     predicted = predict_best_move(example_fen)
#     print(f"Predicted move: {predicted}")
