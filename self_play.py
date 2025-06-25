import chess
import chess.pgn
import datetime
from test import predict_best_move  # import your working function

# === Create a random starting position (optional) ===
board = chess.Board()

# === Setup PGN for saving ===
game = chess.pgn.Game()
game.headers["Event"] = "Magnus vs Magnus Self-Play"
game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
game.setup(board)
node = game

print("Starting Magnus vs Magnus Self-Play\n")

turn = 0
max_turns = 100  # safeguard in case of drawish looping

while not board.is_game_over() and turn < max_turns:
    fen = board.fen()
    move_uci = predict_best_move(fen)
    
    if move_uci in ("[game over]", "[no valid move]"):
        print(f"Turn {turn+1}: {move_uci}")
        break

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        print(f"[!] Invalid move predicted: {move_uci}")
        break

    board.push(move)
    node = node.add_variation(move)

    print(f"Turn {turn+1}: {'White' if board.turn == chess.BLACK else 'Black'} played {move_uci}")
    turn += 1

# === Finalize PGN ===
game.headers["Result"] = board.result()
with open("magnus_vs_magnus.pgn", "w") as f:
    f.write(str(game))

print(f"\nGame over. Result: {board.result()}")
print("Saved as magnus_vs_magnus.pgn")
