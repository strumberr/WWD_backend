from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_best_move
import os
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()



# initial setup
model_path = "model/model.pt"
if not os.path.exists(model_path):
    print("[INFO] Model not found. Running setup.py...")
    result = subprocess.run(["python", "setup.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] setup.py failed:\n", result.stderr)
    else:
        print("[INFO] Model setup complete.")



app = FastAPI()
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify domains here
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


class ModelMetadata(BaseModel):
    """Metadata about the ML model."""

    model_name: str
    version: str
    accuracy: float
    tags: List[str]


class PredictionRequest(BaseModel):
    """Input format for making a prediction."""

    features: List[float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output format for prediction results."""
    
    predicted_move: str
    confidence: float


class BoardInput(BaseModel):
    board: str

class ExplanationResponse(BaseModel):
    explanation: str
    move: str




@app.get("/")
def read_root() -> dict:
    """
    Root endpoint.

    Returns:
        dict: A welcome message.
    """

    return {"message": "Welcome to the ML Model API"}


@app.get("/model-info", response_model=ModelMetadata)
def get_model_metadata() -> ModelMetadata:
    """
    Returns metadata about the machine learning model.

    Returns:
        ModelMetadata: Model name, version, accuracy, and tags.
    """

    return ModelMetadata(
        model_name="IrisClassifier",
        version="1.0.0",
        accuracy=0.97,
        tags=["iris", "scikit-learn", "classifier"]
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_move(request: BoardInput) -> PredictionResponse:
    """
    Predicts the next move based on the input FEN string (current board state).

    Args:
        request (BoardInput): The input FEN string for the current chess board.

    Returns:
        PredictionResponse: Predicted UCI move and confidence score.
    """
    predicted_move, confidence = predict_best_move(request.board)


    return PredictionResponse(
        predicted_move=str(predicted_move),
        confidence=float(confidence)
    )


@app.post("/explain-move", response_model=ExplanationResponse)
def explain_move(request: BoardInput) -> ExplanationResponse:
    """
    Uses OpenAI GPT to explain the predicted best move for the given board state.
    
    Args:
        request (BoardInput): The current FEN board state.

    Returns:
        ExplanationResponse: Short explanation for why this move is strong.
    """
    predicted_move, _ = predict_best_move(request.board)
    # predicted_move = request.predicted_move

    prompt = (
        f"You are Magnus Carlsen, a world chess champion. The board position is: '{request.board}' (FEN). "
        f"The suggested move is '{predicted_move}'. "
        f"Explain in clear, simple terms why this move is recommended. Use 2–3 short sentences. "
        f"Speak confidently, but make it easy for new players to understand. Use a clean format: first describe what the move does, then why it's useful. "
        f"Use basic chess words like 'attack', 'defend', 'center', 'open file', or 'check'. Avoid deep jargon."
    )

  
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324-fast",
            max_tokens=150,
            temperature=0.6,
            top_p=0.9,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Magnus Carlsen: a world chess champion. Speak clearly, confidently, and in a way beginners can understand. "
                        "Explain the suggested move using simple ideas and structure your response to show what the move does and why it's helpful."
                    )
                },
                {"role": "user", "content": prompt}
            ],
        )
        print(response)
        explanation = response.choices[0].message.content.strip()


        return ExplanationResponse(explanation=explanation, move=predicted_move)

    except Exception as e:

        return ExplanationResponse(explanation=f"[ERROR] Failed to get explanation: {str(e)}")
    