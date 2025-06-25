from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_best_move



app = FastAPI()

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
    
    predicted_move: str  # The predicted move in UCI format
    confidence: float     # Confidence value for the prediction


class BoardInput(BaseModel):
    board: str


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