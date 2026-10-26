from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.services.model_service import CivicAIModelService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "grievances.csv"
ARTIFACT = ROOT / "artifacts" / "category_model.joblib"
UI_FILE = ROOT / "app" / "templates" / "index.html"

app = FastAPI(
    title="CivicAI ML Service",
    description="Grievance category classification and rule-based urgency assessment.",
    version="1.1.0",
)
service = CivicAIModelService(ARTIFACT)


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)


class EvaluationExample(BaseModel):
    text: str = Field(min_length=3, max_length=10000)
    category: str = Field(min_length=1, max_length=100)


class EvaluateRequest(BaseModel):
    examples: list[EvaluationExample] = Field(min_length=1, max_length=5000)


class TrainRequest(BaseModel):
    dataset_path: str | None = None


@app.on_event("startup")
def load_model():
    service.load()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.is_loaded, "service": "civicai-ml-service"}


@app.get("/model/info")
def model_info():
    return service.info()


@app.post("/predict")
def predict(request: PredictRequest):
    return service.predict(request.text)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    try:
        return service.evaluate(
            [x.text for x in request.examples],
            [x.category for x in request.examples],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(request: TrainRequest):
    path = Path(request.dataset_path).expanduser() if request.dataset_path else DEFAULT_DATASET
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="Dataset path must be inside the project directory.")
    try:
        return service.train_from_csv(path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
