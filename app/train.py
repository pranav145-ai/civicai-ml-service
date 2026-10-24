from pathlib import Path
from app.services.model_service import CivicAIModelService

ROOT = Path(__file__).resolve().parents[1]
service = CivicAIModelService(ROOT / "artifacts" / "category_model.joblib")

if __name__ == "__main__":
    print(service.train_from_csv(ROOT / "dataset" / "grievances.csv"))
