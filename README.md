# CivicAI ML Service

Standalone Python service for grievance category classification and rule-based urgency prediction.

## Design
- **Category:** TF-IDF + LinearSVC trained on labelled grievance text.
- **Urgency:** deterministic rules in `app/services/model_service.py`; this is not a second trained model.
- **Priority:** mirrors the validated urgency tier for initial triage.
- **Confidence:** normalized SVM decision-margin score for category selection; not a calibrated probability and not confidence in urgency.
- **Human review:** unsupported urgency or unavailable model output is not fabricated; the response requests human review.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.train
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Setup (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.train
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI at http://127.0.0.1:8000/docs.

## Endpoints
- `GET /health` — service health and model status.
- `GET /model/info` — model details and category list.
- `POST /predict` — category, urgency, priority, confidence score, and human-review flag.
- `POST /evaluate` — evaluate supplied labelled examples.
- `POST /train` — retrain using the local CSV dataset by default.

Example `POST /predict` body:
```json
{"text":"A burst water pipeline is flooding the shared residential street."}
```

## Urgency rules
- **Critical:** explicit immediate electrical injury hazards, or water contamination linked to reported illness/health impact.
- **High:** major water infrastructure failure (e.g. burst pipeline), or flooding affecting a shared public/residential area.
- **Medium:** service disruption, or an issue affecting multiple homes/buildings without an immediate hazard.
- **Low:** explicitly minor/unaffected issue, or no sustained disruption, major failure, or immediate hazard stated.
- Ambiguous cases return null urgency and require human review. Critical rules take precedence.

## Limitations
The included dataset is small and illustrative, not a validated benchmark. Do not present its metrics as real-world accuracy. Use a separate held-out dataset for meaningful evaluation. The training endpoint is for local academic use; do not expose it publicly without authentication, authorization, and additional safeguards.
