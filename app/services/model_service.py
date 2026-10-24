from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

VALID_URGENCY = {"Low", "Medium", "High", "Critical"}


def determine_urgency(text: str) -> tuple[Optional[str], str]:
    """Apply documented urgency rules; this is not a trained severity model."""
    t = re.sub(r"\\s+", " ", text.lower()).strip()

    electrical_hazard = (
        any(x in t for x in ("electric shock", "shocked by", "electrocut", "live wire", "exposed live wire"))
        and any(x in t for x in ("injury", "injured", "person", "child", "people", "someone", "shock"))
    )
    water_health = (
        any(x in t for x in ("contaminated water", "dirty drinking water", "unsafe drinking water", "water contamination"))
        and any(x in t for x in ("illness", "sick", "vomit", "diarrhea", "diarrhoea", "fever", "health impact", "hospital"))
    )
    if electrical_hazard or water_health:
        return "Critical", "Immediate electrical injury hazard or water contamination linked to health impact."

    major_water_failure = any(x in t for x in (
        "burst pipeline", "pipeline burst", "burst water pipe", "water main burst", "major pipe leak"
    ))
    shared_flooding = any(x in t for x in ("flood", "flooding", "flooded")) and any(
        x in t for x in ("street", "road", "shared", "residential area", "multiple homes", "neighborhood", "neighbourhood", "building")
    )
    if major_water_failure or shared_flooding:
        return "High", "Major water infrastructure failure or flooding affecting a shared area."

    disruption = any(x in t for x in (
        "not working", "doesn't work", "does not work", "no water", "no electricity",
        "power cut", "power outage", "intermittent", "for days", "since yesterday",
        "not collected", "blocked", "overflowing", "leaking", "unavailable", "broken"
    ))
    multi_property = any(x in t for x in (
        "multiple homes", "many homes", "several homes", "whole building", "multiple buildings",
        "entire street", "whole neighborhood", "whole neighbourhood", "many residents"
    ))
    if disruption or multi_property:
        return "Medium", "Service disruption without an explicit critical hazard."

    explicitly_minor = any(x in t for x in (
        "minor issue", "small issue", "cosmetic", "no one affected", "not affecting anyone"
    ))
    if explicitly_minor or len(t.split()) < 5:
        return "Low", "Explicitly minor issue or insufficient evidence of a major hazard."

    return None, "No documented urgency rule matched; human review is required."


class CivicAIModelService:
    def __init__(self, artifact_path: Path):
        self.artifact_path = Path(artifact_path)
        self.pipeline: Optional[Pipeline] = None
        self.is_loaded = False

    def load(self) -> None:
        if self.artifact_path.exists():
            try:
                self.pipeline = joblib.load(self.artifact_path)
                self.is_loaded = True
            except Exception:
                self.pipeline = None
                self.is_loaded = False

    def info(self) -> dict:
        categories = list(self.pipeline.named_steps["classifier"].classes_) if self.pipeline else []
        return {
            "service": "CivicAI ML Service",
            "category_model": "TF-IDF + LinearSVC",
            "category_model_loaded": self.is_loaded,
            "categories": categories,
            "urgency_method": "deterministic rules; not a trained severity model",
            "confidence_definition": "normalized SVM decision-margin score; not a calibrated probability",
            "endpoints": ["/health", "/model/info", "/predict", "/evaluate", "/train"],
        }

    def train_from_csv(self, dataset_path: Path) -> dict:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        if not {"text", "category"}.issubset(df.columns):
            raise ValueError("CSV must contain 'text' and 'category' columns.")
        df = df.dropna(subset=["text", "category"])
        df["text"] = df["text"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()
        df = df[(df.text != "") & (df.category != "")]
        if len(df) < 4 or df.category.nunique() < 2:
            raise ValueError("Training requires at least 4 rows and 2 categories.")
        if df.category.value_counts().min() < 2:
            raise ValueError("Each category needs at least 2 examples.")

        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ])
        model.fit(df.text, df.category)
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.artifact_path)
        self.pipeline, self.is_loaded = model, True
        return {
            "status": "trained",
            "rows_used": int(len(df)),
            "category_count": int(df.category.nunique()),
            "categories": sorted(df.category.unique().tolist()),
            "artifact": str(self.artifact_path),
            "note": "Training-set metrics do not estimate generalization performance.",
        }

    def predict(self, text: str) -> dict:
        urgency, reason = determine_urgency(text)
        if self.pipeline is None:
            self.load()
        if self.pipeline is None:
            return {
                "category": None, "urgency": urgency,
                "priority": urgency if urgency in VALID_URGENCY else None,
                "confidence": None, "needs_human_review": True,
                "message": "Category model unavailable; train the model and request human review.",
                "urgency_reason": reason,
            }
        try:
            category = str(self.pipeline.predict([text])[0])
            decision = self.pipeline.decision_function([text])
            if decision.ndim == 1:
                margin = abs(float(decision[0]))
            elif decision.shape[1] == 1:
                margin = abs(float(decision[0][0]))
            else:
                scores = sorted((float(x) for x in decision[0]), reverse=True)
                margin = max(0.0, scores[0] - scores[1])
            confidence = round(margin / (1.0 + margin), 4)
        except Exception:
            return {
                "category": None, "urgency": urgency,
                "priority": urgency if urgency in VALID_URGENCY else None,
                "confidence": None, "needs_human_review": True,
                "message": "Category prediction failed; human review is required.",
                "urgency_reason": reason,
            }
        return {
            "category": category, "urgency": urgency,
            "priority": urgency if urgency in VALID_URGENCY else None,
            "confidence": confidence, "needs_human_review": urgency is None,
            "message": "Category uses TF-IDF + LinearSVC; urgency uses deterministic rules.",
            "urgency_reason": reason,
        }

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        if self.pipeline is None:
            self.load()
        if self.pipeline is None:
            raise ValueError("Model unavailable. Train it first using POST /train.")
        if len(texts) != len(labels):
            raise ValueError("Each text needs a corresponding category label.")
        predictions = self.pipeline.predict(texts)
        return {
            "sample_count": len(texts),
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "classification_report": classification_report(
                labels, predictions, output_dict=True, zero_division=0
            ),
            "warning": "Use a separate held-out dataset for meaningful performance estimates.",
        }
