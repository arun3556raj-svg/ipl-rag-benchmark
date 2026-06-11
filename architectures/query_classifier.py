"""Query routing classifier.

Loads the trained logistic regression and predicts which architecture
(text_to_sql, hybrid_rag, light_rag) is best suited for a new question.

The model is loaded once and cached in _state. Thread-safe for the Flask
single-process dev server.

Usage:
    from architectures.query_classifier import route
    arch, confidence = route("Who scored the most runs in 2016?")
    # arch = "text_to_sql", confidence = 0.87
"""

from __future__ import annotations

import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.pkl"

LABELS = ["text_to_sql", "hybrid_rag", "light_rag"]

_state: dict = {
    "clf":     None,
    "le":      None,
    "encoder": None,
}


def ensure_ready() -> None:
    if _state["clf"] is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found at {MODEL_PATH}. "
            "Run: uv run python models/train_classifier.py"
        )

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    _state["clf"] = bundle["clf"]
    _state["le"]  = bundle["label_encoder"]

    from sentence_transformers import SentenceTransformer
    _state["encoder"] = SentenceTransformer("all-MiniLM-L6-v2")


def route(question: str) -> tuple[str, float]:
    """Return (predicted_arch, confidence) for a question.

    confidence is the max class probability from the logistic regression.
    """
    ensure_ready()

    emb = _state["encoder"].encode([question])
    proba = _state["clf"].predict_proba(emb)[0]
    pred_idx = int(proba.argmax())
    arch = _state["le"].inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    return arch, round(confidence, 4)


def route_with_scores(question: str) -> dict:
    """Return arch, confidence, and per-class scores."""
    ensure_ready()

    emb = _state["encoder"].encode([question])
    proba = _state["clf"].predict_proba(emb)[0]
    pred_idx = int(proba.argmax())
    arch = _state["le"].inverse_transform([pred_idx])[0]

    classes = _state["le"].inverse_transform(list(range(len(proba))))
    scores = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}

    return {
        "predicted_arch": arch,
        "confidence":     round(float(proba[pred_idx]), 4),
        "scores":         scores,
    }
