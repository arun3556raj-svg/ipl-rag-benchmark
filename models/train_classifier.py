"""Train the query routing classifier.

Input  : data/ground_truth.json  (50 labelled questions)
Model  : logistic regression on all-MiniLM-L6-v2 sentence embeddings
Output : models/classifier.pkl + models/classifier_meta.json

Evaluation uses leave-one-out cross-validation (LOO-CV). With only 50
samples LOO-CV is the most statistically honest estimate of generalisation
— it trains on 49, predicts on 1, and repeats 50 times.

Final model is trained on all 50 questions and saved to disk.

Usage:
    uv run python models/train_classifier.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GT_PATH   = ROOT / "data" / "ground_truth.json"
MODEL_OUT = ROOT / "models" / "classifier.pkl"
META_OUT  = ROOT / "models" / "classifier_meta.json"

LABELS = ["text_to_sql", "hybrid_rag", "light_rag"]


def _load_data() -> tuple[list[str], list[str]]:
    questions_raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    texts  = [q["question"] for q in questions_raw]
    labels = [q["expected_best"] for q in questions_raw]
    return texts, labels


def _embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def _loo_cv(X, y_labels: list[str]):
    """Leave-one-out cross-validation. Returns predictions list."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(y_labels)

    preds = []
    n = len(X)
    for i in range(n):
        mask = [j for j in range(n) if j != i]
        X_train = X[mask]
        y_train = y[mask]
        X_test  = X[[i]]

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        preds.append(le.inverse_transform([pred])[0])

    return preds


def _confusion(y_true: list[str], y_pred: list[str]) -> dict:
    matrix: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in LABELS} for l in LABELS}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    metrics = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics[label] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   tp + fn,
        }
    return metrics


def train():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    print("Loading questions...", flush=True)
    texts, labels = _load_data()
    print(f"  {len(texts)} questions loaded.", flush=True)

    print("Encoding questions with all-MiniLM-L6-v2...", flush=True)
    X = _embed(texts)

    print("Running leave-one-out cross-validation...", flush=True)
    loo_preds = _loo_cv(X, labels)

    accuracy = sum(1 for t, p in zip(labels, loo_preds) if t == p) / len(labels)
    confusion = _confusion(labels, loo_preds)
    per_class = _per_class_metrics(labels, loo_preds)

    print(f"  LOO-CV accuracy: {accuracy:.1%}", flush=True)

    print("Training final model on all 50 questions...", flush=True)
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(labels)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, y)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"clf": clf, "label_encoder": le}, f)

    meta = {
        "loo_accuracy":   round(accuracy, 4),
        "n_samples":      len(texts),
        "n_features":     int(X.shape[1]),
        "labels":         LABELS,
        "confusion":      confusion,
        "per_class":      per_class,
        "embedding_model": "all-MiniLM-L6-v2",
        "classifier":      "LogisticRegression(C=1.0, max_iter=1000)",
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def _print_report(meta: dict) -> None:
    print("\n=== CLASSIFIER REPORT ===")
    print(f"Samples        : {meta['n_samples']}")
    print(f"Features       : {meta['n_features']}  ({meta['embedding_model']})")
    print(f"LOO-CV Accuracy: {meta['loo_accuracy']:.1%}")
    print()
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print("-" * 56)
    for label, m in meta["per_class"].items():
        print(f"{label:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9}")
    print()
    print("Confusion matrix (rows=true, cols=predicted):")
    header = f"{'':>15}" + "".join(f"{l:>15}" for l in LABELS)
    print(header)
    for true_label in LABELS:
        row = f"{true_label:>15}"
        for pred_label in LABELS:
            row += f"{meta['confusion'][true_label][pred_label]:>15}"
        print(row)
    print(f"\nModel saved to  : {MODEL_OUT}")
    print(f"Meta saved to   : {META_OUT}")


if __name__ == "__main__":
    meta = train()
    _print_report(meta)
