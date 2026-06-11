"""Flask api for the IPL retrieval benchmark.

All phases wired:
  /             Frontend (The Stadium)
  /api/sql      Text2SQL
  /api/hybrid   Hybrid BM25 + Vector RAG
  /api/lightrag LightRAG graph
  /api/route    Auto-route via trained query classifier
  /api/eval     Evaluation summary from eval_results.json
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path

from flask import Flask, jsonify, request, send_from_directory

from architectures import hybrid_rag, light_rag, query_classifier, text_to_sql

_ROOT         = _Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _ROOT / "frontend"
_EVAL_PATH    = _ROOT / "data" / "eval_results.json"

app = Flask(__name__)

_ARCH_FN = {
    "text_to_sql": lambda q: text_to_sql.answer(q, use_mock=True),
    "hybrid_rag":  lambda q: hybrid_rag.answer(q, use_mock=True),
    "light_rag":   lambda q: light_rag.answer(q, use_mock=True),
}


@app.get("/")
def index():
    return send_from_directory(str(_FRONTEND_DIR), "index.html")


@app.get("/api/eval")
def eval_endpoint():
    if not _EVAL_PATH.exists():
        return jsonify({"error": "No eval results found. Run: uv run python evaluation/run_eval.py"}), 404
    data = _json.loads(_EVAL_PATH.read_text(encoding="utf-8"))
    return jsonify(data["summary"])


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "phase": 7,
        "architectures": ["text_to_sql", "hybrid_rag", "light_rag", "auto"],
    })


@app.post("/api/sql")
def sql_endpoint():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    try:
        result = text_to_sql.answer(question, use_mock=True)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.post("/api/hybrid")
def hybrid_endpoint():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400
    top_k = int(payload.get("top_k", 12))

    try:
        result = hybrid_rag.answer(question, use_mock=True, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.post("/api/lightrag")
def lightrag_endpoint():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400
    top_k = int(payload.get("top_k", 30))

    try:
        result = light_rag.answer(question, use_mock=True, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.post("/api/route")
def route_endpoint():
    """Auto-classify the question and call the predicted best architecture."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    try:
        routing = query_classifier.route_with_scores(question)
        arch    = routing["predicted_arch"]
        result  = _ARCH_FN[arch](question)
        result["routing"] = routing
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
