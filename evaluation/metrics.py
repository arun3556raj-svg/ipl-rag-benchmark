"""Evaluation metrics for the IPL RAG benchmark.

Three metrics computable without a live LLM:

  answer_f1       Token-level F1 between generated answer and ground truth.
                  Precision = fraction of answer tokens in ground truth.
                  Recall    = fraction of ground truth tokens in answer.
                  F1        = harmonic mean of the two.
                  Same formula used by SQuAD evaluation.

  context_recall  Fraction of ground truth tokens that appear in the
                  retrieved context string. Measures whether the system
                  fetched the right information, independent of how the
                  LLM uses it.

  latency_ms      Wall time for the full answer() call in milliseconds.

When a real LLM is wired:
  faithfulness    Add: LLM judges whether the answer is grounded in context.
  answer_relevance Add: LLM judges whether the answer addresses the question.
"""

from __future__ import annotations

import re
import string


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "in", "of", "for",
        "to", "with", "and", "or", "at", "by", "on", "from", "has", "have",
        "he", "she", "they", "his", "her", "their", "this", "that", "it",
        "be", "been", "had", "who", "which", "what", "how", "did", "do",
    }
    return [t for t in tokens if t not in STOPWORDS]


def answer_f1(prediction: str, ground_truth: str) -> float:
    """Token F1 between prediction and ground truth. Range [0, 1]."""
    pred_tokens  = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counts  = _count(pred_tokens)
    truth_counts = _count(truth_tokens)

    common = sum(min(pred_counts.get(t, 0), truth_counts.get(t, 0))
                 for t in truth_counts)

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def context_recall(context: str, ground_truth: str) -> float:
    """Fraction of ground truth tokens that appear in context. Range [0, 1]."""
    ctx_tokens   = set(_tokenize(context))
    truth_tokens = _tokenize(ground_truth)

    if not truth_tokens:
        return 0.0

    hits = sum(1 for t in truth_tokens if t in ctx_tokens)
    return hits / len(truth_tokens)


def _count(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts
