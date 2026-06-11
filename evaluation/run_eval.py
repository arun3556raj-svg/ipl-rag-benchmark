"""Run the RAGAS-inspired evaluation across all 50 ground truth questions.

For each question, all three architectures are called and scored.
Results are written to data/eval_results.json.

Usage:
    uv run python evaluation/run_eval.py
    uv run python evaluation/run_eval.py --arch text_to_sql   # single arch
    uv run python evaluation/run_eval.py --quiet              # no progress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from architectures import hybrid_rag, light_rag, text_to_sql
from evaluation.metrics import answer_f1, context_recall

GT_PATH      = ROOT / "data" / "ground_truth.json"
RESULTS_PATH = ROOT / "data" / "eval_results.json"

ARCHS = ["text_to_sql", "hybrid_rag", "light_rag"]


def _get_context(arch: str, result: dict) -> str:
    """Extract the context string from an architecture result dict."""
    if arch == "text_to_sql":
        rows = result.get("rows") or []
        cols = result.get("columns") or []
        if rows and cols:
            lines = []
            for row in rows[:20]:
                if isinstance(row, dict):
                    lines.append(" ".join(str(row.get(c, "")) for c in cols))
                else:
                    lines.append(" ".join(str(v) for v in row))
            return " ".join(lines)
        return result.get("sql", "") + " " + result.get("answer", "")
    if arch == "hybrid_rag":
        return " ".join(result.get("retrieved_ids", []))
    if arch == "light_rag":
        return result.get("answer", "")
    return ""


def _run_arch(arch: str, question: str, use_mock: bool = False) -> dict:
    try:
        if arch == "text_to_sql":
            return text_to_sql.answer(question, use_mock=use_mock)
        if arch == "hybrid_rag":
            return hybrid_rag.answer(question, use_mock=use_mock)
        if arch == "light_rag":
            return light_rag.answer(question, use_mock=use_mock)
    except Exception as exc:
        return {"error": str(exc), "answer": "", "latency_ms": 0.0, "cost_usd": 0.0}
    return {}


def _score_result(arch: str, result: dict, ground_truth: str) -> dict:
    answer = result.get("answer", "")
    context = _get_context(arch, result)

    af1 = answer_f1(answer, ground_truth)
    cr  = context_recall(context, ground_truth)

    latency = result.get("latency_ms", 0.0)

    scored = {
        "answer":          answer,
        "answer_f1":       round(af1, 4),
        "context_recall":  round(cr, 4),
        "latency_ms":      latency,
        "cost_usd":        result.get("cost_usd", 0.0),
        "use_mock":        result.get("use_mock", True),
    }

    # architecture-specific extras
    if arch == "text_to_sql":
        scored["sql"]       = result.get("sql", "")
        scored["row_count"] = len(result.get("rows") or [])
    elif arch == "hybrid_rag":
        scored["retrieved_count"] = result.get("retrieved_count", 0)
        scored["bm25_ms"]         = result.get("bm25_ms", 0)
        scored["vector_ms"]       = result.get("vector_ms", 0)
    elif arch == "light_rag":
        scored["seed_count"]     = result.get("seed_count", 0)
        scored["top_node_count"] = result.get("top_node_count", 0)
        scored["graph_ms"]       = result.get("graph_ms", 0)

    return scored


def evaluate(
    archs: list[str],
    quiet: bool = False,
    use_mock: bool = False,
) -> dict:
    questions = json.loads(GT_PATH.read_text(encoding="utf-8"))
    total = len(questions)

    if not quiet:
        mode = "MOCK" if use_mock else "LIVE (deepseek-v4-flash)"
        print(f"Mode: {mode}", flush=True)
        print(f"Warming up {len(archs)} architecture(s)...", flush=True)
    for arch in archs:
        try:
            _run_arch(arch, "warm up", use_mock=use_mock)
        except Exception:
            pass

    results_per_question = []
    total_cost = 0.0
    if not quiet:
        print(f"Evaluating {total} questions across {archs}...", flush=True)

    for i, q in enumerate(questions):
        if not quiet:
            print(f"  [{i+1:02d}/{total}] {q['id']}  {q['question'][:60]}",
                  flush=True)

        per_arch: dict[str, dict] = {}
        for arch in archs:
            raw = _run_arch(arch, q["question"], use_mock=use_mock)
            scored = _score_result(arch, raw, q["ground_truth"])
            total_cost += scored.get("cost_usd", 0.0)
            per_arch[arch] = scored

        # Determine which arch won on answer_f1 this question
        if len(archs) > 1:
            winner = max(per_arch, key=lambda a: per_arch[a]["answer_f1"])
        else:
            winner = archs[0]

        results_per_question.append({
            "id":            q["id"],
            "question":      q["question"],
            "ground_truth":  q["ground_truth"],
            "expected_best": q["expected_best"],
            "category":      q["category"],
            "difficulty":    q["difficulty"],
            "winner":        winner,
            "results":       per_arch,
        })

    if not quiet:
        print(f"\nTotal LLM cost: ${total_cost:.4f}", flush=True)

    summary = _summarise(results_per_question, archs)

    output = {
        "run_id":       f"eval-{int(time.time())}",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "archs":        archs,
        "use_mock":     use_mock,
        "total_cost_usd": round(total_cost, 4),
        "questions":    results_per_question,
        "summary":      summary,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def _summarise(questions: list[dict], archs: list[str]) -> dict:
    summary: dict[str, dict] = {}

    for arch in archs:
        qs_with_arch = [q for q in questions if arch in q["results"]]
        if not qs_with_arch:
            continue

        af1_scores = [q["results"][arch]["answer_f1"] for q in qs_with_arch]
        cr_scores  = [q["results"][arch]["context_recall"] for q in qs_with_arch]
        lat_scores = [q["results"][arch]["latency_ms"] for q in qs_with_arch]

        wins = sum(1 for q in qs_with_arch if q["winner"] == arch)
        expected_wins = sum(
            1 for q in qs_with_arch
            if q["expected_best"] == arch and q["winner"] == arch
        )
        expected_total = sum(
            1 for q in qs_with_arch if q["expected_best"] == arch
        )

        summary[arch] = {
            "avg_answer_f1":      round(_mean(af1_scores), 4),
            "avg_context_recall": round(_mean(cr_scores), 4),
            "avg_latency_ms":     round(_mean(lat_scores), 1),
            "total_wins":         wins,
            "expected_wins":      expected_wins,
            "expected_total":     expected_total,
            "win_rate":           round(wins / len(qs_with_arch), 3),
        }

    # Per-category breakdown
    category_summary: dict[str, dict] = {}
    for cat in {"factual", "relational", "narrative", "temporal"}:
        cat_qs = [q for q in questions if q["category"] == cat]
        if not cat_qs:
            continue
        cat_winner_counts: dict[str, int] = {}
        for q in cat_qs:
            cat_winner_counts[q["winner"]] = cat_winner_counts.get(q["winner"], 0) + 1
        category_summary[cat] = {
            "total":   len(cat_qs),
            "winners": cat_winner_counts,
        }

    return {"by_arch": summary, "by_category": category_summary}


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _print_summary(output: dict) -> None:
    print("\n=== EVALUATION SUMMARY ===")
    summary = output["summary"]["by_arch"]
    print(f"{'Arch':<15} {'Avg F1':>8} {'Ctx Recall':>12} {'Avg Lat ms':>12} {'Win Rate':>10}")
    print("-" * 60)
    for arch, s in summary.items():
        print(
            f"{arch:<15} {s['avg_answer_f1']:>8.3f} "
            f"{s['avg_context_recall']:>12.3f} "
            f"{s['avg_latency_ms']:>12.1f} "
            f"{s['win_rate']:>10.3f}"
        )
    print("\n=== BY CATEGORY ===")
    for cat, info in output["summary"]["by_category"].items():
        winners = ", ".join(f"{a}: {n}" for a, n in info["winners"].items())
        print(f"  {cat:<12}  total={info['total']}  winners: {winners}")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHS, default=None,
                        help="Evaluate a single architecture only")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM instead of live API")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    archs_to_run = [args.arch] if args.arch else ARCHS
    output = evaluate(archs_to_run, quiet=args.quiet, use_mock=args.mock)
    if not args.quiet:
        _print_summary(output)
