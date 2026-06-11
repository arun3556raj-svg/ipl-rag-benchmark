"""Tests for the evaluation metrics and pipeline."""

from evaluation.metrics import answer_f1, context_recall


def test_answer_f1_exact_match():
    assert answer_f1("Virat Kohli scored 973 runs", "Virat Kohli scored 973 runs") == 1.0


def test_answer_f1_partial_overlap():
    score = answer_f1("Kohli scored 973 runs in 2016", "Virat Kohli 973 runs")
    assert 0.0 < score < 1.0


def test_answer_f1_no_overlap():
    assert answer_f1("MS Dhoni Chennai", "Eden Gardens Kolkata wickets") == 0.0


def test_answer_f1_empty_strings():
    assert answer_f1("", "something") == 0.0
    assert answer_f1("something", "") == 0.0


def test_context_recall_full():
    assert context_recall("Kohli scored 973 runs for RCB in Bangalore", "Kohli 973") == 1.0


def test_context_recall_partial():
    score = context_recall("Kohli played for RCB", "Kohli scored 973 runs in Bangalore")
    assert 0.0 < score < 1.0


def test_context_recall_zero():
    assert context_recall("completely unrelated text", "Kohli runs Bangalore") == 0.0
