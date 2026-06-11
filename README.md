# The Stadium — IPL RAG Benchmark

Three retrieval architectures compete to answer the same questions from the same 295,732-delivery IPL database. A trained classifier decides which one bowls.

![The Stadium](https://img.shields.io/badge/IPL-RAG%20Benchmark-22C55E?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## The three architectures

| Architecture | Strategy | Best at |
|---|---|---|
| **Text2SQL** | LLM writes SQL, SQLite executes it | Precise factual queries (exact counts, totals, rankings) |
| **Hybrid RAG** | BM25 sparse + ChromaDB dense, fused via Reciprocal Rank Fusion | Narrative and contextual queries |
| **LightRAG** | Personalized PageRank over a 899-node knowledge graph | Relational queries (connections between players, teams, venues) |

A LogisticRegression classifier (trained on 50 labelled questions, 90% LOO-CV accuracy) routes each incoming question to the predicted best architecture.

---

## Benchmark results

Scored against 50 hand-verified cricket questions using RAGAS-inspired metrics (Answer F1, Context Recall).

| Architecture | Win Rate | Answer F1 | Context Recall | Avg Latency |
|---|---|---|---|---|
| LightRAG | **54%** | 0.086 | 0.249 | 4.0ms |
| Hybrid RAG | 28% | 0.064 | 0.000 | 28.6ms |
| Text2SQL | 18% | 0.014 | 0.003 | 0.5ms |

*All runs use mock LLM responses. Wire in DeepSeek V4 Pro via `DEEPSEEK_API_KEY` to activate live generation.*

---

## Quick start

```bash
git clone https://github.com/arun3556raj/ipl-rag-benchmark
cd ipl-rag-benchmark
uv sync
```

**Rebuild indexes** (required on first run):

```bash
uv run python architectures/chunk_builder.py   # builds data/chunks.json + ChromaDB
uv run python architectures/graph_builder.py   # builds data/graph.pkl
uv run python models/train_classifier.py        # trains models/classifier.pkl
```

**Start the UI:**

```bash
uv run python api/app.py
# open http://127.0.0.1:5000
```

**Or query via curl:**

```bash
curl -X POST http://127.0.0.1:5000/api/route \
  -H "Content-Type: application/json" \
  -d '{"question": "Who scored the most runs in IPL history?"}'
```

---

## Project structure

```
ipl-rag/
├── api/
│   └── app.py              Flask API + static frontend server
├── architectures/
│   ├── chunk_builder.py    Builds 4,629 text chunks + ChromaDB index
│   ├── graph_builder.py    Builds NetworkX knowledge graph (899 nodes, 2,443 edges)
│   ├── hybrid_rag.py       BM25 + dense vector retrieval via RRF
│   ├── light_rag.py        Personalized PageRank graph traversal
│   ├── text_to_sql.py      Natural language to SQL
│   └── query_classifier.py Routes questions to predicted best architecture
├── data/
│   ├── ipl_universe.db     SQLite: 295,732 deliveries, 6 normalized tables
│   ├── ground_truth.json   50 hand-verified questions with real DB answers
│   ├── eval_results.json   RAGAS evaluation output
│   └── schema.md           Plain English DB schema
├── evaluation/
│   ├── metrics.py          answer_f1, context_recall
│   └── run_eval.py         Full evaluation pipeline
├── frontend/
│   └── index.html          Single-file SPA (The Stadium)
├── models/
│   └── train_classifier.py LOO-CV training, saves classifier.pkl
└── tests/                  28 passing pytest tests
```

---

## Database

`data/ipl_universe.db` is a normalized SQLite database built from the Cricsheet IPL dataset.

| Table | Rows |
|---|---|
| matches | 1,243 |
| deliveries | 295,732 |
| players | 816 |
| teams | 19 |
| venues | 64 |
| player_stats | aggregated per player per season |

---

## Environment variables

```bash
DEEPSEEK_API_KEY=sk-...   # activates live LLM generation (optional, mock works without it)
```

---

## Run the evaluation

```bash
uv run python evaluation/run_eval.py              # all architectures
uv run python evaluation/run_eval.py --arch sql   # single architecture
```

Results saved to `data/eval_results.json`, surfaced at `GET /api/eval`.

---

## Tests

```bash
uv run pytest tests/ -v
# 28 passed
```

---

## License

MIT
