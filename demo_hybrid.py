"""Quick demo: run the hybrid RAG pipeline on a sample question."""
import json
from architectures import hybrid_rag

question = "Who scored the most runs in death overs across all IPL seasons?"
result = hybrid_rag.answer(question, use_mock=True)
print(json.dumps(result, indent=2))
