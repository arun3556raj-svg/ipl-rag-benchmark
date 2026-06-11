"""Quick demo: run the LightRAG graph pipeline on a sample question."""
import json
from architectures import light_rag

question = "How many times did MS Dhoni win player of the match for Chennai Super Kings?"
result = light_rag.answer(question, use_mock=True)
print(json.dumps(result, indent=2))
