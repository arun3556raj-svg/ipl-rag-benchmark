"""Demo: classifier routes 3 question types to the right architecture."""
import json
from architectures import query_classifier, text_to_sql, hybrid_rag, light_rag

questions = [
    "Who scored the most runs in IPL history?",
    "Describe Virat Kohli's performance in the 2016 season.",
    "Which teams has MS Dhoni played for?",
]

for q in questions:
    arch, conf = query_classifier.route(q)
    print(f"Q: {q}")
    print(f"   -> {arch}  (confidence {conf:.0%})")
    print()
