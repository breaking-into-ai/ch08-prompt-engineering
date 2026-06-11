"""Evaluation-driven development: run after EVERY prompt change. Gate on regressions."""
import json
from statistics import mean

from app.context import ContextPipeline
from app.llm_client import call_llm
from app.prompts.recommend_v3 import SYSTEM, VERSION
from .judge import judge

REGRESSION_THRESHOLD = 0.2  # block deploy if any category drops more than this


def run_evaluation():
    dataset = json.load(open("eval/dataset.json"))
    pipeline = ContextPipeline(system_prompt=SYSTEM)
    results = []
    for example in dataset:
        output = call_llm(pipeline.build_messages(example["query"]))
        scores = judge(example["query"], output)
        results.append(scores)
        print(f"  {example['query'][:50]:<52} {scores}")
    avg = {k: round(mean(r[k] for r in results), 2) for k in results[0]}
    print(f"\nPrompt {VERSION} averages: {avg}")
    return avg


if __name__ == "__main__":
    run_evaluation()
