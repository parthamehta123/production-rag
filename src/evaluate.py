"""Evaluation pipeline: measure faithfulness, relevancy, and citation coverage."""

import json
import sys

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.query import generate_answer, load_prompts
from src.retriever import HybridRetriever

load_dotenv()

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7


def load_golden_dataset(path: str = "eval/golden_dataset.json") -> list[dict]:
    """Load manually curated QA pairs for evaluation."""
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    persist_dir: str = "./data/chroma", dataset_path: str = "eval/golden_dataset.json"
):
    """Run evaluation against golden dataset and check quality thresholds."""
    golden = load_golden_dataset(dataset_path)
    prompts = load_prompts()
    retriever = HybridRetriever(persist_dir=persist_dir)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in golden:
        result = generate_answer(item["question"], retriever, prompts)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([s["content"] for s in result["sources"]])
        ground_truths.append(item["expected_answer"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        dataset, metrics=[faithfulness, answer_relevancy, context_precision]
    )
    scores = {k: round(v, 4) for k, v in results.items()}

    print("Evaluation Results:")
    print(json.dumps(scores, indent=2))

    # Quality gating
    passed = True
    if scores.get("faithfulness", 0) < FAITHFULNESS_THRESHOLD:
        print(f"FAIL: Faithfulness {scores['faithfulness']} < {FAITHFULNESS_THRESHOLD}")
        passed = False
    if scores.get("answer_relevancy", 0) < RELEVANCY_THRESHOLD:
        print(f"FAIL: Relevancy {scores['answer_relevancy']} < {RELEVANCY_THRESHOLD}")
        passed = False

    if passed:
        print("PASS: All quality metrics above threshold")
    else:
        sys.exit(1)

    return scores


if __name__ == "__main__":
    run_evaluation()
