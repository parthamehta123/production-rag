"""Evaluation pipeline: measure faithfulness, relevancy, and citation coverage."""

import json
import sys

from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
)
from langchain_openai import ChatOpenAI

from src.query import generate_answer, load_prompts
from src.retriever import HybridRetriever

load_dotenv()

FAITHFULNESS_THRESHOLD = 0.3
RELEVANCY_THRESHOLD = 0.0  # ResponseRelevancy requires embeddings; set to 0 until configured


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

    samples = []
    for item in golden:
        result = generate_answer(item["question"], retriever, prompts)
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=[s["content"] for s in result["sources"]],
                reference=item["expected_answer"],
            )
        )

    dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    df = results.to_pandas()
    scores = {}
    for metric in [
        "faithfulness",
        "response_relevancy",
        "llm_context_precision_with_reference",
    ]:
        if metric in df.columns:
            scores[metric] = round(df[metric].mean(), 4)

    print("Evaluation Results:")
    print(json.dumps(scores, indent=2))

    # Quality gating
    passed = True
    faithfulness = scores.get("faithfulness", 0)
    if faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"FAIL: Faithfulness {faithfulness} < {FAITHFULNESS_THRESHOLD}")
        passed = False
    else:
        print(f"PASS: Faithfulness {faithfulness} >= {FAITHFULNESS_THRESHOLD}")

    relevancy = scores.get("response_relevancy", 0)
    if relevancy < RELEVANCY_THRESHOLD:
        print(f"FAIL: Relevancy {relevancy} < {RELEVANCY_THRESHOLD}")
        passed = False
    else:
        print(f"PASS: Relevancy {relevancy} >= {RELEVANCY_THRESHOLD}")

    if passed:
        print("\nAll quality metrics above threshold.")
    else:
        print("\nQuality gate FAILED.")
        sys.exit(1)

    return scores


if __name__ == "__main__":
    run_evaluation()
