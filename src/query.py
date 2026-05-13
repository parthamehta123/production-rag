"""Query pipeline: retrieve context, generate cited answer with enforcement."""

import argparse
import json

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.retriever import HybridRetriever

load_dotenv()


def load_prompts(config_path: str = "configs/prompts.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source IDs for citation."""
    parts = []
    for chunk in chunks:
        parts.append(f"[Source {chunk['source_id']}]: {chunk['content']}")
    return "\n\n".join(parts)


def generate_answer(query: str, retriever: HybridRetriever, prompts: dict) -> dict:
    """Retrieve context, generate answer with citations, enforce grounding."""
    chunks = retriever.retrieve(query)

    if not chunks:
        return {
            "answer": "I cannot answer this question based on the available documents.",
            "sources": [],
            "query": query,
        }

    context = format_context(chunks)
    prompt_text = prompts["query_template"].format(context=context, question=query)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke(
        [
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": prompt_text},
        ]
    )

    answer = response.content

    # Citation enforcement: check if answer contains citations
    has_citations = "[Source" in answer
    is_decline = "cannot answer" in answer.lower()

    if not has_citations and not is_decline:
        answer = (
            "I cannot answer this question based on the available documents. "
            "The retrieved context did not provide sufficient evidence."
        )

    return {
        "answer": answer,
        "sources": [
            {
                "source_id": c["source_id"],
                "content": c["content"][:200],
                "metadata": c["metadata"],
            }
            for c in chunks
        ],
        "query": query,
    }


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--persist-dir", default="./data/chroma")
    args = parser.parse_args()

    prompts = load_prompts()
    retriever = HybridRetriever(persist_dir=args.persist_dir)
    result = generate_answer(args.question, retriever, prompts)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
