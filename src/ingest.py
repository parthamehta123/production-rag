"""Document ingestion pipeline: load, chunk, embed, and store documents.

Key architectural decisions:
- Enrich metadata at ingestion time (company name, section) so retrieval can filter
- Prepend company context to each chunk so the LLM knows which company it's reading
- Filter out garbage chunks (headers, short fragments) that waste retrieval slots
- Use larger chunk sizes (800 tokens) for financial docs which have dense, connected paragraphs
"""

import argparse
import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

LOADERS = {
    "*.pdf": PyPDFLoader,
    "*.txt": TextLoader,
    "*.md": UnstructuredMarkdownLoader,
}

# Map filenames to company metadata
COMPANY_MAP = {
    "apple": {"company": "Apple Inc.", "ticker": "AAPL"},
    "microsoft": {"company": "Microsoft Corporation", "ticker": "MSFT"},
    "tesla": {"company": "Tesla, Inc.", "ticker": "TSLA"},
    "jpmorgan": {"company": "JPMorgan Chase & Co.", "ticker": "JPM"},
    "goldman_sachs": {"company": "The Goldman Sachs Group, Inc.", "ticker": "GS"},
}

# 10-K section patterns for metadata enrichment
SECTION_PATTERNS = [
    (r"Item 1A\.?\s*Risk Factors", "Risk Factors"),
    (r"Item 1\.?\s*Business", "Business"),
    (r"Item 1C\.?\s*Cybersecurity", "Cybersecurity"),
    (r"Item 7\.?\s*Management", "MD&A"),
    (r"Item 8\.?\s*Financial Statements", "Financial Statements"),
    (r"Competition", "Competition"),
    (r"Risk Factor", "Risk Factors"),
    (r"Gross Margin", "Gross Margin"),
    (r"Revenue|Net Sales", "Revenue"),
]


def detect_company(source_path: str) -> dict:
    """Extract company metadata from the file path."""
    basename = os.path.basename(source_path).lower()
    for key, meta in COMPANY_MAP.items():
        if key in basename:
            return meta
    return {"company": "Unknown", "ticker": "N/A"}


def detect_section(text: str) -> str:
    """Detect which 10-K section a chunk likely belongs to."""
    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return section_name
    return "General"


def load_documents(docs_dir: str) -> list:
    """Load documents from a directory supporting PDF, TXT, and MD files."""
    all_docs = []
    for glob_pattern, loader_cls in LOADERS.items():
        loader = DirectoryLoader(
            docs_dir,
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=True,
        )
        all_docs.extend(loader.load())
    print(f"Loaded {len(all_docs)} documents from {docs_dir}")
    return all_docs


def chunk_documents(
    documents: list, chunk_size: int = 800, chunk_overlap: int = 150
) -> list:
    """Split documents into chunks with overlap and enrich metadata.

    Improvements over naive chunking:
    - Larger chunk size (800) to keep financial paragraphs intact
    - Company name + ticker added to metadata for filtered retrieval
    - Section detection for understanding which part of the 10-K a chunk is from
    - Company context prepended to chunk text so the LLM always knows the source
    - Short/garbage chunks filtered out to avoid wasting retrieval slots
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_documents(documents)

    enriched = []
    filtered_count = 0
    for chunk in raw_chunks:
        # Filter out garbage chunks (headers, short fragments)
        if len(chunk.page_content.strip()) < 80:
            filtered_count += 1
            continue

        # Enrich metadata
        source = chunk.metadata.get("source", "")
        company_meta = detect_company(source)
        section = detect_section(chunk.page_content)

        chunk.metadata["company"] = company_meta["company"]
        chunk.metadata["ticker"] = company_meta["ticker"]
        chunk.metadata["section"] = section

        # Prepend company context to chunk text so LLM always knows the source
        prefix = f"[{company_meta['company']} ({company_meta['ticker']}) - {section}] "
        chunk.page_content = prefix + chunk.page_content

        enriched.append(chunk)

    print(
        f"Split into {len(enriched)} chunks "
        f"(filtered {filtered_count} short fragments, "
        f"size={chunk_size}, overlap={chunk_overlap})"
    )
    return enriched


def store_embeddings(chunks: list, persist_dir: str = "./data/chroma") -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {persist_dir}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--docs-dir", required=True, help="Path to documents directory")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--persist-dir", default="./data/chroma")
    args = parser.parse_args()

    documents = load_documents(args.docs_dir)
    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    store_embeddings(chunks, args.persist_dir)


if __name__ == "__main__":
    main()
