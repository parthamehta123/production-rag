"""Document ingestion pipeline: load, chunk, embed, and store documents."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

LOADERS = {
    "*.pdf": PyPDFLoader,
    "*.txt": TextLoader,
    "*.md": UnstructuredMarkdownLoader,
}

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


def chunk_documents(documents: list, chunk_size: int = 600, chunk_overlap: int = 100) -> list:
    """Split documents into chunks with overlap to preserve context at boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


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
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--persist-dir", default="./data/chroma")
    args = parser.parse_args()

    documents = load_documents(args.docs_dir)
    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    store_embeddings(chunks, args.persist_dir)


if __name__ == "__main__":
    main()
