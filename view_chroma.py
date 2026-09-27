"""
Quick command-line viewer for the local ChromaDB persistent store used by this app.

Usage:
    python view_chroma.py                  # list all collections
    python view_chroma.py <collection>     # peek at chunks in one collection
    python view_chroma.py <collection> N   # show N chunks (default 5)


import sys
import chromadb
from src.config import chroma_dir


def list_collections(client: chromadb.ClientAPI) -> None:
    collections = client.list_collections()
    if not collections:
        print(f"No collections found in {chroma_dir}")
        return

    print(f"Found {len(collections)} collection(s) in {chroma_dir}:\n")
    for col in collections:
        count = client.get_collection(col.name).count()
        print(f"  - {col.name}  ({count} chunks)")


def show_chunks(client: chromadb.ClientAPI, collection_name: str, limit: int) -> None:
    collection = client.get_collection(collection_name)
    result = collection.get(limit=limit, include=["documents", "metadatas"])

    ids = result["ids"]
    docs = result["documents"]
    metas = result["metadatas"]

    if not ids:
        print(f"Collection '{collection_name}' is empty.")
        return

    for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
        print(f"--- Chunk {i} (id={doc_id}) ---")
        print(f"metadata: {meta}")
        preview = (doc or "")[:400]
        print(f"content: {preview}{'...' if doc and len(doc) > 400 else ''}\n")


def main() -> None:
    client = chromadb.PersistentClient(path=chroma_dir)

    if len(sys.argv) == 1:
        list_collections(client)
        return

    collection_name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    show_chunks(client, collection_name, limit)


if __name__ == "__main__":
    main()
"""

from chromaviz import visualize_collection
collection = "academic_documents"
visualize_collection(collection)