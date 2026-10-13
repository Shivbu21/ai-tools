"""ingest.py: load products.csv -> embed -> store in ChromaDB.

Embeddings: chromadb's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2
ONNX, ~90MB, downloaded once from HuggingFace, runs on CPU). Documented in README.

Usage:
    ./catalog-venv/bin/python ingest.py [--reset]
"""
import csv
import os
import sys

# Sandbox NO_PROXY contains bracketed IPv6 entries that crash httpx's
# proxy-pattern parser (chromadb uses httpx internally). Sanitize first.
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "chroma_db")
CSV_PATH = os.path.join(BASE, "products.csv")
COLLECTION = "kirana_catalog"


def row_text(row):
    return (f"{row['name']} | category: {row['category']} | "
            f"price: Rs {row['price']} per {row['unit']} | {row['description']}")


def main():
    import chromadb
    from chromadb.utils import embedding_functions

    reset = "--reset" in sys.argv
    client = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    ef = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2 ONNX
    col = client.get_or_create_collection(COLLECTION, embedding_function=ef)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ids, docs, metas = [], [], []
    for i, r in enumerate(rows):
        ids.append(f"p{i:03d}")
        docs.append(row_text(r))
        metas.append({"name": r["name"], "category": r["category"],
                      "price": r["price"], "unit": r["unit"],
                      "description": r["description"]})
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"ingested {col.count()} products into {DB_DIR}")


if __name__ == "__main__":
    main()
