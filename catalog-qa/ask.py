"""ask.py: CLI catalog Q&A.

Retrieves top-k product rows from ChromaDB, builds a grounded prompt, and asks
the shared llama-server (qwen2.5-1.5b, port 11435). The answer is printed WITH
cited product rows (name + price).

If the LLM server is unreachable, falls back to extractive mode: prints the top
retrieved rows directly, LABELLED as extractive. Never presents retrieval as
generation.

Usage:
    ./catalog-venv/bin/python ask.py "what is the price of Tata Salt?"
    ./catalog-venv/bin/python ask.py --json "cheapest atta"
"""
import csv
import json
import os
import sys

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "chroma_db")
CSV_PATH = os.path.join(BASE, "products.csv")
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:11435/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-1.5b")
TOP_K = 5
# similarity floor: below this, treat as "not in catalog" (honest no-hallucination rule)
MIN_DISTANCE = 1.15


def get_collection():
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection("kirana_catalog",
                                 embedding_function=embedding_functions.DefaultEmbeddingFunction())


def retrieve(question, k=TOP_K):
    col = get_collection()
    res = col.query(query_texts=[question], n_results=k)
    rows = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        rows.append({"distance": round(dist, 3), "doc": doc, **meta})
    return rows


def fmt_row(r):
    return f"- {r['name']} — Rs {r['price']} per {r['unit']} [{r['category']}]"


def extractive_answer(question, rows):
    lines = [f"[extractive mode — LLM unreachable; showing top catalog matches]",
             f"Question: {question}", ""]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def llm_answer(question, rows):
    import httpx
    context = "\n".join(fmt_row(r) for r in rows)
    prompt = (f"You are a helpful kirana shop assistant. Answer ONLY from the catalog below. "
              f"If the item is not in the catalog, say you don't have it. Keep it under 60 words.\n\n"
              f"CATALOG:\n{context}\n\nQuestion: {question}\nAnswer:")
    r = httpx.post(f"{LLM_API_URL}/chat/completions",
                   json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 120, "temperature": 0.1},
                   timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def answer(question):
    rows = retrieve(question)
    best = rows[0]["distance"] if rows else 999
    if not rows or best > MIN_DISTANCE:
        return {"mode": "not-found",
                "answer": "I don't have that in the shop's catalog. I can only answer about items we stock.",
                "cited": [], "best_distance": best}
    try:
        text = llm_answer(question, rows)
        return {"mode": "llm", "answer": text, "cited": rows, "best_distance": best}
    except Exception as e:
        return {"mode": "extractive", "answer": extractive_answer(question, rows),
                "cited": rows, "best_distance": best, "error": str(e)}


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]
    if not args:
        print('usage: ask.py [--json] "your question"')
        sys.exit(2)
    out = answer(" ".join(args))
    if as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"[mode: {out['mode']}]")
        print(out["answer"])
        if out.get("cited"):
            print("\nCited rows:")
            for r in out["cited"][:3]:
                print(" ", fmt_row(r))


if __name__ == "__main__":
    main()
