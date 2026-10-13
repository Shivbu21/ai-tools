# Catalog QA — RAG over a kirana shop catalog (offline)

Ask questions about a shop's product catalog in plain English. Retrieves the
relevant product rows from a local vector DB, then a local 1.5B LLM answers
**grounded on the retrieved rows, with citations**. If the LLM is unreachable,
it falls back to extractive mode (prints the top rows directly) and **labels
which mode produced the answer**. Out-of-catalog questions get an honest
"I don't have that" — never a hallucinated price.

## Architecture

```
products.csv (76 kirana items)
  → ingest.py: row text → all-MiniLM-L6-v2 (ONNX, ~90MB, CPU) → ChromaDB ./chroma_db/
ask.py: question → embed → top-5 rows (cosine distance)
  → if best distance > 1.15: "not in catalog" (no hallucination)
  → elif LLM up: grounded prompt → qwen2.5-1.5b @ :11435 → answer + cited rows   [mode: llm]
  → else: print top rows directly, labelled extractive                          [mode: extractive]
app.py: Flask single-page UI (textbox + Ask + mode badge) on :8767
```

## Run

```bash
cd ~/workspace/ai-tools/catalog-qa

# one shared LLM server for both ai-tools projects (start once)
~/llamacpp/build/bin/llama-server \
  -m /home/hatch/ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4 \
  -t 2 -c 2048 --port 11435

# venv + deps (done once)
/usr/bin/python3 -m venv ../catalog-venv && ../catalog-venv/bin/pip install -r requirements.txt

# ingest (downloads the ONNX embedding model once, ~90MB)
/catalog-venv/bin/python ingest.py          # add --reset to rebuild

# CLI
../catalog-venv/bin/python ask.py "what is the price of Tata Salt?"
../catalog-venv/bin/python ask.py --json "cheapest soap"

# web UI
../catalog-venv/bin/python app.py           # → http://127.0.0.1:8767
```

## Embedding choice

ChromaDB's built-in `DefaultEmbeddingFunction` = **all-MiniLM-L6-v2 as ONNX**
(~90MB, CPU, no torch needed). Chosen over TF-IDF because semantic matching
handles phrasing like "cheapest soap" / "do you have basmati rice" far better,
and it fits comfortably in the RAM budget. First run downloads the model from
HuggingFace (internet); after that it's fully offline, cached at
`~/.cache/chroma/onnx_models/`.

## Verified

5 real questions run end-to-end on 2026-10-14, full transcript in `EVAL.md`:

| question | mode | result |
|---|---|---|
| What is the price of Tata Salt? | llm | Rs 28 per 1 kg ✓ |
| Which atta is the cheapest? | llm | Aashirvaad Whole Wheat Atta, Rs 265/5 kg ✓ |
| Do you have basmati rice? | llm | India Gate Rozzana Basmati Rice, Rs 145/1 kg ✓ |
| Do you sell iPhone 16? | not-found | "I don't have that in the catalog" — no hallucination ✓ |
| cheapest soap | llm | Lifebuoy Total Soap, Rs 32/125 g ✓ (correctly picked over higher-priced Lux despite ranking) |

## Honest limitations

- **1.5B LLM**: answers are short and usually correct when the retrieval is
  right, but the model can misread the retrieved rows on ambiguous questions
  (e.g. comparing per-unit prices across different pack sizes). Every answer
  ships with cited rows so the user can check.
- **Retrieval quality**: MiniLM is good at semantic match but not perfect —
  near-miss rows (same brand, wrong product) appear in top-5; the not-found
  threshold (cosine distance > 1.15) is tuned by hand, not validated on a set.
- **Hindi/Hinglish queries**: not tested. The embedding model is English-first;
  expect degraded retrieval on Hindi questions.
- **Prices are sample data** in `products.csv` — realistic but fictional. Swap
  in the shop's real CSV with the same columns to go live.
- Single-user CLI/web UI; no auth, no multi-shop support.
