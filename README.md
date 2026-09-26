# Plant Disease Detection and Explanation — CNN + Graph RAG

Upload a leaf photo, get the disease **and** a grounded explanation. A ResNet-50 classifier
answers *"which disease is this?"*; a Graph RAG pipeline answers *"what does it mean and what
should I do?"* in Indonesian, using a Neo4j knowledge graph plus semantic retrieval.

Crops: apple, grape, corn, tomato — 12 diseases plus an explicit `unknown` class.

## How it works

```
Leaf photo
  │
  ├─ ResNet-50 (fine-tuned, 13 classes)
  │     └─ open-set rejection: `unknown` class + confidence threshold 0.70
  │
  ├─ Hybrid retrieval
  │     ├─ Graph retrieval   → Neo4j Aura (symptoms, treatment, prevention, care)
  │     └─ Semantic retrieval → sentence-transformers embeddings
  │
  └─ LLM generator (Groq) → grounded answer in Indonesian
```

Gradio interface with three tabs: CNN result, Neo4j graph view, RAG answer.

## Results

**Classifier** (split 80/10/10): test accuracy **0.9786**, with rejection threshold **0.9705**,
F1 of the `unknown` class **0.99**. The rejection gate turned away **100%** of held-out
non-leaf images (96 images).

| Split | Test accuracy | F1 |
|---|---:|---:|
| 50/25/25 | 0.9679 | 0.9679 |
| 60/20/20 | 0.9759 | 0.9761 |
| 70/15/15 | 0.9785 | 0.9787 |
| 80/10/10 | 0.9786 | 0.9785 |

**RAG evaluation with RAGAS** (60 questions, golden dataset in `app/golden_dataset.json`):

| Metric | Score |
|---|---:|
| context_recall | 0.9725 |
| faithfulness | 0.9554 |
| semantic_similarity | 0.8814 |
| factual_correctness | 0.8775 |
| answer_relevancy | 0.8406 |
| llm_context_precision_with_reference | 0.5714 |

## Run it

```bash
cd app
pip install -r requirements.txt
cp .env.example .env      # fill in Neo4j, Groq and OpenRouter credentials
python app.py
```

RAGAS evaluation: `pip install -r requirements_eval.txt`, then `python evaluate_ragas.py`.
Step-by-step guide: [app/docs/cara-menjalankan.md](app/docs/cara-menjalankan.md).

## Repo layout

| Path | Content |
|---|---|
| `app/` | Gradio app, CNN wrapper, RAG engine, semantic retriever, RAGAS evaluation |
| `app/GRAPHRAG.json` | Knowledge base loaded into Neo4j |
| `app/models/resnet50_final.pt` | Final classifier weights |
| `hasil-training/` | Metrics, confusion matrices and training report for all four splits |
| `notebook/` | Training notebooks (Kaggle) |

No credentials are stored in the repo; everything is read from `.env`.

## Stack

Python · PyTorch · ResNet-50 · Hugging Face sentence-transformers · Neo4j · Groq LLM · RAGAS · Gradio

---

Built by **Dwi Yuda** · [dwiyuda.is-a.dev](https://dwiyuda.is-a.dev)
