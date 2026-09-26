"""
semantic_retriever.py — Semantic Retrieval Module (Hybrid RAG)
Melengkapi Graph RAG (Neo4j) dengan pencarian berbasis kemiripan makna
(embedding + cosine similarity) memakai SentenceTransformer.
"""
# Bagian ini mengimpor seluruh library yang diperlukan oleh modul semantic retrieval.

import os
import json
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# Konfigurasi path knowledge base, cache embedding, dan model embedding
# yang dipakai oleh SemanticRetriever. Bisa dioverride lewat environment
# variable, konsisten dengan pola konfigurasi di rag_engine.py.
# ─────────────────────────────────────────────
GRAPHRAG_JSON_PATH   = os.getenv("GRAPHRAG_JSON_PATH", "GRAPHRAG.json")
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "embedding_cache.pkl")
SEMANTIC_MODEL_NAME  = os.getenv("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


class SemanticRetriever:
    """
    Semantic Retrieval menggunakan embedding
    SentenceTransformer dan Cosine Similarity.

    Fungsi utama:
    1. Membaca knowledge base JSON
    2. Membuat document chunk
    3. Membuat embedding
    4. Menyimpan embedding
    5. Melakukan semantic retrieval
    """

    def __init__(
        self,
        json_path=GRAPHRAG_JSON_PATH,
        embedding_path=EMBEDDING_CACHE_PATH,
        model_name=SEMANTIC_MODEL_NAME
    ):
        self.json_path = json_path
        self.embedding_path = embedding_path

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded.")

        self.documents = []
        self.embeddings = None

        self.load_documents()
        self.load_or_create_embeddings()

    # ── 1. LOAD DOCUMENTS ────────────────────
    def load_documents(self):
        """
        Membaca seluruh data penyakit dari file GRAPHRAG_JSON_PATH.

        Mendukung struktur asli GRAPHRAG.json: {"meta": {...},
        "penyakit_tanaman": [...]}. Tetap kompatibel kalau suatu saat
        filenya berupa list polos (tanpa wrapper "meta").
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(
                f"File {self.json_path} tidak ditemukan."
            )

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        knowledge = raw.get("penyakit_tanaman", []) if isinstance(raw, dict) else raw

        print(f"Total penyakit : {len(knowledge)}")

        self.documents = [self.build_document(disease) for disease in knowledge]

        print(f"Document semantic berhasil dibuat : {len(self.documents)}")

    # ── 2. BUILD DOCUMENT ────────────────────
    def build_document(self, disease):
        """
        Mengubah satu objek JSON penyakit menjadi satu semantic document.

        Sesuai Gambar 3.6 (Pipeline Hybrid RAG), similarity search
        dihitung terhadap field 'chunk_rag' — jadi teks yang di-embed
        adalah chunk_rag itu sendiri (narasi penyakit yang sudah jadi).
        Kalau chunk_rag kosong/tidak ada, fallback ke rekonstruksi teks
        dari field terstruktur (gejala, penanganan, dst) sebagai jaring
        pengaman.
        """
        chunk_rag = disease.get("chunk_rag", "")
        text = chunk_rag.strip() if chunk_rag and chunk_rag.strip() else self._build_fallback_text(disease)

        return {
            "id": disease.get("id"),
            "label": disease.get("nama_tampil", ""),
            "text": text
        }

    def _build_fallback_text(self, disease):
        """
        FALLBACK — hanya dipakai kalau 'chunk_rag' kosong. Merangkai
        manual field terstruktur jadi satu blok teks naratif.
        """
        nama    = disease.get("nama_tampil", "")
        ilmiah  = disease.get("nama_ilmiah", "")
        tanaman = disease.get("tanaman", "")

        patogen    = disease.get("patogen", {}) or {}
        gejala     = disease.get("gejala", []) or []
        penanganan = disease.get("penanganan", []) or []
        pencegahan = disease.get("pencegahan", []) or []
        perawatan  = disease.get("perawatan", []) or []

        patogen_text = ""
        if isinstance(patogen, dict):
            kondisi = patogen.get("kondisi_pemicu", [])
            kondisi_text = ", ".join(kondisi) if isinstance(kondisi, list) else str(kondisi)
            patogen_text = (
                f"Nama Patogen : {patogen.get('nama','')}\n"
                f"Tipe : {patogen.get('tipe','')}\n"
                f"Penyebab : {patogen.get('penyebab_utama','')}\n"
                f"Kondisi Pemicu : {kondisi_text}"
            )

        gejala_text = "\n".join(gejala)

        # 'penanganan' berisi list of dict (tipe, produk, dosis, dst),
        # bukan list string — perlu dirangkai jadi baris teks dulu.
        penanganan_lines = []
        for h in penanganan:
            if isinstance(h, dict):
                penanganan_lines.append(
                    f"{h.get('tipe','')} - {h.get('produk','')} "
                    f"(bahan aktif: {h.get('bahan_aktif','')}, dosis: {h.get('dosis','')})"
                )
            else:
                penanganan_lines.append(str(h))
        penanganan_text = "\n".join(penanganan_lines)

        pencegahan_text = "\n".join(pencegahan)
        perawatan_text  = "\n".join(perawatan)

        return f"""
Nama Penyakit :
{nama}

Nama Ilmiah :
{ilmiah}

Tanaman Inang :
{tanaman}

==========================
PATOGEN
==========================
{patogen_text}

==========================
GEJALA
==========================
{gejala_text}

==========================
PENANGANAN
==========================
{penanganan_text}

==========================
PENCEGAHAN
==========================
{pencegahan_text}

==========================
PERAWATAN
==========================
{perawatan_text}
"""

    # ── 3. CREATE EMBEDDINGS ─────────────────
    def create_embeddings(self):
        """
        Membuat embedding seluruh document menggunakan SentenceTransformer.
        """
        print("\nMembuat embedding dokumen...")

        texts = [doc["text"] for doc in self.documents]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        print(f"Embedding berhasil dibuat : {len(self.embeddings)} document")

    # ── 4. SAVE / LOAD EMBEDDINGS (cache) ────
    def save_embeddings(self):
        """
        Menyimpan embedding agar tidak dihitung ulang setiap start aplikasi.
        """
        with open(self.embedding_path, "wb") as f:
            pickle.dump(
                {"documents": self.documents, "embeddings": self.embeddings}, f
            )

        print(f"Embedding disimpan pada {self.embedding_path}")

    def load_embeddings(self):
        """
        Membaca embedding yang sudah pernah dibuat dari cache.
        """
        with open(self.embedding_path, "rb") as f:
            data = pickle.load(f)

        self.documents  = data["documents"]
        self.embeddings = data["embeddings"]

        print(f"Embedding berhasil dimuat : {len(self.documents)} document")

    def load_or_create_embeddings(self):
        """
        Memuat embedding dari cache jika sudah ada.
        Jika belum ada, embedding dibuat lalu disimpan ke cache.
        """
        if os.path.exists(self.embedding_path):
            print("Embedding cache ditemukan.")
            self.load_embeddings()
        else:
            print("Embedding cache belum ada.")
            self.create_embeddings()
            self.save_embeddings()

    # ── 5. QUERY & SEARCH ────────────────────
    def embed_query(self, question):
        """
        Mengubah pertanyaan pengguna menjadi embedding.
        """
        return self.model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def similarity_search(self, question, top_k=3):
        """
        Menghitung cosine similarity antara query dan seluruh embedding
        dokumen, lalu mengembalikan top_k dokumen paling mirip.
        """
        query_embedding = self.embed_query(question)

        scores = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.documents[idx]["id"],
                "label": self.documents[idx]["label"],
                "text": self.documents[idx]["text"],
                "score": float(scores[idx])
            })

        return results

    def retrieve(self, question, top_k=3):
        """
        Fungsi utama Semantic Retrieval.

        Input  : question (str)
        Output : list top-K semantic chunks, contoh struktur hasil:
            [
                {
                    "id": "tomato_early_blight",
                    "label": "Tomato Early Blight",
                    "score": 0.95,
                    "text": "..."
                },
                ...
            ]
        """
        if question is None or question.strip() == "":
            return []

        results = self.similarity_search(question, top_k)

        print("\n===== SEMANTIC RETRIEVAL =====")
        for i, item in enumerate(results, 1):
            print(f"{i}. {item['label']} (Similarity : {item['score']:.4f})")

        return results