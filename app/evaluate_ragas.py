# -*- coding: utf-8 -*-
"""
evaluate_ragas.py
Evaluasi komponen Graph RAG (Neo4j + Groq) menggunakan framework RAGAS.

Sistem yang dievaluasi:
- Knowledge Graph penyakit tumbuhan hortikultura (Neo4j Aura)
- Generator jawaban: Groq openai/gpt-oss-120b (di rag_engine.py)
- Evaluator (judge): model OpenRouter (lihat JUDGE_MODEL) -- sengaja beda
  provider & model dari generator untuk menghindari self-preference bias.

Catatan metodologis:
- RAGAS HANYA mengevaluasi sub-sistem RAG, BUKAN model CNN.
- Suhu judge diset 0 untuk meminimalkan variansi hasil penilaian.

Diuji terhadap: ragas==0.2.x (lihat requirements_eval.txt).
Jika versi RAGAS Anda berbeda, nama kelas metrik dapat berubah;
sesuaikan baris impor pada bagian METRIK.
"""

import os
import sys
import types
import json
import hashlib
import argparse
import traceback
import pandas as pd

# =====================================================================
# 0. SHIM KOMPATIBILITAS (WAJIB sebelum impor ragas)
#    Sebagian versi ragas meng-impor ChatVertexAI dari lokasi lama
#    'langchain_community.chat_models.vertexai' yang sudah dihapus di
#    langchain-community baru -> ModuleNotFoundError.
#    Sistem ini memakai Groq, tidak butuh Vertex AI, jadi kita sediakan
#    modul tiruan agar impor ragas tidak gagal. Lihat issue ragas #2741.
# =====================================================================
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        # pakai kelas asli jika kebetulan tersedia
        from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    except Exception:
        class _ChatVertexAI:  # kelas dummy; tidak pernah dipakai (kita pakai Groq)
            pass
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

# --- RAGAS core ---
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# --- METRIK (RAGAS v0.2.x) ---
# Empat metrik inti RAG + dua metrik berbasis acuan (opsional).
from ragas.metrics import (
    Faithfulness,                       # kesetiaan jawaban thd konteks KG
    ResponseRelevancy,                  # relevansi jawaban thd pertanyaan
    LLMContextPrecisionWithReference,   # presisi konteks ter-retrieve
    LLMContextRecall,                   # kelengkapan konteks vs acuan
    FactualCorrectness,                 # kebenaran faktual (F1) vs acuan
    SemanticSimilarity,                 # kemiripan semantik jawaban vs acuan
)

# --- LangChain wrapper: judge via OpenRouter (API OpenAI-compatible), embedding lokal ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_core.rate_limiters import InMemoryRateLimiter
except Exception:
    InMemoryRateLimiter = None


# =====================================================================
# 1. KONFIGURASI
# =====================================================================
# Pakai key dari environment; bila kosong, pinjam dari rag_engine
# (catatan: sebaiknya pindahkan semua key ke .env demi keamanan).
try:
    from rag_engine import GROQ_API_KEY as _RAG_GROQ_KEY
except Exception:
    _RAG_GROQ_KEY = None
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY") or _RAG_GROQ_KEY  # dipakai rag_engine (generator RAG)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")             # dipakai judge RAGAS di bawah
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL    = os.environ.get("RAGAS_JUDGE_MODEL", "openai/gpt-4o-mini")
# gpt-4o-mini dipilih karena murah (~$0.15/$0.60 per 1 juta token input/output)
# tapi cukup andal untuk tugas judge terstruktur RAGAS -- penting saat kuota
# terbatas ($3 kredit OpenRouter). Judge sengaja beda model dari generator
# (anti-bias self-preference). Cek harga/ketersediaan terkini & override lewat
# env var RAGAS_JUDGE_MODEL bila model ini berubah nama/harga di openrouter.ai/models.
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # mendukung Bahasa Indonesia
GOLDEN_PATH    = "golden_dataset.json"                   # test set buatan
OUTPUT_CSV     = "hasil_ragas_per_item.csv"              # skor tiap item
OUTPUT_SUMMARY = "hasil_ragas_ringkasan.csv"             # rata-rata per metrik
SCORE_CACHE    = "ragas_scores_cache.json"               # skor per item (utk resume)

# Instance RAGEngine dibuat sekali (koneksi Neo4j + Groq), dipakai ulang.
_ENGINE = None


# =====================================================================
# 2. ADAPTER KE rag_engine.py  (disesuaikan dengan kelas RAGEngine Anda)
# =====================================================================
def _get_engine():
    """Inisialisasi RAGEngine sekali saja (lazy singleton)."""
    global _ENGINE
    if _ENGINE is None:
        from rag_engine import RAGEngine
        _ENGINE = RAGEngine()
    return _ENGINE


def _contexts_dari_retrieved(d: dict) -> list:
    """
    Memecah subgraph yang ditarik dari Neo4j menjadi potongan-potongan
    teks terpisah (patogen, tiap gejala, tiap penanganan, dst). RAGAS
    menghitung context_precision/recall PER potongan, sehingga konteks
    granular jauh lebih informatif daripada satu blok teks panjang.
    """
    ctx = []
    pat = d.get("patogen", {}) or {}
    if pat.get("nama"):
        bagian = f"Patogen: {pat.get('nama','')} (tipe: {pat.get('tipe','-')})."
        if pat.get("penyebab_utama"):
            bagian += f" Penyebab utama: {pat['penyebab_utama']}."
        kondisi = pat.get("kondisi_pemicu") or []
        if kondisi:
            bagian += f" Kondisi pemicu: {', '.join(kondisi)}."
        ctx.append(bagian)
    for g in d.get("gejala", []):
        ctx.append(f"Gejala: {g}")
    for h in d.get("penanganan", []):
        bagian = (f"Penanganan (tipe {h.get('tipe','-')}): produk {h.get('produk','-')}, "
                  f"bahan aktif {h.get('bahan_aktif','-')}, dosis {h.get('dosis','-')}.")
        if h.get("cara"):
            bagian += f" Cara: {h['cara']}."
        ctx.append(bagian)
    for c in d.get("pencegahan", []):
        ctx.append(f"Pencegahan: {c}")
    for r in d.get("perawatan", []):
        ctx.append(f"Perawatan: {r}")
    if d.get("chunk_rag"):
        ctx.append(str(d["chunk_rag"]))
    return [c for c in ctx if c and c.strip()]


def jalankan_rag(pertanyaan: str, neo4j_id: str):
    """
    Menjalankan pipeline Graph RAG untuk satu pertanyaan dan mengembalikan:
      1) response : jawaban LLM MURNI (tanpa ringkasan KG markdown)
      2) konteks  : list potongan teks dari subgraph Neo4j

    Alur mengikuti RAGEngine: retrieve(id) -> build_context -> generate.
    ID penyakit disuplai dari golden dataset agar RAG diuji terpisah dari CNN.
    """
    engine = _get_engine()

    retrieved = engine.retrieve(neo4j_id)
    if not retrieved:
        # id tidak ditemukan di Neo4j -> kembalikan kosong + sinyal jelas
        print(f"  ! ID '{neo4j_id}' tidak ada di Neo4j (cek kolom 'kelas' di golden_dataset).")
        return "", []

    # potongan konteks untuk RAGAS
    konteks = _contexts_dari_retrieved(retrieved)

    # prediksi 'dummy' (1.0) hanya agar build_context tidak memunculkan
    # peringatan confidence rendah; evaluasi RAG tidak bergantung pada CNN.
    prediction = {"predicted_class": neo4j_id, "confidence_resnet": 1.0}
    context_str = engine.build_context(retrieved, prediction)

    # generate() mengembalikan: ringkasan_KG + "### 💬 Jawaban AI\n\n" + jawaban_LLM
    keluaran = engine.generate(context_str, pertanyaan, retrieved=retrieved)

    # ambil HANYA jawaban LLM agar penilaian RAGAS jujur
    penanda = "### 💬 Jawaban AI"
    if penanda in keluaran:
        jawaban = keluaran.split(penanda, 1)[1].strip()
    else:
        jawaban = keluaran.strip()

    return jawaban, konteks


# =====================================================================
# 3. BANGUN DATASET EVALUASI
# =====================================================================
def bangun_records(golden: list) -> list:
    """
    Untuk tiap pertanyaan pada golden dataset, jalankan RAG dan susun
    record sesuai skema RAGAS v0.2:
        user_input          : pertanyaan
        retrieved_contexts  : konteks dari Neo4j (list[str])
        response            : jawaban sistem
        reference           : jawaban acuan (ground truth)
    """
    records = []
    gagal = 0
    for i, item in enumerate(golden, start=1):
        pertanyaan = item["pertanyaan"]
        acuan      = item["acuan"]
        # ID node Penyakit di Neo4j; diambil dari kolom 'kelas' (atau 'id')
        neo4j_id   = item.get("kelas") or item.get("id")
        print(f"[{i}/{len(golden)}] [{neo4j_id}] {pertanyaan[:55]}...")

        try:
            jawaban, konteks = jalankan_rag(pertanyaan, neo4j_id)
        except Exception:
            print(f"  ! GAGAL menjalankan RAG untuk pertanyaan ini. Traceback:")
            traceback.print_exc()   # tampilkan error sebenarnya, jangan ditelan
            jawaban, konteks = "", []

        # generate() di rag_engine.py mengembalikan pesan galat sebagai STRING
        # biasa (bukan exception) saat panggilan Groq gagal (mis. rate limit),
        # supaya UI Gradio tetap menampilkan sesuatu. String itu tidak kosong,
        # jadi tanpa pengecekan ini akan lolos sebagai "jawaban" sah dan ikut
        # dinilai RAGAS -- mengukur kemiripan pesan error dengan acuan, bukan
        # kualitas RAG. Normalisasi ke "" di sini menyalurkannya ke jalur
        # "gagal" yang sama seperti jawaban kosong.
        if str(jawaban).lstrip().startswith("❌ Gagal generate jawaban dari Groq:"):
            jawaban = ""

        if not str(jawaban).strip():
            gagal += 1
            print(f"  ! PERINGATAN: jawaban KOSONG (konteks={len(konteks)} potongan)")

        records.append({
            "user_input": pertanyaan,
            "retrieved_contexts": konteks,
            "response": jawaban,
            "reference": acuan,
        })

    # Hentikan lebih awal bila mayoritas jawaban kosong -> hemat kuota Groq
    if gagal == len(golden):
        raise SystemExit(
            "\nSEMUA jawaban kosong/gagal digenerate. Kemungkinan penyebab:\n"
            "1. GROQ_API_KEY belum diset di .env, atau kuota harian (TPD) Groq\n"
            "   sudah habis -- cek pesan error 429 di atas kalau ada.\n"
            "2. Kolom 'kelas'/'id' di golden_dataset.json tidak cocok dengan\n"
            "   id node Penyakit di Neo4j, sehingga retrieve() mengembalikan None.\n"
            "Cek dengan menjalankan RAGEngine().retrieve(<neo4j_id>) secara manual\n"
            "di Python REPL untuk memastikan datanya ditemukan, lalu sesuaikan\n"
            "jalankan_rag() di file ini bila strukturnya berubah."
        )
    elif gagal > 0:
        print(f"\n! {gagal}/{len(golden)} jawaban kosong — skor item tsb akan 0/NaN.\n")
    return records


def _cache_key(rec: dict) -> str:
    """Kunci cache berbasis isi pertanyaan (bukan indeks urutan), supaya
    perubahan urutan/isi golden_dataset.json tidak salah menautkan skor lama
    ke pertanyaan yang berbeda."""
    return hashlib.md5(rec["user_input"].encode("utf-8")).hexdigest()[:12]


def evaluasi_bertahap(records, metrics, evaluator_llm, evaluator_emb, run_config,
                       cache_path=SCORE_CACHE):
    """
    Evaluasi RAGAS satu item per satu, skor disimpan ke cache_path setelah
    tiap item selesai. Item yang skornya sudah lengkap di cache dilewati,
    jadi kalau proses berhenti (rate limit dsb) run berikutnya otomatis
    melanjutkan dari item yang belum selesai, bukan mengulang dari awal.
    """
    skor_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            skor_cache = json.load(f)

    metric_names = [m.name for m in metrics]

    def _ambil_skor(baris: dict, nama: str):
        """Ambil nilai metrik dari baris pandas RAGAS. Sebagian metrik (mis.
        FactualCorrectness) memberi nama kolom bersuffix mode — 'factual_correctness(mode=f1)'
        — sehingga baris.get('factual_correctness') = None. Cocokkan juga kolom
        yang berawalan '<nama>(' supaya nilainya tidak hilang jadi NaN."""
        if nama in baris:
            return baris[nama]
        for kolom in baris:
            if kolom == nama or kolom.startswith(nama + "("):
                return baris[kolom]
        return None

    for i, rec in enumerate(records):
        key = _cache_key(rec)
        sudah = skor_cache.get(key)
        if sudah and all(m in sudah for m in metric_names):
            print(f"  [{i+1}/{len(records)}] sudah ada di cache skor, lewati.")
            continue

        print(f"  [{i+1}/{len(records)}] mengevaluasi...")
        try:
            hasil = evaluate(
                dataset=EvaluationDataset.from_list([rec]),
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_emb,
                run_config=run_config,
            )
            baris = hasil.to_pandas().iloc[0].to_dict()
            skor_cache[key] = {**rec, **{m: _ambil_skor(baris, m) for m in metric_names}}
        except Exception:
            print(f"  ! GAGAL item {i+1}, dilewati untuk sekarang (coba lagi di run berikutnya).")
            traceback.print_exc()

        # simpan setelah tiap item (bukan cuma di akhir) supaya progres tidak hilang
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(skor_cache, f, ensure_ascii=False, indent=2)

    baris_akhir = [skor_cache.get(_cache_key(rec), dict(rec)) for rec in records]
    return pd.DataFrame(baris_akhir)


# =====================================================================
# 4. MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluasi RAGAS untuk Graph RAG")
    parser.add_argument("--golden", default=GOLDEN_PATH, help="path golden dataset JSON")
    parser.add_argument("--cache", default=None,
                        help="path JSON cache record (lewati pemanggilan RAG ulang)")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY belum diset. Jalankan: export GROQ_API_KEY=...")
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY belum diset (dipakai judge RAGAS). "
                          "Set di .env atau: export OPENROUTER_API_KEY=...")

    # --- 4a. Siapkan evaluator LLM (judge) via OpenRouter ---
    # temperature=0 -> hasil judge stabil & reproducible.
    # rate_limiter menahan laju panggilan agar tidak menembus batas rate limit OpenRouter.
    # max_tokens tinggi: metrik FactualCorrectness mendekomposisi jawaban jadi banyak
    # klaim; jawaban panjang bisa membuat output judge terpotong (LLMDidNotFinishException
    # -> skor NaN). 8000 memberi ruang cukup untuk dekomposisi klaim jawaban terpanjang.
    kwargs = dict(model=JUDGE_MODEL, temperature=0, api_key=OPENROUTER_API_KEY,
                  base_url=OPENROUTER_BASE_URL, max_retries=8, max_tokens=8000)
    if InMemoryRateLimiter is not None:
        # ~0.5 req/detik (≈ 30 req/menit). Turunkan bila masih kena limit.
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.5,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    judge = ChatOpenAI(**kwargs)
    evaluator_llm = LangchainLLMWrapper(judge)

    # --- 4b. Siapkan embedding (lokal, multibahasa, mendukung Bhs Indonesia) ---
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    evaluator_emb = LangchainEmbeddingsWrapper(embeddings)

    # --- 4c. Susun record (jalankan RAG, atau muat dari cache) ---
    if args.cache and os.path.exists(args.cache):
        print(f"Memuat record dari cache: {args.cache}")
        with open(args.cache, encoding="utf-8") as f:
            records = json.load(f)
    else:
        with open(args.golden, encoding="utf-8") as f:
            golden = json.load(f)
        records = bangun_records(golden)
        # simpan cache agar tidak perlu memanggil RAG ulang saat tuning evaluasi
        with open("records_cache.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print("Record disimpan ke records_cache.json")

    # --- 4d. Daftar metrik ---
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),          # mode default = F1
        SemanticSimilarity(embeddings=evaluator_emb),
    ]

    # --- 4e. Konfigurasi eksekusi (hindari rate limit OpenRouter) ---
    # max_workers=1 -> panggilan diserialkan (tidak burst); cocok utk free tier.
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=10)

    print(f"\nMenjalankan evaluasi RAGAS per item (bisa beberapa menit; skor "
          f"disimpan bertahap ke {SCORE_CACHE}, aman dilanjutkan bila terhenti)...")
    df = evaluasi_bertahap(records, metrics, evaluator_llm, evaluator_emb, run_config)

    # --- 4f. Simpan & cetak hasil ---
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "retrieved_contexts", "response", "reference")]
    total = len(df)
    ringkasan = df[metric_cols].mean().round(4)        # skip NaN otomatis
    terhitung = df[metric_cols].notna().sum()          # berapa item valid per metrik
    out = pd.DataFrame({"rata_rata": ringkasan, "item_terhitung": terhitung,
                        "dari_total": total})
    out.to_csv(OUTPUT_SUMMARY, encoding="utf-8-sig")

    print("\n================ RINGKASAN SKOR RAGAS ================")
    print(f"  {'METRIK':<38}{'RATA²':>8}  TERHITUNG")
    ada_masalah = False
    for nama in metric_cols:
        n = int(terhitung[nama])
        nilai = ringkasan[nama]
        tanda = "" if n == total else "  <-- TIDAK LENGKAP!"
        if n < total:
            ada_masalah = True
        nilai_str = f"{nilai:.4f}" if pd.notna(nilai) else "  NaN "
        print(f"  {nama:<38}{nilai_str:>8}  {n}/{total}{tanda}")
    print("=====================================================")
    if ada_masalah:
        print("PERINGATAN: sebagian metrik tidak terhitung penuh (kemungkinan\n"
              "rate limit OpenRouter/model judge). Rata-rata di atas TIDAK valid untuk dilaporkan\n"
              "sampai 'item_terhitung' = total. Jalankan lagi perintah yang sama\n"
              f"({SCORE_CACHE} akan membuat item yang sudah lengkap dilewati,\n"
              "sisanya otomatis dicoba lagi), atau kurangi requests_per_second di 4a.")
    print(f"Detail per item : {OUTPUT_CSV}")
    print(f"Ringkasan       : {OUTPUT_SUMMARY}")

    # Tutup koneksi Neo4j bila engine sempat dibuat
    if _ENGINE is not None:
        try:
            _ENGINE.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()