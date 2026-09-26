# Rangkuman & Pembelajaran Kode — Sistem Deteksi & Interpretasi Penyakit Tanaman

Dokumen ini merangkum **seluruh kode di proyek ini**: apa fungsinya, di mana letaknya, bagaimana saling terhubung, konsep dasar di baliknya, dan bagaimana semuanya berjalan dari awal sampai akhir. Tujuannya supaya siapa pun (termasuk Anda sendiri saat evaluasi) bisa menjelaskan sistem ini tanpa harus membuka semua file satu-satu.

---

## 1. Gambaran Besar — Apa dan Kenapa

Sistem ini punya dua "otak" yang bekerja berurutan:

1. **CNN (ResNet-50)** menjawab pertanyaan *"penyakit apa ini?"* dari gambar daun — tugas klasifikasi citra.
2. **Hybrid RAG (Neo4j + Semantic + Groq LLM)** menjawab pertanyaan *"apa artinya, dan apa yang harus dilakukan?"* — tugas mengambil pengetahuan terstruktur lalu menuliskannya jadi jawaban natural bahasa Indonesia.

Kenapa tidak cukup CNN saja? Karena CNN cuma bisa mengeluarkan **label** (mis. `"early_blight"`), bukan penjelasan, dosis obat, atau jawaban atas pertanyaan spesifik petani. Karena itu labelnya "dijembatani" ke Knowledge Graph (Neo4j) yang berisi data terstruktur per penyakit, lalu LLM (Groq) merangkainya jadi kalimat.

Kenapa RAG-nya "hybrid" (dua jalur, bukan satu)? Karena Graph Retrieval (Neo4j) hanya tahu **satu** penyakit — yang sudah ditentukan CNN. Kalau petani bertanya sesuatu yang menyinggung penyakit lain, atau CNN kurang yakin, sistem butuh cara mencari lintas seluruh basis pengetahuan (12 penyakit) berdasarkan kemiripan makna teks — itu peran Semantic Retrieval.

---

## 2. Peta File & Keterhubungan (Import Graph)

```
app.py                     ← dijalankan langsung (python app.py)
 ├── cnn_model.py           (load_model, predict_disease)
 └── rag_engine.py           (RAGEngine)
      └── semantic_retriever.py  (SemanticRetriever)

evaluate_ragas.py           ← dijalankan langsung, TERPISAH dari app.py
 └── rag_engine.py           (reuse RAGEngine yang sama)
      └── semantic_retriever.py

download_vis.py             ← utilitas sekali-pakai, tidak terhubung ke apa pun
                               (download vis-network.min.js untuk visualisasi graph)
```

Poin penting: `rag_engine.py` dipakai **dua kali** oleh dua "pintu masuk" berbeda (`app.py` untuk produksi, `evaluate_ragas.py` untuk evaluasi) — tapi isinya file yang sama persis, tidak ada duplikasi logika. Ini contoh desain yang baik: logika RAG cuma ditulis sekali, dipakai ulang.

---

## 3. Konsep Dasar di Balik Tiap Bagian

| Bagian | Konsep | Penjelasan singkat |
|---|---|---|
| `cnn_model.py` | **Transfer Learning** | Alih-alih melatih CNN dari nol, pakai `microsoft/resnet-50` yang sudah dilatih di ImageNet (1000 kelas objek umum), lalu classifier head-nya diganti untuk **13 kelas** (12 penyakit + 1 kelas `unknown`) dan dilatih ulang (fine-tuning) — jauh lebih cepat & butuh data lebih sedikit daripada training dari nol. |
| `cnn_model.py` | **Open-Set Recognition** | Model dilatih dengan kelas `unknown` eksplisit **+** ambang confidence `0.70`: gambar di luar objek penelitian (bukan daun apel/anggur/jagung/tomat, atau model tidak yakin) ditolak (`is_unknown=True`) sebelum masuk RAG. Ini menggantikan pendekatan gate eksternal — penolakan sekarang murni dari model yang dilatih pakai data latih. |
| `cnn_model.py` | **Softmax + Argmax** | Output CNN mentah (logit) diubah jadi probabilitas (0-1, total=1) lewat Softmax, lalu kelas dengan probabilitas tertinggi dipilih lewat Argmax — itulah `predicted_class` dan `confidence`. |
| `rag_engine.py` | **Knowledge Graph & Cypher** | Neo4j menyimpan data sebagai node (entitas, mis. `:Penyakit`, `:Gejala`) dan relasi berarah (mis. `MEMILIKI_GEJALA`). Cypher adalah bahasa query-nya — satu query bisa menarik satu penyakit beserta *seluruh* relasinya sekaligus, beda dengan tabel SQL yang butuh banyak JOIN. |
| `semantic_retriever.py` | **Embedding & Cosine Similarity** | Teks diubah jadi vektor angka (embedding) lewat model `SentenceTransformer` — teks yang maknanya mirip akan punya vektor yang "berdekatan". Cosine similarity mengukur seberapa dekat dua vektor (1 = identik, 0 = tidak berkaitan). Ini cara komputer "mengerti" kemiripan makna, bukan sekadar mencocokkan kata. |
| `rag_engine.py` (generate) | **Prompt Engineering & LLM** | LLM (Groq, model `openai/gpt-oss-120b`) tidak tahu apa-apa soal Neo4j — dia cuma menerima teks (`system_prompt` + `user_prompt`) dan mengembalikan teks. "Konteks" dari Neo4j/semantic sengaja disusun rapi jadi bagian dari prompt, supaya jawaban LLM berbasis fakta yang benar-benar ada, bukan karangan (mengurangi *hallucination*). |
| `rag_engine.py` (RAG) | **Retrieval-Augmented Generation** | Pola umum: **ambil dulu** data yang relevan (retrieval) baru **serahkan ke LLM** untuk dirangkai jadi jawaban (generation) — bukan LLM menjawab dari "ingatannya" sendiri yang bisa salah/ketinggalan zaman. |
| `evaluate_ragas.py` | **LLM-as-a-Judge** | Karena jawaban RAG berupa teks bebas (bukan angka), kualitasnya dinilai pakai LLM lain yang bertindak sebagai "juri" — bukan dihitung manual satu-satu. |

---

## 4. Titik Masuk (Entry Point)

**`app.py`** adalah satu-satunya file yang dijalankan langsung (`python app.py`). File lain hanya **diimpor**:

- `cnn_model.py` — diimpor oleh `app.py`
- `rag_engine.py` — diimpor oleh `app.py` **dan** `evaluate_ragas.py`
- `semantic_retriever.py` — diimpor oleh `rag_engine.py`
- `evaluate_ragas.py` — jalur terpisah, dijalankan manual, **tidak pernah** dipanggil `app.py`

---

## 5. Yang Berjalan Sekali Saat Startup (`python app.py`)

| Urutan | Kode | Yang Terjadi |
|---|---|---|
| 1 | `app.py` `load_model()` → `cnn_model.py` | Download arsitektur `microsoft/resnet-50` dari HuggingFace, ganti classifier head jadi 13 kelas (`num_labels=13`), lalu timpa semua bobot dari `models/resnet50_final.pt`. Daftar kelas (`classes`) & ambang `unknown_threshold` (0.70) ikut dibaca dari checkpoint agar konsisten dengan bobot |
| 2 | `app.py:21` `RAGEngine()` → `rag_engine.py:69-84` | Muat `.env` (kredensial), buka koneksi ke Neo4j Aura, buat client Groq, lalu instansiasi `SemanticRetriever()` |
| 2a | ↳ `semantic_retriever.py:39-56` | Load `GRAPHRAG.json` (12 penyakit) → `build_document()` per penyakit (pakai field `chunk_rag`) → load embedding dari cache `embedding_cache.pkl` (atau buat baru kalau belum ada) |
| 3 | `app.py:155-240` | Bangun UI Gradio: 3 tab (Hasil CNN, Graph Neo4j, Jawaban RAG) |
| 4 | `app.py:248-250` `demo.launch()` | Web server Gradio aktif, siap menerima request dari pengguna |

Semua langkah di atas terjadi **satu kali saja**, sebelum pengguna melakukan apa pun. Ini penting untuk performa: model besar (CNN, embedding) mahal untuk dimuat, jadi dimuat sekali lalu dipakai berulang-ulang, bukan dimuat ulang tiap request.

---

## 6. Pipeline Runtime — Setiap Kali Pengguna Upload Gambar + Klik "Analisis"

Trigger: `submit_btn.click()` (`app.py:228-233`) memanggil `analyze_image()` (`app.py:29-137`) — sebuah **generator function** (pakai `yield`, bukan `return`) supaya tiap tab UI terisi bertahap, tanpa menunggu seluruh pipeline (termasuk panggilan Groq yang paling lambat) selesai dulu. Ini pola penting untuk UX: pengguna melihat hasil CNN dalam hitungan detik, sementara jawaban LLM (yang bisa perlu beberapa detik lagi) masih diproses di belakang layar.

```
Gambar daun + pertanyaan (opsional)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 1 — CNN Inference                                      │
│ app.py:60-63 → cnn_model.py:predict_disease()                │
│  • _preprocess(): RGB + resize 224x224 (cnn_model.py:88-93) │
│  • forward pass ResNet-50, no_grad (cnn_model.py:104-110)   │
│  • softmax → argmax → predicted_class, confidence          │
│  • Open-set gate: kelas `unknown` ATAU confidence < 0.70   │
│    → is_unknown=True → TOLAK, pipeline berhenti (app.py)    │
│  • CLASS_TO_NEO4J mapping (cnn_model.py) → neo4j_id         │
│    (None jika unknown)                                      │
└───────────────────────────────────────────────────────────┘
        │  jika unknown → TAB penolakan, STOP
        │  jika valid → yield → TAB "Hasil CNN" tampil
        ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 2 — Graph Retrieval (Jalur A Hybrid RAG)                │
│ app.py:85 → rag_engine.py:retrieve(neo4j_id)                  │
│  • Query Cypher ke Neo4j pakai neo4j_id (rag_engine.py:170) │
│  • Ambil node Penyakit + Patogen + collect Gejala/          │
│    Penanganan/Pencegahan/Perawatan — SATU query, SATU       │
│    penyakit, SEMUA relasinya (data utama/otoritatif)        │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 3 — Deteksi Intent Pertanyaan (LLM)                     │
│ app.py:100 → rag_engine.py:detect_question_intent()           │
│  • Pertanyaan user dikirim ke Groq (openai/gpt-oss-120b)    │
│    sebagai classifier → kategori relevan (identitas/        │
│    patogen/gejala/penanganan/pencegahan/perawatan)          │
│  • Gagal 2x berturut-turut → fallback ke SEMUA kategori     │
│  • Tujuan: jawaban tidak melebar ke topik yang tak ditanya  │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 4 — Semantic Retrieval (Jalur B Hybrid RAG)             │
│ app.py:107 → rag_engine.py:retrieve_semantic()                 │
│              → semantic_retriever.py:retrieve()                │
│  • query = predicted_class + question                      │
│  • embed pakai all-MiniLM-L6-v2 (semantic_retriever.py:23)  │
│  • cosine similarity vs 12 dokumen penyakit                 │
│  • exclude_id: SENGAJA buang penyakit yang sama dgn hasil   │
│    Graph RAG — jalur ini mencari penyakit LAIN sebagai      │
│    pembanding, bukan pelengkap penyakit yang terdeteksi     │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 5 — Hybrid Fusion                                       │
│ app.py:113 → rag_engine.py:hybrid_fusion()                     │
│  • Gabung hasil Jalur A (graph) + Jalur B (semantic)        │
│    jadi satu dict `fused` — CUMA DITEMPEL, tidak ada        │
│    pembobotan/skor gabungan (bukan weighted blending)       │
└───────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│ STEP 6a — Build Graph HTML │  │ STEP 6b — Build Context        │
│ app.py:118                 │  │ app.py:126                     │
│ rag_engine.py:              │  │ rag_engine.py:                  │
│ build_graph_html()          │  │ build_context()                 │
│  • vis.js network sesuai    │  │  • susun teks sesuai intent      │
│    intent → HTML iframe     │  │    (info penyakit = utama, hasil │
│                             │  │    semantic = section terpisah,  │
│                             │  │    ditandai "pembanding")        │
└──────────────────────────┘  └──────────────────────────────┘
        │  yield → TAB "Graph Neo4j" (app.py:122)          │
        │                                                  ▼
        │                              ┌──────────────────────────────┐
        │                              │ STEP 7 — Generate Jawaban       │
        │                              │ app.py:127 → rag_engine.py:      │
        │                              │ generate()                      │
        │                              │  • system prompt eksplisit       │
        │                              │    melarang LLM jadikan hasil    │
        │                              │    semantic sbg diagnosis utama  │
        │                              │  • kirim ke Groq llama-3.3-70b-  │
        │                              │    versatile (temperature=0.2,   │
        │                              │    max_tokens=1024)               │
        │                              │  • gabung neo4j_summary + LLM    │
        │                              └──────────────────────────────┘
        │                                        │
        └────────────────────────────────────────┤
                                                   ▼
                          yield → TAB "Jawaban RAG" tampil (app.py:137)
                                    PIPELINE SELESAI
```

### Struktur data yang mengalir antar fungsi (biar tidak bingung baca kode)

- `prediction` (dari Step 1): `{predicted_class, neo4j_id, confidence_resnet, is_unknown, unknown_reason, low_confidence, probabilities_resnet}` — `is_unknown=True` menghentikan pipeline sebelum Step 2 (`neo4j_id=None`); `unknown_reason` = `"kelas_unknown"` | `"confidence_rendah"` | `None`
- `retrieved` (dari Step 2): `{id, nama_tampil, nama_ilmiah, tanaman, tingkat_keparahan, chunk_rag, patogen{...}, gejala[...], penanganan[...], pencegahan[...], perawatan[...]}`
- `intents` (dari Step 3): sebuah `set` berisi subset dari `{"identitas","patogen","gejala","penanganan","pencegahan","perawatan"}`
- `semantic_hits` (dari Step 4): list of `{id, label, text, score}`
- `fused` (dari Step 5): sama seperti `retrieved`, ditambah key `"semantic_chunks"` berisi `semantic_hits`
- `context` (dari Step 6b): satu string panjang berformat `=== JUDUL ===\n isi...` — inilah yang dikirim ke LLM

---

## 7. Jalur Terpisah — Evaluasi RAGAS (Bukan Bagian Pipeline Produksi)

`evaluate_ragas.py` **tidak pernah dipanggil oleh `app.py`** — dijalankan manual untuk mengukur kualitas Step 2, 6b, dan 7 di atas secara offline, terlepas dari akurasi CNN:

1. Baca `golden_dataset.json` (12 pertanyaan + `neo4j_id` + jawaban acuan) — `neo4j_id` disuplai langsung dari file, **Step 1 (CNN) di-skip sepenuhnya**, supaya kualitas RAG diuji terpisah dari akurasi CNN.
2. `jalankan_rag()` (`evaluate_ragas.py:144-179`) memanggil ulang `RAGEngine().retrieve()` → `build_context()` → `generate()` — fungsi yang **persis sama** dengan pipeline produksi (reuse Step 2, 6b, 7), tanpa Step 1/3/4/5/6a.
3. Kumpulan hasil dinilai oleh RAGAS memakai judge model **berbeda** dari generator (judge `openai/gpt-4o-mini` via OpenRouter vs generator `openai/gpt-oss-120b` via Groq, untuk menghindari bias self-preference) dan embedding **berbeda** dari yang dipakai produksi (`paraphrase-multilingual-mpnet-base-v2` vs `all-MiniLM-L6-v2`).
4. Output: `hasil_ragas_per_item.csv` (skor tiap pertanyaan) dan `hasil_ragas_ringkasan.csv` (rata-rata tiap metrik).
5. Sebelumnya ditemukan bug: pesan error rate-limit Groq (string, bukan exception) ikut dinilai RAGAS seolah jawaban sah — sudah diperbaiki (lihat bagian 9).

Dependency evaluasi (`ragas`, `langchain-groq`, dst.) sengaja dipisah ke `requirements_eval.txt` dan venv terpisah (`venv_eval/`) — supaya install/upgrade paket evaluasi tidak berisiko mengubah versi `torch`/`transformers` yang dipakai pipeline produksi (`venv/`).

---

## 8. Ringkasan Pemetaan File → Tanggung Jawab

| File | Tanggung Jawab | Dipanggil Oleh |
|---|---|---|
| `app.py` | UI Gradio + orkestrasi seluruh pipeline (Step 1-7) | Dijalankan langsung (`python app.py`) |
| `cnn_model.py` | Load model + inferensi ResNet-50 (Step 1) | `app.py` |
| `rag_engine.py` | Koneksi Neo4j (Step 2), deteksi intent (Step 3), hybrid fusion (Step 5), visualisasi graph (Step 6a), susun konteks + generate jawaban Groq (Step 6b, 7) | `app.py`, `evaluate_ragas.py` |
| `semantic_retriever.py` | Embedding dokumen/query + cosine similarity (Step 4) | `rag_engine.py` |
| `evaluate_ragas.py` | Jalur evaluasi kualitas RAG secara offline, reuse fungsi dari `rag_engine.py` | Dijalankan langsung (manual, terpisah dari `app.py`) |
| `download_vis.py` | Utilitas sekali-pakai untuk mengunduh `vis-network.min.js` | Dijalankan manual, tidak terhubung ke pipeline |

---

## 9. Pelajaran Penting dari Menelusuri Kode Ini

Beberapa insight yang baru terlihat setelah membaca seluruh kode secara menyeluruh (bukan sekadar membaca laporan):

1. **`hybrid_fusion()` bukan penggabungan berbobot** — dia cuma menempelkan hasil semantic sebagai key tambahan (`rag_engine.py:264-273`). Tidak ada rumus persentase/skor gabungan antara Graph dan Semantic Retrieval. Proporsi kontribusinya bervariasi per pertanyaan, bukan rasio tetap.
2. **Semantic Retrieval by design mencari penyakit LAIN**, bukan pelengkap penyakit yang terdeteksi (`exclude_id=neo4j_id`, `rag_engine.py:255-256`). Menaikkan bobot/porsi semantic tanpa memperbaiki desain ini berisiko mencampur informasi penyakit yang salah ke jawaban — signifikan karena kontennya menyangkut dosis bahan kimia pertanian.
3. **Tidak ada ambang skor minimum di `semantic_retriever.py`** — sistem selalu mengembalikan top-k dokumen meski skor kemiripannya rendah, sehingga "ada hasil semantic" tidak berarti hasilnya benar-benar relevan.
4. **Model embedding produksi (`all-MiniLM-L6-v2`) berorientasi Bahasa Inggris**, dipakai untuk 100% teks Bahasa Indonesia — sementara model embedding di evaluasi RAGAS berbeda dan memang dipilih untuk mendukung Bahasa Indonesia. Ketidakkonsistenan ini membuat skor evaluasi tidak sepenuhnya mencerminkan retrieval produksi.
5. **Kredensial sempat hardcoded di `rag_engine.py`** (Neo4j password, Groq API key) — sudah dipindah ke `.env` dengan loader stdlib (tanpa dependency baru), lihat bagian 10.
6. **`evaluate_ragas.py` sempat punya bug kritis**: pesan error Groq (string, bukan exception) ikut dinilai RAGAS seolah jawaban sah, membuat skor kualitas RAG di run lama tidak valid. Sudah diperbaiki dengan menormalkan pesan error jadi string kosong sebelum dinilai.
7. **`cnn_model.py` sempat membuang hasil `load_state_dict()` tanpa diperiksa** — berisiko model diam-diam memakai bobot acak kalau checkpoint tidak cocok penuh dengan arsitektur, tanpa ada peringatan. Sudah diperbaiki dengan memeriksa & mencetak `missing_keys`/`unexpected_keys`; hasil pengujian nyata menunjukkan checkpoint `resnet50_final.pt` saat ini cocok 100% (tidak ada key yang hilang).
8. **`requirements.txt` awalnya tidak lengkap** — `sentence-transformers` dan `scikit-learn` dipakai langsung oleh `semantic_retriever.py` tapi tidak tercantum, sehingga instalasi dari nol akan gagal. Sudah diperbaiki.

---

## 10. Konfigurasi & Kredensial

Kredensial (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GROQ_API_KEY`) **tidak lagi hardcoded** di `rag_engine.py` — dibaca dari file `.env` di root proyek lewat loader manual (`rag_engine.py`, fungsi `_load_env_file()`), tanpa dependency tambahan (`python-dotenv` tidak terpasang di venv, jadi dibuat versi stdlib-only). Template ada di `.env.example`; `.env` asli sudah masuk `.gitignore` supaya tidak ikut ter-commit kalau proyek ini di-push ke Git.

Dua environment terpisah:
- `venv/` — dependency produksi (`requirements.txt`): torch, transformers, gradio, neo4j, groq, sentence-transformers, scikit-learn.
- `venv_eval/` — dependency evaluasi (`requirements_eval.txt` + neo4j/groq/scikit-learn karena `evaluate_ragas.py` memanggil ulang `rag_engine.py`): ragas, langchain-groq, langchain-huggingface, datasets, pandas.
