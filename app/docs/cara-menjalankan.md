# Cara Menjalankan Sistem Ini

Panduan menjalankan aplikasi web (`app.py`) dan evaluasi RAGAS (`evaluate_ragas.py`).
Semua perintah di bawah sudah diuji nyata, bukan teori.

> Terakhir diverifikasi **25 Agustus 2026** pada struktur folder baru: server start bersih,
> gambar `unknown` ditolak gerbang, daun apel terklasifikasi dan dijawab RAG.

---

## 0. Struktur Folder

Seluruh berkas penelitian ada di bawah satu akar: `.\`

```
user\
├── app\                        KODE APLIKASI — jalankan semuanya dari sini
│   ├── app.py                  antarmuka Gradio (titik masuk utama)
│   ├── cnn_model.py            ResNet-50 13 kelas + gerbang unknown
│   ├── rag_engine.py           Neo4j + Groq + hybrid fusion
│   ├── semantic_retriever.py   embedding & cosine similarity
│   ├── evaluate_ragas.py       evaluasi kualitas RAG (jalur terpisah)
│   ├── download_vis.py         utilitas sekali pakai
│   ├── models\                 resnet50_final.pt (13 kelas) + backup 12 kelas
│   ├── docs\                   dokumentasi proyek (termasuk berkas ini)
│   ├── lib\                    aset vis-network lokal (tidak dipakai, kode pakai CDN)
│   ├── GRAPHRAG.json           isi Knowledge Graph
│   ├── golden_dataset.json     60 pertanyaan uji untuk RAGAS
│   ├── embedding_cache.pkl     cache embedding 12 penyakit
│   ├── hasil_ragas_*.csv       keluaran evaluasi
│   ├── requirements.txt        dependency aplikasi
│   ├── requirements_eval.txt   dependency evaluasi
│   └── .env                    kredensial (Neo4j, Groq, OpenRouter)
│
├── dataset\                    data gambar (eks NEW\)
│   ├── apel\ grape\ jagung\ tomat\      data mentah per komoditas
│   ├── unknown_lengkap_new\             498 gambar kelas unknown
│   └── SPLIT_80_10_10\ dan 3 split lain train\ val\ test\
│
├── hasil-training\             keluaran training Kaggle (eks results\)
│   ├── MODEL_SPLIT_80_10_10\   model terpakai — resnet50_final.pt, metrics.json
│   ├── MODEL_SPLIT_70_15_15\ 60_20_20\ 50_25_25\
│   └── LAPORAN_HASIL_TRAINING.html
│
├── notebook\                   RESNET_KAGGLE_4SPLIT.ipynb dan notebook lama
│
├── venv\                       environment aplikasi   (komputer ini)
├── venv_eval\                  environment evaluasi   (komputer ini)
├── _env_cache\                 cache unduhan pip & HuggingFace
└── _venv-lama-laptop\    venv warisan laptop, tidak dipakai di sini
```

**Aturan penting:** semua perintah dijalankan dari dalam `app\`, karena kode memakai
path relatif (`models/resnet50_final.pt`, `GRAPHRAG.json`, `golden_dataset.json`).

---

## 1. Prasyarat

- **Python 3.11 atau 3.12**.
- **Koneksi internet** — sistem memakai Neo4j Aura (cloud), Groq API (cloud), dan
  mengunduh `microsoft/resnet-50` dari HuggingFace saat pertama kali start.
- `app\models\resnet50_final.pt` harus ada (bobot 13 kelas). Tanpa ini `app.py` gagal start.
- `app\.env` harus terisi. Salin dari `.env.example` bila belum ada:
  ```
  NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
  NEO4J_USER=<username>
  NEO4J_PASSWORD=<password>
  GROQ_API_KEY=<groq-key>
  OPENROUTER_API_KEY=<openrouter-key>
  ```

Ada **dua environment terpisah** — jangan dicampur:

| Environment | Untuk apa | Dependency |
|---|---|---|
| `venv\` | aplikasi web (`app.py`) | `requirements.txt` |
| `venv_eval\` | evaluasi RAGAS (`evaluate_ragas.py`) | `requirements_eval.txt` + `neo4j`, `groq`, `scikit-learn` |

Alasan dipisah: paket evaluasi (`ragas`, ekosistem `langchain`) bisa menarik versi
`torch`/`transformers` berbeda dari yang dipakai pipeline CNN produksi.

> **Di laptop** environment-nya ada **di dalam** folder project (`.venv\` dan
> `venv_eval\`), bukan di akar seperti di komputer ini. Sesuaikan path aktivasinya.

---

## 2. Menjalankan Aplikasi Web

```powershell
cd .\app
& ".\venv\Scripts\python.exe" app.py
```

Atau dengan mengaktifkan environment lebih dulu:

```powershell
cd .\app
& ".\venv\Scripts\Activate.ps1"
python app.py
```

Kalau `venv\` belum ada:

```powershell
python -m venv .\venv
& ".\venv\Scripts\python.exe" -m pip install -r .\app\requirements.txt
```

Yang terjadi saat start (±1–2 menit pada run pertama):

1. Muat `microsoft/resnet-50`, timpa bobotnya dari `models/resnet50_final.pt` (13 kelas).
2. Konek Neo4j Aura + buat client Groq.
3. Muat embedding 12 dokumen penyakit dari `embedding_cache.pkl`.
4. Server Gradio aktif di `http://127.0.0.1:7860` plus URL publik sementara
   (karena `demo.launch(share=True)`).

Hentikan dengan `Ctrl+C`.

### Memakai aplikasinya

Upload foto daun (apel, anggur, jagung, tomat), tulis pertanyaan (opsional), klik
**Analisis**. Tiga tab terisi bertahap: Hasil CNN → Graph Neo4j → Jawaban RAG.

Kalau gambar bukan daun komoditas yang didukung, atau model tidak cukup yakin
(confidence < 70%), muncul **"🚫 Gambar di Luar Konteks — Ditolak Model"** dan
analisis dihentikan sebelum masuk RAG.

---

## 3. Menjalankan Evaluasi RAGAS

```powershell
cd .\app
& ".\venv_eval\Scripts\python.exe" evaluate_ragas.py
```

Kalau `venv_eval\` belum ada:

```powershell
python -m venv .\venv_eval
& ".\venv_eval\Scripts\python.exe" -m pip install -r .\app\requirements_eval.txt neo4j groq scikit-learn
```

`requirements_eval.txt` sendiri **tidak cukup** — `evaluate_ragas.py` memanggil ulang
`rag_engine.py` yang butuh `neo4j` dan `groq`, serta `semantic_retriever.py` yang butuh
`scikit-learn`. Karena itu ketiganya ditambahkan manual.

Yang terjadi:

1. Baca `golden_dataset.json` (60 pertanyaan: 12 kelas × 5 relasi).
2. Untuk tiap pertanyaan jalankan `retrieve()` → `build_context()` → `generate()`
   (CNN di-skip, `neo4j_id` diambil dari file). Hasil disimpan ke `records_cache.json`.
3. Penilaian RAGAS: 6 metrik × 60 item = 360 penilaian oleh judge OpenRouter
   (`openai/gpt-4o-mini`), dibatasi ±0,5 request/detik. Skor disimpan bertahap ke
   `ragas_scores_cache.json` — kalau terhenti, jalankan ulang perintah yang sama,
   item yang sudah lengkap otomatis dilewati.
4. Hasil akhir: `hasil_ragas_per_item.csv` dan `hasil_ragas_ringkasan.csv`.

Mengulang penilaian tanpa memanggil Groq lagi:

```powershell
& ".\venv_eval\Scripts\python.exe" evaluate_ragas.py --cache records_cache.json
```

### Hasil evaluasi terakhir (60/60, semua metrik lengkap)

| Metrik | Skor |
|---|---|
| faithfulness | 0,9554 |
| answer_relevancy | 0,8406 |
| llm_context_precision_with_reference | 0,5714 |
| context_recall | 0,9725 |
| factual_correctness | 0,8775 |
| semantic_similarity | 0,8814 |

Generator: Groq `openai/gpt-oss-120b` · Judge: OpenRouter `openai/gpt-4o-mini` ·
Embedding evaluasi: `paraphrase-multilingual-mpnet-base-v2`.

---

## 3b. Menjalankan di laptop

Perapian struktur folder di komputer ini **tidak mempengaruhi** laptop. Alasannya
sudah diverifikasi, bukan dugaan:

| Yang diperiksa | Hasil |
|---|---|
| Path absolut di kode (`C:\`, `D:\`) | **Tidak ada satu pun.** Semua path relatif ke folder `app\` |
| Environment bawaan `app\.venv` | Ikut berpindah, isinya utuh — Python 3.11.7 |
| Paket di `.venv` | gradio, torch, torchvision, transformers, Pillow, neo4j, groq, sentence-transformers, scikit-learn, numpy, httpx — **semua ada** |
| Berkas runtime | `models\`, `GRAPHRAG.json`, `golden_dataset.json`, `embedding_cache.pkl` tetap di akar `app\` |

Artinya folder `app\` bisa diletakkan di mana saja, dengan nama apa saja. Yang wajib
dipenuhi hanya dua hal: **isi `app\` tidak dipecah**, dan **perintah dijalankan dari dalam
`app\`**.

Di laptop environment ada di dalam folder project, jadi perintahnya:

```powershell
cd <lokasi folder app di laptop>
.\.venv\Scripts\Activate.ps1
python app.py
```

Untuk evaluasi:

```powershell
cd <lokasi folder app di laptop>
.\venv_eval\Scripts\Activate.ps1
python evaluate_ragas.py
```

**Satu-satunya hal yang harus dipastikan** adalah isi `.env` terisi key yang masih aktif,
dan `ood_gate.py` sudah tidak ada. Daftar berkas yang perlu disalin ada di
`docs\panduan-update-laptop.md`.

> Catatan: folder `_venv-lama-laptop\` di akar `user\` adalah venv warisan yang
> nyasar saat perapian. Tidak dipakai di komputer ini, tapi sengaja tidak dihapus.

---

## 4. Kalau Gagal

| Gejala | Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError` | salah environment, atau dependency belum dipasang | pastikan memakai python dari `venv\` (aplikasi) atau `venv_eval\` (evaluasi) |
| `FileNotFoundError: models/resnet50_final.pt` | dijalankan bukan dari dalam `app\` | `cd .\app` dulu |
| `Error code: 404 ... model does not exist` | nama model Groq sudah dipensiunkan | ganti `GROQ_MODEL` di `rag_engine.py` ke model aktif; cek daftar di `https://api.groq.com/openai/v1/models` |
| `Error code: 401` | API key salah atau kedaluwarsa | perbarui `GROQ_API_KEY` di `.env` |
| `Error code: 429` dari Groq | kuota harian habis | tunggu reset, atau pakai key lain |
| `Error code: 429` dari OpenRouter saat evaluasi | rate limit judge | jalankan ulang `evaluate_ragas.py --cache records_cache.json`, atau turunkan `requests_per_second` |
| `factual_correctness` = NaN | pernah jadi bug pembacaan kolom | sudah diperbaiki di `evaluate_ragas.py`; pastikan memakai versi terbaru |
| `LLMDidNotFinishException` | keluaran judge terpotong | sudah diatasi lewat `max_tokens=8000` pada judge |
| `RAGEngine()` gagal konek | `.env` kosong atau kredensial salah | bandingkan dengan `.env.example` |
| `WinError 206 ... filename too long` saat `pip install` | path folder terlalu dalam | pasang environment di path pendek, mis. `.\venv` |
| Start pertama lama sekali | mengunduh `microsoft/resnet-50` | normal; run berikutnya memakai cache di `_env_cache\huggingface` |
| Daun asli ikut ditolak "unknown" | confidence < 70% (foto kurang jelas) | foto ulang dengan pencahayaan lebih baik — ini perilaku gerbang yang disengaja |

---

## 5. Ringkasan Perintah Cepat

```powershell
# Aplikasi web
cd .\app
& ".\venv\Scripts\python.exe" app.py

# Evaluasi RAGAS
cd .\app
& ".\venv_eval\Scripts\python.exe" evaluate_ragas.py
```

---

