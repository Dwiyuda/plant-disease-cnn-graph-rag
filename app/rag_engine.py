"""
rag_engine.py — Graph RAG Engine: Neo4j retrieval + Groq LLM generation + Vis.js Graph
"""
# Bagian ini mengimpor seluruh library yang diperlukan oleh sistem 

import os 
import json
import html
from neo4j import GraphDatabase
from groq import Groq
from semantic_retriever import SemanticRetriever

# ─────────────────────────────────────────────
# Muat variabel dari .env (kredensial TIDAK boleh hardcoded di source).
# Loader manual (stdlib saja) supaya tidak perlu dependency tambahan
# (python-dotenv) yang belum terpasang di venv proyek ini.
# ─────────────────────────────────────────────
def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_env_file()

# ─────────────────────────────────────────────
# Konfigurasi koneksi ke database Neo4j dan model LLM Groq yang akan digunakan oleh sistem.
# Wajib diset lewat .env (lihat .env.example) — tidak ada fallback hardcoded.
# ─────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = "openai/gpt-oss-120b"   # llama-3.3-70b-versatile dipensiunkan Groq (404); diganti ke model aktif

# Jumlah dokumen semantic retrieval yang diambil sebagai referensi
# tambahan/pembanding di luar hasil Graph RAG (Neo4j).
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "3"))



# ─────────────────────────────────────────────
# DETEKSI INTENT PERTANYAAN
# Menentukan section konteks mana yang relevan dengan pertanyaan
# pengguna, supaya jawaban LLM (dan ringkasan panel) tidak melebar
# ke topik yang tidak ditanyakan (mis. tanya "penyakit apa ini"
# tidak perlu menampilkan/membahas dosis obat atau pencegahan).
#
# Deteksi dilakukan SEPENUHNYA lewat PROMPTING LLM (lihat method
# RAGEngine.detect_question_intent di bawah) — tidak ada lagi keyword
# matching / rule-based di sistem ini. ALL_SECTIONS di bawah hanyalah
# daftar nama kategori yang valid (taksonomi skema Neo4j), bukan logika
# aturan; dipakai untuk memvalidasi keluaran LLM dan sebagai nilai
# default yang aman jika panggilan LLM gagal total.
# ─────────────────────────────────────────────
ALL_SECTIONS = {"identitas", "patogen", "gejala", "penanganan", "pencegahan", "perawatan"}


# ─────────────────────────────────────────────
# HELPER: Neo4j Connection 
#Kelas ini berfungsi membuka koneksi ke database Neo4j, 
#menjalankan query Cypher, dan menutup koneksi setelah proses selesai.
# ─────────────────────────────────────────────
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, cypher: str, params: dict = None):
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]


# ─────────────────────────────────────────────
# RAG ENGINE
#kelas utama yang mengelola seluruh proses Graph RAG,
#mulai dari mengambil data dari Neo4j hingga menghasilkan jawaban menggunakan LLM Groq.
# ─────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        kurang = [k for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GROQ_API_KEY") if not os.getenv(k)]
        if kurang:
            raise RuntimeError(
                f"Kredensial belum di-set: {', '.join(kurang)}. "
                "Salin app/.env.example menjadi app/.env lalu isi nilainya."
            )
        self.neo4j  = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self.groq   = Groq(api_key=GROQ_API_KEY)
        print("  🔗 Neo4j terhubung")
        print("  🤖 Groq client siap")

        # Semantic Retrieval (bagian kedua Hybrid RAG). Dibungkus try/except
        # supaya kalau GRAPHRAG.json/model embedding belum siap, aplikasi
        # TETAP jalan dengan Graph RAG saja (degradasi anggun / graceful
        # degradation) — bukan crash total.
        try:
            self.semantic = SemanticRetriever()
            print("  🧠 Semantic Retriever siap (Hybrid RAG aktif)")
        except Exception as e:
            self.semantic = None
            print(f"  ⚠️  Semantic Retriever gagal dimuat, lanjut dengan Graph RAG saja: {e}")

    # ── 0. DETEKSI INTENT PERTANYAAN (LLM-based) ──
    #Memahami maksud pertanyaan petani menggunakan LLM (Groq), bukan lagi
    #keyword matching manual. LLM diminta mengklasifikasikan pertanyaan
    #ke satu/lebih kategori (identitas, patogen, gejala, penanganan,
    #pencegahan, perawatan) dan menjawab HANYA dalam format JSON supaya
    #mudah di-parse. Hasilnya dipakai oleh build_context, build_graph_html,
    #dan generate() untuk menyaring informasi yang relevan saja.
    def detect_question_intent(self, question: str) -> set:  #mengubah LLM menjadi pengklasifikasi intent. Outputnya bukan jawaban, melainkan kategori seperti patogen, gejala, atau penanganan.
        """
        Menggunakan LLM untuk memahami maksud pertanyaan petani, lalu
        memetakannya ke section konteks yang relevan. Mengembalikan
        subset dari ALL_SECTIONS. Retry otomatis 1x kalau panggilan/parsing
        gagal; kalau tetap gagal, default ke ALL_SECTIONS (bukan keyword
        matching) supaya jawaban tetap informatif.
        """
        if not question or not question.strip():
            return set(ALL_SECTIONS)

        system_prompt = (
            "Kamu adalah pengklasifikasi maksud pertanyaan untuk sistem deteksi "
            "penyakit tanaman. Baca pertanyaan petani lalu tentukan kategori "
            "informasi apa saja yang relevan untuk menjawabnya.\n\n"
            "Kategori yang tersedia:\n"
            "- identitas  : nama/jenis penyakit, ringkasan umum, diagnosis\n"
            "- patogen    : penyebab, sumber penyakit, kenapa/mengapa bisa muncul\n"
            "- gejala     : ciri-ciri, tanda-tanda, tampilan fisik penyakit\n"
            "- penanganan : obat, fungisida/pestisida, dosis, cara mengobati\n"
            "- pencegahan : cara mencegah agar tidak terkena/menyebar\n"
            "- perawatan  : perawatan lanjutan setelah penanganan\n\n"
            "Aturan:\n"
            "1. Jika pertanyaan kosong, umum/ambigu (mis. \"apa penyakit ini dan "
            "bagaimana cara mengatasinya\"), atau eksplisit minta info "
            "lengkap/semua/detail, kembalikan SEMUA kategori.\n"
            "2. Jika pertanyaan spesifik ke satu/dua topik saja, kembalikan HANYA "
            "kategori yang benar-benar relevan — jangan berlebihan.\n"
            "3. Balas HANYA dengan JSON valid, tanpa penjelasan/markdown apa pun, "
            "persis format ini:\n"
            '{"kategori": ["kategori1", "kategori2"]}'
        )

        last_error = None
        for attempt in range(2):  # coba lagi 1x kalau percobaan pertama gagal/response invalid
            try:
                response = self.groq.chat.completions.create(   #terlihat bahwa sistem mengirim pertanyaan pengguna ke LLM.
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": f"Pertanyaan petani: {question}"},
                    ],
                    temperature=0,
                    max_tokens=100,
                )
                raw = response.choices[0].message.content.strip()

                # Bersihkan kalau LLM tetap membungkus jawaban dengan code fence
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    if raw.lower().startswith("json"):
                        raw = raw[4:].strip()

                data     = json.loads(raw)
                kategori = set(data.get("kategori", [])) & ALL_SECTIONS #LLM mengembalikan JSON kategori.

                if not kategori:
                    return set(ALL_SECTIONS)

                kategori.add("identitas")
                return kategori

            except Exception as e:
                last_error = e
                print(f"⚠️  Percobaan {attempt + 1} deteksi intent via LLM gagal: {e}")

        # LLM gagal 2x berturut-turut (network/API bermasalah) — fallback ke
        # nilai default yang aman (tampilkan semua section) tanpa keyword
        # matching apa pun, supaya jawaban tetap informatif walau tidak
        # tersaring secara spesifik.
        print(f"⚠️  Deteksi intent via LLM gagal total ({last_error}), pakai default aman (semua section).")
        return set(ALL_SECTIONS)

    # ── 1. RETRIEVE ──────────────────────────
    #Mengambil informasi penyakit dari Knowledge Graph Neo4j berdasarkan neo4j_id yang diperoleh dari hasil klasifikasi CNN. 
    #Data yang diambil meliputi penyakit,patogen, gejala, penanganan, pencegahan, dan perawatan.
    def retrieve(self, neo4j_id: str) -> dict | None:
        cypher = """
            MATCH (p:Penyakit {id: $id})
            OPTIONAL MATCH (p)-[:DISEBABKAN_OLEH]->(pat:Patogen)
            OPTIONAL MATCH (p)-[:MEMILIKI_GEJALA]->(g:Gejala)
            OPTIONAL MATCH (p)-[:DITANGANI_DENGAN]->(h:Penanganan)
            OPTIONAL MATCH (p)-[:DICEGAH_DENGAN]->(prev:Pencegahan)
            OPTIONAL MATCH (p)-[:DIRAWAT_DENGAN]->(raw:Perawatan)
            RETURN p, pat,
                   collect(DISTINCT g.deskripsi) AS gejala_list,
                   collect(DISTINCT {
                       tipe: h.tipe, produk: h.produk,
                       bahan_aktif: h.bahan_aktif, dosis: h.dosis,
                       cara: h.cara, waktu: h.waktu, catatan: h.catatan
                   }) AS penanganan_list,
                   collect(DISTINCT prev.cara) AS pencegahan_list,
                   collect(DISTINCT raw.cara)  AS perawatan_list
        """
        try:
            rows = self.neo4j.query(cypher, {"id": neo4j_id})
        except Exception as e:
            print(f"❌ Query Neo4j gagal: {e}")
            return None

        if not rows:
            print(f"⚠️  ID '{neo4j_id}' tidak ditemukan di Neo4j.")
            return None

        row      = rows[0]
        p_node   = row.get("p", {})
        pat_node = row.get("pat", {}) or {}

        return {
            "id"                : p_node.get("id", ""),
            "nama_tampil"       : p_node.get("nama_tampil", ""),
            "nama_ilmiah"       : p_node.get("nama_ilmiah", ""),
            "tanaman"           : p_node.get("tanaman", ""),
            "tingkat_keparahan" : p_node.get("tingkat_keparahan", ""),
            "chunk_rag"         : p_node.get("chunk_rag", ""),
            "patogen": {
                "nama"           : pat_node.get("nama", ""),
                "tipe"           : pat_node.get("tipe", ""),
                "penyebab_utama" : pat_node.get("penyebab_utama", ""),
                "kondisi_pemicu" : pat_node.get("kondisi_pemicu", []),
            },
            "gejala"      : [g for g in row.get("gejala_list", [])      if g],
            "penanganan"  : [h for h in row.get("penanganan_list", [])   if h and h.get("tipe")],
            "pencegahan"  : [c for c in row.get("pencegahan_list", [])   if c],
            "perawatan"   : [r for r in row.get("perawatan_list", [])    if r],
        }

    # ── 1b. RETRIEVE SEMANTIC (Hybrid RAG — bagian kedua) ──
    #Mengambil dokumen-dokumen lain dari seluruh knowledge base yang secara
    #makna mirip dengan pertanyaan petani (bukan cuma penyakit hasil deteksi
    #CNN). Berguna sebagai pembanding/pelengkap, misalnya saat confidence CNN
    #rendah, atau pertanyaan menyinggung penyakit/gejala lain yang mirip.
    def retrieve_semantic(self, question: str, predicted_class: str = None, top_k: int = SEMANTIC_TOP_K, exclude_id: str = None) -> list:
        """
        Wrapper Jalur B (Semantic Retrieval) — sesuai Gambar 3.6 Pipeline
        Hybrid RAG:
        - Query yang di-embed adalah gabungan `predicted_class + question`
          (bukan pertanyaan mentah saja), supaya pencarian tetap
          "berjangkar" pada konteks hasil deteksi CNN sambil tetap peka
          pada maksud spesifik pertanyaan petani.
        - Similarity dihitung terhadap field 'chunk_rag' tiap penyakit
          (lihat SemanticRetriever.build_document).

        Tambahan:
        - Aman dipanggil walau semantic retriever gagal dimuat (return []).
        - `exclude_id` membuang penyakit yang SAMA dengan hasil Graph RAG
          (neo4j_id) dari daftar hasil, supaya tidak duplikat — bagian ini
          memang dimaksudkan untuk penyakit LAIN sebagai pembanding.
        """
        if self.semantic is None or not question or not question.strip():
            return []

        query_text = f"{predicted_class} {question}".strip() if predicted_class else question
        #query embedding bukan hanya pertanyaan.
        try:
            # Ambil sedikit lebih banyak dari top_k sebagai buffer, supaya
            # setelah exclude_id dibuang, hasil akhir tetap top_k item.
            hits = self.semantic.retrieve(query_text, top_k=top_k + 1)
        except Exception as e:
            print(f"⚠️  Semantic retrieval gagal: {e}")
            return []

        if exclude_id:
            hits = [h for h in hits if h.get("id") != exclude_id]

        return hits[:top_k]

    # ── 1c. HYBRID FUSION (Gambar 3.6) ──
    #Menggabungkan hasil Jalur A (Graph Retrieval) dan Jalur B (Semantic
    #Retrieval) menjadi satu struktur data terpadu, sebelum diteruskan ke
    #dua fungsi output paralel: build_graph_html() dan build_context().
    def hybrid_fusion(self, retrieved: dict, semantic_hits: list) -> dict:
        """
        Langkah 'Hybrid Fusion' pada Gambar 3.6. `build_graph_html()`
        cukup mengabaikan key 'semantic_chunks' (dia hanya pakai data
        graph), sementara `build_context()`/`generate()` memakainya
        sebagai referensi tambahan/pembanding.
        """
        fused = dict(retrieved) if retrieved else {}
        fused["semantic_chunks"] = semantic_hits or []
        return fused

    # ── 2. BUILD GRAPH HTML (vis.js inline) ──
    #Membuat visualisasi Knowledge Graph dalam bentuk graf interaktif menggunakan Vis.js,
    #kemudian menampilkannya pada tab Graph di Gradio.

    def build_graph_html(self, retrieved: dict, question: str = None, intents: set = None) -> str:
        """
        Bangun visualisasi graph. Hanya node/edge kategori yang relevan
        dengan `question` (intent-aware, sama seperti build_context) yang
        ditampilkan, supaya graph tidak penuh dengan cabang yang tidak
        ditanyakan. Node Penyakit (pusat) dan Patogen selalu ditampilkan
        sebagai anchor dasar identitas; node Gejala/Penanganan/Pencegahan/
        Perawatan hanya muncul kalau memang ditanyakan (atau pertanyaan
        bersifat umum/komprehensif).

        `intents` bisa dikirim langsung (hasil dari detect_question_intent
        yang sudah dihitung sebelumnya di app.py) supaya tidak perlu
        memanggil LLM lagi di sini. Kalau tidak dikirim, dihitung sendiri.
        """
        nama_tampil = retrieved.get("nama_tampil", retrieved.get("id", "Penyakit"))
        pat         = retrieved.get("patogen", {})
        intents     = intents if intents is not None else self.detect_question_intent(question)

        nodes = []
        edges = []
        nid   = 1

        # Node Penyakit (pusat) — selalu tampil,node pusat pada Knowledge Graph
        pid = nid
        nodes.append({
            "id": pid, "label": nama_tampil,
            "title": f"<b>{nama_tampil}</b><br>Tanaman: {retrieved.get('tanaman','-')}<br>Keparahan: {retrieved.get('tingkat_keparahan','-')}<br>Ilmiah: {retrieved.get('nama_ilmiah','-')}",
            "color": {"background": "#e94560", "border": "#ff6b8a"},
            "size": 40, "shape": "dot",
            "font": {"size": 16, "color": "#ffffff", "bold": True},
        })
        nid += 1

        legend_items = ["<span>🔴 Penyakit</span>"]

        # Node Patogen — anchor dasar identitas, selalu ditampilkan
        # (satu edge saja, tidak membuat graph melebar)
        if pat.get("nama"):
            nodes.append({
                "id": nid, "label": pat["nama"],
                "title": f"<b>Patogen</b><br>Nama: {pat.get('nama','-')}<br>Tipe: {pat.get('tipe','-')}<br>Penyebab: {pat.get('penyebab_utama','-')}",
                "color": {"background": "#f5a623", "border": "#ffc94d"},
                "size": 28, "shape": "diamond",
                "font": {"size": 13, "color": "#ffffff"},
            })
            edges.append({"from": pid, "to": nid, "label": "DISEBABKAN_OLEH", "color": {"color": "#f5a623"}, "width": 2})
            nid += 1
            legend_items.append("<span>🟠 Patogen</span>")

        # Node Gejala — hanya jika ditanyakan
        if "gejala" in intents:
            for i, g in enumerate(retrieved.get("gejala", [])[:6], 1):
                short = g[:30] + "…" if len(g) > 30 else g
                nodes.append({
                    "id": nid, "label": short,
                    "title": f"<b>Gejala {i}</b><br>{g}",
                    "color": {"background": "#00b894", "border": "#55efc4"},
                    "size": 20, "shape": "ellipse",
                    "font": {"size": 11, "color": "#ffffff"},
                })
                edges.append({"from": pid, "to": nid, "label": "MEMILIKI_GEJALA", "color": {"color": "#00b894"}, "width": 1})
                nid += 1
            if retrieved.get("gejala"):
                legend_items.append("<span>🟢 Gejala</span>")

        # Node Penanganan — hanya jika ditanyakan
        if "penanganan" in intents:
            for i, h in enumerate(retrieved.get("penanganan", [])[:4], 1):
                produk = h.get("produk") or h.get("tipe") or f"Penanganan {i}"
                short  = produk[:28] + "…" if len(produk) > 28 else produk
                nodes.append({
                    "id": nid, "label": short,
                    "title": f"<b>Penanganan {i}</b><br>Tipe: {h.get('tipe','-')}<br>Produk: {h.get('produk','-')}<br>Bahan Aktif: {h.get('bahan_aktif','-')}<br>Dosis: {h.get('dosis','-')}",
                    "color": {"background": "#6c5ce7", "border": "#a29bfe"},
                    "size": 20, "shape": "box",
                    "font": {"size": 11, "color": "#ffffff"},
                })
                edges.append({"from": pid, "to": nid, "label": "DITANGANI_DENGAN", "color": {"color": "#6c5ce7"}, "width": 1})
                nid += 1
            if retrieved.get("penanganan"):
                legend_items.append("<span>🟣 Penanganan</span>")

        # Node Pencegahan — hanya jika ditanyakan
        if "pencegahan" in intents:
            for i, c in enumerate(retrieved.get("pencegahan", [])[:4], 1):
                short = c[:28] + "…" if len(c) > 28 else c
                nodes.append({
                    "id": nid, "label": short,
                    "title": f"<b>Pencegahan {i}</b><br>{c}",
                    "color": {"background": "#0984e3", "border": "#74b9ff"},
                    "size": 20, "shape": "triangle",
                    "font": {"size": 11, "color": "#ffffff"},
                })
                edges.append({"from": pid, "to": nid, "label": "DICEGAH_DENGAN", "color": {"color": "#0984e3"}, "width": 1})
                nid += 1
            if retrieved.get("pencegahan"):
                legend_items.append("<span>🔵 Pencegahan</span>")

        # Node Perawatan — hanya jika ditanyakan
        if "perawatan" in intents:
            for i, r in enumerate(retrieved.get("perawatan", [])[:3], 1):
                short = r[:28] + "…" if len(r) > 28 else r
                nodes.append({
                    "id": nid, "label": short,
                    "title": f"<b>Perawatan {i}</b><br>{r}",
                    "color": {"background": "#e84393", "border": "#fd79a8"},
                    "size": 20, "shape": "triangleDown",
                    "font": {"size": 11, "color": "#ffffff"},
                })
                edges.append({"from": pid, "to": nid, "label": "DIRAWAT_DENGAN", "color": {"color": "#e84393"}, "width": 1})
                nid += 1
            if retrieved.get("perawatan"):
                legend_items.append("<span>🩷 Perawatan</span>")

        nodes_json = json.dumps(nodes, ensure_ascii=False)
        edges_json = json.dumps(edges, ensure_ascii=False)
        legend_html = "\n  ".join(legend_items)

        # Selalu gunakan CDN karena aplikasi ini tetap butuh internet untuk Neo4j dan Groq
        vis_script = '<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>'

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
{vis_script}
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; background:#0f0f1a; overflow:hidden; }}
  #gc {{ width:100%; height:460px; background:#0f0f1a; }}
  #leg {{ background:#1a1a2e; color:#ccc; padding:8px 16px; font-size:12px;
          display:flex; flex-wrap:wrap; gap:14px; border-top:1px solid #2d2d4e; }}
  .tip {{ color:#555; font-size:11px; text-align:center; padding:3px 0; background:#0f0f1a; }}
</style>
</head>
<body>
<div id="gc"></div>
<div id="leg">
  {legend_html}
</div>
<div class="tip">💡 Scroll = zoom &nbsp;·&nbsp; Drag = pindahkan &nbsp;·&nbsp; Hover = detail</div>
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
new vis.Network(
  document.getElementById("gc"),
  {{ nodes: nodes, edges: edges }},
  {{
    physics: {{
      solver: "forceAtlas2Based",
      forceAtlas2Based: {{
        gravitationalConstant: -60,
        centralGravity: 0.01,
        springLength: 160,
        springConstant: 0.06,
        damping: 0.4
      }},
      stabilization: {{ iterations: 150 }}
    }},
    edges: {{
      arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
      font: {{ size: 10, color: "#aaaaaa", align: "middle" }},
      smooth: {{ type: "dynamic" }}
    }},
    nodes: {{ borderWidth: 2, shadow: true }},
    interaction: {{
      hover: true,
      tooltipDelay: 100,
      navigationButtons: true,
      keyboard: true
    }}
  }}
);
</script>
</body>
</html>"""
        
        # Bungkus menggunakan iframe agar script tereksekusi dengan benar di Gradio
        # dan tidak mengganggu performa DOM utama (tidak berat)
        safe_html = html.escape(html_content)
        iframe_html = f'<iframe style="width:100%; height:480px; border:none; border-radius:8px; overflow:hidden; background:#0f0f1a;" srcdoc="{safe_html}"></iframe>'
        return iframe_html

    # ── 3. BUILD CONTEXT ─────────────────────
    #Menyusun konteks yang akan dikirim ke LLM. Konteks berisi hasil Graph Retrieval, hasil prediksi CNN,
    #serta informasi yang sesuai dengan pertanyaan pengguna.

    def build_context(self, retrieved: dict, prediction: dict, question: str = None, intents: set = None, semantic_hits: list = None) -> str:
        """
        Bangun konteks untuk LLM. Hanya section yang relevan dengan
        `question` (intent-aware, dideteksi lewat LLM prompting) yang
        disertakan, supaya jawaban LLM tidak melebar ke topik yang tidak
        ditanyakan.

        `intents` bisa dikirim langsung (hasil dari detect_question_intent
        yang sudah dihitung sebelumnya di app.py) supaya tidak perlu
        memanggil LLM lagi di sini. Kalau tidak dikirim, dihitung sendiri.

        `semantic_hits` (Hybrid RAG — bagian kedua): hasil dari
        retrieve_semantic(), berisi penyakit LAIN yang mirip secara makna
        dengan pertanyaan. Disisipkan sebagai section terpisah, jelas
        ditandai hanya sebagai pembanding — bukan sumber utama.
        """
        d   = retrieved
        pat = d.get("patogen", {})
        intents = intents if intents is not None else self.detect_question_intent(question)
        # Kalau `retrieved` adalah hasil hybrid_fusion() (punya key
        # 'semantic_chunks') dan semantic_hits tidak dikirim eksplisit,
        # ambil otomatis dari situ.
        if semantic_hits is None:
            semantic_hits = d.get("semantic_chunks", [])

        lines = ["=== INFORMASI PENYAKIT ==="]
        lines.append(f"Nama             : {d.get('nama_tampil', '')}")
        lines.append(f"Nama Ilmiah      : {d.get('nama_ilmiah', '')}")
        lines.append(f"Tanaman          : {d.get('tanaman', '')}")
        lines.append(f"Tingkat Keparahan: {d.get('tingkat_keparahan', '')}")

        if "patogen" in intents:
            lines.append("\n=== PATOGEN PENYEBAB ===")
            lines.append(f"Nama Patogen     : {pat.get('nama', '')}")
            lines.append(f"Tipe             : {pat.get('tipe', '')}")
            lines.append(f"Penyebab Utama   : {pat.get('penyebab_utama', '')}")
            kondisi = pat.get("kondisi_pemicu", [])
            lines.append(f"Kondisi Pemicu   : {', '.join(kondisi) if kondisi else '-'}")

        if "gejala" in intents:
            lines.append("\n=== GEJALA ===")
            for i, g in enumerate(d.get("gejala", []), 1):
                lines.append(f"{i}. {g}")

        if "penanganan" in intents:
            lines.append("\n=== PENANGANAN ===")
            for i, h in enumerate(d.get("penanganan", []), 1):
                lines.append(f"{i}. Tipe: {h.get('tipe','')} | Produk: {h.get('produk','')} | Bahan Aktif: {h.get('bahan_aktif','')} | Dosis: {h.get('dosis','')}")
                if h.get("cara"):
                    lines.append(f"   Cara: {h.get('cara','')}")

        if "pencegahan" in intents:
            lines.append("\n=== PENCEGAHAN ===")
            for i, c in enumerate(d.get("pencegahan", []), 1):
                lines.append(f"{i}. {c}")

        if "perawatan" in intents:
            lines.append("\n=== PERAWATAN LANJUTAN ===")
            for i, r in enumerate(d.get("perawatan", []), 1):
                lines.append(f"{i}. {r}")

        res_c = prediction.get("confidence_resnet", 0)
        lines.append("\n=== HASIL DETEKSI CNN ===")
        lines.append(f"Kelas           : {prediction.get('predicted_class', '')}")
        lines.append(f"Conf ResNet-50   : {res_c:.2%}")

        # Ringkasan naratif (chunk_rag) sering memuat campuran topik
        # (gejala, pencegahan, dst). Hanya disertakan untuk pertanyaan
        # komprehensif/umum agar tidak membocorkan topik yang tidak
        # ditanyakan saat pertanyaan spesifik (mis. hanya "penanganan").
        if d.get("chunk_rag") and intents == ALL_SECTIONS:
            lines.append("\n=== RINGKASAN PENYAKIT ===")
            lines.append(d["chunk_rag"])

        # ── Hybrid RAG: referensi tambahan dari Semantic Retrieval ──
        # Hanya disisipkan kalau memang ada hasilnya (semantic retriever
        # siap & menemukan penyakit lain yang cukup mirip). Ditulis
        # sebagai section terpisah dengan catatan eksplisit "pembanding",
        # supaya LLM tidak keliru menganggapnya sebagai diagnosis utama.
        if semantic_hits:
            lines.append("\n=== REFERENSI TAMBAHAN (Semantic Retrieval) ===")
            lines.append(
                "Catatan: penyakit di bawah ini HANYA pembanding/pelengkap "
                "dari pencarian kemiripan makna terhadap seluruh knowledge "
                "base. Sumber utama & paling akurat tetap bagian "
                "'INFORMASI PENYAKIT' di atas (hasil deteksi CNN + Graph RAG)."
            )
            for i, hit in enumerate(semantic_hits, 1):
                ringkas = " ".join(hit.get("text", "").split())
                if len(ringkas) > 300:
                    ringkas = ringkas[:300] + "..."
                lines.append(f"\n{i}. {hit.get('label','')} (skor kemiripan: {hit.get('score', 0):.2f})")
                lines.append(f"   {ringkas}")

        return "\n".join(lines)

    # ── 4. GENERATE ──────────────────────────
    #Melakukan prompt engineering dengan membuat System Prompt dan User Prompt,
    #kemudian mengirimkannya ke model Groq (openai/gpt-oss-120b) untuk menghasilkan jawaban
    #berdasarkan konteks yang telah disusun.

    def generate(self, context: str, question: str, retrieved: dict = None, low_confidence: bool = False, intents: set = None, semantic_hits: list = None) -> str:
        confidence_note = (
            "\n\n⚠️ PENTING: Confidence model CNN rendah (<50%). "
            "Sampaikan kepada pengguna bahwa hasil ini perlu dikonfirmasi lebih lanjut."
            if low_confidence else ""
        )
        hybrid_note = (
            "\n5. Jika konteks memuat bagian 'REFERENSI TAMBAHAN (Semantic "
            "Retrieval)', perlakukan sebagai INFORMASI PEMBANDING saja — "
            "JANGAN jadikan itu sebagai diagnosis utama, dan jangan campur "
            "adukkan dengan penyakit yang sudah terdeteksi CNN. Sebut "
            "penyakit pembanding itu hanya jika pertanyaan petani secara "
            "eksplisit minta perbandingan/alternatif kemungkinan lain."
            if semantic_hits else ""
        )
        system_prompt = (
            "Kamu adalah asisten ahli pertanian yang membantu petani mendeteksi dan menangani "
            "penyakit tanaman. Jawab dalam bahasa Indonesia yang jelas dan mudah dipahami petani.\n\n"
            "ATURAN WAJIB:\n"
            "1. Jawab HANYA berdasarkan konteks yang diberikan. Jika informasi tidak tersedia "
            "dalam konteks, sampaikan dengan jujur — jangan mengarang.\n"
            "2. Jawab HANYA sesuai dengan apa yang ditanyakan petani. Jika petani hanya bertanya "
            "nama/jenis penyakit, jawab nama penyakit dan penjelasan singkatnya saja.\n"
            "3. JANGAN otomatis membahas gejala, penyebab, penanganan, dosis obat, pencegahan, "
            "atau perawatan kecuali pertanyaan secara eksplisit meminta hal tersebut, meskipun "
            "informasi itu tersedia di konteks.\n"
            "4. Jangan menambahkan bagian/topik lain di luar yang ditanyakan. Jawaban harus "
            "fokus, ringkas, dan tidak melebar." + hybrid_note + confidence_note
        )
        user_prompt = (
            f"Pertanyaan petani: {question}\n\n"
            f"Konteks informasi penyakit (gunakan hanya bagian yang relevan dengan pertanyaan "
            f"di atas, JANGAN membahas bagian lain yang tidak ditanyakan):\n{context}"
        )

        # Intent pertanyaan (dideteksi via LLM prompting) dipakai juga untuk
        # menyaring ringkasan panel "Data Penyakit dari Knowledge Graph" di
        # bawah ini, supaya panel itu tidak selalu menampilkan seluruh data
        # (mis. gejala) walau tidak ditanyakan. Kalau sudah dihitung
        # sebelumnya (dikirim dari app.py), pakai itu — tidak perlu LLM call lagi.
        intents = intents if intents is not None else self.detect_question_intent(question)
        # Sama seperti build_context(): kalau `retrieved` hasil hybrid_fusion()
        # dan semantic_hits tidak dikirim eksplisit, ambil otomatis dari situ.
        if semantic_hits is None and retrieved:
            semantic_hits = retrieved.get("semantic_chunks", [])

        neo4j_summary = ""
        if retrieved:
            nama_tampil = retrieved.get("nama_tampil", "-")
            tanaman     = retrieved.get("tanaman", "-")
            keparahan   = retrieved.get("tingkat_keparahan", "-")
            nama_ilmiah = retrieved.get("nama_ilmiah", "-")
            gejala_list = retrieved.get("gejala", [])
            pat         = retrieved.get("patogen", {})

            neo4j_summary = (
                f"### 🗄️ Data Penyakit dari Knowledge Graph\n\n"              
                f"**{nama_tampil}** *(Nama Ilmiah: {nama_ilmiah})*\n\n"
                f"- 🌱 **Tanaman**   : {tanaman}\n"
                f"- ⚠️ **Keparahan** : {keparahan}\n"
            )
            if "patogen" in intents:
                neo4j_summary += f"- 🦠 **Patogen**   : {pat.get('nama', '-')} ({pat.get('tipe', '-')})\n"
            if "gejala" in intents and gejala_list:
                neo4j_summary += (
                    "\n**Gejala Utama:**\n"
                    + "\n".join(f"- {g}" for g in gejala_list[:5])
                    + "\n"
                )
            neo4j_summary += "\n---\n\n"

        # Ringkasan singkat hasil Semantic Retrieval (Hybrid RAG), sekadar
        # transparansi ke user — daftar penyakit lain yang mirip secara
        # makna, TANPA teks lengkapnya (itu sudah masuk ke context untuk LLM).
        if semantic_hits:
            neo4j_summary += "### 🔍 Penyakit Lain yang Mirip (Semantic Search)\n\n"
            for hit in semantic_hits:
                neo4j_summary += f"- {hit.get('label','-')} — kemiripan {hit.get('score', 0):.0%}\n"
            neo4j_summary += "\n---\n\n"

        try:
            response = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            llm_answer = response.choices[0].message.content
        except Exception as e:
            llm_answer = f"❌ Gagal generate jawaban dari Groq: {e}"

        return neo4j_summary + "### 💬 Jawaban AI\n\n" + llm_answer
    
    #Menutup koneksi ke database Neo4j setelah seluruh proses selesai.
    def close(self):
        self.neo4j.close()