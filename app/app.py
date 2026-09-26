"""
app.py — Gradio Interface untuk Sistem Deteksi Penyakit Tanaman
CNN (ResNet-50) + Graph RAG (Neo4j + Groq)
"""
#Mengimpor library Gradio sebagai antarmuka aplikasi, model CNN untuk klasifikasi penyakit,
#dan RAG Engine untuk proses Graph RAG serta LLM.

import gradio as gr
from cnn_model import load_model, predict_disease, UNKNOWN_THRESHOLD
from rag_engine import RAGEngine

# ─────────────────────────────────────────────
# 1. INISIALISASI (dijalankan sekali saat startup)
#Memuat model ResNet-50 yang telah dilatih dan menginisialisasi koneksi ke Neo4j serta Groq.
# ─────────────────────────────────────────────
print("⏳ Memuat model CNN...")
resnet_model, resnet_processor = load_model()
print("✅ Model CNN siap!")

print("⏳ Menghubungkan ke Neo4j + Groq...")
rag = RAGEngine()
print("✅ RAG Engine siap!")


# ─────────────────────────────────────────────
# 2. FUNGSI UTAMA (dipanggil setiap ada input)
# menjalankan seluruh alur sistem, mulai dari klasifikasi gambar hingga menghasilkan jawaban dari LLM.
# ─────────────────────────────────────────────
def analyze_image(image, user_question, progress=gr.Progress()):
    """
    Pipeline utama:
    1. Prediksi penyakit dengan CNN (ResNet-50)
    2. Retrieve konteks dari Neo4j
    3. Build graph HTML (pyvis)
    4. Generate jawaban dengan Groq LLM (+ ringkasan Neo4j)

    Fungsi ini berupa GENERATOR (pakai `yield`, bukan `return`) supaya
    setiap tab di UI update segera setelah hasilnya siap — tidak perlu
    menunggu seluruh pipeline (termasuk panggilan Groq yang paling lambat)
    selesai dulu baru semua tab terisi bersamaan.
    """
    EMPTY_GRAPH = "<div style='color:#aaa; text-align:center; padding:60px; background:#1a1a2e;'>📊 Graph akan muncul setelah analisis</div>"

#Memastikan pengguna telah mengunggah gambar sebelum proses analisis dilakukan.
    if image is None:
        yield (
            "⚠️ Silakan upload gambar daun terlebih dahulu.",
            EMPTY_GRAPH,
            ""
        )
        return

    # Default pertanyaan jika kosong
    if not user_question.strip():
        user_question = "Apa penyakit ini dan bagaimana cara mengatasinya?"

    # ── Step 1: Prediksi CNN ──
    #Melakukan klasifikasi citra daun menggunakan model ResNet-50 dan menghasilkan predicted_class, neo4j_id, serta confidence.
    progress(0.1, desc="🤖 Memproses gambar dengan CNN (ResNet-50)...")
    prediction = predict_disease(   
        image,                      
        resnet_model, resnet_processor
    )

    predicted_class = prediction["predicted_class"]
    neo4j_id        = prediction["neo4j_id"]
    conf_res        = prediction["confidence_resnet"]
    low_confidence  = prediction["low_confidence"]

    # Format hasil prediksi CNN
    cnn_result = (
        f"🌿 **Kelas Terdeteksi:** `{predicted_class}`\n\n"
        f"🔗 **Neo4j ID:** `{neo4j_id}`\n\n"
        f"📊 **Confidence:**\n"
        f"- ResNet-50 : `{conf_res:.2%}`\n\n"
    )
    if low_confidence:
        cnn_result += "⚠️ *Confidence rendah — hasil mungkin kurang akurat. Coba foto ulang dengan pencahayaan lebih baik.*"

    # ── Step 1b: Open-Set Rejection (kelas `unknown` eksplisit + threshold confidence) ──
    # Penolakan input di luar objek penelitian dilakukan OLEH MODEL ITU SENDIRI: ResNet-50
    # dilatih dengan kelas `unknown` eksplisit (13 kelas), diperkuat threshold confidence.
    # Jika gambar bukan daun komoditas yang didukung / model tidak cukup yakin, hasil
    # ditandai `unknown` dan analisis penyakit (Graph RAG + LLM) TIDAK dilanjutkan.
    # (Menggantikan gate LVLM lama — tidak ada model/layanan eksternal untuk validasi.)
    if prediction["is_unknown"]:
        if prediction.get("unknown_reason") == "confidence_rendah":
            alasan = (
                f"Confidence tertinggi model ({conf_res:.2%}) di bawah ambang batas "
                f"({UNKNOWN_THRESHOLD:.0%}) — model tidak cukup yakin gambar ini termasuk "
                "salah satu penyakit yang didukung."
            )
        else:
            alasan = (
                "Model mengklasifikasikan gambar sebagai kelas **unknown** — bukan daun "
                "komoditas yang didukung (apel, anggur, jagung, tomat)."
            )

        rejected_cnn = (
            "## 🚫 Gambar di Luar Konteks — Ditolak Model\n\n"
            "Model ResNet-50 menilai foto ini **bukan** daun komoditas yang didukung "
            "(apel, anggur, jagung, tomat) atau tidak cukup yakin. Karena itu **analisis "
            "penyakit tidak dilanjutkan**.\n\n"
            "---\n\n"
            "**Output mentah model:**\n\n"
            f"🌿 Kelas Terdeteksi: `{predicted_class}`\n\n"
            f"📊 Confidence ResNet-50: `{conf_res:.2%}` (ambang unknown: `{UNKNOWN_THRESHOLD:.0%}`)"
        )
        rejected_graph = (
            "<div style='color:#e94560; text-align:center; padding:60px; background:#1a1a2e; "
            "border-radius:8px;'>🚫 Analisis dihentikan — gambar di luar konteks sistem.</div>"
        )
        rejected_answer = (
            "## 🚫 Gambar di Luar Konteks Sistem\n\n"
            f"**Alasan:** {alasan}\n\n"
            "Sistem ini hanya menganalisis **foto daun** dari **apel, anggur, jagung, atau tomat**. "
            "Analisis penyakit (Knowledge Graph + LLM) tidak dilanjutkan untuk gambar ini.\n\n"
            "💡 *Silakan unggah foto daun yang jelas dari salah satu komoditas tersebut, dengan "
            "pencahayaan baik dan daun memenuhi sebagian besar frame.*"
        )
        progress(1.0, desc="🚫 Gambar di luar konteks — analisis dihentikan.")
        yield rejected_cnn, rejected_graph, rejected_answer
        return

    # Tab "Hasil CNN" langsung tampil di sini, tanpa menunggu Neo4j/Groq
    yield cnn_result, EMPTY_GRAPH, "⏳ Mengambil data dari Knowledge Graph..."

    # ── Step 2: Retrieve dari Neo4j ──
    progress(0.4, desc="🗄️ Mengambil data penyakit dari Knowledge Graph (Neo4j)...")
    retrieved = rag.retrieve(neo4j_id)  #Mengambil informasi penyakit dari Knowledge Graph Neo4j berdasarkan neo4j_id hasil klasifikasi.
    if retrieved is None:
        error_graph = f"<div style='color:#e94560; text-align:center; padding:60px; background:#1a1a2e;'>❌ Data untuk <b>{neo4j_id}</b> tidak ditemukan di Neo4j.</div>"
        yield (
            cnn_result,
            error_graph,
            f"❌ Data `{neo4j_id}` tidak ditemukan di Neo4j. Tidak bisa generate jawaban."
        )
        return

    # ── Step 3: Pahami maksud pertanyaan (LLM prompting) ──
    #Memahami maksud pertanyaan petani menggunakan LLM (Groq), bukan lagi rule-based
    #keyword matching. Dihitung SEKALI di sini lalu dipakai bersama oleh graph,
    #context, dan generate — supaya tidak 3x panggilan LLM terpisah untuk 1 request.
    progress(0.55, desc="🧠 Memahami maksud pertanyaan dengan LLM...")
    intents = rag.detect_question_intent(user_question)

    # ── Step 3b: Semantic Retrieval (Hybrid RAG — Jalur B) ──
    #Mencari penyakit lain yang mirip secara makna dengan "predicted_class +
    #question" (sesuai Gambar 3.6), di luar hasil deteksi CNN. Dipakai
    #sebagai pembanding/pelengkap, bukan pengganti hasil Graph RAG.
    progress(0.6, desc="🔎 Mencari referensi tambahan (semantic retrieval)...")
    semantic_hits = rag.retrieve_semantic(user_question, predicted_class=predicted_class, exclude_id=neo4j_id)

    # ── Step 3c: Hybrid Fusion (Gambar 3.6) ──
    #Menggabungkan hasil Jalur A (Graph Retrieval) dan Jalur B (Semantic
    #Retrieval) menjadi satu struktur data terpadu, sebelum diteruskan ke
    #dua fungsi output paralel: build_graph_html() dan build_context().
    fused = rag.hybrid_fusion(retrieved, semantic_hits)

    # ── Step 4: Build Graph HTML ──
    #Membangun visualisasi Knowledge Graph dalam bentuk graf interaktif untuk ditampilkan pada Gradio.
    progress(0.7, desc="🕸️ Membangun visualisasi graph interaktif...")
    graph_html = rag.build_graph_html(fused, question=user_question, intents=intents)

    # Tab "Graph Neo4j" langsung tampil di sini, sementara Groq LLM
    # masih diproses di belakang (biasanya bagian paling lambat)
    yield cnn_result, graph_html, "⏳ Menghasilkan jawaban dengan Groq LLM..."

    # ── Step 5: Generate RAG Answer (sudah include ringkasan Neo4j) ──
    progress(0.85, desc="💬 Menghasilkan jawaban AI berdasarkan konteks (Groq LLM)...")
    context = rag.build_context(fused, prediction, question=user_question, intents=intents, semantic_hits=semantic_hits)  #Menyusun konteks dari hasil retrieval dan prediksi CNN sebagai masukan untuk LLM.
    answer  = rag.generate(     #Mengirim konteks dan pertanyaan pengguna ke model Groq untuk menghasilkan jawaban dalam bahasa alami.
        context,
        user_question,
        retrieved=fused,
        low_confidence=low_confidence,
        intents=intents,
        semantic_hits=semantic_hits
    )

    progress(1.0, desc="✅ Analisis Selesai!")
    yield cnn_result, graph_html, answer



# ─────────────────────────────────────────────
# 3. GRADIO UI
# ─────────────────────────────────────────────
DISEASE_CLASSES = [
    "apel__apple_scab", "apel__black_rot", "apel__cedar_apple_rust",
    "grape__grape_black_rot", "grape__grape_esca_black_measles",
    "grape__grape_leaf_blight_isariopsis_leaf_spot",
    "jagung__cercospora", "jagung__common_rust", "jagung__leaf_blight",
    "tomat__bacterial_spot", "tomat__early_blight", "tomat__late_blight",
    "unknown"
]

EMPTY_GRAPH = "<div style='color:#aaa; text-align:center; padding:60px; background:#1a1a2e; border-radius:8px;'>📊 Graph akan muncul setelah analisis</div>"

#Membangun antarmuka aplikasi yang terdiri dari input gambar, kolom pertanyaan, tombol analisis, 
#serta tiga tab hasil yaitu Hasil CNN, Graph Neo4j, dan Jawaban RAG.
with gr.Blocks() as demo:   

    # ── Header ──
    gr.HTML("""
        <div class="header-box">
            <h1>🌿 Sistem Deteksi & interpretasi Penyakit Tumbuhan Hortikultura</h1>
            <p>CNN (ResNet-50) + Graph RAG (Neo4j + Groq)</p>
            <p><small>13 Kelas (12 Penyakit + Unknown) — Apel, Anggur, Jagung, Tomat</small></p>
        </div>
    """)

    # ── Info kelas ──
    with gr.Accordion("ℹ️ Kelas Penyakit yang Didukung (klik untuk lihat)", open=False):
        gr.Dataframe(
            headers=["No", "Kelas"],
            value=[[i+1, c] for i, c in enumerate(DISEASE_CLASSES)],
            interactive=False
        )

    gr.Markdown("---")

    # ── Input ──
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 Upload Foto Daun",
                height=300
            )
            question_input = gr.Textbox(
                label="❓ Pertanyaan (opsional)",
                placeholder="Contoh: Apa obat yang tepat untuk penyakit ini?",
                lines=2
            )
            submit_btn = gr.Button("🔍 Analisis", variant="primary", size="lg")
            clear_btn  = gr.ClearButton(
                [image_input, question_input],
                value="🗑️ Bersihkan"
            )

        # ── Output ──
        with gr.Column(scale=2):
            with gr.Tabs():

                # Tab 1: Hasil CNN
                with gr.TabItem("🤖 Hasil CNN"):
                    cnn_output = gr.Markdown(
                        label="Prediksi Model",
                        elem_classes=["result-box"]
                    )

                # Tab 2: Graph Neo4j (interaktif)
                with gr.TabItem("🕸️ Graph Neo4j"):
                    graph_output = gr.HTML(
                        value=EMPTY_GRAPH,
                        label="Visualisasi Knowledge Graph"
                    )

                # Tab 3: Jawaban RAG (ringkasan Neo4j + LLM)
                with gr.TabItem("💬 Jawaban RAG"):
                    rag_output = gr.Markdown(
                        label="Ringkasan Data + Jawaban dari LLM (Groq)",
                        elem_classes=["result-box"]
                    )

    # ── Contoh gambar ──
    gr.Examples(
        examples=[],
        inputs=image_input,
        label="📂 Contoh Gambar (tambahkan di folder examples/)"
    )

    # ── Event handler ──
    submit_btn.click(   #Menghubungkan tombol Analisis dengan fungsi analyze_image(), sehingga proses dijalankan ketika tombol ditekan.
        fn=analyze_image,
        inputs=[image_input, question_input],
        outputs=[cnn_output, graph_output, rag_output],
        show_progress="full"
    )

    # ── Footer ──
    gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:gray; font-size:0.85em;">
            ResNet-50 97.86% (13 kelas, ambang unknown 70%) | Neo4j Aura | Groq openai/gpt-oss-120b
        </div>
    """)


# ─────────────────────────────────────────────
# 4. JALANKAN
#Menjalankan aplikasi Gradio sehingga sistem dapat diakses melalui browser.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True
    )