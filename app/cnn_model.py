"""
cnn_model.py — Load ResNet-50 dan fungsi prediksi penyakit tanaman
"""
#Mengimpor library yang digunakan untuk menjalankan model ResNet-50, mengolah citra, serta melakukan prediksi penyakit.
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ─────────────────────────────────────────────
# KONFIGURASI — sesuaikan path model .pt
#Menentukan lokasi model ResNet-50 dan perangkat (GPU/CPU). Model ini dilatih dengan
#13 kelas: 12 penyakit + 1 kelas `unknown` eksplisit untuk menolak input di luar objek
#penelitian (open-set recognition), MENGGANTIKAN gate LVLM lama.
# ─────────────────────────────────────────────
RESNET_PT    = "models/resnet50_final.pt"          # Path relatif dari root project

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ambang confidence untuk pendekatan GABUNGAN: kelas `unknown` eksplisit + threshold.
# Jika probabilitas tertinggi < ambang ini → hasil dianggap `unknown` (tolak), walau
# argmax jatuh ke salah satu penyakit. Nilai default dioverride dari checkpoint bila ada.
UNKNOWN_THRESHOLD = 0.70
UNKNOWN_LABEL     = "unknown"

#Daftar 13 kelas (12 penyakit + `unknown`) sesuai model hasil training SPLIT_80_10_10.
#Urutan HARUS sama dengan class_to_idx saat training; nilai ini dioverride dari
#checkpoint (`classes`) di load_model() supaya selalu konsisten dengan bobot.
CLASS_NAMES = [
    "apel__apple_scab",                                    # 0
    "apel__black_rot",                                     # 1
    "apel__cedar_apple_rust",                              # 2
    "grape__grape_black_rot",                              # 3
    "grape__grape_esca_black_measles",                     # 4
    "grape__grape_leaf_blight_isariopsis_leaf_spot",       # 5
    "jagung__cercospora",                                  # 6
    "jagung__common_rust",                                 # 7
    "jagung__leaf_blight",                                 # 8
    "tomat__bacterial_spot",                               # 9
    "tomat__early_blight",                                 # 10
    "tomat__late_blight",                                  # 11
    "unknown",                                             # 12
]

#Menghubungkan hasil klasifikasi CNN (nama kelas 13-kelas <tanaman>__<penyakit>) dengan
#ID penyakit pada Knowledge Graph Neo4j, sehingga hasil prediksi dapat digunakan pada
#proses Graph Retrieval. Kelas `unknown` sengaja TIDAK dipetakan (ditolak sebelum RAG).
CLASS_TO_NEO4J = {
    "apel__apple_scab"                             : "apple_scab",
    "apel__black_rot"                              : "apple_black_rot",
    "apel__cedar_apple_rust"                       : "apple_cedar_rust",
    "grape__grape_black_rot"                       : "grape_black_rot",
    "grape__grape_esca_black_measles"              : "grape_esca_black_measles",
    "grape__grape_leaf_blight_isariopsis_leaf_spot": "grape_leaf_blight_isariopsis_leaf_spot",
    "jagung__cercospora"                           : "corn_cercospora",
    "jagung__common_rust"                          : "corn_common_rust",
    "jagung__leaf_blight"                          : "corn_northern_leaf_blight",
    "tomat__bacterial_spot"                        : "tomato_bacterial_spot",
    "tomat__early_blight"                          : "tomato_early_blight",
    "tomat__late_blight"                           : "tomato_late_blight",
}

NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# LOAD MODEL
#Memuat model ResNet-50 beserta AutoImageProcessor, membaca bobot + metadata dari
#resnet50_final.pt (class list & threshold ikut dibaca dari checkpoint) agar konsisten.
# ─────────────────────────────────────────────
def load_model():
    """
    Load ResNet-50 dari file .pt lokal.
    Dipanggil sekali saat startup app.
    """
    global CLASS_NAMES, NUM_CLASSES, UNKNOWN_THRESHOLD

    print(f"🖥️  Device: {DEVICE}")

    # ── Baca checkpoint dulu: state_dict + metadata (classes, threshold) ──
    print("📦 Memuat ResNet-50...")
    ckpt = torch.load(RESNET_PT, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        res_state = ckpt["model_state_dict"]
        # Checkpoint menyimpan metadata training → pakai sebagai sumber kebenaran.
        if ckpt.get("classes"):
            CLASS_NAMES = list(ckpt["classes"])
            NUM_CLASSES = len(CLASS_NAMES)
        if isinstance(ckpt.get("unknown_threshold"), (int, float)):
            UNKNOWN_THRESHOLD = float(ckpt["unknown_threshold"])
    else:
        res_state = ckpt

    res_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    res_model     = AutoModelForImageClassification.from_pretrained(    #model pretrained dari Hugging Face.
        "microsoft/resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # strict=False dipertahankan (mengubah ke strict=True berisiko membuat
    # startup gagal total kalau ada mismatch buffer yang sebetulnya tidak
    # berbahaya, mis. num_batches_tracked). Tapi hasilnya WAJIB diperiksa --
    # kalau bobot classifier head hasil fine-tuning gagal termuat, model akan
    # diam-diam memakai bobot acak/pretrained ImageNet tanpa ada peringatan.
    load_result = res_model.load_state_dict(res_state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"  ⚠️  PERINGATAN: checkpoint '{RESNET_PT}' tidak cocok penuh dengan arsitektur model!")
        if load_result.missing_keys:
            print(f"     Missing keys (TIDAK ter-load dari checkpoint, masih bobot lama/acak): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"     Unexpected keys (ada di checkpoint tapi tidak dipakai model): {load_result.unexpected_keys}")
        print("     Periksa apakah nama layer di checkpoint sesuai arsitektur -- prediksi bisa tidak akurat kalau head classifier termasuk yang gagal termuat.")

    res_model.eval().to(DEVICE)
    print(f"  ✅ ResNet-50 siap ({NUM_CLASSES} kelas, ambang unknown {UNKNOWN_THRESHOLD:.0%})")

    return res_model, res_processor


# ─────────────────────────────────────────────
# PREDIKSI
#Melakukan prapengolahan gambar, yaitu mengubah gambar menjadi format RGB, kemudian mengubahnya menjadi tensor sesuai format input model ResNet-50.
# ─────────────────────────────────────────────
def _preprocess(image: Image.Image, processor) -> dict:
    """Konversi PIL Image → tensor input model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in inputs.items()}

#Melakukan klasifikasi citra daun menggunakan model ResNet-50. Hasil prediksi dihitung
#menggunakan Softmax, kemudian kelas dengan probabilitas tertinggi dipilih menggunakan
#argmax. Pendekatan GABUNGAN menentukan `unknown`: kelas unknown eksplisit ATAU
#confidence tertinggi < UNKNOWN_THRESHOLD.
def predict_disease(
    image: Image.Image,
    res_model, res_processor
) -> dict:
    """
    Prediksi penyakit menggunakan ResNet-50.
    Return dict berisi kelas, confidence, neo4j_id, dan penanda `unknown`.
    """
    with torch.no_grad():
        # ResNet inference
        res_inputs  = _preprocess(image, res_processor)
        res_outputs = res_model(**res_inputs)
        res_probs   = torch.softmax(res_outputs.logits, dim=-1)[0].cpu().numpy()
        res_idx     = int(np.argmax(res_probs))
        res_conf    = float(res_probs[res_idx])

    predicted_class = CLASS_NAMES[res_idx]

    # ── Deteksi `unknown` (open-set): kelas unknown eksplisit ATAU confidence rendah ──
    if predicted_class == UNKNOWN_LABEL:
        is_unknown, unknown_reason = True, "kelas_unknown"
    elif res_conf < UNKNOWN_THRESHOLD:
        is_unknown, unknown_reason = True, "confidence_rendah"
    else:
        is_unknown, unknown_reason = False, None

    neo4j_id = None if is_unknown else CLASS_TO_NEO4J.get(predicted_class, predicted_class)

    return {
        "predicted_class"      : predicted_class,
        "neo4j_id"             : neo4j_id,
        "confidence_resnet"    : res_conf,
        "is_unknown"           : is_unknown,
        "unknown_reason"       : unknown_reason,          # "kelas_unknown" | "confidence_rendah" | None
        "low_confidence"       : res_conf < UNKNOWN_THRESHOLD,
        "probabilities_resnet" : {CLASS_NAMES[i]: float(res_probs[i]) for i in range(NUM_CLASSES)},
    }
#Mengembalikan hasil prediksi berupa
#predicted_class → nama kelas hasil klasifikasi (13 kelas, termasuk `unknown`).
#neo4j_id → ID penyakit untuk Graph Retrieval (None jika `unknown`).
#confidence_resnet → tingkat keyakinan model.
#is_unknown → True jika input ditolak (kelas unknown / confidence < ambang).
#unknown_reason → alasan penolakan.
#low_confidence → penanda jika confidence di bawah ambang unknown.
#probabilities_resnet → probabilitas seluruh kelas.
