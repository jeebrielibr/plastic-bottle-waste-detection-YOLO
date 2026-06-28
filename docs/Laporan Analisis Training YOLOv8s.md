# Laporan Analisis Hasil Training — YOLOv8s

## Deteksi Sampah Botol Plastik

| Informasi | Detail |
|-----------|--------|
| Model Utama | **YOLOv8s** (small) — 11.2M parameter |
| Model Pembanding | YOLOv8n (nano) — 3.2M parameter |
| Framework | Ultralytics 8.4.75 + PyTorch 2.6.0 |
| GPU | NVIDIA RTX 3050 6GB Laptop GPU |
| Dataset | Plastic Bottle — 1 class (`bottle`) |
| Split Data | Train: 2177 · Valid: 1174 · Test: 648 |

---

## 1. Dataset

Dataset terdiri dari **3.999 gambar** botol plastik dengan anotasi bounding box format YOLO.

| Split | Gambar | Label | Status |
|-------|--------|-------|--------|
| Train | 2,177 | 3,186 objek | ✅ |
| Valid | 1,174 | 1,550 objek | ✅ |
| Test | 648 | 792 objek | ✅ |

### Distribusi Bounding Box

Rata-rata ukuran bounding box (ternormalisasi):
- **Width:** 0.349 (std: 0.241)
- **Height:** 0.571 (std: 0.291)

Botol plastik cenderung memiliki bounding box **lebih tinggi daripada lebar**, sesuai bentuk botol pada umumnya. Variasi ukuran cukup besar (std ~0.24–0.29), menunjukkan dataset mencakup botol dari berbagai jarak dan sudut pandang.

### Catatan Dataset

- Terdapat **2 gambar korup** di train set (koordinat label di luar rentang [0,1]) — diabaikan otomatis oleh Ultralytics
- Dataset mengandung campuran format **detect + segment** — segmen diabaikan, hanya bounding box digunakan
- Kategori: 1 kelas (`bottle`)

---

## 2. Konfigurasi Training

### Parameter Training YOLOv8s

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Model | `yolov8s.pt` | Pre-trained COCO |
| Epochs | 100 (early stop di 71) | Patience = 30 |
| Batch | 8 | Dibatasi VRAM 6 GB |
| Image Size | 640 × 640 | Standar YOLO |
| Optimizer | SGD | LR 0.01, momentum 0.937, weight decay 0.0005 |
| LR Schedule | Cosine decay | `cos_lr=True` |
| Warmup | 3 epochs | Momentum 0.8 → 0.937 |
| Augmentasi | Mosaic (close at epoch 15), RandAugment, HSV, flip |
| Cache | RAM (disk fallback — RAM < safety margin) |
| Seed | 42 |

### Parameter Training YOLOv8n (pembanding)

| Parameter | Nilai |
|-----------|-------|
| Model | `yolov8n.pt` |
| Batch | 13 (auto-batch) |
| Optimizer | AdamW (LR 0.002 dari auto) |
| Label Smoothing | 0.1 |

---

## 3. Proses Training

### YOLOv8s — Training Log

| Epoch | Box Loss | Cls Loss | DFL Loss | Precision | Recall | mAP@50 | mAP@50-95 | Waktu |
|-------|----------|----------|----------|-----------|--------|--------|-----------|-------|
| 1 | 1.474 | 2.019 | 1.701 | 0.544 | 0.506 | 0.383 | 0.257 | 70 s |
| 10 | 1.471 | 1.559 | 1.629 | 0.532 | 0.508 | 0.344 | 0.213 | 663 s |
| 20 | 1.346 | 1.348 | 1.518 | 0.589 | 0.512 | 0.372 | 0.235 | 1350 s |
| 30 | 1.238 | 1.234 | 1.439 | 0.579 | 0.526 | 0.362 | 0.234 | 2024 s |
| **41** | **1.150** | **1.153** | **1.376** | **0.627** | **0.529** | **0.455** | **0.311** | **2765 s** |
| 50 | 1.091 | 1.080 | 1.337 | 0.581 | 0.546 | 0.411 | 0.276 | 3372 s |
| 60 | 1.026 | 0.981 | 1.288 | 0.577 | 0.515 | 0.421 | 0.291 | 4051 s |
| 70 | 0.975 | 0.924 | 1.256 | 0.571 | 0.535 | 0.412 | 0.282 | 4716 s |
| **71** | **0.935** | **0.890** | **1.240** | **0.603** | **0.512** | **0.430** | **0.296** | **4787 s** |

### Kurva Loss

Semua loss (box, cls, dfl) menunjukkan tren **menurun stabil** selama 71 epoch — tidak ada overfitting signifikan. Training berhenti lebih awal (patience=30) karena metrik validasi tidak membaik secara konsisten setelah epoch 41.

### Total Waktu Training

| Model | Total Waktu | Epoch | Rata-rata per Epoch |
|-------|-------------|-------|---------------------|
| **YOLOv8s** (batch 8) | **~80 menit** (4.787 s) | 71 | ~67 detik |
| YOLOv8n (batch 13) | ~57 menit (3.428 s) | 100 | ~34 detik |

YOLOv8s ~2× lebih lambat per epoch dari YOLOv8n, konsisten dengan perbedaan jumlah parameter (11.2M vs 3.2M).

---

## 4. Hasil Evaluasi

### 4.1 Best Validation Metrics (YOLOv8s)

| Metrik | Nilai Terbaik | Epoch | Target | Status |
|--------|---------------|-------|--------|--------|
| Precision | **0.629** | 56 | ≥ 0.85 | ❌ |
| Recall | **0.557** | 29 | ≥ 0.85 | ❌ |
| mAP@50 | **0.455** | 41 | ≥ 0.90 | ❌ |
| mAP@50-95 | **0.311** | 41 | ≥ 0.60 | ❌ |

### 4.2 Test Set Evaluation (YOLOv8n — Pembanding)

| Metrik | YOLOv8n | Target | Keterangan |
|--------|---------|--------|------------|
| Precision | **0.578 – 0.643** | ≥ 0.85 | Bervariasi antar evaluasi |
| Recall | **0.458 – 0.499** | ≥ 0.85 | |
| mAP@50 | **0.446 – 0.447** | ≥ 0.90 | |
| mAP@50-95 | **0.319 – 0.330** | ≥ 0.60 | |

> **Catatan:** Kedua model menunjukkan performa yang mendekati sama pada dataset ini, dengan YOLOv8s unggul tipis pada Recall (+~0.06). Namun, semua metrik **jauh di bawah target** yang ditetapkan (precision/recall ≥ 0.85, mAP@50 ≥ 0.90).

### 4.3 Perbandingan YOLOv8n vs YOLOv8s

| Aspek | YOLOv8n (nano) | YOLOv8s (small) |
|-------|----------------|-----------------|
| Parameter | 3.2M | 11.2M (3.5×) |
| GFLOPs | 8.2 | ~28.6 (3.5×) |
| Batch size | 13 | 8 |
| Waktu training | ~57 menit (100 epoch) | ~80 menit (71 epoch) |
| **mAP@50 (val)** | **~0.45** | **~0.46** |
| **mAP@50-95 (val)** | **~0.33** | **~0.31** |
| Ukuran model | 5.9 MB | 18.4 MB |

Perbedaan performa antara YOLOv8n dan YOLOv8s **tidak signifikan** pada dataset ini, menunjukkan bahwa bottleneck utama bukan pada kapasitas model, melainkan pada **kualitas atau konsistensi dataset**.

---

## 5. Analisis

### 5.1 Mengapa Akurasi Rendah?

Beberapa faktor yang berkontribusi terhadap rendahnya metrik:

1. **Dataset kecil & tidak seragam**
   - Hanya 2.177 gambar train untuk 1 kelas — relatif sedikit
   - Gambar berasal dari berbagai sumber (Roboflow) dengan kualitas, resolusi, dan pencahayaan yang bervariasi

2. **Anotasi tidak konsisten**
   - Ditemukan 2 gambar dengan koordinat label di luar batas [0,1] (korup)
   - Dataset mengandung campuran format detect + segment, menandakan proses anotasi yang tidak seragam

3. **Objek kecil & bervariasi**
   - Distribusi ukuran bounding box bervariasi besar (std width 0.24, height 0.29)
   - Botol plastik dengan latar belakang kompleks sulit dibedakan (botol transparan, refleksi cahaya)

4. **Kelas tunggal — tantangan berbeda**
   - Dengan hanya 1 kelas, model tidak mendapat informasi kontekstual dari multiple classes
   - False positive lebih sulit ditekan tanpa negative class

5. **Overfitting pada epoch lanjut**
   - Loss training terus menurun sementara mAP validasi stagnan/plateau

### 5.2 Analisis Kurva Precision-Recall

Berdasarkan kurva PR Box dari training:
- Area di bawah kurva (AUC) yang kecil mengonfirmasi model kesulitan memisahkan botol dari background pada confidence threshold yang lebih tinggi
- Pada confidence > 0.5, recall turun drastis — banyak botol tidak terdeteksi sebagai trade-off precision

### 5.3 Analisis Confusion Matrix

Confusion matrix menunjukkan bahwa model menghasilkan **false positive (background predicted as bottle)** dan **false negative (bottle missed)** dalam jumlah signifikan — konsisten dengan precision dan recall yang rendah.

---

## 6. Kesimpulan

### Hasil Training YOLOv8s

| Metrik | Hasil | Target |
|--------|-------|--------|
| Precision | **0.629** | ≥ 0.85 |
| Recall | **0.557** | ≥ 0.85 |
| mAP@50 | **0.455** | ≥ 0.90 |
| mAP@50-95 | **0.311** | ≥ 0.60 |
| Waktu Training | **~80 menit** | — |
| Model Size | **~18 MB** (.pt) / **~12 MB** (.onnx) |

### Temuan Utama

1. YOLOv8s **tidak memberikan peningkatan signifikan** dibanding YOLOv8n pada dataset ini — keduanya menghasilkan mAP@50 ~0.45
2. **Bottleneck utama ada pada dataset**, bukan arsitektur model
3. Kapasitas model 11.2M (YOLOv8s) tidak terpakai optimal karena keterbatasan data

### Rekomendasi

Untuk meningkatkan akurasi:

**Jangka Pendek — Optimalisasi Dataset:**
- **Bersihkan anotasi**: Validasi ulang seluruh bounding box, perbaiki yang tidak konsisten
- **Tambah data**: Augmentasi agresif (rotasi, scaling, brightness, cutout) untuk memperkaya variasi
- **Negative mining**: Tambahkan gambar tanpa botol sebagai background class

**Jangka Menengah — Hyperparameter Tuning:**
- Naikkan `epochs` dengan learning rate yang lebih kecil
- Coba optimizer AdamW (terbukti lebih stabil di YOLOv8n)
- Turunkan `patience` jika loss validasi mulai naik
- Atur `label_smoothing` untuk mencegah overconfidence

**Jangka Panjang — Arsitektur:**
- Jika dataset sudah diperbaiki, coba YOLOv8m (25.9M) dengan batch size 4

---

## 7. Output Model

Model berhasil di-export ke format ONNX untuk deployment:

| Format | Ukuran | Path |
|--------|--------|------|
| PyTorch (.pt) | ~5.9 MB (v8n) / ~18 MB (v8s) | `notebook/runs/detect/{v8n,v8s}/weights/best.pt` |
| ONNX | ~12 MB | `notebook/runs/detect/v8s/weights/best.onnx` |

Model ONNX mendukung dynamic batch size dan dapat dijalankan di CPU dengan ONNX Runtime.