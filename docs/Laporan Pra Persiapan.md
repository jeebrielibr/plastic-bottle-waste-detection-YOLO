# Laporan Pra Persiapan — Plastic Bottle Waste Detection YOLO

## 1. Pendahuluan

Dokumen ini berisi rencana kerja lengkap untuk proyek deteksi sampah botol plastik menggunakan YOLO, mulai dari persiapan lingkungan kerja hingga implementasi akhir. Laporan ini disusun sebagai panduan bagi seluruh anggota tim agar proses pengerjaan berjalan terstruktur dan efisien.

**Tema:** Deteksi botol plastik untuk daur ulang
**Deadline pengumpulan:** 28 Juni 2025

---

## 2. Pemilihan Versi YOLO

### 2.1 Kendala Hardware Tim

| Anggota       | GPU                  | VRAM    | Keterangan          |
|---------------|----------------------|---------|---------------------|
| Jibril        | NVIDIA RTX 3050      | 6 GB    | **Satu-satunya untuk training** |
| Anggota lain  | Integrated / None    | —       | Hanya untuk development & inferensi CPU |

### 2.2 Model yang Digunakan: YOLOv8 Small (`yolov8s`)

Dengan VRAM 6 GB pada RTX 3050, varian yang sesuai adalah:

| Varian    | Parameter | VRAM (est.) | mAP@50 COCO | Hasil Aktual |
|-----------|-----------|-------------|-------------|--------------|
| YOLOv8n   | 3.2 M     | ~1.5 GB     | 0.88        | **0.447** ✅ |
| YOLOv8s   | 11.2 M    | ~3.0 GB     | 0.92        | **0.455** ✅ **Dipilih** |
| YOLOv8m   | 25.9 M    | ~5.5 GB     | 0.95        | Tidak dicoba  |
| YOLOv8l   | 43.7 M    | ~8.0 GB     | 0.96        | Tidak muat   |
| YOLOv8x   | 68.2 M    | ~12.0 GB    | 0.97        | Tidak muat   |

> **Catatan:** mAP@50 pada tabel adalah benchmark COCO. Hasil aktual pada dataset botol plastik lebih rendah (lihat laporan analisis).

**Alasan memilih YOLOv8s (aktual):**
- VRAM 6 GB cukup (batch 8, ~4 GB terpakai)
- Training selesai dalam ~80 menit untuk 71 epoch
- Performa lebih stabil dibanding YOLOv8n pada validation set

**Eksperimen dilakukan pada 2 varian:**
1. **YOLOv8n** — batch 13, AdamW, 100 epoch, mAP@50 test = 0.447
2. **YOLOv8s** — batch 8, SGD + cos_lr, 71 epoch (early stop), mAP@50 val = 0.455

Hasil detail ada di `docs/Laporan Analisis Training YOLOv8s.md`.

**Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — library Python resmi dengan dokumentasi lengkap dan dukungan CUDA out-of-the-box.

---

## 3. Tahapan Kerja

### Tahap 1 — Persiapan Environment

#### 3.1.1 Install Python & CUDA

| Komponen         | Versi Minimum | Keterangan                              |
|------------------|---------------|-----------------------------------------|
| Python           | 3.12          | Sudah terinstall di sistem              |
| PyTorch          | 2.6           | Sudah terinstall, mendukung CUDA        |
| NVIDIA Driver    | ≥ 525.x       | Cek dengan `nvidia-smi`                 |
| CUDA Toolkit     | 11.8 atau 12.1 | Harus cocok dengan versi PyTorch        |
| cuDNN            | 8.x / 9.x     | Ikuti versi yang kompatibel dengan CUDA |

Verifikasi instalasi:

```bash
python --version
nvidia-smi
nvcc --version
```

#### 3.1.2 Buat Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3.1.3 Install Dependencies

```bash
pip install ultralytics opencv-python matplotlib pandas seaborn
```

Perintah di atas otomatis menginstall PyTorch dengan dukungan CUDA. Untuk memastikan GPU terdeteksi:

```python
import torch
print(torch.cuda.is_available())       # Harus True
print(torch.cuda.get_device_name(0))   # Harus menampilkan "NVIDIA GeForce RTX 3050 ..."
```

---

### Tahap 2 — Persiapan Dataset

Dataset sudah tersedia di folder `datasets/` dengan struktur:

```
datasets/
├── data.yaml
├── train/
│   ├── images/    # Gambar latih
│   └── labels/    # Label YOLO (.txt)
├── valid/
│   ├── images/    # Gambar validasi
│   └── labels/    # Label YOLO (.txt)
└── test/
    ├── images/    # Gambar uji
    └── labels/    # Label YOLO (.txt)
```

#### 3.2.1 Verifikasi Dataset

Sebelum training, lakukan pengecekan:

1. **Cek jumlah gambar per split:**

```python
import os

for split in ['train', 'valid', 'test']:
    img_count = len(os.listdir(f'datasets/{split}/images'))
    lbl_count = len(os.listdir(f'datasets/{split}/labels'))
    print(f"{split}: {img_count} images, {lbl_count} labels")
    assert img_count == lbl_count, f"Mismatch di {split}!"
```

2. **Cek isi `data.yaml`** — pastikan path sesuai:

```yaml
train: ./datasets/train/images
val: ./datasets/valid/images
test: ./datasets/test/images

nc: 1
names: ['bottle']
```

> **Catatan:** Path di `data.yaml` saat ini mengarah ke path Kaggle (`/kaggle/input/...`). Path ini **harus diubah** menjadi path lokal agar training berjalan di laptop.

3. **Visualisasi sampel label** — pastikan format bounding box benar:

```
# Format YOLO: <class_id> <x_center> <y_center> <width> <height>
# Semua nilai dinormalisasi (0-1)
0 0.452 0.331 0.214 0.567
```

4. **Visualisasi sampel gambar + bounding box:**

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('datasets/train/images/<nama_file>.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
```

#### 3.2.2 Perbaikan `data.yaml`

File `data.yaml` perlu diperbarui agar menggunakan path lokal relatif. Perubahan ini akan dilakukan sebelum training dimulai.

---

### Tahap 3 — Training Model

#### 3.3.1 Konfigurasi Training (Aktual)

Training dilakukan dengan 2 varian model. Berikut parameter aktual yang digunakan:

**Eksperimen 1: YOLOv8n**
| Parameter    | Nilai   | Keterangan                                    |
|--------------|---------|-----------------------------------------------|
| `model`      | yolov8n.pt | Nano variant (3.2M params) — eksperimen awal |
| `batch`      | 13      | Auto-batch — ~68% VRAM                        |
| `optimizer`  | AdamW   | LR 0.002 (auto-detect)                        |
| `epochs`     | 100     | Selesai penuh                                 |
| `cos_lr`     | true    | Cosine learning rate decay                    |

**Eksperimen 2: YOLOv8s (Final)**
| Parameter    | Nilai   | Keterangan                                    |
|--------------|---------|-----------------------------------------------|
| `model`      | yolov8s.pt | Small variant (11.2M params) — final dipilih |
| `batch`      | 8       | Dibutuhkan karena VRAM lebih terpakai         |
| `optimizer`  | SGD     | Momentum 0.937, weight decay 0.0005           |
| `epochs`     | 100     | Early stop di epoch 71 (patience 30)          |
| `cos_lr`     | true    | Cosine learning rate decay                    |

**Parameter umum:**
| Parameter    | Nilai   |
|--------------|---------|
| `imgsz`      | 640     |
| `device`     | 0 (RTX 3050) |
| `workers`    | 4       |
| `close_mosaic` | 15   |
| `cache`      | true    |
| `seed`       | 42      |

#### 3.3.2 Hasil Training

Ringkasan hasil training:

| Metrik | YOLOv8n (Test) | YOLOv8s (Val) | Target |
|--------|----------------|---------------|--------|
| Precision | 0.643 | **0.629** | ≥ 0.85 |
| Recall | 0.458 | **0.557** | ≥ 0.85 |
| mAP@50 | 0.447 | **0.455** | ≥ 0.90 |
| mAP@50-95 | 0.330 | **0.311** | ≥ 0.60 |
| Waktu | ~57 menit | ~80 menit | — |

> **Catatan:** YOLOv8s tidak memberikan peningkatan signifikan dibanding YOLOv8n. Bottleneck utama ada pada dataset, bukan arsitektur model.

Lihat laporan analisis lengkap di `docs/Laporan Analisis Training YOLOv8s.md`.

#### 3.3.3 Monitoring Training

Selama training, Ultralytics otomatis menghasilkan:
- **Loss curves** — `results.png` (box_loss, cls_loss, dfl_loss)
- **Metric curves** — precision, recall, mAP per epoch
- **Confusion matrix** — `confusion_matrix.png`
- **Best weights** — `notebook/runs/detect/v8s/train_bottle/weights/best.pt`

Output training disimpan di:
- YOLOv8n (notebook): `notebook/runs/detect/v8n/train_bottle/`
- YOLOv8s (notebook): `notebook/runs/detect/v8s/train_bottle/`

---

### Tahap 4 — Evaluasi Model

#### 3.4.1 Evaluasi pada Test Set (Aktual)

Evaluasi dilakukan pada **test set** (648 gambar, 792 instance) menggunakan model terbaik:

```python
from ultralytics import YOLO

best_model = YOLO("notebook/runs/detect/v8n/train_bottle/weights/best.pt")
metrics = best_model.val(data="../dataset/data.yaml", split="test")
```

| Metrik       | Hasil (YOLOv8n) | Target   | Status |
|--------------|-----------------|----------|--------|
| Precision    | 0.578 – 0.643  | ≥ 0.85   | ❌ Tidak tercapai |
| Recall       | 0.458 – 0.499  | ≥ 0.85   | ❌ Tidak tercapai |
| mAP@50       | 0.446 – 0.447  | ≥ 0.90   | ❌ Tidak tercapai |
| mAP@50-95    | 0.319 – 0.330  | ≥ 0.60   | ❌ Tidak tercapai |

> **Analisis:** Semua metrik belum mencapai target. Faktor utama: dataset kecil (~2.177 train), anotasi tidak konsisten, dan variasi objek tinggi. Lihat laporan analisis untuk rekomendasi perbaikan.

---

### Tahap 5 — Inferensi & Implementasi

#### 3.5.1 Deteksi pada Gambar

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train_bottle/weights/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    conf=0.5,
    save=True,
    project="runs/detect",
    name="predict_image",
)
```

#### 3.5.2 Deteksi pada Video

```python
results = model.predict(
    source="path/to/video.mp4",
    conf=0.5,
    save=True,
    project="runs/detect",
    name="predict_video",
)
```

#### 3.5.3 Deteksi Real-time via Webcam

```python
results = model.predict(
    source=0,
    stream=True,
    conf=0.5,
)

for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"bottle {conf:.2f}"
        # Tampilkan bounding box + label
```

#### 3.5.4 Output Implementasi

Program harus menampilkan:
- Bounding box di sekitar objek botol plastik yang terdeteksi
- Label kelas (`bottle`) dan confidence score
- Output berupa gambar/video/webcam dengan anotasi deteksi
- Informasi jumlah botol yang terdeteksi (opsional, nilai tambah)

---

## 4. Rencana Pembagian Tugas

| Tugas                          | Penanggung Jawab | Status    |
|--------------------------------|------------------|-----------|
| Setup environment & CUDA       | Jibril           | ✅ Selesai |
| Verifikasi & perbaikan dataset | Jibril           | ✅ Selesai |
| Training model (YOLOv8n & v8s) | Jibril           | ✅ Selesai |
| Evaluasi model                 | Jibril           | ✅ Selesai |
| Implementasi deteksi (gambar)  | Jibril           | ✅ Selesai |
| Implementasi deteksi (webcam)  | Jibril           | ✅ Selesai |
| Dokumentasi pengerjaan         | Seluruh tim      | ✅ Selesai |
| Penyusunan laporan akhir       | Seluruh tim      | Diproses   |
| Slide presentasi               | TBD              | Diproses   |
| Video presentasi / demo        | TBD              | Diproses   |

> **Catatan:** Training dan evaluasi dilakukan eksklusif di laptop Jibril (RTX 3050). Anggota lain dapat membantu implementasi inferensi dan penyusunan laporan.

---

## 5. Timeline Pengerjaan

| Tanggal        | Kegiatan                                    | Output                    | Realisasi |
|----------------|---------------------------------------------|---------------------------|-----------|
| 17 – 18 Juni  | Setup environment, verifikasi dataset       | Environment siap, dataset valid | ✅ |
| 19 – 20 Juni  | Training model (YOLOv8n eksperimen 1)       | Model YOLOv8n terlatih    | ✅ |
| 22 – 23 Juni  | Training model (YOLOv8s eksperimen 2)       | Model YOLOv8s terlatih    | ✅ |
| 24 Juni       | Implementasi deteksi (gambar, webcam)       | Program berfungsi         | ✅ |
| 25 – 27 Juni  | Evaluasi, dokumentasi, penyusunan laporan   | Metrik evaluasi + laporan | ✅ |
| 27 – 28 Juni  | Slide presentasi, video demo                | Slide + video demo        | Diproses |
| 28 Juni       | **Pengumpulan seluruh berkas**              | ✅ Submit                  | ✅ |

---

## 6. Checklist Kesiapan

### Environment
- [x] Python 3.12 terinstall
- [x] PyTorch 2.6 terinstall
- [x] NVIDIA Driver & CUDA Toolkit terinstall (`nvidia-smi` OK)
- [x] Virtual environment dibuat
- [x] `ultralytics`, `opencv-python`, `matplotlib` terinstall
- [x] `torch.cuda.is_available()` → `True`

### Dataset
- [x] Dataset sudah di-extract ke folder `dataset/` (bukan `datasets/`)
- [x] `data.yaml` sudah menggunakan path lokal
- [x] Jumlah images = jumlah labels di setiap split (✅ train: 2177/2177, valid: 1174/1174, test: 648/648)
- [x] Format label YOLO sudah diverifikasi
- [x] Visualisasi sampel gambar + bbox sudah dicek

### Training
- [x] YOLOv8n — training selesai (100 epoch, mAP@50 = 0.447)
- [x] YOLOv8s — training selesai (71 epoch, early stop, mAP@50 = 0.455)
- [x] VRAM monitoring — OK (batch 8, ~4 GB terpakai)
- [x] Model terbaik terseleksi — `weights/best.pt`
- [x] Model di-export ke format ONNX

### Deliverables
- [x] Laporan proyek (dokumen ini + laporan analisis)
- [x] Kode program (repository GitHub)
- [ ] Slide presentasi
- [x] Dokumentasi pengerjaan (screenshot)
- [ ] Video presentasi / demo

---

## 7. Risiko & Mitigasi

| Risiko                          | Dampak          | Mitigasi                                         |
|---------------------------------|-----------------|--------------------------------------------------|
| VRAM penuh (OOM) saat training  | Training gagal  | Turunkan batch size (16→8→4) atau imgsz (640→416), atau fallback ke yolov8n |
| Dataset tidak seimbang          | Model bias      | Augmentasi data (flip, rotate, brightness)       |
| Akurasi rendah                  | Tidak memenuhi target | Tambah epochs, tuning LR, augmentasi, atau naik ke yolov8m dengan batch kecil |
| Laptop training bermasalah      | Timeline mundur | Backup weights rutin, siapkan Colab sebagai fallback |
| Anggota lain tidak bisa training | Bottleneck di 1 orang | Export model → anggota lain bisa inferensi CPU |

---

## 8. Kesimpulan

Proyek ini menggunakan **YOLOv8** (varian nano dan small) melalui framework **Ultralytics** dengan lingkungan **Python 3.12** dan **PyTorch 2.6**. VRAM 6 GB pada RTX 3050 cukup untuk menjalankan training YOLOv8s dengan batch 8. Dataset sudah tersedia dalam format YOLO dan telah diverifikasi. 

**Hasil training:**
- **YOLOv8n** (3.2M params, batch 13, AdamW): mAP@50 = **0.447** (test set)
- **YOLOv8s** (11.2M params, batch 8, SGD + cos_lr): mAP@50 = **0.455** (validation set)

Kedua varian menghasilkan performa yang mendekati sama, menunjukkan bottleneck pada dataset. Proses utama meliputi 5 tahap: setup environment, verifikasi dataset, training (2 varian), evaluasi, dan implementasi inferensi. Seluruh pengerjaan ditargetkan selesai pada 28 Juni 2025. Lihat laporan analisis terpisah di `docs/Laporan Analisis Training YOLOv8s.md`.
