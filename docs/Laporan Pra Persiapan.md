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

### 2.2 Rekomendasi: YOLOv8 Small (`yolov8s`)

Dengan VRAM 6 GB pada RTX 3050, varian yang sesuai adalah:

| Varian    | Parameter | VRAM (est.) | mAP@50   | Status       |
|-----------|-----------|-------------|----------|--------------|
| YOLOv8n   | 3.2 M     | ~1.5 GB     | 0.88     | Opsi ringan  |
| YOLOv8s   | 11.2 M    | ~3.0 GB     | 0.92     | **Dipilih**  |
| YOLOv8m   | 25.9 M    | ~5.5 GB     | 0.95     | Ketat, perlu batch kecil |
| YOLOv8l   | 43.7 M    | ~8.0 GB     | 0.96     | Tidak muat   |
| YOLOv8x   | 68.2 M    | ~12.0 GB    | 0.97     | Tidak muat   |

**Alasan memilih YOLOv8s:**
- VRAM 6 GB cukup untuk menjalankan YOLOv8s dengan leluasa (butuh ~3 GB)
- Akurasi lebih tinggi dibanding YOLOv8n (mAP@50 0.92 vs 0.88)
- Jumlah parameter (11.2M) masih ringan, training tetap cepat
- YOLOv8m (~5.5 GB) bisa dicoba sebagai eksperimen lanjutan dengan batch size kecil (4)
- Framework Ultralytics mendukung YOLOv8 secara native dengan API yang sederhana

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

#### 3.3.1 Konfigurasi Training

Parameter training yang direkomendasikan untuk RTX 3050 (6 GB VRAM):

| Parameter    | Nilai   | Alasan                                        |
|--------------|---------|-----------------------------------------------|
| `model`      | yolov8s.pt | Varian small, akurasi lebih tinggi, muat di VRAM 6 GB |
| `data`       | datasets/data.yaml | Konfigurasi dataset lokal           |
| `epochs`     | 50–100  | Cukup untuk 1 kelas, bisa early stop          |
| `imgsz`      | 640     | Ukuran standar, keseimbangan akurasi vs speed |
| `batch`      | 16–32   | VRAM 6 GB cukup untuk batch lebih besar       |
| `device`     | 0       | GPU pertama (RTX 3050)                        |
| `workers`    | 4       | Data loading parallel                         |
| `patience`   | 15      | Early stopping jika tidak ada improvement     |

#### 3.3.2 Script Training

```python
from ultralytics import YOLO

# Load model pre-trained
model = YOLO("yolov8s.pt")

# Training
results = model.train(
    data="datasets/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    patience=15,
    project="runs/detect",
    name="train_bottle",
    pretrained=True,
    optimizer="auto",
    verbose=True,
)
```

#### 3.3.3 Monitoring Training

Selama training, Ultralytics otomatis menghasilkan:
- **Loss curves** — `results.png` (box_loss, cls_loss, dfl_loss)
- **Metric curves** — precision, recall, mAP per epoch
- **Confusion matrix** — `confusion_matrix.png`
- **Best weights** — `runs/detect/train_bottle/weights/best.pt`

Monitor VRAM usage dengan `nvidia-smi` di terminal terpisah. Jika OOM (Out of Memory), turunkan `batch` ke 8 atau `imgsz` ke 416.

---

### Tahap 4 — Evaluasi Model

#### 3.4.1 Evaluasi pada Test Set

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train_bottle/weights/best.pt")

# Evaluasi pada test set
metrics = model.val(data="datasets/data.yaml", split="test")

print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")
```

#### 3.4.2 Metrik yang Dilaporkan

| Metrik       | Deskripsi                                                  | Target   |
|--------------|------------------------------------------------------------|----------|
| Precision    | Dari semua deteksi positif, berapa yang benar              | ≥ 0.85   |
| Recall       | Dari semua objek asli, berapa yang berhasil terdeteksi     | ≥ 0.85   |
| mAP@50       | Mean Average Precision pada IoU threshold 0.50             | ≥ 0.90   |
| mAP@50-95    | Mean Average Precision pada IoU 0.50 s.d. 0.95            | ≥ 0.60   |

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
| Setup environment & CUDA       | Jibril           | Pending   |
| Verifikasi & perbaikan dataset | Jibril           | Pending   |
| Training model                 | Jibril           | Pending   |
| Evaluasi model                 | Jibril           | Pending   |
| Implementasi deteksi (gambar)  | TBD              | Pending   |
| Implementasi deteksi (webcam)  | TBD              | Pending   |
| Dokumentasi pengerjaan         | Seluruh tim      | Pending   |
| Penyusunan laporan akhir       | Seluruh tim      | Pending   |
| Slide presentasi               | TBD              | Pending   |
| Video presentasi / demo        | TBD              | Pending   |

> **Catatan:** Training dan evaluasi dilakukan eksklusif di laptop Jibril (RTX 3050). Anggota lain dapat membantu implementasi inferensi dan penyusunan laporan.

---

## 5. Timeline Pengerjaan

| Tanggal        | Kegiatan                                    | Output                    |
|----------------|---------------------------------------------|---------------------------|
| 17 – 18 Juni  | Setup environment, verifikasi dataset       | Environment siap, dataset valid |
| 19 – 20 Juni  | Training model, tuning hyperparameter       | Model terbaik (.pt)       |
| 21 – 22 Juni  | Evaluasi model, analisis hasil              | Metrik evaluasi lengkap   |
| 23 – 24 Juni  | Implementasi deteksi (gambar, video, webcam)| Program berfungsi         |
| 25 – 26 Juni  | Dokumentasi, penyusunan laporan & slide     | Draft laporan & slide     |
| 27 Juni       | Video demo, review akhir                    | Semua berkas lengkap      |
| 28 Juni       | **Pengumpulan seluruh berkas**              | ✅ Submit                  |

---

## 6. Checklist Kesiapan

### Environment
- [x] Python 3.12 terinstall
- [x] PyTorch 2.6 terinstall
- [ ] NVIDIA Driver & CUDA Toolkit terinstall (`nvidia-smi` OK)
- [ ] Virtual environment dibuat
- [ ] `ultralytics`, `opencv-python`, `matplotlib` terinstall
- [ ] `torch.cuda.is_available()` → `True`

### Dataset
- [ ] Dataset sudah di-extract ke folder `datasets/`
- [ ] `data.yaml` sudah menggunakan path lokal (bukan path Kaggle)
- [ ] Jumlah images = jumlah labels di setiap split
- [ ] Format label YOLO sudah benar (class_id, x_center, y_center, w, h)
- [ ] Visualisasi sampel gambar + bbox sudah dicek

### Training
- [ ] Pre-trained weight `yolov8s.pt` tersedia (auto-download saat pertama kali run)
- [ ] Script training sudah disiapkan di notebook/Python
- [ ] VRAM monitoring siap (`nvidia-smi`)
- [ ] Plan fallback jika OOM (turunkan batch/imgsz)

### Deliverables
- [ ] Laporan proyek
- [ ] Kode program (repository GitHub)
- [ ] Slide presentasi
- [ ] Dokumentasi pengerjaan (screenshot)
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

Proyek ini menggunakan **YOLOv8 Small** melalui framework **Ultralytics** dengan lingkungan **Python 3.12** dan **PyTorch 2.6**. VRAM 6 GB pada RTX 3050 cukup leluasa untuk menjalankan YOLOv8s (butuh ~3 GB), memberikan akurasi lebih tinggi dibanding varian nano. Dataset sudah tersedia dalam format YOLO dan hanya memerlukan perbaikan path pada `data.yaml`. Proses utama meliputi 5 tahap: setup environment, verifikasi dataset, training, evaluasi, dan implementasi inferensi. Seluruh pengerjaan ditargetkan selesai dalam 12 hari (17–28 Juni 2025) dengan pembagian tugas yang jelas antar anggota tim.
