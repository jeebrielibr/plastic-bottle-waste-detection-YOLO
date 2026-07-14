<div align="center">

# 🍶 Plastic Bottle Waste Detection — YOLO

**Deteksi otomatis sampah botol plastik menggunakan YOLOv8**  
_Tugas Akhir — Pengolahan Citra_

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4-00C853?logo=yolo)](https://ultralytics.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)]()
[![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316?logo=gradio)](https://huggingface.co)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED?logo=onnx&logoColor=white)]()
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Deployed-FFD21E?logo=huggingface&logoColor=black)]()
[![License](https://img.shields.io/badge/License-Academic-808080)]()

</div>

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur](#-fitur)
- [Hasil Model](#-hasil-model)
- [Anggota Kelompok](#-anggota-kelompok)
- [Struktur Repository](#-struktur-repository)
- [Dataset](#-dataset)
- [Cara Menjalankan](#-cara-menjalankan)
- [Deploy Aplikasi](#-deploy-aplikasi)
- [Demo Website](#-demo-website)
- [Lisensi](#-lisensi)

---

## 🎯 Tentang Proyek

Sistem deteksi objek botol plastik menggunakan **YOLOv8** (You Only Look Once) untuk mendukung efisiensi pemilahan dan daur ulang sampah. Model mampu mendeteksi botol plastik dari **gambar, video, dan webcam** secara real-time.

| Komponen | Spesifikasi |
|----------|------------|
| **Model** | `YOLOv8n` (nano) & `YOLOv8s` (small) |
| **Framework** | Ultralytics 8.4.75 + PyTorch 2.6 |
| **GPU** | NVIDIA RTX 3050 6GB Laptop GPU |
| **Kelas** | 1 kelas: `bottle` |
| **Export** | ONNX (dynamic batch, 12 MB) |

---

## ✨ Fitur

- ✅ **Deteksi gambar** — Upload gambar, model mendeteksi botol plastik
- ✅ **Webcam real-time** — Deteksi langsung dari kamera
- ✅ **Dua varian model** — YOLOv8n (cepat) & YOLOv8s (akurat)
- ✅ **Export ONNX** — Siap deployment lintas platform
- ✅ **Deploy Gradio** — Aplikasi web di Hugging Face Spaces

---

## 📊 Hasil Model

| Metrik | YOLOv8n | YOLOv8s | Target |
|--------|:-------:|:-------:|:------:|
| **Precision** | 0.643 | 0.629 | ≥ 0.85 |
| **Recall** | 0.458 | 0.557 | ≥ 0.85 |
| **mAP@50** | 0.447 | 0.455 | ≥ 0.90 |
| **mAP@50-95** | 0.330 | 0.311 | ≥ 0.60 |
| **Parameter** | 3.2M | 11.2M | — |
| **Ukuran model (.pt)** | ~6 MB | ~18 MB | — |
| **Waktu training** | ~57 menit | ~80 menit | — |

> 🔍 **Catatan:** Kedua varian menunjukkan performa serupa (~0.45 mAP@50). Bottleneck utama ada pada **kualitas dan konsistensi dataset**, bukan arsitektur model. Lihat [laporan analisis lengkap](docs/Laporan%20Analisis%20Training%20YOLOv8s.md).

### Kurva Training (YOLOv8s)

Loss menurun stabil selama 71 epoch — tidak ada overfitting signifikan.

```
Box Loss:  1.474 → 0.935  📉
Cls Loss:  2.019 → 0.890  📉
DFL Loss:  1.701 → 1.240  📉
```

---

## 👥 Anggota Kelompok

| ![][avatar1] | ![][avatar2] | ![][avatar3] | ![][avatar4] | ![][avatar5] | ![][avatar6] | ![][avatar7] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Jibril Ibrahim** | **Yanti Elnaya Putri** | **Silvia Zahro Diniah** | **Muhamad Ridwan Karim** | **Achmad Muflih Alrasyid** | **Anwar Maulana** | **Rohmatul Hidayat** |

[avatar1]: https://api.dicebear.com/9.x/initials/svg?seed=JI&backgroundColor=3b82f6
[avatar2]: https://api.dicebear.com/9.x/initials/svg?seed=YE&backgroundColor=ec4899
[avatar3]: https://api.dicebear.com/9.x/initials/svg?seed=SD&backgroundColor=a855f7
[avatar4]: https://api.dicebear.com/9.x/initials/svg?seed=MR&backgroundColor=22c55e
[avatar5]: https://api.dicebear.com/9.x/initials/svg?seed=AA&backgroundColor=eab308
[avatar6]: https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=f97316
[avatar7]: https://api.dicebear.com/9.x/initials/svg?seed=RH&backgroundColor=06b6d4

---

## 📁 Struktur Repository

```
📦 plastic-bottle-waste-detection-yolo
├── 📂 dataset/                  # Dataset YOLO (train / valid / test)
│   ├── data.yaml                # Konfigurasi dataset
│   ├── train/                   # 2.177 gambar + label
│   ├── valid/                   # 1.174 gambar + label
│   └── test/                    # 648 gambar + label
├── 📂 deploy/                   # Aplikasi Gradio (Hugging Face)
│   ├── app.py                   # App web untuk deteksi
│   ├── best.pt                  # Model terbaik untuk deploy
│   └── requirements.txt         # Dependensi deploy
├── 📂 docs/                     # Dokumentasi proyek
│   ├── Deskripsi Projek.md
│   ├── Laporan Pra Persiapan.md
│   └── Laporan Analisis Training YOLOv8s.md
├── 📂 model/                    # Model backup (Google Colab)
│   └── v8n_colab/
├── 📂 notebook/                 # Jupyter Notebooks
│   ├── implementationYOLOv8n.ipynb
│   ├── colabeval.ipynb
│   └── implementcamera.ipynb
└── README.md
```

---

## 📊 Dataset

| Split | Gambar | Objek (Instances) |
|:------|:------:|:-----------------:|
| **Train** | 2.177 | 3.186 |
| **Valid** | 1.174 | 1.550 |
| **Test** | 648 | 792 |
| **Total** | **3.999** | **5.528** |

- **Sumber**: [Plastic Bottle Image Dataset](https://universe.roboflow.com/) (Roboflow Universe)
- **Kelas**: 1 kelas — `bottle`
- **Format**: YOLO annotation (`.txt`), koordinat ternormalisasi [0, 1]
- **Distribusi bbox**: Width rata-rata 0.35, Height rata-rata 0.57 (botol lebih tinggi dari lebar)

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.12+
- NVIDIA GPU dengan CUDA (opsional, untuk training)
- 6 GB VRAM atau lebih (untuk training)

### 1. Clone & Setup

```bash
git clone https://github.com/<username>/plastic-bottle-waste-detection-yolo.git
cd plastic-bottle-waste-detection-yolo
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python matplotlib pandas seaborn
```

### 3. Training Model

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # atau yolov8n.pt
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
)
```

### 4. Deteksi Gambar

```python
from ultralytics import YOLO

model = YOLO("notebook/runs/detect/v8s/train_bottle/weights/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.5)
```

### 5. Deteksi Webcam

```bash
python -c "
from ultralytics import YOLO
import cv2

model = YOLO('notebook/runs/detect/v8s/train_bottle/weights/best.pt')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    results = model.predict(frame, conf=0.5, verbose=False)
    cv2.imshow('Detection', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
"
```

---

## 🚢 Deploy Aplikasi

Aplikasi web untuk deteksi botol plastik sudah siap di-deploy:

### Gradio App (`deploy/`)

```bash
cd deploy
pip install -r requirements.txt
python app.py
```

Aplikasi Gradio menyediakan:
- **Upload gambar** — deteksi botol dari file gambar
- **Kamera langsung** — ambil foto dan deteksi real-time

### Hugging Face Spaces
Link : https://huggingface.co/spaces/brielibr/Plastic-Bottle-Detection-YOLOv8n
    
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Deployed-FFD21E)](https://huggingface.co/spaces/jeebri/Plastic-Bottle-Waste-Detection)

---


## 📝 Lisensi

Proyek ini dikembangkan untuk **keperluan akademik** — Tugas Akhir Mata Kuliah Pengolahan Citra.

---

<div align="center">

**© 2026 — Kelompok Deteksi Botol Plastik**

</div>
