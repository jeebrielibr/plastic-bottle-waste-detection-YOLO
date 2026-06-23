# 🍶 Plastic Bottle Waste Detection — YOLO

Sistem deteksi dan pengenali objek sampah botol plastik menggunakan model **YOLO (You Only Look Once)**. Proyek ini dikembangkan sebagai tugas akhir mata kuliah **Pengolahan Citra**.

## 🎯 Tujuan

Membangun sistem yang mampu mendeteksi botol plastik secara otomatis dari gambar, video, atau webcam, guna mendukung efisiensi proses pemilahan dan daur ulang sampah.

## 👥 Anggota Kelompok

| No | Nama                        |
|----|-----------------------------|
| 1  | Jibril Ibrahim              |
| 2  | Yanti Elnaya Putri          |
| 3  | Silvia Zahro Diniah         |
| 4  | Muhamad Ridwan Karim        |
| 5  | Achmad Muflih Alrasyid      |
| 6  | Anwar Maulana               |
| 7  | Rohmatul Hidayat            |

## 📁 Struktur Repository

```
├── datasets/           # Dataset (train, valid, test) beserta konfigurasi data.yaml
│   ├── data.yaml       # Konfigurasi dataset untuk YOLO
│   ├── train/          # Data latih (images + labels)
│   ├── valid/          # Data validasi (images + labels)
│   └── test/           # Data uji (images + labels)
├── docs/               # Dokumentasi dan catatan proyek
│   └── Deskripsi Projek.md
├── notebook/           # File Jupyter Notebook (.ipynb) untuk eksperimen & training
└── README.md
```

## 📊 Dataset

- **Sumber**: [Plastic Bottle Image Dataset](https://www.kaggle.com/datasets/) (Kaggle)
- **Kelas**: 1 kelas → `bottle`
- **Pembagian**: train / valid / test
- **Format label**: YOLO annotation (`.txt`)

Konfigurasi dataset dapat dilihat pada file [`datasets/data.yaml`](datasets/data.yaml).

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/<username>/plastic-bottle-waste-detection-yolo.git
cd plastic-bottle-waste-detection-yolo
```

### 2. Install dependency

```bash
pip install ultralytics opencv-python matplotlib
```

### 3. Training model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # atau varian YOLO lainnya
results = model.train(
    data="datasets/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)
```

### 4. Inferensi / Deteksi

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Deteksi dari gambar
results = model.predict(source="path/to/image.jpg", conf=0.5)

# Deteksi dari webcam
results = model.predict(source=0, stream=True)
```

## 📈 Metrik Evaluasi

Model dievaluasi menggunakan metrik standar YOLO:

- **Precision** — ketepatan deteksi positif
- **Recall** — kemampuan mendeteksi seluruh objek positif
- **mAP (mean Average Precision)** — akurasi keseluruhan model

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik.
