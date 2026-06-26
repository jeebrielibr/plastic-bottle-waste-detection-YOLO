import os

import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
CONF_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)


def detect(image):
    if image is None:
        return None, "Tidak ada gambar yang diberikan."

    results = model.predict(source=image, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    count = len(results[0].boxes)
    summary = f"Ditemukan {count} botol plastik."

    return annotated, summary


def detect_stream(frame):
    if frame is None:
        return None

    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    return results[0].plot()


with gr.Blocks(title="Plastic Bottle Waste Detection") as demo:
    gr.Markdown("# 🍶 Plastic Bottle Waste Detection")
    gr.Markdown("Deteksi botol plastik menggunakan model YOLOv8n.")

    with gr.Tab("Upload / Kamera"):
        with gr.Row():
            input_img = gr.Image(
                sources=["upload", "webcam", "clipboard"],
                type="numpy",
                label="Pilih Gambar",
            )
            output_img = gr.Image(type="numpy", label="Hasil Deteksi")
        info_text = gr.Textbox(label="Info Deteksi")
        detect_btn = gr.Button("Deteksi", variant="primary")
        detect_btn.click(
            fn=detect,
            inputs=input_img,
            outputs=[output_img, info_text],
        )

    with gr.Tab("Kamera Live"):
        cam_stream = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Webcam Streaming",
            mirror_webcam=True,
        )
        cam_stream_output = gr.Image(type="numpy", label="Hasil Deteksi Live")
        cam_stream.stream(
            fn=detect_stream,
            inputs=cam_stream,
            outputs=cam_stream_output,
        )

if __name__ == "__main__":
    demo.launch()
