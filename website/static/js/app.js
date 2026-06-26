document.addEventListener("DOMContentLoaded", () => {
    // Tab switching
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.tab;
            tabBtns.forEach((b) => b.classList.remove("active"));
            tabContents.forEach((c) => c.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(tab).classList.add("active");
        });
    });

    // Upload functionality
    initUpload();

    // Webcam functionality
    initWebcam();
});

/* ---------- Upload Image ---------- */
function initUpload() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const resultSection = document.getElementById("uploadResult");
    const resultImage = document.getElementById("uploadResultImage");
    const countEl = document.getElementById("uploadCount");
    const detectionsEl = document.getElementById("uploadDetections");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleUpload(e.target.files[0]);
        }
    });

    async function handleUpload(file) {
        resultSection.style.display = "flex";
        resultImage.style.display = "none";
        detectionsEl.innerHTML = "";
        countEl.textContent = "—";

        const loadingDiv = document.createElement("div");
        loadingDiv.className = "loading";
        loadingDiv.textContent = "Memproses gambar...";
        resultImage.parentElement.appendChild(loadingDiv);

        const formData = new FormData();
        formData.append("image", file);

        try {
            const res = await fetch("/detect/image", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            loadingDiv.remove();

            if (data.success) {
                resultImage.src = data.image;
                resultImage.style.display = "block";
                countEl.textContent = data.count;
                renderDetections(detectionsEl, data.detections);
            } else {
                countEl.textContent = "Error";
                detectionsEl.innerHTML = `<p style="color:#e74c3c">${data.error}</p>`;
            }
        } catch (err) {
            loadingDiv.remove();
            countEl.textContent = "Error";
            detectionsEl.innerHTML = `<p style="color:#e74c3c">Gagal terhubung ke server</p>`;
        }
    }
}

/* ---------- Webcam Real-time ---------- */
function initWebcam() {
    const startBtn = document.getElementById("startWebcam");
    const stopBtn = document.getElementById("stopWebcam");
    const container = document.getElementById("webcamContainer");
    const video = document.getElementById("webcamVideo");
    const canvas = document.getElementById("webcamCanvas");
    const resultImage = document.getElementById("webcamResultImage");
    const countEl = document.getElementById("webcamCount");
    const statusEl = document.getElementById("webcamStatus");
    const detectionsEl = document.getElementById("webcamDetections");

    let stream = null;
    let intervalId = null;
    let processing = false;

    startBtn.addEventListener("click", async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            container.style.display = "flex";
            startBtn.style.display = "none";
            stopBtn.style.display = "inline-block";
            statusEl.textContent = "Aktif";

            intervalId = setInterval(captureAndDetect, 500);
        } catch (err) {
            statusEl.textContent = "Gagal akses kamera";
            alert("Tidak dapat mengakses webcam. Pastikan izin kamera diberikan.");
        }
    });

    stopBtn.addEventListener("click", () => {
        stopWebcam();
    });

    function stopWebcam() {
        if (intervalId) clearInterval(intervalId);
        intervalId = null;
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        container.style.display = "none";
        startBtn.style.display = "inline-block";
        stopBtn.style.display = "none";
        statusEl.textContent = "—";
        countEl.textContent = "0";
        detectionsEl.innerHTML = "";
    }

    async function captureAndDetect() {
        if (processing) return;
        processing = true;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.8);

        try {
            const res = await fetch("/detect/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl }),
            });
            const data = await res.json();

            if (data.success) {
                resultImage.src = data.image;
                countEl.textContent = data.count;
                renderDetections(detectionsEl, data.detections);
            }
        } catch (err) {
            // silently skip frame on error
        } finally {
            processing = false;
        }
    }
}

/* ---------- Helpers ---------- */
function renderDetections(container, detections) {
    if (detections.length === 0) {
        container.innerHTML = "<p style='color:#a0a0b0'>Tidak ada botol terdeteksi</p>";
        return;
    }

    container.innerHTML = detections
        .map(
            (d, i) => `
        <div class="detection-item">
            <span>Bottle #${i + 1}</span>
            <span>
                ${(d.confidence * 100).toFixed(1)}%
                <span class="confidence-bar">
                    <span class="confidence-fill" style="width:${d.confidence * 100}%"></span>
                </span>
            </span>
        </div>`
        )
        .join("");
}
