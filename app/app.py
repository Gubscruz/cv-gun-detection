import streamlit as st
import cv2
import av
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


st.set_page_config(page_title="Gun Detection - YOLO26", page_icon="🔫", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def resolve_model_path() -> Path:
    """Resolve um caminho de modelo existente para evitar erros por cwd/typos."""
    candidates = [
        APP_DIR / "best.pt",
        PROJECT_ROOT / "results" / "best.pt",
        PROJECT_ROOT / "yolo26n.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Nenhum arquivo de modelo encontrado. Procurei em: "
        + ", ".join(str(p) for p in candidates)
    )


MODEL_PATH = resolve_model_path()


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def draw_detections(image_bgr, results, conf_threshold):
    """Desenha bounding boxes na imagem e retorna imagem RGB + lista de deteccoes."""
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            name = r.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Desenhar box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label com fundo
            label = f"{name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_bgr, (x1, y1 - h - 8), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(image_bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({"classe": name, "confianca": f"{conf:.2%}", "bbox": f"[{x1}, {y1}, {x2}, {y2}]"})

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, detections


# --- Sidebar ---
st.sidebar.title("Configuracoes")
modo = st.sidebar.radio("Fonte de entrada", ["Imagem", "Video", "Camera"])
conf_threshold = st.sidebar.slider("Confianca minima", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Modelo:** `{MODEL_PATH}`")


st.title("Gun Detection - YOLO26")
st.caption("Detecta armas em imagens, videos ou camera usando YOLO26")

# Carregar modelo
model = load_model()

# ==========================
# MODO IMAGEM
# ==========================
if modo == "Imagem":
    uploaded = st.file_uploader("Upload de imagem", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None:
        # Ler imagem
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Rodar modelo
        with st.spinner("Rodando deteccao..."):
            results = model.predict(img_bgr, conf=conf_threshold, verbose=False)

        # Desenhar e mostrar
        img_result, detections = draw_detections(img_bgr.copy(), results, conf_threshold)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width='stretch')
        with col2:
            st.subheader(f"Deteccoes: {len(detections)}")
            st.image(img_result, width='stretch')

        if detections:
            st.dataframe(detections, width='stretch')
        else:
            st.info("Nenhuma arma detectada.")

# ==========================
# MODO VIDEO
# ==========================
elif modo == "Video":
    uploaded = st.file_uploader("Upload de video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        # Salvar video em arquivo temporario
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        st.info(f"Video: {total_frames} frames | {fps:.0f} FPS")

        frame_placeholder = st.empty()
        status_text = st.empty()
        stop_btn = st.button("Parar")

        frame_count = 0
        detection_count = 0

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Processar a cada 2 frames para performance
            if frame_count % 2 == 0:
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                frame_result, detections = draw_detections(frame.copy(), results, conf_threshold)
                detection_count += len(detections)
                frame_placeholder.image(frame_result, width='stretch')
                status_text.text(f"Frame {frame_count}/{total_frames} | Deteccoes neste frame: {len(detections)}")

        cap.release()
        Path(tmp.name).unlink(missing_ok=True)
        st.success(f"Video processado! {frame_count} frames | {detection_count} deteccoes totais")

# ==========================
# MODO CAMERA (REALTIME)
# ==========================
elif modo == "Camera":
    st.markdown("Deteccao em tempo real pela webcam. Clique **START** para iniciar:")

    class GunDetector(VideoProcessorBase):
        def __init__(self):
            self.model = load_model()
            self.conf = conf_threshold

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")

            results = self.model.predict(img_bgr, conf=self.conf, verbose=False)

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf:
                        continue
                    cls = int(box.cls[0])
                    name = r.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{name} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img_bgr, (x1, y1 - h - 8), (x1 + w, y1), (0, 0, 255), -1)
                    cv2.putText(img_bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    webrtc_streamer(
        key="gun-detection",
        video_processor_factory=GunDetector,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        async_processing=True,
    )
