import streamlit as st
import cv2
import av
import numpy as np
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import requests
import base64


st.set_page_config(page_title="Gun Detection - YOLO26", page_icon="🔫", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = "BbgTrEnq95LA9drZNK2f"
PEOPLE_MODEL_ID = "people-detection-general/7"


def resolve_model_path() -> Path:
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


def detect_people(image_bgr, conf_threshold):
    """Roda o modelo Roboflow de pessoas via API REST."""
    return []

    _, buffer = cv2.imencode(".jpg", image_bgr)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    url = f"{ROBOFLOW_API_URL}/{PEOPLE_MODEL_ID}"
    response = requests.post(
        url,
        params={"api_key": ROBOFLOW_API_KEY, "confidence": int(conf_threshold * 100)},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()

    predictions = result.get("predictions", [])
    people = []
    for pred in predictions:
        conf = float(pred.get("confidence", 0))
        if conf < conf_threshold:
            continue
        class_name = str(pred.get("class", "")).lower()
        if class_name and class_name not in {"person", "people"}:
            continue
        people.append(pred)

    return people


def draw_detections(image_bgr, results, conf_threshold):
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            name = r.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_bgr, (x1, y1 - h - 8), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(image_bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({"classe": name, "confianca": f"{conf:.2%}", "bbox": f"[{x1}, {y1}, {x2}, {y2}]"})

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, detections


def roboflow_bbox_to_xyxy(prediction, image_shape):
    height, width = image_shape[:2]

    if all(key in prediction for key in ("x", "y", "width", "height")):
        center_x = float(prediction["x"])
        center_y = float(prediction["y"])
        box_width = float(prediction["width"])
        box_height = float(prediction["height"])
        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2
    elif all(key in prediction for key in ("x_min", "y_min", "x_max", "y_max")):
        x1 = float(prediction["x_min"])
        y1 = float(prediction["y_min"])
        x2 = float(prediction["x_max"])
        y2 = float(prediction["y_max"])
    else:
        return None

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_people_detections(image, people_detections):
    for person in people_detections:
        bbox = roboflow_bbox_to_xyxy(person, image.shape)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        conf = float(person.get("confidence", 0))
        label = f"person {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - h - 8), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image


def robbery_detected(gun_detections, people_detections):
    return bool(gun_detections) and bool(people_detections)


def show_robbery_message(is_detected):
    if is_detected:
        st.error("robbery detected!")


def render_metrics(placeholder, gun_ms=None, people_ms=None, history_ms=None):
    """Renderiza painel compacto de metricas dentro de um placeholder unico.

    Usa placeholder.container() para garantir que cada chamada SUBSTITUI o
    conteudo anterior em vez de empilhar.
    """
    def fmt(v):
        return f"{v:.0f} ms" if v is not None else "—"

    with placeholder.container():
        st.markdown("##### ⏱️ Tempo de inferencia")

        # Linha 1: ultima inferencia (arma | pessoa) lado a lado
        c1, c2 = st.columns(2)
        c1.markdown(
            f"<div style='font-size:0.78rem;color:#888'>Arma</div>"
            f"<div style='font-size:1.05rem;font-weight:600'>{fmt(gun_ms)}</div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div style='font-size:0.78rem;color:#888'>Pessoa</div>"
            f"<div style='font-size:1.05rem;font-weight:600'>{fmt(people_ms)}</div>",
            unsafe_allow_html=True,
        )

        # Linha 2: estatisticas agregadas
        if history_ms:
            avg = sum(history_ms) / len(history_ms)
            mn = min(history_ms)
            mx = max(history_ms)
            fps = 1000.0 / avg if avg > 0 else 0
            st.markdown(
                f"<div style='margin-top:0.6rem;padding:0.4rem 0.6rem;"
                f"background:rgba(255,255,255,0.04);border-radius:6px;"
                f"font-size:0.82rem;line-height:1.5'>"
                f"<b>Media:</b> {avg:.0f} ms <span style='color:#888'>(~{fps:.1f} FPS)</span><br>"
                f"<span style='color:#888'>min {mn:.0f} · max {mx:.0f} · n={len(history_ms)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='margin-top:0.6rem;font-size:0.78rem;color:#888'>"
                "Sem dados ainda</div>",
                unsafe_allow_html=True,
            )


# --- Sidebar ---
st.sidebar.title("Configuracoes")
modo = st.sidebar.radio("Fonte de entrada", ["Imagem", "Video", "Camera"])
conf_threshold = st.sidebar.slider("Confianca minima", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Modelo:** `{MODEL_PATH.name}`")

# UM placeholder unico para metricas (sera reutilizado em todas as atualizacoes)
st.sidebar.markdown("---")
metrics_placeholder = st.sidebar.empty()


st.title("Gun Detection - YOLO26")
st.caption("Detecta armas em imagens, videos ou camera usando YOLO26")

model = load_model()

# ==========================
# MODO IMAGEM
# ==========================
if modo == "Imagem":
    uploaded = st.file_uploader("Upload de imagem", type=["jpg", "jpeg", "png", "bmp", "webp"])

    render_metrics(metrics_placeholder)

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Rodando deteccao..."):
            t0 = time.perf_counter()
            results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
            gun_ms = (time.perf_counter() - t0) * 1000

        img_result, detections = draw_detections(img_bgr.copy(), results, conf_threshold)

        people_ms = None
        people_detections = []
        if detections:
            try:
                t1 = time.perf_counter()
                people_detections = detect_people(img_bgr, conf_threshold)
                people_ms = (time.perf_counter() - t1) * 1000
            except Exception as exc:
                st.warning(f"Nao foi possivel rodar o modelo de pessoas: {exc}")
        img_result = draw_people_detections(img_result, people_detections)

        total_ms = gun_ms + (people_ms or 0)
        render_metrics(metrics_placeholder, gun_ms=gun_ms, people_ms=people_ms, history_ms=[total_ms])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width='stretch')
        with col2:
            st.subheader(f"Deteccoes: {len(detections)}")
            st.image(img_result, width='stretch')

        show_robbery_message(robbery_detected(detections, people_detections))

        if detections:
            st.dataframe(detections, width='stretch')
        else:
            st.info("Nenhuma arma detectada.")

# ==========================
# MODO VIDEO
# ==========================
elif modo == "Video":
    uploaded = st.file_uploader("Upload de video", type=["mp4", "avi", "mov", "mkv"])

    process_every_seconds = st.sidebar.slider(
        "Intervalo entre inferencias - video (s)", 0.0, 2.0, 0.3, 0.05,
        help="0 = processa todos os frames. Maior = mais rapido, menos preciso."
    )

    render_metrics(metrics_placeholder)

    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_meta = cap.get(cv2.CAP_PROP_FPS) or 0
        duration_s = (total_frames / fps_meta) if fps_meta > 0 else 0

        st.info(
            f"Video: {total_frames} frames | FPS (metadado): {fps_meta:.1f} | "
            f"Duracao estimada: {duration_s:.1f}s"
        )

        frame_placeholder = st.empty()
        status_text = st.empty()
        robbery_placeholder = st.empty()
        stop_btn = st.button("Parar")

        frame_count = 0
        processed_count = 0
        detection_count = 0
        robbery_count = 0
        last_processed_ts = -float("inf")
        inference_times_ms = []

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if timestamp_s - last_processed_ts < process_every_seconds:
                continue

            last_processed_ts = timestamp_s
            processed_count += 1

            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            gun_inference_ms = (time.perf_counter() - t0) * 1000

            frame_result, detections = draw_detections(frame.copy(), results, conf_threshold)

            people_inference_ms = None
            people_detections = []
            if detections:
                t1 = time.perf_counter()
                try:
                    people_detections = detect_people(frame, conf_threshold)
                except Exception:
                    people_detections = []
                people_inference_ms = (time.perf_counter() - t1) * 1000

            total_inference_ms = gun_inference_ms + (people_inference_ms or 0)
            inference_times_ms.append(total_inference_ms)

            frame_result = draw_people_detections(frame_result, people_detections)
            has_robbery = robbery_detected(detections, people_detections)
            detection_count += len(detections)
            robbery_count += int(has_robbery)

            frame_placeholder.image(frame_result, width='stretch')

            status_text.text(
                f"Tempo do video: {timestamp_s:.2f}s | "
                f"Frame {frame_count}/{total_frames} (processados: {processed_count}) | "
                f"Deteccoes neste frame: {len(detections)}"
            )

            render_metrics(
                metrics_placeholder,
                gun_ms=gun_inference_ms,
                people_ms=people_inference_ms,
                history_ms=inference_times_ms,
            )

            if has_robbery:
                robbery_placeholder.error("robbery detected!")
            else:
                robbery_placeholder.empty()

        cap.release()
        Path(tmp.name).unlink(missing_ok=True)

        if inference_times_ms:
            avg_ms = sum(inference_times_ms) / len(inference_times_ms)
            min_ms = min(inference_times_ms)
            max_ms = max(inference_times_ms)
            st.success(
                f"Video processado! {frame_count} frames lidos | "
                f"{processed_count} processados | {detection_count} deteccoes totais | "
                f"{robbery_count} alertas\n\n"
                f"Tempo de inferencia - media: {avg_ms:.0f}ms | "
                f"min: {min_ms:.0f}ms | max: {max_ms:.0f}ms"
            )
        else:
            st.success(f"Video processado! {frame_count} frames lidos (nenhum processado)")

# ==========================
# MODO CAMERA (REALTIME)
# ==========================
elif modo == "Camera":
    st.markdown("Deteccao em tempo real pela webcam. Clique **START** para iniciar:")

    # Default 0.3s (300 ms) conforme pedido
    camera_interval_s = st.sidebar.slider(
        "Intervalo entre inferencias - camera (s)", 0.0, 2.0, 0.3, 0.05,
        help="0 = roda em todo frame. Maior = menos chamadas HTTP, video mais fluido."
    )

    LOG_COOLDOWN_S = 2.0
    MAX_LOG_ENTRIES = 100

    render_metrics(metrics_placeholder)

    class GunDetector(VideoProcessorBase):
        def __init__(self):
            self.model = load_model()
            self.conf = conf_threshold
            self.interval_s = camera_interval_s
            self.robbery_detected = False

            self.last_inference_time = 0.0
            self.cached_gun_dets = []
            self.cached_people = []

            self.metrics_lock = threading.Lock()
            self.last_gun_ms = None
            self.last_people_ms = None
            self.history_ms = []

            self.detection_log = []
            self.log_lock = threading.Lock()
            self.last_log_time_by_class = {}

        def _maybe_log(self, class_name, conf, has_people):
            now = time.time()
            last = self.last_log_time_by_class.get(class_name, 0)
            if now - last < LOG_COOLDOWN_S:
                return
            self.last_log_time_by_class[class_name] = now

            entry = {
                "horario": datetime.now().strftime("%H:%M:%S"),
                "classe": class_name,
                "confianca": f"{conf:.2%}",
                "pessoa_no_frame": "Sim" if has_people else "Nao",
                "alerta_roubo": "SIM" if has_people else "Nao",
            }
            with self.log_lock:
                self.detection_log.append(entry)
                if len(self.detection_log) > MAX_LOG_ENTRIES:
                    self.detection_log = self.detection_log[-MAX_LOG_ENTRIES:]

        def get_log_snapshot(self):
            with self.log_lock:
                return list(self.detection_log)

        def clear_log(self):
            with self.log_lock:
                self.detection_log.clear()
            self.last_log_time_by_class.clear()

        def get_metrics_snapshot(self):
            with self.metrics_lock:
                return self.last_gun_ms, self.last_people_ms, list(self.history_ms)

        def _draw_cached_guns(self, img):
            for det in self.cached_gun_dets:
                x1, y1, x2, y2 = det["bbox"]
                name = det["classe"]
                conf = det["conf"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return img

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            now = time.time()

            # FRAME SKIP por tempo
            if now - self.last_inference_time < self.interval_s:
                img_bgr = self._draw_cached_guns(img_bgr)
                img_bgr = draw_people_detections(img_bgr, self.cached_people)
                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

            self.last_inference_time = now
            raw_img_bgr = img_bgr.copy()

            t0 = time.perf_counter()
            results = self.model.predict(img_bgr, conf=self.conf, verbose=False)
            gun_ms = (time.perf_counter() - t0) * 1000

            gun_detections = []
            cached_gun_dets = []

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

                    gun_detections.append({"classe": name, "confianca": conf})
                    cached_gun_dets.append({"classe": name, "conf": conf, "bbox": (x1, y1, x2, y2)})

            people_detections = []
            people_ms = None
            if gun_detections:
                t1 = time.perf_counter()
                try:
                    people_detections = detect_people(raw_img_bgr, self.conf)
                except Exception:
                    people_detections = []
                people_ms = (time.perf_counter() - t1) * 1000

            img_bgr = draw_people_detections(img_bgr, people_detections)

            self.cached_gun_dets = cached_gun_dets
            self.cached_people = people_detections

            total_ms = gun_ms + (people_ms or 0)
            with self.metrics_lock:
                self.last_gun_ms = gun_ms
                self.last_people_ms = people_ms
                self.history_ms.append(total_ms)
                if len(self.history_ms) > 200:
                    self.history_ms = self.history_ms[-200:]

            self.robbery_detected = robbery_detected(gun_detections, people_detections)

            has_people = bool(people_detections)
            for det in gun_detections:
                self._maybe_log(det["classe"], det["confianca"], has_people)

            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    ctx = webrtc_streamer(
        key="gun-detection",
        video_processor_factory=GunDetector,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        async_processing=True,
    )

    col_alert, col_log = st.columns([1, 2])

    with col_alert:
        robbery_placeholder = st.empty()

    with col_log:
        st.subheader("Log de deteccoes")
        log_placeholder = st.empty()
        clear_btn = st.button("Limpar log")
        if clear_btn and ctx.video_processor:
            ctx.video_processor.clear_log()

    while ctx.state.playing:
        if ctx.video_processor:
            if ctx.video_processor.robbery_detected:
                robbery_placeholder.error("robbery detected!")
            else:
                robbery_placeholder.empty()

            gun_ms, people_ms, history_ms = ctx.video_processor.get_metrics_snapshot()
            render_metrics(
                metrics_placeholder,
                gun_ms=gun_ms,
                people_ms=people_ms,
                history_ms=history_ms,
            )

            log_entries = ctx.video_processor.get_log_snapshot()
            if log_entries:
                log_placeholder.dataframe(
                    list(reversed(log_entries)),
                    width='stretch',
                    height=400,
                )
            else:
                log_placeholder.info("Nenhuma arma detectada ainda.")
        time.sleep(0.5)