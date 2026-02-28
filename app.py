import threading
import time
import concurrent.futures
import streamlit as st
import av, cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from gemini_analyzer import analyze_frame
from voice_alerts import play_alert
from signal_controller import TrafficSignalController

st.set_page_config(page_title="Smart Traffic Analyzer", layout="wide")
if "controller" not in st.session_state:
    st.session_state.controller = TrafficSignalController()

# Thread-safe state shared between the WebRTC callback thread and the main Streamlit thread
_lock = threading.Lock()
_last_analysis = {}
_next_allowed_call = 0.0  # epoch seconds; enforces minimum gap between API calls
_CALL_INTERVAL = 5.0      # seconds between Gemini requests
_analyzing = False
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _draw_boxes(img, analysis):
    h, w = img.shape[:2]
    for vehicle in analysis.get("vehicles", []):
        box = vehicle.get("box_2d", [])
        if len(box) == 4:
            y_min, x_min, y_max, x_max = box
            x1, y1 = int(x_min * w / 1000), int(y_min * h / 1000)
            x2, y2 = int(x_max * w / 1000), int(y_max * h / 1000)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, vehicle.get("type", "vehicle"), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for ev in analysis.get("emergency_vehicles", []):
        box = ev.get("box_2d", [])
        if len(box) == 4:
            y_min, x_min, y_max, x_max = box
            x1, y1 = int(x_min * w / 1000), int(y_min * h / 1000)
            x2, y2 = int(x_max * w / 1000), int(y_max * h / 1000)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, ev.get("type", "emergency"), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for ped in analysis.get("pedestrians", []):
        box = ped.get("box_2d", [])
        if len(box) == 4:
            y_min, x_min, y_max, x_max = box
            x1, y1 = int(x_min * w / 1000), int(y_min * h / 1000)
            x2, y2 = int(x_max * w / 1000), int(y_max * h / 1000)
            color = (0, 0, 255) if ped.get("crossing") else (255, 165, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "pedestrian", (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for obj in analysis.get("test_objects", []):
        box = obj.get("box_2d", [])
        if len(box) == 4:
            y_min, x_min, y_max, x_max = box
            x1, y1 = int(x_min * w / 1000), int(y_min * h / 1000)
            x2, y2 = int(x_max * w / 1000), int(y_max * h / 1000)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, obj.get("label", "object"), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

def _run_analysis(frame_bytes):
    global _last_analysis, _next_allowed_call, _analyzing
    try:
        result = analyze_frame(frame_bytes)
        with _lock:
            _last_analysis = result
            _next_allowed_call = time.time() + _CALL_INTERVAL
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        backoff = _CALL_INTERVAL * 3 if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else _CALL_INTERVAL
        with _lock:
            _next_allowed_call = time.time() + backoff
    finally:
        with _lock:
            _analyzing = False

def video_frame_callback(frame):
    global _analyzing
    img = frame.to_ndarray(format="bgr24")

    with _lock:
        analysis = dict(_last_analysis)
        ready = time.time() >= _next_allowed_call and not _analyzing

    if ready:
        small = cv2.resize(img, (640, 360))
        _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _analyzing = True
        _executor.submit(_run_analysis, buffer.tobytes())

    _draw_boxes(img, analysis)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="traffic-cam", mode=WebRtcMode.SENDRECV, video_frame_callback=video_frame_callback)

with _lock:
    analysis = dict(_last_analysis)

if analysis:
    st.metric("Vehicles", len(analysis.get("vehicles", [])))
    signal = st.session_state.controller.update(analysis, {})
    st.info(signal.get("message"))
    if st.button("Announce Status"):
        play_alert("status", signal.get("message"))
