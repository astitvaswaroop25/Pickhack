import io
import threading
import time
import re as _re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import av, cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from gemini_analyzer import analyze_frame
from voice_alerts import play_alert
from signal_controller import TrafficSignalController
from arduino_controller import ArduinoController
import state  # persistent shared state — survives Streamlit reruns

# ── Page config & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Traffic Analyzer", layout="wide", page_icon="🚦")

# ── Non-blocking auto-refresh (every 5 s) ────────────────────────────────────────
st_autorefresh(interval=5000, key="traffic_refresh")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .header {
        background: linear-gradient(90deg, #1a1f2e, #2d3561);
        border-radius: 12px; padding: 18px 28px; margin-bottom: 20px;
    }
    .header h1 { margin: 0; font-size: 1.8rem; color: #ffffff; }
    .header p  { margin: 0; font-size: 0.9rem; color: #8b9ab8; }
    [data-testid="metric-container"] {
        background: #1a1f2e; border: 1px solid #2d3561;
        border-radius: 10px; padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #8b9ab8 !important; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem; font-weight: 700; }
    .section-label { color: #8b9ab8; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.08em; margin: 18px 0 8px 0; }
    .stButton > button {
        background: linear-gradient(90deg, #3b4fd8, #5a6ef0); color: white;
        border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600;
    }
    .stButton > button:hover { background: linear-gradient(90deg, #5a6ef0, #7b8ef8); }
    #MainMenu, footer { visibility: hidden; }
</style>
<div class="header">
    <h1>🚦 Smart Traffic Analyzer</h1>
    <p>Powered by Gemini AI &nbsp;·&nbsp; Real-time intersection monitoring</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "controller"       not in st.session_state:
    st.session_state.controller       = TrafficSignalController()
if "last_alert_state" not in st.session_state:
    st.session_state.last_alert_state = None
if "arduino"          not in st.session_state:
    st.session_state.arduino          = None  # connected on demand via sidebar

# ── Sidebar: Arduino connection ───────────────────────────────────────────────
with st.sidebar:
    st.header("🔌 Arduino")
    available_ports = ArduinoController.list_ports()
    port_options    = ["(none)"] + available_ports
    selected_port   = st.selectbox("Serial port", port_options)

    if st.button("Connect"):
        if selected_port != "(none)":
            if st.session_state.arduino:
                st.session_state.arduino.close()
            st.session_state.arduino = ArduinoController(selected_port)
        else:
            st.warning("Select a port first.")

    ard = st.session_state.arduino
    if ard and ard.connected:
        st.success(f"Connected — {selected_port}")
        st.write(f"PIR sensor: {'**MOTION**' if ard.sensor_triggered else 'clear'}")
    else:
        st.info("Not connected")

# ── Constants ────────────────────────────────────────────────────────────────
_CALL_INTERVAL = 12.0          # every 12 seconds
_429_BACKOFF   = 120.0

# ── Sidebar: Mock Mode ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛠️ Testing")
    st.session_state["use_mock_analysis"] = st.checkbox("Enable Mock AI Analysis", value=st.session_state.get("use_mock_analysis", False))

# ── Drawing helpers ──────────────────────────────────────────────────────────
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

def _draw_map(analysis, signal):
    """Top-down intersection map rendered with matplotlib."""
    fig, ax = plt.subplots(figsize=(3.2, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Road ──
    road = mpatches.FancyBboxPatch((2.5, 0), 5, 15,
                                   boxstyle="square,pad=0",
                                   facecolor='#3a3a3a', edgecolor='#555', linewidth=1)
    ax.add_patch(road)

    # Lane divider (dashed yellow centre line)
    for y in range(0, 15, 2):
        ax.plot([5, 5], [y, y + 1], color='#f5c542', linewidth=2, linestyle='-', alpha=0.8)

    # Road edge lines (white)
    ax.plot([2.5, 2.5], [0, 15], color='white', linewidth=1.5, alpha=0.5)
    ax.plot([7.5, 7.5], [0, 15], color='white', linewidth=1.5, alpha=0.5)

    # ── Direction arrows ──
    ax.annotate('', xy=(3.75, 12.5), xytext=(3.75, 10.5),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))
    ax.annotate('', xy=(6.25, 2.5), xytext=(6.25, 4.5),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    # ── Crosswalk (5 white stripes) ──
    for i in range(5):
        stripe = mpatches.Rectangle((2.5, 6.5 + i * 0.45), 5, 0.3,
                                    facecolor='white', alpha=0.9, edgecolor='none')
        ax.add_patch(stripe)

    # ── Traffic light (top-right corner) ──
    light_col = '#00dd55' if signal.get('light_state', 'green') == 'green' else '#dd2200'
    housing = mpatches.FancyBboxPatch((8.0, 13.0), 1.5, 1.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#111', edgecolor='#444', linewidth=1)
    ax.add_patch(housing)
    light = plt.Circle((8.75, 14.1), 0.55, color=light_col, zorder=5)
    ax.add_patch(light)
    ax.text(8.75, 14.1, 'G' if signal.get('light_state','green')=='green' else 'R',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', zorder=6)

    # ── Walk sign (left of crosswalk) ──
    walk_on    = signal.get('walk_sign', False)
    sign_color = '#00dd55' if walk_on else '#dd2200'
    sign_text  = 'WALK' if walk_on else "DON'T\nWALK"
    sign_box   = mpatches.FancyBboxPatch((0.1, 6.3), 1.8, 1.5,
                                         boxstyle="round,pad=0.15",
                                         facecolor='#111', edgecolor=sign_color, linewidth=2)
    ax.add_patch(sign_box)
    ax.text(1.0, 7.05, sign_text, ha='center', va='center',
            color=sign_color, fontsize=6.5, fontweight='bold', linespacing=1.3)

    # ── Sensor indicator ──
    sensor_box = mpatches.FancyBboxPatch((0.1, 5.3), 1.8, 0.7,
                                          boxstyle="round,pad=0.1",
                                          facecolor='#0d2540', edgecolor='#3a7bd5', linewidth=1.5)
    ax.add_patch(sensor_box)
    ard_ref = st.session_state.get("arduino")
    pir_active = ard_ref and getattr(ard_ref, "sensor_triggered", False)
    sensor_label = "[!] MOTION" if pir_active else "SENSOR"
    sensor_color = '#ff6633' if pir_active else '#3a7bd5'
    ax.text(1.0, 5.65, sensor_label, ha='center', va='center', color=sensor_color, fontsize=5)

    # ── Vehicles on road ──
    emojis  = {'car': 'CAR', 'truck': 'TRK', 'bus': 'BUS',
                'ambulance': 'AMB', 'police': 'POL'}
    vehicles  = analysis.get('vehicles', [])
    emergency = analysis.get('emergency_vehicles', [])

    # Slot positions: lane 1 (left, going up) and lane 2 (right, going down)
    lane1_slots = [(3.75, 11.5), (3.75, 9.5)]
    lane2_slots = [(6.25, 3.5),  (6.25, 1.5)]
    slot1, slot2 = 0, 0

    for v in (vehicles + emergency):
        vtype = v.get('type', 'car')
        label = emojis.get(vtype, 'CAR')
        v_color = '#ff4444' if vtype in ('ambulance','police') else '#00ccff'
        if slot1 < len(lane1_slots):
            x, y = lane1_slots[slot1]; slot1 += 1
        elif slot2 < len(lane2_slots):
            x, y = lane2_slots[slot2]; slot2 += 1
        else:
            break
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color=v_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', edgecolor=v_color, lw=1))

    # ── Pedestrians on crosswalk ──
    ped_slots = [(3.0, 7.3), (4.2, 6.8), (5.5, 7.4), (6.8, 7.0)]
    for i, _ in enumerate(analysis.get('pedestrians', [])[:4]):
        px, py = ped_slots[i]
        ax.text(px, py, 'PED', ha='center', va='center', fontsize=7,
                fontweight='bold', color='#ffaa00',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', edgecolor='#ffaa00', lw=1))

    ax.set_title('Intersection — Top Down', color='#8b9ab8', fontsize=8, pad=6)
    fig.tight_layout(pad=0.3)
    return fig

# ── Background analysis ──────────────────────────────────────────────────────
def _run_analysis(frame_bytes):
    with state.lock:
        state.api_calls_made += 1
    try:
        if st.session_state.get("use_mock_analysis", False):
            # Simulate a slight delay then return fake data
            time.sleep(0.5)
            import random
            result = {
                "vehicles": [{"type": "car", "box_2d": [400, 400, 600, 600]}] if random.random() > 0.5 else [],
                "emergency_vehicles": [],
                "pedestrians": [{"box_2d": [700, 300, 800, 400], "crossing": True}] if random.random() > 0.5 else [],
                "traffic_density": random.choice(["low", "medium", "high"]),
                "recommended_action": "Mock mode active",
                "emergency_priority": False
            }
        else:
            result = analyze_frame(frame_bytes)
            
        with state.lock:
            state.last_analysis     = result
            state.next_allowed_call = time.time() + _CALL_INTERVAL
            state.analysis_version += 1
    except Exception as e:
        err = str(e)
        print(f"Gemini analysis error: {e}")
        backoff = _CALL_INTERVAL
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            # Detect DAILY quota exhaustion vs per-minute rate limit
            if "PerDay" in err or "per_day" in err.lower():
                print("  → DAILY quota exhausted — stopping API calls")
                with state.lock:
                    state.daily_quota_hit = True
                    state.next_allowed_call = float('inf')  # never retry
                return
            # Per-minute limit: parse retry delay
            match = _re.search(r"retryDelay.*?(\d+)", err)
            backoff = float(match.group(1)) + 5 if match else _429_BACKOFF
            print(f"  → per-minute rate limit, backing off {backoff:.0f}s")
        with state.lock:
            state.next_allowed_call = time.time() + backoff
    finally:
        with state.lock:
            state.analyzing = False

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    state.camera_active = True          # mark camera as live
    with state.lock:
        analysis = dict(state.last_analysis)
        ready = (time.time() >= state.next_allowed_call
                 and not state.analyzing
                 and not state.daily_quota_hit)
    if ready:
        resized = cv2.resize(img, (960, 540))
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with state.lock:
            state.analyzing = True
        state.executor.submit(_run_analysis, buffer.tobytes())
    _draw_boxes(img, analysis)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── UI Layout ────────────────────────────────────────────────────────────────
cam_col, map_col = st.columns([3, 2])

with cam_col:
    st.markdown('<p class="section-label">📷 Live Camera</p>', unsafe_allow_html=True)
    webrtc_streamer(key="traffic-cam", mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_frame_callback)

with state.lock:
    analysis = dict(state.last_analysis)

# ── Merge Arduino PIR sensor into analysis ────────────────────────────────────
ard = st.session_state.arduino
if ard and ard.connected and ard.sensor_triggered:
    # PIR fired — inject a synthetic pedestrian so signal logic reacts
    if not analysis.get("pedestrians"):
        analysis = dict(analysis)
        analysis["pedestrians"] = [{"box_2d": [], "crossing": True, "source": "pir"}]

signal = st.session_state.controller.update(analysis, {}) if analysis else {
    "action": "idle", "light_state": "green", "walk_sign": False,
    "message": "⚠️ Daily API quota exhausted (20/20). Restart tomorrow."
               if state.daily_quota_hit
               else ("Camera active — waiting for first analysis..."
                     if state.camera_active
                     else "Waiting for camera...")
}

# ── Send WALK / STOP to Arduino ───────────────────────────────────────────────
if ard and ard.connected:
    ard.send("WALK" if signal.get("walk_sign") else "STOP")

with map_col:
    st.markdown('<p class="section-label">🗺️ Intersection Map</p>', unsafe_allow_html=True)
    fig = _draw_map(analysis, signal)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)

# ── Auto voice alerts ────────────────────────────────────────────────────────
if analysis:
    has_ambulance = any(e.get("type") == "ambulance"
                        for e in analysis.get("emergency_vehicles", []))
    has_cars = bool(analysis.get("vehicles"))
    has_peds = bool(analysis.get("pedestrians"))
    action   = signal.get("action")

    alert_state = f"{action}_{has_peds}_{has_cars}_{has_ambulance}"

    if alert_state != st.session_state.last_alert_state:
        st.session_state.last_alert_state = alert_state

        if has_ambulance:
            threading.Thread(target=play_alert, args=("do_not_cross",), daemon=True).start()
        elif has_peds and not has_cars:
            def _walk_alerts():
                play_alert("pedestrians_on_road")
                play_alert("walk")
            threading.Thread(target=_walk_alerts, daemon=True).start()
        elif has_peds and has_cars:
            def _wait_alerts():
                play_alert("pedestrians_on_road")
                play_alert("wait")
            threading.Thread(target=_wait_alerts, daemon=True).start()

# ── Metrics ──────────────────────────────────────────────────────────────────
vehicles  = analysis.get("vehicles", [])
emergency = analysis.get("emergency_vehicles", [])

cars       = sum(1 for v in vehicles if v.get("type") == "car")
buses      = sum(1 for v in vehicles if v.get("type") == "bus")
trucks     = sum(1 for v in vehicles if v.get("type") == "truck")
police     = sum(1 for e in emergency if e.get("type") == "police")
ambulances = sum(1 for e in emergency if e.get("type") == "ambulance")
peds       = len(analysis.get("pedestrians", []))

st.markdown('<p class="section-label">🚗 Vehicles detected</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("🚗 Cars", cars)
c2.metric("🚌 Buses", buses)
c3.metric("🚛 Trucks", trucks)

st.markdown('<p class="section-label">🚨 Priority</p>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
c4.metric("🚶 Pedestrians", peds)
c5.metric("🚔 Police", police)
c6.metric("🚑 Ambulances", ambulances)

# ── Signal status ─────────────────────────────────────────────────────────────
density = analysis.get("traffic_density", "") if analysis else ""
density_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(density, "⚪")
walk_icon    = "🟢 WALK" if signal.get("walk_sign") else "🔴 DON'T WALK"

st.markdown(f"**{density_icon} Signal:** {signal.get('message', '')} &nbsp;&nbsp; **Walk sign:** {walk_icon}")

# ── API quota info ──────────────────────────────────────────────────────────────
with state.lock:
    _calls = state.api_calls_made
    _quota_hit = state.daily_quota_hit
if _quota_hit:
    st.error(f"🚫 Daily API quota exhausted (20 RPD). Used {_calls} calls this session. Quota resets tomorrow.")
else:
    _secs_left = max(0, state.next_allowed_call - time.time())
    if _secs_left > 0 and _calls > 0:
        st.info(f"⏳ Next API call in {_secs_left:.0f}s  ·  {_calls} calls this session  ·  Limit: 20/day")
    elif _calls > 0:
        st.success(f"✅ API ready  ·  {_calls} calls this session  ·  Limit: 20/day")

if st.button("📢 Announce Status"):
    threading.Thread(target=play_alert, args=("status", signal.get("message", "")),
                     daemon=True).start()


