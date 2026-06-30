"""
Flask server con MJPEG streaming para ARIA Guard.

Solo rutas HTTP — la logica de procesamiento esta en pipeline.py.
"""
import threading
import time

import cv2
from flask import Flask, Response, render_template, jsonify, request

import logging

app = Flask(__name__)

# Disable Flask request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Shared state (populated by pipeline.py via process_loop)
state = {
    "frame_lock": threading.Lock(),
    "stats_lock": threading.Lock(),
    "current_frame": None,
    "current_depth": None,
    "current_eye": None,
    "current_detections": [],
    "current_gaze": None,
    "system_stats": {
        "server_fps": 0,
        "detector_fps": 0,
        "observer_fps": 0,
        "latency_ms": 0,
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "input_queue_size": 0,
        "output_queue_size": 0,
        "uptime_sec": 0,
    },
}

# TurboJPEG for faster encoding
# CRITICAL: NO cv2.cuda en main process — CUDA_VISIBLE_DEVICES="" para Aria SDK/FastDDS
_TURBOJPEG = None
try:
    from turbojpeg import TurboJPEG
    _TURBOJPEG = TurboJPEG()
    print("[SERVER] TurboJPEG habilitado (encoding rapido)")
except Exception:
    pass


def generate_frames(feed_type="rgb"):
    """Generator para MJPEG streaming con encoding optimizado.

    Rate-limited a ~12 FPS y solo codifica frames nuevos: sin esto, cada
    stream del navegador entra en un busy-loop de copy+resize+JPEG que
    compite por el GIL con process_loop y hunde el pipeline a ~2 FPS.
    """
    MIN_INTERVAL = 1.0 / 12  # cap ~12 FPS por stream (la fuente da ~10)
    last_id = None
    while True:
        t_iter = time.time()
        with state["frame_lock"]:
            if feed_type == "rgb" and state["current_frame"] is not None:
                src = state["current_frame"]
            elif feed_type == "depth" and state["current_depth"] is not None:
                src = state["current_depth"]
            elif feed_type == "eye" and state["current_eye"] is not None:
                src = state["current_eye"]
            elif feed_type in ("slam1", "slam2"):
                src = state.get(f"current_{feed_type}")
            else:
                src = None
            if src is None or id(src) == last_id:
                frame = None  # nada nuevo que codificar
            else:
                last_id = id(src)
                frame = src.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # CPU resize preserving aspect ratio (max 936p height, ~30% more than 720p)
        h, w = frame.shape[:2]
        if h > 936:
            scale = 936 / h
            frame = cv2.resize(frame, (int(w * scale), 936))

        if _TURBOJPEG:
            buffer = _TURBOJPEG.encode(frame, quality=75)
        else:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            buffer = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')

        # Rate limit: cede el GIL el tiempo restante del intervalo
        remaining = MIN_INTERVAL - (time.time() - t_iter)
        if remaining > 0:
            time.sleep(remaining)


def generate_dashboard():
    """MJPEG of the FULL dashboard — all panels tiled into ONE frame.

    Lets `ffmpeg -f mpjpeg -i .../dashboard_feed` record the whole dashboard
    (RGB+detections, depth, eye, SLAM, status) in a single synced video, instead
    of only the RGB feed. Same ~12 FPS rate-limit as the per-panel feeds.
    """
    from src.output.dashboard import compose_dashboard_frame
    MIN_INTERVAL = 1.0 / 12
    while True:
        t_iter = time.time()
        with state["frame_lock"]:
            panels = {
                "rgb": state.get("current_frame"),
                "depth": state.get("current_depth"),
                "eye": state.get("current_eye"),
                "slam1": state.get("current_slam1"),
                "slam2": state.get("current_slam2"),
            }
            panels = {k: (v.copy() if v is not None else None)
                      for k, v in panels.items()}
        if not any(p is not None for p in panels.values()):
            time.sleep(0.05)
            continue
        with state["stats_lock"]:
            st = dict(state.get("system_stats") or {})
        status_lines = [
            f"FPS det: {float(st.get('detector_fps', 0) or 0):.1f}",
            f"Latencia: {float(st.get('latency_ms', 0) or 0):.0f} ms",
            f"Uptime: {float(st.get('uptime_sec', 0) or 0):.0f} s",
        ]
        composite = compose_dashboard_frame(panels, status_lines)
        if _TURBOJPEG:
            buffer = _TURBOJPEG.encode(composite, quality=75)
        else:
            _, buf = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 75])
            buffer = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')
        remaining = MIN_INTERVAL - (time.time() - t_iter)
        if remaining > 0:
            time.sleep(remaining)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames("rgb"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/depth_feed')
def depth_feed():
    return Response(generate_frames("depth"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/eye_feed')
def eye_feed():
    return Response(generate_frames("eye"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/slam1_feed')
def slam1_feed():
    return Response(generate_frames("slam1"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/slam2_feed')
def slam2_feed():
    return Response(generate_frames("slam2"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dashboard_feed')
def dashboard_feed():
    """Full dashboard (all panels tiled) as one MJPEG stream — for recording."""
    return Response(generate_dashboard(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sensors')
def sensors():
    """IMU/magnetometer/barometer snapshot (aria:bridge source only)."""
    return jsonify(state.get("sensors") or {})


@app.route('/status')
def status():
    with state["frame_lock"]:
        dets = [{
            'name': d.name,
            'zone': d.zone,
            'distance': d.distance,
            'is_gazed': getattr(d, 'is_gazed', False),
            'traffic_light_state': getattr(d, 'traffic_light_state', None),
            'threat_level': getattr(d, 'threat_level', 'NONE'),
            'collision_risk': round(getattr(d, 'collision_risk', 0.0), 3),
        } for d in state["current_detections"]] if state["current_detections"] else []
    audio_ref = state.get("audio_ref")
    audio_stats = audio_ref.get_stats() if audio_ref is not None else None
    return jsonify({
        'fps': state["system_stats"]["server_fps"],
        'detections': dets,
        'gaze': list(state["current_gaze"]) if state["current_gaze"] else None,
        'audio': audio_stats,
    })


# NOTE: /tts/test and /audio/sim are dashboard control endpoints with NO auth.
# They assume a trusted local/LAN network (the dashboard is local-only). Do not
# expose them to the public internet. A minimal cooldown guards /tts/test against
# accidental synthesis floods (each NeMo utterance is GPU-expensive).
_SIM_ZONES = {"left", "center", "right"}
_SIM_DISTANCES = {"very_close", "close", "medium", "far", "unknown"}
_SIM_LEVELS = {"DANGER", "WARNING", "ATTENTION"}

_tts_test_lock = threading.Lock()
_last_tts_test = {"t": 0.0}
_TTS_TEST_COOLDOWN = 0.5  # s — anti-spam for on-demand synthesis


@app.route('/tts/test', methods=['POST'])
def tts_test():
    """Speak an arbitrary phrase on demand (dashboard 'probar voz' button).

    Doubles as the by-ear voice A/B tool and the seed for aria-scene's
    dynamic (VLM) text path.
    """
    audio_ref = state.get("audio_ref")
    if audio_ref is None:
        return jsonify({"ok": False, "error": "audio not ready"}), 503
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "peligro izquierda").strip()
    if not text or len(text) > 200:
        return jsonify({"ok": False, "error": "text must be 1..200 chars"}), 400

    # Anti-spam: drop on-demand requests within the cooldown window.
    now = time.time()
    with _tts_test_lock:
        if now - _last_tts_test["t"] < _TTS_TEST_COOLDOWN:
            return jsonify({"ok": False, "error": "rate limited"}), 429
        _last_tts_test["t"] = now

    audio_ref.speak(text, force=True, detected_ts=now)
    return jsonify({"ok": True, "text": text})


@app.route('/audio/sim', methods=['POST'])
def audio_sim():
    """Trigger an exact beep scenario without standing in front of the camera.

    The methodology's scenario matrix: pick (level x zone x distance) and
    confirm dashboard <-> ear correspondence.
    """
    audio_ref = state.get("audio_ref")
    if audio_ref is None:
        return jsonify({"ok": False, "error": "audio not ready"}), 503
    data = request.get_json(silent=True) or {}
    zone = data.get("zone", "center")
    distance = data.get("distance", "medium")
    threat_level = data.get("threat_level", "WARNING")
    if zone not in _SIM_ZONES or distance not in _SIM_DISTANCES \
            or threat_level not in _SIM_LEVELS:
        return jsonify({
            "ok": False,
            "error": "invalid params",
            "allowed": {"zone": sorted(_SIM_ZONES),
                        "distance": sorted(_SIM_DISTANCES),
                        "threat_level": sorted(_SIM_LEVELS)},
        }), 400
    audio_ref.play_spatial_beep(
        zone=zone, distance=distance, threat_level=threat_level,
        detected_ts=time.time(),
    )
    return jsonify({"ok": True, "zone": zone, "distance": distance,
                    "threat_level": threat_level})


@app.route('/stats')
def stats_endpoint():
    """Endpoint con estadisticas detalladas del sistema."""
    vram_used = 0
    vram_total = 0
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            vram_used = int(parts[0].strip())
            vram_total = int(parts[1].strip())
    except Exception:
        pass

    if vram_total == 0:
        # Jetson/Tegra: no nvidia-smi; GPU shares unified memory with the
        # system — report it from /proc/meminfo (real numbers, not 0/0)
        try:
            mem = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    mem[k] = int(v.strip().split()[0])  # kB
            vram_total = mem.get('MemTotal', 0) // 1024
            vram_used = (mem.get('MemTotal', 0) - mem.get('MemAvailable', 0)) // 1024
        except Exception:
            pass

    with state["stats_lock"]:
        stats_copy = state["system_stats"].copy()
        stats_copy["vram_used_mb"] = vram_used
        stats_copy["vram_total_mb"] = vram_total

    import os
    stats_copy["profile"] = os.environ.get("ARIA_PROFILE", "")
    return jsonify(stats_copy)
