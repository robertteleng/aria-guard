"""
Flask server con MJPEG streaming para ARIA Guard.

Solo rutas HTTP — la logica de procesamiento esta en pipeline.py.
"""
import threading
import time

import cv2
from flask import Flask, Response, render_template, jsonify

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
    return jsonify({
        'fps': state["system_stats"]["server_fps"],
        'detections': dets,
        'gaze': list(state["current_gaze"]) if state["current_gaze"] else None
    })


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
    except:
        pass

    with state["stats_lock"]:
        stats_copy = state["system_stats"].copy()
        stats_copy["vram_used_mb"] = vram_used
        stats_copy["vram_total_mb"] = vram_total

    return jsonify(stats_copy)
