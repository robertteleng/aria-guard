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
    """Generator para MJPEG streaming con encoding optimizado."""
    while True:
        with state["frame_lock"]:
            if feed_type == "rgb" and state["current_frame"] is not None:
                frame = state["current_frame"].copy()
            elif feed_type == "depth" and state["current_depth"] is not None:
                frame = state["current_depth"].copy()
            elif feed_type == "eye" and state["current_eye"] is not None:
                frame = state["current_eye"].copy()
            else:
                time.sleep(0.01)
                continue

        # CPU resize preserving aspect ratio (max 720p height)
        h, w = frame.shape[:2]
        if h > 720:
            scale = 720 / h
            frame = cv2.resize(frame, (int(w * scale), 720))

        if _TURBOJPEG:
            buffer = _TURBOJPEG.encode(frame, quality=75)
        else:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            buffer = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')


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
