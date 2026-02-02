"""Server MJPEG simple para streaming de video con audio feedback."""
import cv2
import time
import threading
from flask import Flask, Response, render_template_string

from observer import MockObserver
from detector import ParallelDetector
from dashboard import Dashboard
from audio import AudioFeedback

app = Flask(__name__)

# Estado global
frame_lock = threading.Lock()
current_frame = None
current_depth = None
fps = 0
current_detections = []
current_gaze = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ARIA Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0a0a; color: white; font-family: sans-serif; height: 100vh; overflow: hidden; }
        .main-container { display: flex; height: calc(100vh - 120px); }
        .video-area { flex: 1; padding: 10px; display: flex; align-items: center; justify-content: center; }
        .video-area img { max-width: 100%; max-height: 100%; border: 2px solid #333; border-radius: 4px; }
        .panel {
            width: 280px;
            background: #151515;
            padding: 15px;
            border-left: 2px solid #333;
            overflow-y: auto;
        }
        .panel h2 { color: #4a9eff; font-size: 14px; margin-bottom: 15px; }
        .fps { font-size: 32px; color: #4aff4a; margin-bottom: 15px; font-weight: bold; }
        .section-title { color: #666; font-size: 11px; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase; }
        .detection { padding: 8px 10px; margin: 4px 0; border-radius: 4px; background: #222; font-size: 13px; }
        .very_close { border-left: 3px solid #ff4a4a; }
        .close { border-left: 3px solid #ffa54a; }
        .medium { border-left: 3px solid #ffff4a; }
        .far { border-left: 3px solid #4aff4a; }
        .gazed { background: #2a2a3a; }
        .gaze-row {
            height: 120px;
            background: #151515;
            border-top: 2px solid #333;
            display: flex;
            align-items: center;
            padding: 0 30px;
            gap: 40px;
        }
        .gaze-indicator {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .gaze-dot { width: 18px; height: 18px; background: #ff4aff; border-radius: 50%; }
        .gaze-coords { font-family: monospace; font-size: 18px; color: #ccc; }
        .eye-status { color: #666; font-size: 13px; }
        .legend { display: flex; gap: 15px; margin-left: auto; }
        .legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #888; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="video-area">
            <img src="/video_feed" />
        </div>
        <div class="panel">
            <h2>ARIA DEMO</h2>
            <div class="fps" id="fps">-- FPS</div>
            <div class="section-title">Objetos detectados (<span id="count">0</span>)</div>
            <div id="detections"></div>
        </div>
    </div>
    <div class="gaze-row">
        <div class="gaze-indicator">
            <div class="gaze-dot"></div>
            <span class="gaze-coords" id="gaze-text">Gaze: --</span>
        </div>
        <div class="eye-status" id="eye-status">Sin eye tracking</div>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#ff4a4a"></div>Muy cerca</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ffa54a"></div>Cerca</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ffff4a"></div>Medio</div>
            <div class="legend-item"><div class="legend-dot" style="background:#4aff4a"></div>Lejos</div>
        </div>
    </div>
    <script>
        setInterval(async () => {
            const res = await fetch('/status');
            const data = await res.json();
            document.getElementById('fps').textContent = data.fps.toFixed(1) + ' FPS';
            document.getElementById('count').textContent = data.detections.length;

            let html = '';
            data.detections.slice(0, 10).forEach(d => {
                const gazeClass = d.is_gazed ? ' gazed' : '';
                const gazeIcon = d.is_gazed ? '👁 ' : '';
                html += `<div class="detection ${d.distance}${gazeClass}">${gazeIcon}${d.name} - ${d.zone}</div>`;
            });
            document.getElementById('detections').innerHTML = html;

            if (data.gaze) {
                document.getElementById('gaze-text').textContent =
                    `Gaze: (${data.gaze[0].toFixed(2)}, ${data.gaze[1].toFixed(2)})`;
                document.getElementById('eye-status').textContent = 'Eye tracking activo';
                document.getElementById('eye-status').style.color = '#4aff4a';
            }
        }, 200);
    </script>
</body>
</html>
"""


def generate_frames(feed_type="rgb"):
    """Generator para MJPEG streaming."""
    global current_frame, current_depth
    while True:
        with frame_lock:
            if feed_type == "rgb" and current_frame is not None:
                frame = current_frame.copy()
            elif feed_type == "depth" and current_depth is not None:
                frame = current_depth.copy()
            else:
                time.sleep(0.01)
                continue

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def process_loop(video_path: str, enable_audio: bool = True):
    """Loop de procesamiento en background."""
    global current_frame, current_depth, fps, current_detections, current_gaze

    print(f"[SERVER] Cargando video: {video_path}")
    observer = MockObserver(source="video", video_path=video_path)
    print("[SERVER] Cargando detector...")
    detector = ParallelDetector(enable_depth=True)
    dashboard = Dashboard()
    audio = AudioFeedback(enabled=enable_audio)
    print("[SERVER] Iniciando procesamiento...")

    frame_count = 0
    start_time = time.time()

    while True:
        rgb = observer.get_frame("rgb")
        if rgb is None:
            time.sleep(0.01)
            continue

        # Get eye tracking frame (None for video/webcam, available with Aria)
        eye_frame = observer.get_frame("eye")

        # Process all in parallel (YOLO + Depth + Gaze)
        detections, depth_map, gaze_point = detector.process(rgb, eye_frame)

        # Audio feedback for dangers
        for det in detections:
            if det.distance in ("very_close", "close"):
                user_looking = getattr(det, 'is_gazed', False)
                audio.alert_danger(
                    object_name=det.name,
                    zone=det.zone,
                    distance=det.distance,
                    user_looking=user_looking
                )
                break  # Only alert for most dangerous object

        # Renderizar
        rgb_out, depth_out, _, _ = dashboard.render(
            rgb, depth_map, eye_frame, detections, gaze_point, fps
        )

        with frame_lock:
            current_frame = rgb_out
            current_depth = depth_out if depth_out is not None else rgb_out
            current_detections = detections
            current_gaze = gaze_point

        # FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[SERVER] Frame {frame_count}, {fps:.1f} FPS, {len(detections)} objetos")


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames("rgb"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/depth_feed')
def depth_feed():
    return Response(generate_frames("depth"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    from flask import jsonify
    with frame_lock:
        dets = [{
            'name': d.name,
            'zone': d.zone,
            'distance': d.distance,
            'is_gazed': getattr(d, 'is_gazed', False)
        } for d in current_detections] if current_detections else []
    return jsonify({
        'fps': fps,
        'detections': dets,
        'gaze': list(current_gaze) if current_gaze else None
    })


if __name__ == '__main__':
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

    # Iniciar procesamiento en background
    thread = threading.Thread(target=process_loop, args=(video_path,), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
