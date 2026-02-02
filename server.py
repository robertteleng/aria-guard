"""Server MJPEG simple para streaming de video."""
import cv2
import time
import threading
from flask import Flask, Response, render_template_string

from observer import MockObserver
from detector import ParallelDetector
from dashboard import Dashboard

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
        body { background: #1a1a1a; color: white; font-family: sans-serif; margin: 20px; }
        h1 { color: #4a9eff; margin-bottom: 20px; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video-box img { max-width: 640px; border: 2px solid #333; border-radius: 4px; }
        .eye-box {
            width: 320px;
            height: 120px;
            background: #252525;
            border: 2px solid #333;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        .status-box {
            background: #252525;
            padding: 15px;
            border-radius: 8px;
            min-width: 280px;
            font-family: monospace;
            border: 2px solid #333;
        }
        .fps { font-size: 28px; color: #4aff4a; margin-bottom: 10px; }
        .section-title { color: #888; font-size: 12px; margin-top: 15px; margin-bottom: 5px; }
        .detection { padding: 5px 8px; margin: 2px 0; border-radius: 4px; background: #333; }
        .very_close { border-left: 3px solid #ff4a4a; }
        .close { border-left: 3px solid #ffa54a; }
        .medium { border-left: 3px solid #ffff4a; }
        .far { border-left: 3px solid #4aff4a; }
        .gaze-info { margin-top: 15px; padding: 10px; background: #333; border-radius: 4px; }
        .gaze-dot { display: inline-block; width: 10px; height: 10px; background: #ff4aff; border-radius: 50%; margin-right: 8px; }
    </style>
</head>
<body>
    <h1>🎯 ARIA Demo - Streaming</h1>
    <div class="container">
        <div class="video-box">
            <h3>RGB + Detecciones</h3>
            <img src="/video_feed" />
        </div>
        <div class="video-box">
            <h3>Profundidad</h3>
            <img src="/depth_feed" />
        </div>
        <div>
            <h3>Eye Tracking</h3>
            <div class="eye-box" id="eye-box">
                <span>Sin datos de eye tracking</span>
            </div>
            <div class="gaze-info" id="gaze-info" style="display:none;">
                <span class="gaze-dot"></span>
                <span id="gaze-text">Gaze: --</span>
            </div>
        </div>
        <div class="status-box">
            <div class="fps" id="fps">-- FPS</div>
            <div class="section-title">OBJETOS DETECTADOS</div>
            <p>Total: <span id="count">0</span></p>
            <div id="detections"></div>
        </div>
    </div>
    <script>
        setInterval(async () => {
            const res = await fetch('/status');
            const data = await res.json();
            document.getElementById('fps').textContent = data.fps.toFixed(1) + ' FPS';
            document.getElementById('count').textContent = data.detections.length;

            let html = '';
            data.detections.slice(0, 8).forEach(d => {
                html += `<div class="detection ${d.distance}">${d.name} - ${d.zone} (${d.distance})</div>`;
            });
            document.getElementById('detections').innerHTML = html;

            if (data.gaze) {
                document.getElementById('gaze-info').style.display = 'block';
                document.getElementById('gaze-text').textContent =
                    `Gaze: (${data.gaze[0].toFixed(2)}, ${data.gaze[1].toFixed(2)})`;
                document.getElementById('eye-box').innerHTML =
                    `<span style="color:#4aff4a;">Eye tracking activo</span>`;
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


def process_loop(video_path: str):
    """Loop de procesamiento en background."""
    global current_frame, current_depth, fps, current_detections

    print(f"[SERVER] Cargando video: {video_path}")
    observer = MockObserver(source="video", video_path=video_path)
    print("[SERVER] Cargando detector...")
    detector = ParallelDetector(enable_depth=True)
    dashboard = Dashboard()
    print("[SERVER] Iniciando procesamiento...")

    frame_count = 0
    start_time = time.time()

    while True:
        rgb = observer.get_frame("rgb")
        if rgb is None:
            time.sleep(0.01)
            continue

        detections, depth_map = detector.process(rgb)

        # Renderizar
        rgb_out, depth_out, _, _ = dashboard.render(
            rgb, depth_map, None, detections, None, fps
        )

        with frame_lock:
            current_frame = rgb_out
            current_depth = depth_out if depth_out is not None else rgb_out
            current_detections = detections

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
        dets = [{'name': d.name, 'zone': d.zone, 'distance': d.distance}
                for d in current_detections] if current_detections else []
    return jsonify({
        'fps': fps,
        'detections': dets,
        'gaze': current_gaze
    })


if __name__ == '__main__':
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

    # Iniciar procesamiento en background
    thread = threading.Thread(target=process_loop, args=(video_path,), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
