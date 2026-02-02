"""Server MJPEG simple para streaming de video con audio feedback."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import time
import threading
from flask import Flask, Response, render_template

from src.core import MockObserver, AriaDemoObserver, AriaDatasetObserver, ParallelDetector, Dashboard, AudioFeedback

app = Flask(__name__)

# Disable Flask request logging for cleaner output
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Estado global
frame_lock = threading.Lock()
current_frame = None
current_depth = None
fps = 0
current_detections = []
current_gaze = None


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


def process_loop(source: str, mode: str = "all", enable_audio: bool = True):
    """Loop de procesamiento en background."""
    global current_frame, current_depth, fps, current_detections, current_gaze

    print(f"[SERVER] Iniciando con fuente: {source}, modo: {mode}")

    # Seleccionar observer según fuente
    use_precomputed_gaze = False
    if source.startswith("dataset:"):
        # Parse dataset:vrs_path:gaze_csv
        parts = source.split(":", 2)
        vrs_path = parts[1]
        gaze_csv = parts[2] if len(parts) > 2 and parts[2] else None
        print(f"[SERVER] Cargando Aria Dataset: {vrs_path}")
        observer = AriaDatasetObserver(vrs_path, gaze_csv, target_fps=10.0)
        use_precomputed_gaze = gaze_csv is not None
    elif source == "aria":
        print("[SERVER] Conectando con gafas Aria...")
        observer = AriaDemoObserver()
    elif source == "webcam":
        observer = MockObserver(source="webcam")
    else:
        observer = MockObserver(source="video", video_path=source)
    print("[SERVER] Cargando detector...")
    detector = ParallelDetector(enable_depth=True, mode=mode)
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

        # Check for precomputed gaze from dataset
        precomputed_gaze = None
        if use_precomputed_gaze and hasattr(observer, 'get_precomputed_gaze'):
            gaze_data = observer.get_precomputed_gaze()
            if gaze_data:
                # Convert (yaw, pitch, depth) to normalized 2D coords
                # Aria RGB camera has ~120 degree FOV
                import math
                yaw, pitch, depth = gaze_data
                fov_h = math.pi * 2 / 3  # ~120 degrees
                fov_v = math.pi * 2 / 3
                gaze_x = 0.5 + yaw / fov_h
                gaze_y = 0.5 - pitch / fov_v
                # Clamp to [0, 1]
                gaze_x = max(0, min(1, gaze_x))
                gaze_y = max(0, min(1, gaze_y))
                precomputed_gaze = (gaze_x, gaze_y)

        # Process all in parallel (YOLO + Depth + Gaze)
        detections, depth_map, gaze_point = detector.process(rgb, eye_frame)

        # Use precomputed gaze if available
        if precomputed_gaze:
            gaze_point = precomputed_gaze

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
    return render_template('index.html')


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

    # Parsear source desde argumentos
    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    print()
    print("╔══════════════════════════════════════╗")
    print("║          ARIA DEMO v1.0              ║")
    print("║   Visual Assistance System           ║")
    print("╚══════════════════════════════════════╝")
    print()
    print(f"  Fuente: {source}")
    print()
    print("  Selecciona el modo de detección:")
    print()
    print("    [1] Indoor  - persona, silla, sofá, mesa, tv, puerta...")
    print("    [2] Outdoor - persona, coche, bici, moto, bus, semáforo...")
    print("    [3] All     - todas las clases (80 objetos)")
    print()

    while True:
        choice = input("  Modo [1/2/3]: ").strip()
        if choice == "1":
            mode = "indoor"
            break
        elif choice == "2":
            mode = "outdoor"
            break
        elif choice == "3":
            mode = "all"
            break
        else:
            print("  Opción no válida. Introduce 1, 2 o 3.")

    print()
    print(f"  → Modo seleccionado: {mode.upper()}")
    print()

    # Iniciar procesamiento en background
    thread = threading.Thread(target=process_loop, args=(source, mode), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
