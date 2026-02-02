"""Server MJPEG simple para streaming de video con audio feedback."""
import cv2
import time
import threading
from flask import Flask, Response, render_template

from observer import MockObserver, AriaDemoObserver
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


def process_loop(source: str, enable_audio: bool = True):
    """Loop de procesamiento en background."""
    global current_frame, current_depth, fps, current_detections, current_gaze

    print(f"[SERVER] Iniciando con fuente: {source}")

    if source == "aria":
        print("[SERVER] Conectando con gafas Aria...")
        observer = AriaDemoObserver()
    elif source == "webcam":
        observer = MockObserver(source="webcam")
    else:
        observer = MockObserver(source="video", video_path=source)
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

    # Opciones: "aria", "webcam", o ruta a video
    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    print(f"Uso: python main.py [aria|webcam|video.mp4]")
    print(f"  - aria: Conectar con gafas Meta Aria")
    print(f"  - webcam: Usar webcam local")
    print(f"  - video.mp4: Usar archivo de video")
    print()

    # Iniciar procesamiento en background
    thread = threading.Thread(target=process_loop, args=(source,), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
