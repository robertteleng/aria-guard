"""Server MJPEG simple para streaming de video con audio feedback.

Architecture (aria-nav pattern):
    Main Process (NO CUDA)     DetectorProcess (spawn)     TTSProcess (spawn)
    - Aria SDK                 - YOLO                      - NeMo TTS
    - Flask                    - Depth Anything
    - Dashboard                - Eye Gaze
    - AudioFeedback wrapper
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import time
import threading
from flask import Flask, Response, render_template

# NO CUDA imports here - main process must be CUDA-free for Aria SDK compatibility
from src.core import Dashboard, DetectorProcess
from src.core import MockObserver, AriaDemoObserver, AriaDatasetObserver, RealSenseObserver
from src.core.alert_engine import AlertDecisionEngine
from src.core.tracker import SimpleTracker

# AudioFeedback is safe - TTS runs in separate process
from src.core.audio import AudioFeedback

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

# Stats globales
stats_lock = threading.Lock()
system_stats = {
    "server_fps": 0,
    "detector_fps": 0,
    "observer_fps": 0,
    "latency_ms": 0,
    "vram_used_mb": 0,
    "vram_total_mb": 0,
    "input_queue_size": 0,
    "output_queue_size": 0,
    "uptime_sec": 0,
}


# Check for OpenCV CUDA and turbojpeg availability
_OPENCV_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
_TURBOJPEG = None
try:
    from turbojpeg import TurboJPEG
    _TURBOJPEG = TurboJPEG()
    print("[SERVER] TurboJPEG habilitado (encoding rápido)")
except ImportError:
    pass


def generate_frames(feed_type="rgb"):
    """Generator para MJPEG streaming con encoding optimizado."""
    global current_frame, current_depth

    # GPU frame resize buffer (reuse to avoid allocations)
    gpu_frame = cv2.cuda_GpuMat() if _OPENCV_CUDA else None

    while True:
        with frame_lock:
            if feed_type == "rgb" and current_frame is not None:
                frame = current_frame.copy()
            elif feed_type == "depth" and current_depth is not None:
                frame = current_depth.copy()
            else:
                time.sleep(0.01)
                continue

        # Resize on GPU if frame is large (reduces CPU JPEG encoding load)
        if _OPENCV_CUDA and frame.shape[0] > 720:
            gpu_frame.upload(frame)
            gpu_small = cv2.cuda.resize(gpu_frame, (1280, 720))
            frame = gpu_small.download()

        # Encode JPEG (use TurboJPEG if available, ~2x faster)
        if _TURBOJPEG:
            buffer = _TURBOJPEG.encode(frame, quality=75)
        else:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            buffer = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')


def process_loop(source: str, mode: str = "all", enable_audio: bool = True):
    """Loop de procesamiento en background."""
    global current_frame, current_depth, fps, current_detections, current_gaze, system_stats

    print(f"[SERVER] Iniciando con fuente: {source}, modo: {mode}")

    # Start CUDA in separate process FIRST (before any Aria SDK)
    print("[SERVER] Iniciando DetectorProcess (CUDA en proceso separado)...")
    detector = DetectorProcess(mode=mode, enable_depth=True)
    if not detector.start(timeout=60):
        print("[SERVER] ✗ Failed to start DetectorProcess")
        return

    # Non-CUDA components in main process
    dashboard = Dashboard()
    audio = AudioFeedback(enabled=enable_audio, use_nemo=enable_audio)
    alert_engine = AlertDecisionEngine()
    tracker = SimpleTracker()
    print("[SERVER] ✓ Componentes inicializados")

    # NOW create observer (Aria SDK in main process - safe because no CUDA here)
    use_precomputed_gaze = False
    if source.startswith("dataset:"):
        parts = source.split(":", 2)
        vrs_path = parts[1]
        gaze_csv = parts[2] if len(parts) > 2 and parts[2] else None
        print(f"[SERVER] Cargando Aria Dataset: {vrs_path}")
        observer = AriaDatasetObserver(vrs_path, gaze_csv, target_fps=10.0)
        use_precomputed_gaze = gaze_csv is not None
    elif source == "aria" or source == "aria:usb":
        print("[SERVER] Conectando con Aria (USB)...")
        observer = AriaDemoObserver(interface="usb")
    elif source.startswith("aria:wifi"):
        parts = source.split(":")
        ip = parts[2] if len(parts) > 2 else None
        print(f"[SERVER] Conectando con Aria (WiFi{': ' + ip if ip else ''})...")
        observer = AriaDemoObserver(interface="wifi", ip_address=ip)
    elif source == "webcam":
        observer = MockObserver(source="webcam")
    elif source == "realsense":
        print("[SERVER] Conectando con Intel RealSense D435...")
        observer = RealSenseObserver()
    else:
        observer = MockObserver(source="video", video_path=source)

    print("[SERVER] ✓ Observer listo")
    print("[SERVER] Iniciando procesamiento...")

    frame_count = 0
    start_time = time.time()

    # Check if observer provides hardware depth (RealSense D435)
    has_hardware_depth = hasattr(observer, 'get_depth')
    if has_hardware_depth:
        print("[SERVER] Hardware depth disponible (RealSense) - desactivando modelo de depth IA")
        print("[SERVER] Gaze no disponible (RealSense no tiene eye tracking)")

    while True:
        # Get frame from observer
        rgb = observer.get_frame("rgb")
        if rgb is None:
            time.sleep(0.01)
            continue

        # Eye tracking solo disponible con Aria (RealSense no tiene)
        eye_frame = None if has_hardware_depth else observer.get_frame("eye")

        # Get hardware depth if available (RealSense D435)
        hardware_depth = observer.get_depth() if has_hardware_depth else None

        # Send frame to DetectorProcess
        detector.send_frame(rgb, eye_frame, hardware_depth)

        # Get results from DetectorProcess (non-blocking)
        result = detector.get_result()

        detections = []
        depth_map = None
        gaze_point = None

        if result:
            detections = result.get("detections", [])
            depth_map = result.get("depth")
            gaze_point = result.get("gaze")

        # Check for precomputed gaze from dataset
        if use_precomputed_gaze and hasattr(observer, 'get_precomputed_gaze'):
            gaze_data = observer.get_precomputed_gaze()
            if gaze_data:
                import math
                yaw, pitch, depth_val = gaze_data
                fov_h = math.pi * 2 / 3
                fov_v = math.pi * 2 / 3
                gaze_x = max(0, min(1, 0.5 + yaw / fov_h))
                gaze_y = max(0, min(1, 0.5 - pitch / fov_v))
                gaze_point = (gaze_x, gaze_y)

        # Update tracker (gaze info is already in detections)
        tracked = tracker.update(detections)

        # Audio feedback via decision engine
        if tracked:
            vehicle_alert, other_alert = alert_engine.decide(tracker)

            if vehicle_alert and vehicle_alert.should_alert:
                obj = vehicle_alert.object
                audio.alert_danger(
                    object_name=obj.name,
                    zone=obj.zone,
                    distance=obj.distance,
                    user_looking=obj.is_gazed,
                    force_tts=True
                )
            elif other_alert and other_alert.should_alert:
                obj = other_alert.object
                audio.alert_danger(
                    object_name=obj.name,
                    zone=obj.zone,
                    distance=obj.distance,
                    user_looking=obj.is_gazed
                )

        # Render dashboard - ALWAYS update frame even without detections
        rgb_out, depth_out, _, _ = dashboard.render(
            rgb, depth_map, eye_frame, detections, gaze_point, fps
        )

        with frame_lock:
            current_frame = rgb_out
            current_depth = depth_out if depth_out is not None else rgb_out
            current_detections = detections
            current_gaze = gaze_point

        # FPS and stats
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Calculate latency from result timestamp
            latency = 0
            if result and "timestamp" in result:
                latency = (time.time() - result["timestamp"]) * 1000  # ms

            # Get queue sizes from detector
            input_q_size = 0
            output_q_size = 0
            if hasattr(detector, '_input_queue') and detector._input_queue:
                try:
                    input_q_size = detector._input_queue.qsize()
                except:
                    pass
            if hasattr(detector, '_output_queue') and detector._output_queue:
                try:
                    output_q_size = detector._output_queue.qsize()
                except:
                    pass

            # Get observer stats
            obs_fps = 0
            if hasattr(observer, 'get_stats'):
                obs_stats = observer.get_stats()
                obs_fps = obs_stats.get('fps', 0)

            # Update global stats
            with stats_lock:
                system_stats["server_fps"] = fps
                system_stats["latency_ms"] = latency
                system_stats["input_queue_size"] = input_q_size
                system_stats["output_queue_size"] = output_q_size
                system_stats["observer_fps"] = obs_fps
                system_stats["uptime_sec"] = elapsed

            print(f"[SERVER] Frame {frame_count}, {fps:.1f} FPS")


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


@app.route('/stats')
def stats():
    """Endpoint con estadísticas detalladas del sistema."""
    from flask import jsonify

    # Try to get VRAM usage (requires pynvml)
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

    with stats_lock:
        stats_copy = system_stats.copy()
        stats_copy["vram_used_mb"] = vram_used
        stats_copy["vram_total_mb"] = vram_total

    return jsonify(stats_copy)


if __name__ == '__main__':
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    print()
    print("╔══════════════════════════════════════╗")
    print("║          ARIA DEMO v1.0              ║")
    print("║   Visual Assistance System           ║")
    print("╚══════════════════════════════════════╝")
    print()
    print(f"  Fuente: {source}")
    print()

    while True:
        choice = input("  Modo [1=Indoor/2=Outdoor/3=All]: ").strip()
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
            print("  Opción no válida.")

    print(f"  → Modo: {mode.upper()}")
    print()

    thread = threading.Thread(target=process_loop, args=(source, mode), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
