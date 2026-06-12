"""
Pipeline de procesamiento: Observer -> Detector -> Tracker -> Alerts -> Dashboard.

Corre en un thread background, alimenta el servidor MJPEG con frames renderizados.
"""
import sys
import time
from pathlib import Path

import cv2

from src.input import MockObserver, AriaDemoObserver, AriaDatasetObserver, RealSenseObserver
from src.detection import DetectorProcess
from src.domain import SimpleTracker, AlertArbiter
from src.output import Dashboard, AudioFeedback

# AriaBridgeObserver for Jetson ARM64 (frames via ZMQ from FEX-Emu receiver)
try:
    sys.path.insert(0, str(Path.home() / "Projects" / "aria-arm64-bridge" / "src" / "bridge"))
    from aria_bridge_observer import AriaBridgeObserver
except ImportError:
    AriaBridgeObserver = None


def process_loop(source: str, mode: str = "all", enable_audio: bool = True, state: dict = None):
    """Loop de procesamiento en background.

    Args:
        source: Input source string (webcam, aria:usb, aria:wifi:IP, video path, etc.)
        mode: Detection mode (indoor, outdoor, all)
        enable_audio: Enable TTS audio alerts
        state: Shared state dict for server (frame_lock, current_frame, etc.)
    """
    print(f"[PIPELINE] Iniciando con fuente: {source}, modo: {mode}")

    # Create observer
    use_precomputed_gaze = False
    if source.startswith("dataset:"):
        parts = source.split(":", 2)
        vrs_path = parts[1]
        gaze_csv = parts[2] if len(parts) > 2 and parts[2] else None
        print(f"[PIPELINE] Cargando Aria Dataset: {vrs_path}")
        observer = AriaDatasetObserver(vrs_path, gaze_csv, target_fps=10.0)
        use_precomputed_gaze = gaze_csv is not None
    elif source.startswith("aria:bridge"):
        parts = source.split(":")
        endpoint = parts[2] if len(parts) > 2 else "tcp://127.0.0.1:5555"
        if AriaBridgeObserver is None:
            print("[PIPELINE] AriaBridgeObserver not available. Install aria-arm64-bridge.")
            return
        print(f"[PIPELINE] Conectando con Aria Bridge ({endpoint})...")
        observer = AriaBridgeObserver(zmq_endpoint=endpoint)
    elif source == "aria" or source == "aria:usb":
        print("[PIPELINE] Conectando con Aria (USB)...")
        observer = AriaDemoObserver(interface="usb", auto_subscribe=False, enable_slam=False)
    elif source.startswith("aria:wifi"):
        parts = source.split(":")
        ip = parts[2] if len(parts) > 2 else None
        print(f"[PIPELINE] Conectando con Aria (WiFi{': ' + ip if ip else ''})...")
        observer = AriaDemoObserver(interface="wifi", ip_address=ip, auto_subscribe=False, enable_slam=False)
    elif source == "webcam":
        observer = MockObserver(source="webcam")
    elif source == "realsense":
        print("[PIPELINE] Conectando con Intel RealSense D435...")
        observer = RealSenseObserver()
    else:
        observer = MockObserver(source="video", video_path=source)

    print("[PIPELINE] Observer listo")

    # Check if observer provides hardware depth (RealSense D435)
    has_hardware_depth = hasattr(observer, 'get_depth')
    if has_hardware_depth:
        print("[PIPELINE] Hardware depth disponible (RealSense) - desactivando modelo de depth IA")

    # Start CUDA in separate process
    is_aria = isinstance(observer, AriaDemoObserver) or (AriaBridgeObserver and isinstance(observer, AriaBridgeObserver))
    if is_aria:
        frame_shape = (1408, 1408, 3)
        print(f"[PIPELINE] Frame shape (Aria): {frame_shape}")
    else:
        test_frame = None
        for _ in range(30):
            test_frame = observer.get_frame("rgb")
            if test_frame is not None:
                break
            time.sleep(0.1)
        frame_shape = test_frame.shape if test_frame is not None else (720, 1280, 3)
        print(f"[PIPELINE] Frame shape: {frame_shape}")

    print("[PIPELINE] Iniciando DetectorProcess (CUDA en proceso separado)...")
    detector = DetectorProcess(mode=mode, enable_depth=True, has_hardware_depth=has_hardware_depth)
    if not detector.start(timeout=60, frame_shape=frame_shape):
        print("[PIPELINE] Failed to start DetectorProcess")
        if hasattr(observer, 'stop'):
            observer.stop()
        return

    # Non-CUDA components en main process — ANTES de resume_streaming
    print("[PIPELINE] Iniciando componentes...")
    dashboard = Dashboard()
    audio = AudioFeedback(enabled=enable_audio, use_nemo=enable_audio)
    alert_engine = AlertArbiter()
    tracker = SimpleTracker()
    print("[PIPELINE] Componentes inicializados")

    # Suscribir a DDS DESPUES de que TODOS los componentes esten listos
    if isinstance(observer, AriaDemoObserver):
        observer.resume_streaming()
        print("[PIPELINE] DDS suscrito")

    print("[PIPELINE] Iniciando procesamiento...")

    frame_count = 0
    start_time = time.time()
    fps = 0
    last_inferred_id = None

    try:
     while True:
        rgb = observer.get_frame("rgb")
        if rgb is None:
            time.sleep(0.01)
            continue

        eye_frame = None if has_hardware_depth else observer.get_frame("eye")
        hardware_depth = observer.get_depth() if has_hardware_depth else None

        # Check detector process health
        if hasattr(detector, '_process') and detector._process and not detector._process.is_alive():
            print(f"\n[PIPELINE] *** DetectorProcess MURIO en frame {frame_count} (exitcode={detector._process.exitcode}) ***")
            break

        # El input llega a ~10 FPS pero este bucle gira a 21+ it/s: re-inferir el
        # mismo frame duplica el trabajo GPU y la contencion del bus de memoria
        # unificada estrangula la recepcion DDS del receiver (bridge Exp 007).
        if id(rgb) != last_inferred_id:
            last_inferred_id = id(rgb)
            detector.send_frame(rgb, eye_frame, hardware_depth)
        else:
            time.sleep(0.005)
        result = detector.get_result()

        detections = []
        depth_map = None
        gaze_point = None

        if result:
            detections = result.get("detections", [])
            depth_map = result.get("depth")
            gaze_point = result.get("gaze")

        if has_hardware_depth:
            depth_visual = observer.get_depth_visual()
            if depth_visual is not None:
                depth_map = depth_visual

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

        # Update tracker
        frame_w = rgb.shape[1] if rgb is not None else 1280
        tracked = tracker.update(detections, frame_width=frame_w, fov_h=observer.fov_h)

        # Enrich detections with tracker info
        track_by_bbox = {t.bbox: t for t in tracked} if tracked else {}
        for det in detections:
            t = track_by_bbox.get(det.bbox)
            if t:
                det.threat_level = t.threat_level
                det.collision_risk = t.collision_risk

        # Audio feedback via 2-channel arbiter
        if tracked:
            channel_a, channel_b = alert_engine.decide(tracker)

            if channel_a and channel_a.should_alert:
                obj = channel_a.object
                audio.alert_danger(
                    object_name=obj.name,
                    zone=obj.zone,
                    distance=obj.distance,
                    user_looking=obj.is_gazed,
                    force_tts=channel_a.use_tts,
                    threat_level=channel_a.threat_level,
                )

            if channel_b and channel_b.should_alert:
                obj = channel_b.object
                if obj.name == "traffic light" and obj.traffic_light_state:
                    audio.alert_traffic_light(
                        state=obj.traffic_light_state,
                        zone=obj.zone,
                    )
                elif obj.name == "stop sign":
                    audio.alert_sign(
                        sign_name=obj.name,
                        zone=obj.zone,
                        distance=obj.distance,
                    )

        # Render dashboard
        rgb_out, depth_out, _, _ = dashboard.render(
            rgb, depth_map, eye_frame, detections, gaze_point, fps
        )

        if state is not None:
            # SLAM frames + sensors only exist with the aria:bridge source
            has_sensors = hasattr(observer, "get_sensors")
            slam1 = observer.get_frame("slam1") if has_sensors else None
            slam2 = observer.get_frame("slam2") if has_sensors else None
            with state["frame_lock"]:
                state["current_frame"] = rgb_out
                state["current_depth"] = depth_out if depth_out is not None else rgb_out
                state["current_eye"] = eye_frame
                state["current_detections"] = detections
                state["current_gaze"] = gaze_point
                if slam1 is not None:
                    state["current_slam1"] = slam1
                if slam2 is not None:
                    state["current_slam2"] = slam2
            if has_sensors:
                # plain dict swap — atomic enough for the /sensors endpoint
                state["sensors"] = observer.get_sensors()

        # FPS and stats
        frame_count += 1
        if result and "detector_fps" in result:
            fps = result["detector_fps"]

        if frame_count % 30 == 0:
            if frame_count == 30:
                print(f"[PIPELINE] Frame shape: {rgb.shape}")
            elapsed = time.time() - start_time

            latency = 0
            if result and "timestamp" in result:
                latency = (time.time() - result["timestamp"]) * 1000

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

            obs_fps = 0
            if hasattr(observer, 'get_stats'):
                obs_stats = observer.get_stats()
                fps_data = obs_stats.get('fps', 0)
                if isinstance(fps_data, dict):
                    obs_fps = fps_data.get('rgb', 0)
                else:
                    obs_fps = fps_data

            if state is not None:
                with state["stats_lock"]:
                    state["system_stats"]["server_fps"] = fps
                    state["system_stats"]["detector_fps"] = fps
                    state["system_stats"]["latency_ms"] = latency
                    state["system_stats"]["input_queue_size"] = input_q_size
                    state["system_stats"]["output_queue_size"] = output_q_size
                    state["system_stats"]["observer_fps"] = obs_fps
                    state["system_stats"]["uptime_sec"] = elapsed

            print(f"[PIPELINE] Frame {frame_count}, detector {fps:.1f} FPS")

        time.sleep(0.001)

    except Exception as e:
        import traceback
        print(f"\n[PIPELINE] *** CRASH en process_loop en frame {frame_count} ***")
        print(f"[PIPELINE] Excepcion: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        print("[PIPELINE] Cleanup: deteniendo componentes...")
        try:
            detector.stop()
        except Exception:
            pass
        try:
            audio.shutdown()
        except Exception:
            pass
        if hasattr(observer, 'stop'):
            try:
                observer.stop()
            except Exception:
                pass
        print("[PIPELINE] Cleanup completo")
