"""
Test rápido de Aria streaming para diagnosticar segfaults del SDK (heap corruption).

  uv run python scripts/test_aria_streaming.py                                  # sin fix
  MALLOC_CHECK_=0 uv run python scripts/test_aria_streaming.py                  # glibc sin chequeo
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 uv run python scripts/test_aria_streaming.py

scripts/test_aria_streaming.sh ejecuta todas las variantes.
"""
import os
import sys
import signal
import time

# Mostrar info de entorno
print(f"Python: {sys.version}")
print(f"MALLOC_CHECK_: {os.environ.get('MALLOC_CHECK_', 'not set')}")
print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'not set')}")

import aria.sdk as aria

print("\n--- Aria Streaming Test ---")

# Timeout de seguridad (30 segundos)
def timeout_handler(signum, frame):
    print("\n[TIMEOUT] 30s sin frames. Abortando.")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)


class AriaObserver:
    """Observer que cuenta frames recibidos."""
    def __init__(self):
        self.frame_count = 0
        self.first_frame_time = None
        self.start_time = time.time()

    def on_image_received(self, image, record):
        if self.frame_count == 0:
            self.first_frame_time = time.time()
            elapsed = self.first_frame_time - self.start_time
            print(f"[OK] Primer frame recibido! (en {elapsed:.2f}s)")

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.first_frame_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[OK] {self.frame_count} frames recibidos ({fps:.1f} FPS)")

        # Después de 100 frames, éxito
        if self.frame_count >= 100:
            elapsed = time.time() - self.first_frame_time
            fps = self.frame_count / elapsed
            print(f"\n[EXITO] 100 frames recibidos en {elapsed:.2f}s ({fps:.1f} FPS)")
            print("Streaming funciona correctamente!")
            os._exit(0)

    def on_imu_received(self, samples, imu_idx):
        pass


def main():
    # Configurar cliente (API igual que aria_process.py)
    device_client = aria.DeviceClient()

    print("Buscando dispositivo Aria...")
    device = device_client.connect()
    print("Conectado a dispositivo Aria")

    # Configurar streaming
    streaming_manager = device.streaming_manager
    config = aria.StreamingConfig()
    config.profile_name = "profile28"  # USB 30fps
    config.streaming_interface = aria.StreamingInterface.Usb
    config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = config

    print("Iniciando streaming...")
    print("(Si hay segfault, aparecerá aquí)")

    # Timeout 30s
    signal.alarm(30)

    streaming_manager.start_streaming()
    print("Streaming iniciado OK")

    # Observer para frames
    observer = AriaObserver()
    streaming_client = streaming_manager.streaming_client
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    print("Subscrito a RGB, esperando frames...")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\nInterrumpido. Frames recibidos: {observer.frame_count}")
    finally:
        streaming_manager.stop_streaming()
        streaming_client.unsubscribe()
        device_client.disconnect(device)
        print("Desconectado.")


if __name__ == "__main__":
    main()
