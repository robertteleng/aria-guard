# Jetson Orin Nano

What was verified on 2026-09-17: the replay benchmark on a Jetson Orin Nano
Super (JetPack 6.2, L4T R36.5.2, TensorRT 10.3, MAXN_SUPER power mode). Live
streaming from the glasses on the Jetson with the native aarch64
`projectaria-client-sdk` 2.5.0 has **not** been validated yet.

## Environment

The Ultralytics JetPack 6 container provides PyTorch, TensorRT and OpenCV built
for the device; only the Aria tools are added, without touching its numpy:

```bash
mkdir -p ~/bench/pydeps
docker run --rm --runtime nvidia -u $(id -u):$(id -g) -e HOME=/tmp -v ~/bench:/bench \
    ultralytics/ultralytics:8.4.14-jetson-jetpack6 \
    pip install --no-deps --target /bench/pydeps projectaria-tools==2.3.0 easydict
```

Run as your user (`-u`) so the files it creates stay yours, and set
`--hostname` so replay records carry the machine name instead of the container id.

## Engines

Build every engine on the Jetson; engines from another GPU or TensorRT version
do not load. The ONNX files can come from any machine.

```bash
python3 scripts/build_trt_engine.py models/gaze.onnx models/gaze.engine
python3 scripts/build_trt_engine.py models/depth_anything_v2_vits.onnx \
    models/depth_anything_v2_vits.engine --shape pixel_values:1x3x518x518
yolo export model=models/yolo26n_nav.pt format=engine half=True imgsz=640
```

## Replay benchmark

```bash
docker run --rm --runtime nvidia --ipc=host --hostname jetson-orin -u $(id -u):$(id -g) \
    -e HOME=/tmp -e PYTHON=python3 -e JETSON_POWER_MODE=MAXN_SUPER \
    -e PYTHONPATH=/bench/pydeps:/bench/projectaria_eyetracking \
    -v ~/bench:/bench -v ~/Datasets/aria/ritw:/data:ro -w /bench/aria-guard \
    ultralytics/ultralytics:8.4.14-jetson-jetpack6 \
    bash -c "ARIA_DEPTH_ASYNC=1 PACES=realtime TAG=async ./scripts/replay_benchmark.sh /data benchmarks/replay"
```

## What limits it (measured)

- **CPU, not GPU.** Decoding 1408x1408 JPEG takes ~39 ms per frame on the CPU.
- **`import ultralytics` makes OpenCV single-threaded.** A 1408->640 resize goes
  from 1.2 to 6.6 ms. The detector restores the threads (`ARIA_CV2_THREADS`).
- **GPU frequency scaling.** Without `sudo jetson_clocks`, an idle GPU between
  frames runs the YOLO engine in 11.3 ms instead of 6.7 ms and depth in 71 ms
  instead of 43 ms.
- **Depth is the slowest model.** `ARIA_DEPTH_ASYNC=1` runs it on its own thread
  so no frame waits for it.
