# Third-party components

aria-guard is licensed under the [GNU AGPL-3.0](LICENSE) because it links
Ultralytics YOLO (AGPL-3.0) and runs weights derived from it. The components
below keep their own licenses. **No model weights, voices or recordings are
included in this repository**: the scripts download or build them locally.

## Code and runtimes

| Component | Use | License |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO inference and TensorRT export | AGPL-3.0 |
| [projectaria_tools](https://github.com/facebookresearch/projectaria_tools) | VRS reading, calibration | Apache-2.0 |
| [projectaria_client_sdk](https://pypi.org/project/projectaria-client-sdk/) | Live streaming from the glasses | Meta Project Aria SDK license |
| [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking) | Eye-gaze model code | Apache-2.0 |
| [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) | Engine build and inference | NVIDIA proprietary |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | Optional TTS engine | Apache-2.0 |

## Models (downloaded or built locally)

| Model | Source | License | Notes |
|---|---|---|---|
| `yolo26n_nav` / `yolo26s_nav` | [vision-fine-tuning](https://github.com/robertteleng/vision-fine-tuning) | AGPL-3.0 | 24-class navigation detector fine-tuned from YOLO26 |
| Depth Anything V2 **Small** | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 | Only the Small checkpoint is Apache-2.0; Base and Large are CC BY-NC 4.0 (non-commercial) |
| Meta eye gaze `social_eyes_uncertainty_v1` | [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking) | Apache-2.0 | Weights fetched from the upstream repository |
| Piper voice `es_ES-davefx-medium` | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | Repository MIT; voice dataset CC0 | Fine-tuned from the `en_US-lessac` voice, whose training data (Blizzard 2013 Lessac) is licensed for **research purposes only**. Review before any commercial use |

## Data used for evaluation

| Dataset | Use | License |
|---|---|---|
| [Reading in the Wild](https://www.projectaria.com/datasets/) (Project Aria), including its MPS output | Replay benchmark and alert evaluation (6 outdoor walking recordings) | CC BY-NC 4.0. Recordings are not redistributed; metrics are published, and `docs/media/` contains rendered frames under the same CC BY-NC 4.0 terms |
