#!/usr/bin/env python3
"""
Test CUDA compatibility between NeMo TTS and PyTorch models.
Run this before the full system to verify no crashes.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

# Fix /tmp full issue - NeMo uses tempfile
_tmp_dir = Path.home() / "tmp"
_tmp_dir.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_tmp_dir)
os.environ["TEMP"] = str(_tmp_dir)
os.environ["TMP"] = str(_tmp_dir)
tempfile.tempdir = str(_tmp_dir)

def test_cuda_basic():
    """Test basic CUDA operations."""
    print("\n[1/4] Testing basic CUDA...")
    import torch

    if not torch.cuda.is_available():
        print("  CUDA not available!")
        return False

    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Simple tensor ops
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("  Basic CUDA: OK")
    return True


def test_nemo_load():
    """Test NeMo TTS loading."""
    print("\n[2/4] Loading NeMo TTS...")
    try:
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel

        spec_gen = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
        vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan")

        spec_gen = spec_gen.cuda().eval()
        vocoder = vocoder.cuda().eval()

        print("  NeMo TTS loaded on CUDA: OK")
        return spec_gen, vocoder
    except Exception as e:
        print(f"  NeMo load failed: {e}")
        return None, None


def test_nemo_generate(spec_gen, vocoder):
    """Test NeMo TTS generation."""
    print("\n[3/4] Testing NeMo generation...")
    import torch

    try:
        with torch.no_grad():
            parsed = spec_gen.parse("Testing one two three")
            spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        audio_np = audio.squeeze().cpu().numpy()
        print(f"  Generated {len(audio_np)} samples: OK")
        return True
    except Exception as e:
        print(f"  NeMo generation failed: {e}")
        return False


def test_sequential_cuda(spec_gen, vocoder):
    """Test sequential CUDA ops (simulating detector + TTS)."""
    print("\n[4/4] Testing sequential CUDA (detector simulation + TTS)...")
    import torch

    try:
        for i in range(5):
            # Simulate detector CUDA ops
            with torch.cuda.stream(torch.cuda.Stream()):
                x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.float16)
                # Simulate conv operations
                conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda().half()
                y = conv(x)
                torch.cuda.synchronize()

            # Now TTS (sequential, same thread)
            with torch.no_grad():
                parsed = spec_gen.parse(f"Test {i}")
                spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
                audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
                audio_np = audio.squeeze().cpu().numpy()

            print(f"  Iteration {i+1}/5: OK ({len(audio_np)} samples)")

        print("  Sequential CUDA: OK")
        return True
    except Exception as e:
        print(f"  Sequential CUDA failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("CUDA Compatibility Test: NeMo TTS + PyTorch")
    print("=" * 50)

    # Test 1: Basic CUDA
    if not test_cuda_basic():
        sys.exit(1)

    # Test 2: Load NeMo
    spec_gen, vocoder = test_nemo_load()
    if spec_gen is None:
        sys.exit(1)

    # Test 3: Generate audio
    if not test_nemo_generate(spec_gen, vocoder):
        sys.exit(1)

    # Test 4: Sequential operations
    if not test_sequential_cuda(spec_gen, vocoder):
        sys.exit(1)

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
