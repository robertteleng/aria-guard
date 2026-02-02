"""Benchmark sin display."""
import time
import cv2
from detector import ParallelDetector

video = cv2.VideoCapture("test_video.mp4")

# Test 1: Solo YOLO
print("=== Test 1: Solo YOLO ===")
detector = ParallelDetector(enable_depth=False)
frames = 0
start = time.time()
while frames < 50:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    detector.process(frame)
    frames += 1
print(f"YOLO solo: {frames/(time.time()-start):.1f} FPS")

# Test 2: YOLO + Depth cada 5 frames
print("\n=== Test 2: YOLO + Depth (interval=5) ===")
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
detector2 = ParallelDetector(enable_depth=True, depth_interval=5)
frames = 0
start = time.time()
while frames < 50:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    detector2.process(frame)
    frames += 1
print(f"YOLO + Depth(5): {frames/(time.time()-start):.1f} FPS")

# Test 3: YOLO + Depth cada 10 frames
print("\n=== Test 3: YOLO + Depth (interval=10) ===")
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
detector3 = ParallelDetector(enable_depth=True, depth_interval=10)
frames = 0
start = time.time()
while frames < 50:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    detector3.process(frame)
    frames += 1
print(f"YOLO + Depth(10): {frames/(time.time()-start):.1f} FPS")

video.release()
print("\nListo!")
