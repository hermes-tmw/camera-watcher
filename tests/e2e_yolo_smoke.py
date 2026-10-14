"""End-to-end test: run the real yolo11n on a real frame (host python).

Requires: ultralytics installed for this interpreter.
This is run INSIDE the rebuilt classifier container as a smoke test.
"""
import sys
sys.path.insert(0, '/app/camera-watcher')

from watcher import classify_yolo

weights = classify_yolo._weights_path()
print('weights resolved:', weights)
assert str(weights) == '/data/video/watcher/stealthcam/yolo11n.pt', weights

det = classify_yolo._detector()
print('detector loaded:', det is not None)

from PIL import Image
img = Image.open('/data/video/watcher/stealthcam/20261014_153653_WichitaDriveway_102_1.JPG') \
    if __import__('os').path.exists('/data/video/watcher/stealthcam/20261014_153653_WichitaDriveway_102_1.JPG') \
    else Image.new('RGB', (640, 480), (40, 80, 40))

r = classify_yolo._classify_yolo(img, 'e2eSmokeTest')
print('classify result:', r)
assert isinstance(r, dict) and 'category' in r and 'confidence' in r
print('E2E YOLO CLASSIFY OK')