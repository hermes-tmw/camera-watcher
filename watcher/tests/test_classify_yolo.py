"""Unit tests for the yolo-primary classifier path (no ollama needed)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watcher import classify_yolo
from watcher.classify_yolo import _classify_yolo, _weights_path, CLASS_TO_CATEGORY


class _FakeBox:
    def __init__(self, cls_id, conf):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [[10.0, 10.0, 100.0, 100.0]]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.names = {0: "person", 2: "car", 15: "cat", 5: "bus"}

    def predict(self, img, verbose=False, device="cpu"):
        return [_FakeResult([
            _FakeBox(2, 0.81),   # car -> vehicle
            _FakeBox(15, 0.44),  # cat -> animal (lower conf)
        ])]


class WeightsResolution(unittest.TestCase):
    def test_default_weights_missing_falls_back(self):
        with mock.patch.dict('os.environ', {'WATCHER_CLASSIFICATION_YOLO_WEIGHTS': ''}):
            # application_config mocked to return the VLM name (not a path)
            with mock.patch('watcher.connection.application_config', return_value='llama3.2-vision:latest'):
                p = _weights_path()
        self.assertEqual(str(p), classify_yolo.DEFAULT_CLASSIFIER_WEIGHTS)

    def test_env_override_wins_when_exists(self):
        with mock.patch.dict('os.environ', {'WATCHER_CLASSIFICATION_YOLO_WEIGHTS':
                                            classify_yolo.DEFAULT_CLASSIFIER_WEIGHTS}):
            with mock.patch('watcher.connection.application_config', return_value='/nonexistent/x.pt'):
                p = _weights_path()
        self.assertEqual(str(p), classify_yolo.DEFAULT_CLASSIFIER_WEIGHTS)


class ClassifyMapping(unittest.TestCase):
    def setUp(self):
        classify_yolo._DETECTOR = _FakeYOLO('fake.pt')
        classify_yolo._DETECTOR_WEIGHTS = 'fake.pt'

    def tearDown(self):
        classify_yolo._DETECTOR = None
        classify_yolo._DETECTOR_WEIGHTS = None

    def test_person_conf(self):
        self.assertIn('person', CLASS_TO_CATEGORY)

    def test_best_detection_wins(self):
        result = _classify_yolo(None, 'fake_event')
        self.assertEqual(result['category'], 'vehicle')
        self.assertTrue(result['interesting'])
        self.assertGreater(result['confidence'], 0.8)
        self.assertIn('car', result['description'])


if __name__ == '__main__':
    unittest.main()