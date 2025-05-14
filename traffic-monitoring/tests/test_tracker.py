import unittest
import numpy as np
from src.tracking.deep_sort import VehicleTracker

class TestVehicleTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = VehicleTracker()
        self.test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
    def test_tracker_initialization(self):
        self.assertIsNotNone(self.tracker)
        self.assertIsNotNone(self.tracker.tracker)
        
    def test_update_with_empty_detections(self):
        tracks = self.tracker.update([], self.test_image)
        self.assertIsInstance(tracks, list)
        
    def test_update_with_detections(self):
        test_detections = [{
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'car'
        }]
        
        tracks = self.tracker.update(test_detections, self.test_image)
        self.assertIsInstance(tracks, list)
        
    def test_track_continuity(self):
        # Test xem tracker có duy trì ID qua các frame không
        test_detection = [{
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'car'
        }]
        
        # Frame 1
        tracks1 = self.tracker.update(test_detection, self.test_image)
        # Frame 2 - detection gần giống
        test_detection[0]['bbox'] = [105, 105, 205, 205]
        tracks2 = self.tracker.update(test_detection, self.test_image)
        
        if tracks1 and tracks2:
            self.assertEqual(
                tracks1[0].track_id,
                tracks2[0].track_id
            )

if __name__ == '__main__':
    unittest.main()