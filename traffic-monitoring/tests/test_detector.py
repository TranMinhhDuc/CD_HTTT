import unittest
import cv2
import numpy as np
from src.detection.detector import VehicleDetector

class TestVehicleDetector(unittest.TestCase):
    def setUp(self):
        self.detector = VehicleDetector("yolov8n.pt")
        # Tạo ảnh test đơn giản
        self.test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.classes), 3)
        
    def test_detection_output_format(self):
        detections = self.detector.detect(self.test_image)
        
        # Kiểm tra output có đúng format không
        self.assertIsInstance(detections, list)
        if len(detections) > 0:
            detection = detections[0]
            self.assertIn('bbox', detection)
            self.assertIn('score', detection)
            self.assertIn('class', detection)
            
            # Kiểm tra bbox format
            self.assertEqual(len(detection['bbox']), 4)
            
    def test_valid_class_names(self):
        detections = self.detector.detect(self.test_image)
        valid_classes = ['motorcycle', 'car', 'truck']
        
        for detection in detections:
            self.assertIn(detection['class'], valid_classes)

if __name__ == '__main__':
    unittest.main()