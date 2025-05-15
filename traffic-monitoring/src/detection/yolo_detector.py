from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", classes=[2, 7], conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, classes=self.classes)
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                if conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    detections.append(([x1, y1, width, height], conf, cls))
        return detections
