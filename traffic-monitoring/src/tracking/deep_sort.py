from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class VehicleTracker:
    def __init__(self):
        # Khởi tạo DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None
        )
        
    def update(self, detections, frame):
        # Cập nhật tracking từ các detection
        detection_list = []
        
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            class_name = det['class']
            
            detection_list.append([
                bbox[0], bbox[1], bbox[2], bbox[3], score, class_name
            ])
            
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        return tracks