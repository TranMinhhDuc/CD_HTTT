from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None
        )
        self.tracks = []

    def update(self, detections, frame):
        detection_list = []
        
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            class_name = det['class']
            
            # Convert bbox format
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            detection_list.append(([x1, y1, w, h], score, class_name))

        # Update tracks
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Store class information in tracks
        for track, (_, _, class_name) in zip(tracks, detection_list):
            if track.is_confirmed():
                track.det_class = class_name
        
        self.tracks = tracks
        return [t for t in tracks if t.is_confirmed()]

    def get_counts(self):
        counts = {
            'car': 0,
            'motorcycle': 0,
            'truck': 0
        }
        
        for track in self.tracks:
            if track.is_confirmed() and hasattr(track, 'det_class'):
                if track.det_class in counts:
                    counts[track.det_class] += 1
                    
        return counts