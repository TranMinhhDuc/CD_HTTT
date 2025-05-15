from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self, max_age=30, nn_budget=100, max_iou_distance=0.7, max_cosine_distance=0.3, embedder="mobilenet"):
        self.deepsort = DeepSort(
            max_age=max_age,
            nn_budget=nn_budget,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder
        )

    def update(self, detections, frame):
        return self.deepsort.update_tracks(detections, frame=frame)
