import cv2
from detection.yolo_detector import YoloDetector
from tracking.deepsort_tracker import DeepSortTracker
from utils.helper import get_counting_line

def main():
    video_path = "E:\\CD_HTTT\\test\\CameraTracking\\traffic-monitoring\\uploads\\videotest2.mp4"

    detector = YoloDetector()
    tracker = DeepSortTracker()

    cap = cv2.VideoCapture(video_path)

    car_count = 0
    truck_count = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        y_line = get_counting_line(frame)

        cv2.line(frame, (0, y_line), (frame.shape[1], y_line), (0, 255, 0), 2)

        tracks = tracker.update(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            cls = track.det_class
            vehicle_type = "Car" if cls == 2 else "Truck"

            center_y = (y1 + y2) // 2

            if y_line - 10 < center_y < y_line + 10 and track_id not in tracked_ids:
                if cls == 2:
                    car_count += 1
                else:
                    truck_count += 1
                tracked_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{vehicle_type} ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, f"Car Count: {car_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Truck Count: {truck_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Car and Truck Counting App", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
