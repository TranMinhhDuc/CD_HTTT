import cv2
import time
from datetime import datetime
# Sửa lại imports để tương đối với thư mục src
from .detection.detector import VehicleDetector
from .utils.visualizer import Visualizer
from .utils.logger import Logger
from .tracking.tracker import ObjectTracker
from .detection.model import load_tracker_config

class TrafficMonitor:
    def __init__(self, video_path, model_path, tracker_config_path):
        # Kiểm tra video path có tồn tại
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")
            
        print(f"Đã mở video thành công: {video_path}")
        
        self.detector = VehicleDetector(model_path)
        self.visualizer = Visualizer() 
        self.logger = Logger()
        
        # Load config
        self.tracker_config = load_tracker_config(tracker_config_path)
        self.tracker = ObjectTracker(self.tracker_config)
        
    def run(self):
        last_count_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Kết thúc video")
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # In thông tin mỗi 30 frame
                print(f"Đang xử lý frame thứ {frame_count}")
                
            # Phát hiện phương tiện
            detections = self.detector.detect(frame)
            print(f"Số lượng phương tiện phát hiện được: {len(detections)}")
            
            # Tracking phương tiện
            tracks = self.tracker.update(detections, frame)
            
            # Hiển thị frame
            self.visualizer.draw(frame, tracks, {})
            cv2.imshow('Traffic Monitoring', frame)
            
            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Đường dẫn tuyệt đối đến các file
    video_path = r"C:\Users\Admin\Downloads\Tracking camera\traffic-monitoring\data\video.mp4"  # Thay đổi path tới video của bạn
    model_path = r"C:\Users\Admin\Downloads\Tracking camera\traffic-monitoring\yolov8n.pt"
    config_path = r"C:\Users\Admin\Downloads\Tracking camera\traffic-monitoring\configs\tracker_config.yaml"
    
    print("Khởi tạo chương trình...")
    
    try:
        monitor = TrafficMonitor(
            video_path=video_path,
            model_path=model_path,
            tracker_config_path=config_path
        )
        print("Bắt đầu xử lý video...")
        monitor.run()
    except Exception as e:
        print(f"Lỗi: {str(e)}")