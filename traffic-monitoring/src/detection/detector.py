from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Chỉ định nghĩa các loại xe cần đếm
        self.classes = {
            2: 'car',          # Ô tô
            3: 'motorcycle',   # Xe máy  
            7: 'truck',        # Xe tải
            5: 'bus'          # Xe buýt
        }

    def detect(self, frame):
        # Thực hiện detection
        results = self.model(frame, verbose=False)[0]
        detections = []

        # Xử lý kết quả
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)
            
            # Chỉ lấy các loại xe đã định nghĩa
            if class_id in self.classes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_name = self.classes[class_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': score,
                    'class': class_name
                })
            
        return detections