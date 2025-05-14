import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'person': (255, 0, 0),     # Màu cho người
            'motorcycle': (0, 255, 0),  # Màu cho xe máy
            'vehicle': (0, 0, 255)      # Màu cho ô tô
        }
        self.line_y = None  # Sẽ tính theo chiều cao frame
        self.line_thickness = 2
        self.direction_offset = 50  # Khoảng cách để xác định hướng
        self.uncounted_color = (0, 0, 255)  # Đỏ cho xe chưa qua line
        self.counted_color = (0, 255, 0)    # Xanh cho xe đã qua line

    def draw(self, frame, tracks, counts, vehicle_states):
        h, w = frame.shape[:2]
        if self.line_y is None:
            self.line_y = h // 2

        # Vẽ đường phân cách giữa và line đếm
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        cv2.line(frame, (0, self.line_y), (w, self.line_y), (255, 255, 0), self.line_thickness)

        # Vẽ tracks và bounding boxes
        for track in tracks:
            if track.is_confirmed() and hasattr(track, 'det_class'):
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Thu nhỏ bbox để fit sát với xe
                width = x2 - x1
                height = y2 - y1
                
                # Điều chỉnh bbox nhỏ hơn - giảm padding xuống 5%
                x_padding = int(width * 0.05)
                y_padding = int(height * 0.05)
                
                x1 = max(0, x1 + x_padding)
                x2 = min(w, x2 - x_padding)
                y1 = max(0, y1 + y_padding)
                y2 = min(h, y2 - y_padding)
                
                # Vẽ bbox nhỏ gọn với độ dày 1px
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # Label nhỏ và gọn hơn
                class_name = track.det_class
                if class_name in ['car', 'truck', 'bus']:
                    class_name = 'vehicle'
                cv2.putText(frame, class_name, (x1, y1-3), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Hiển thị số lượng cho từng làn
        padding = 10
        y_offset = 30
        
        # Làn bên trái
        cv2.putText(frame, "Left Lane:", (padding, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += 30
        
        for obj_type in ['person', 'motorcycle', 'vehicle']:
            count = counts.get(f'left_{obj_type}', 0)
            cv2.putText(frame, f"{obj_type}: {count}", 
                        (padding, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, self.colors[obj_type], 2)
            y_offset += 30

        # Làn bên phải
        right_x = w - 200
        y_offset = 30
        
        cv2.putText(frame, "Right Lane:", (right_x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += 30
        
        for obj_type in ['person', 'motorcycle', 'vehicle']:
            count = counts.get(f'right_{obj_type}', 0)
            cv2.putText(frame, f"{obj_type}: {count}", 
                        (right_x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, self.colors[obj_type], 2)
            y_offset += 30