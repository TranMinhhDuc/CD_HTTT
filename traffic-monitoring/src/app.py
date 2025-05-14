from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import os
from detection.detector import VehicleDetector
from tracking.tracker import ObjectTracker
from utils.visualizer import Visualizer
import cv2
from datetime import datetime
import time

app = Flask(__name__)

# Cấu hình upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # giới hạn 16MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

detector = None
tracker = None
visualizer = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Lấy danh sách video từ thư mục uploads
    video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                  if f.lower().endswith(tuple(app.config['ALLOWED_EXTENSIONS']))]
    return render_template('index.html', video_files=video_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return 'Không có file nào được tải lên', 400
    
    file = request.files['video']
    if file.filename == '':
        return 'Không có file nào được chọn', 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))
        
    return 'File không hợp lệ', 400

@app.route('/process_video', methods=['GET', 'POST'])
def process_video():
    global detector, tracker, visualizer
    
    if request.method == 'POST':
        video_file = request.form.get('video_file')
        if not video_file:
            return 'Không tìm thấy video', 400
            
        # Khởi tạo detector và tracker nếu chưa có
        if not all([detector, tracker, visualizer]):
            detector = VehicleDetector("yolov8n.pt")
            tracker = ObjectTracker()
            visualizer = Visualizer()
            
        return render_template('process.html', video_file=video_file)
    
    return redirect(url_for('index'))

@app.route('/video_feed/<video_file>')
def video_feed(video_file):
    global detector, tracker, visualizer
    
    if not all([detector, tracker, visualizer]):
        return 'Chưa khởi tạo detector/tracker', 400

    def generate_frames():
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Không thể mở video")
            return

        # Đặt FPS cho video
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Lưu trữ số lượng đã đếm
        counted_vehicles = {
            'left_person': 0, 'left_motorcycle': 0, 'left_vehicle': 0,
            'right_person': 0, 'right_motorcycle': 0, 'right_vehicle': 0
        }
        
        # Lưu trữ ID xe đã đếm
        counted_ids = set()
        
        # Thêm biến lưu trạng thái xe
        vehicle_states = {}  # {track_id: has_crossed_line}
        
        # Lưu tọa độ y trước đó của mỗi track
        prev_y = {}
        
        try:    
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Kết thúc video")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Quay lại từ đầu
                    # Reset bộ đếm khi video lặp lại
                    counted_vehicles = {k: 0 for k in counted_vehicles}
                    counted_ids.clear()
                    prev_y.clear()
                    vehicle_states.clear()  # Reset trạng thái xe
                    continue
                    
                # Phát hiện và tracking
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)
                
                # Đếm xe khi đi qua line
                line_y = visualizer.line_y
                for track in tracks:
                    if track.is_confirmed() and hasattr(track, 'det_class'):
                        class_name = track.det_class
                        # Gộp các loại xe thành vehicle
                        if class_name in ['car', 'truck', 'bus']:
                            class_name = 'vehicle'
                        
                        bbox = track.to_tlbr()
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        track_id = track.track_id
                        
                        # Xác định làn đường (trái/phải)
                        direction = "left" if center_x < frame.shape[1]//2 else "right"
                        
                        # Kiểm tra hướng đi qua line
                        if track_id in prev_y:
                            prev_center_y = prev_y[track_id]
                            # Xe đi từ trên xuống
                            crossed_down = prev_center_y <= line_y and center_y > line_y
                            # Xe đi từ dưới lên
                            crossed_up = prev_center_y >= line_y and center_y < line_y
                            
                            if (crossed_down or crossed_up) and track_id not in counted_ids:
                                vehicle_key = f"{direction}_{class_name}"
                                if vehicle_key in counted_vehicles:
                                    counted_vehicles[vehicle_key] += 1
                                    counted_ids.add(track_id)
                                vehicle_states[track_id] = True
                        
                        # Cập nhật vị trí y trước đó
                        prev_y[track_id] = center_y
                
                # Vẽ kết quả với trạng thái xe
                visualizer.draw(frame, tracks, counted_vehicles, vehicle_states)
                
                # Chuyển frame thành jpeg với chất lượng cao hơn
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                # Giảm delay để tăng FPS
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Lỗi: {str(e)}")
        finally:
            cap.release()

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    global tracker
    if tracker:
        counts = {
            'motorcycle': len([t for t in tracker.tracks if hasattr(t, 'det_class') and t.det_class == 'motorcycle']),
            'car': len([t for t in tracker.tracks if hasattr(t, 'det_class') and t.det_class == 'car']),
            'truck': len([t for t in tracker.tracks if hasattr(t, 'det_class') and t.det_class == 'truck'])
        }
        return jsonify(counts)
    return jsonify({'motorcycle': 0, 'car': 0, 'truck': 0})

@app.route('/download_stats')
def download_stats():
    if tracker:
        stats = get_stats().json
        output = 'Timestamp,Motorcycles,Cars,Trucks\n'
        output += f"{datetime.now()},{stats['motorcycle']},{stats['car']},{stats['truck']}\n"
        
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=traffic_stats.csv"}
        )
    return 'Không có dữ liệu', 400

if __name__ == '__main__':
    app.run(debug=True)