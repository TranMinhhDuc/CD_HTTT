import csv
from datetime import datetime
import os

class Logger:
    def __init__(self):
        self.log_file = "traffic_counts.csv"
        self._initialize_log_file()
        
    def _initialize_log_file(self):
        # Tạo file CSV nếu chưa tồn tại
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Motorcycles', 'Cars', 'Trucks'])
                
    def log_counts(self, counts):
        # Ghi số lượng phương tiện vào file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                counts['motorcycle'],
                counts['car'], 
                counts['truck']
            ])