# Traffic Monitoring System

## Overview
This project implements a camera tracking system that monitors traffic through input video. It utilizes YOLOv8 for object detection and Deep SORT for real-time tracking of vehicles.

## Features
- Count the current number of objects in the frame.
- Log the number of vehicles at each hour interval.
- Categorize the current count by type (motorcycles, cars, trucks).

## Project Structure
```
traffic-monitoring
├── src
│   ├── detection          # Contains object detection related files
│   ├── tracking           # Contains tracking related files
│   ├── utils              # Contains utility functions and classes
│   └── main.py            # Entry point of the application
├── configs                # Configuration files for detector and tracker
├── tests                  # Unit tests for the project
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd traffic-monitoring
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the detector and tracker settings in the `configs` directory.

4. Run the application:
   ```
   python src/main.py
   ```

## Usage
- The application will start processing the video feed from the camera.
- Detected objects will be logged and visualized in real-time.

## License
This project is licensed under the MIT License.
