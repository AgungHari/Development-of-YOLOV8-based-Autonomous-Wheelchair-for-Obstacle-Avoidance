
# Development of YOLOv8-based Autonomous Wheelchair for Obstacle Avoidance.

Detection is performed by combining two approaches: Yolo bounding box and pose landmarks, where both outputs are mapped into a 10x10 grid (made with OpenCV), which serves as a reference for the wheelchair to avoid obstacles. Commands are sent from the NUC to the ESP32, which then moves the motor.
## Demo

![agungyolo](https://github.com/user-attachments/assets/cb7d43ea-688f-4ce9-a24b-5c50c62da9d3)

## Installation


```bash
  python --version
  python -m venv nama_venv
  nama_venv\Scripts\activate
  pip install mediapipe
  pip install ultralytics
  pip install opencv-python
```
    
## Features

- Grid mapping that can map all detected humans.
- Two avoidance options: human avoidance with a route returning to the main path and multiple human avoidance with a default avoidance route.


![LOGO](https://github.com/user-attachments/assets/65b4bfc3-7e93-4ee3-bf2f-a3f66d561f57)



## Authors

- [@AgungHari](https://github.com/AgungHari)

