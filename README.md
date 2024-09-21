
# Development of YOLOv8-based Autonomous Wheelchair for Obstacle Avoidance (Pengembangan Kursi Roda Otonom Berbasis YOLOv8 untuk Penghindaran Obstacle). 

Detection is performed by combining two approaches: Yolo bounding box and pose landmarks, where both outputs are mapped into a 10x10 grid (made with OpenCV), which serves as a reference for the wheelchair to avoid obstacles. Commands are sent from the NUC to the ESP32, which then moves the motor.
## Demo

![agungyolo](https://github.com/user-attachments/assets/cb7d43ea-688f-4ce9-a24b-5c50c62da9d3)

Watch full video here : https://youtu.be/inr0SE0PDJg?feature=shared

## Installation
Actually, you need an ESP32 and the wheelchair to run it.
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

![LOGO](https://github.com/user-attachments/assets/95a6c264-e6cd-4ea9-b378-208966d44ba6)



## Authors

- [@AgungHari](https://github.com/AgungHari)

