
# Development of YOLOv8-based Autonomous Wheelchair for Obstacle Avoidance (Pengembangan Kursi Roda Otonom Berbasis YOLOv8 untuk Penghindaran Obstacle). 

🚀 Detection is performed by combining two approaches: Yolo bounding box and pose landmarks, where both outputs are mapped into a 10x10 grid (made with OpenCV), which serves as a reference for the wheelchair to avoid obstacles. Commands are sent from the NUC to the ESP32, which then moves the motor.
## Demo

![agungyolo](https://github.com/user-attachments/assets/cb7d43ea-688f-4ce9-a24b-5c50c62da9d3)

Watch full video here : https://youtu.be/inr0SE0PDJg?feature=shared

## Installation
I recommend a separate folder and venv for creating the model and the wheelchair program. for training venv setup :
```bash
  python --version
  python -m venv nama_venv
  nama_venv\Scripts\activate
  pip install ipykernel
  pip install ultralytics roboflow opencv-python
```
YOLOv8 need an absolute path. so change the data path for train, val, test in data.yaml example :
```bash
  names:
  -Manusia
  nc: 1
  roboflow:
    license: CC BY 4.0
    project: deteksi-manusia-yolov8-dataset
    url: https://universe.roboflow.com/hari-vijaya-kusuma/deteksi-manusia-yolov8-dataset/dataset/1
    version: 1
    workspace: hari-vijaya-kusuma
  test: D:/train yolo/Deteksi-Manusia-YoloV8-Dataset-1/test/images
  train: D:/train yolo/Deteksi-Manusia-YoloV8-Dataset-1/train/images
  val: D:/train yolo/Deteksi-Manusia-YoloV8-Dataset-1/valid/images
```
If you feel the training is taking too long, but you have a computer with NASA Super Computer like specs, it means you haven't utilized your GPU for the training process. check for cuda by running this
```bash
  import torch
  print(torch.cuda.is_available())
```
if the output is false then you need to install PyTorch with CUDA support. Check your driver version before implementing CUDA. example for installing CUDA 11.8
```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
You need an ESP32 and the wheelchair to run it.
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
- Optimized for use with GPU with the help of CUDA.
- Two avoidance options: human avoidance with a route returning to the main path and multiple human avoidance with a default avoidance route.

![LOGO](https://github.com/user-attachments/assets/95a6c264-e6cd-4ea9-b378-208966d44ba6)



## Authors

- [@AgungHari](https://github.com/AgungHari)

