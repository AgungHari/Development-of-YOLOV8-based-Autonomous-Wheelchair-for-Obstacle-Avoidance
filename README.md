[![banner1](banner1.png)](https://www.agungg.com/)

# Development of YOLOv8-based Autonomous Wheelchair for Obstacle Avoidance (Pengembangan Kursi Roda Otonom Berbasis YOLOv8 untuk Penghindaran Obstacle). 

![MediaPipe version](https://img.shields.io/badge/MediaPipe-v0.10.14-blue)
![Ultralytics version](https://img.shields.io/badge/Ultralytics-v8.1.42-red)
![Tensorflow version](https://img.shields.io/badge/Tensorflow-v2.10.1-orange)
![OpenCV version](https://img.shields.io/badge/OpenCV-v4.9.0.80-darkred)
![IPyKernel version](https://img.shields.io/badge/IPyKernel-v6.29.4-yellow)
![License](https://img.shields.io/badge/License-MIT-darkblue)

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="900">

Detection is performed by combining two approaches: Yolo bounding box and pose landmarks, where both outputs are mapped into a 10x10 grid (made with OpenCV), which serves as a reference for the wheelchair to avoid obstacles. Commands are sent from the NUC to the ESP32, which then moves the motor.

## Project Result

This test involves testing the wheelchair's ability to avoid humans in real-time. The test will be conducted at Tower 2 ITS. The detected human remains still and does not move. The following video is a sample of the test. The full video can be accessed by clicking the video or the YouTube button below.

<div align="center">
  <img src="https://github.com/user-attachments/assets/cb7d43ea-688f-4ce9-a24b-5c50c62da9d3" alt="agungyolo" />
  <br>
</div>

[![YouTube](https://img.shields.io/badge/YouTube-black?style=flat-square&logo=youtube)](https://youtu.be/inr0SE0PDJg)

---

Based on the test results, the following conclusions can be drawn:

![test](https://img.shields.io/badge/Test-30_Test_100%_Results-green)
![FPS](https://img.shields.io/badge/FPS_Diff-7.029fps-red)
![Delay](https://img.shields.io/badge/Delay-0.2494seconds-blue)
![Inference](https://img.shields.io/badge/Inference-139.4899ms-darkblue)

- The model with the highest metrics trained with various configurations is the model with the highest mAP score at IoU 0.5, achieving 81.85%. This score is quite good for performing avoidance, as seen in the very good avoidance performance results.
- The FPS performance test of the NUC resulted in a lower value compared to the Author's personal laptop, with a difference of 7.029 fps.
- The average delay obtained in the test was approximately 0.2494 seconds, with an average inference time of 139.4899 ms or 0.1394 seconds.
- The results show that detection using bounding boxes and shoulder landmarks is more accurate at greater distances (150 cm and 100 cm), while arm landmarks are more accurate at closer distances (50 cm). The best average difference for the bounding box was 3.2 cm at a distance of 150 cm, the best average difference for the shoulder landmark was 2.2 cm at 100 cm, and the best average difference for the arm landmark was 1.93 cm at 50 cm.
- The detection performance results were satisfying, with a 100% success rate across 30 test samples, indicating that the system is highly effective in avoiding humans.


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

## Contributing

I am open to contributions and collaboration. If you would like to contribute, please create a pull request or contact me directly!
- Fork this repo.
- Create a new feature branch:

```bash
git checkout -b new-feature
```

- Commit your changes.
```bash
git commit -m "ver..."
```

- Push to the branch:
```bash
git push origin new-feature
```

## Features

- Grid mapping that can map all detected humans.
- Optimized for use with GPU with the help of CUDA.
- Two avoidance options: human avoidance with a route returning to the main path and multiple human avoidance with a default avoidance route.

<p align="center">
  <img src="https://github.com/user-attachments/assets/95a6c264-e6cd-4ea9-b378-208966d44ba6" alt="LOGO" width="300">
</p>




## Authors
<img alt="Static Badge" src="https://img.shields.io/badge/AgungHari-black?style=social&logo=github&link=https%3A%2F%2Fgithub.com%2FAgungHari">



## License

<img alt="GitHub License" src="https://img.shields.io/github/license/AgungHari/Development-of-YOLOV8-based-Autonomous-Wheelchair-for-Obstacle-Avoidance">



