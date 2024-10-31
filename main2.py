from enum import Enum
from ultralytics import YOLO
import cv2
import math
import mediapipe as mp
import numpy as np
import socket
import time
import datetime
import csv
from mediapipe.python.solutions.pose import PoseLandmark


def delay(seconds):
    start_time = time.time()
    while time.time() - start_time < seconds:
        pass

csv_file_path = "log.csv"
csv_columns = ["Direction", "Distance", "Timestamp"]
host = "192.168.18.110" 
port = 8080 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1366)
cap.set(4, 768)

k = 10.622
k2 = 24.222

focal_length_pixel = 481
tinggi_objek_nyata = 181

frame_count = 0
start_time = time.time()
fps = 0

model = YOLO("100Epoch16batch.pt")

class SocketCommunicator:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.socket = None
        self.connect()
        pass

    def connect(self):
        s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((self.host, self.port))
            print("Terkoneksi dengan kursi roda")
            self.socket = s
        except socket.error:
            print("Mode Remote anjay")
            
    def send(self, data):
        if self.socket:
            self.socket.send(data)

s = SocketCommunicator(host, port)

def hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata):
    if tinggi_bounding_box == 0:
        return float('inf')
    jarak = (tinggi_objek_nyata * focal_length_pixel) / tinggi_bounding_box
    return jarak / 100

def hitung_lebar_objek(lebar_bounding_box, jarak_objek, focal_length_pixel):
    if lebar_bounding_box == 0 or focal_length_pixel == 0:
        return 0
    lebar_objek = (lebar_bounding_box * jarak_objek) / focal_length_pixel
    return lebar_objek

def convert_coordinates(outputs, img_width, img_height):
    boxes = []
    for detection in outputs:
        x_center, y_center, width, height = detection['x_center'], detection['y_center'], detection['width'], detection['height']
        
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height
        
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def hitung_jarak_euclidean(landmark1, landmark2, lebar_img):
    jarak_pix = math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) * lebar_img
    return jarak_pix

def hitung_lebar_mediapipe(pose_results, lebar_img):
    if pose_results.pose_landmarks:
        bahu_kiri = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        bahu_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        jarak_pix = hitung_jarak_euclidean(bahu_kiri, bahu_kanan, lebar_img)
        
        faktor_konversi = 0.00087
        lebar_m = jarak_pix * faktor_konversi
        
        return lebar_m
    return 0

def hitung_jarak_vertikal(pose_results, tinggi_img):
    if pose_results.pose_landmarks:
        pusar = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        
        jarak_vertikal = (1 - pusar.y) * tinggi_img
        return jarak_vertikal
    return 0

def draw_grid(img, detection_status, camera_position):
    grid_size = 100
    start_x = img.shape[1] - grid_size - 10
    start_y = img.shape[0] - grid_size - 10

    cell_size = int(grid_size / 10)
    
    for i in range(10):
        for j in range(10):
            cell_color = (0, 0, 255) if detection_status[i][j] else (0, 0, 0)
            cv2.rectangle(img, (start_x + j * cell_size, start_y + i * cell_size),
                          (start_x + (j + 1) * cell_size, start_y + (i + 1) * cell_size), cell_color, -1)

    for pos in camera_position:
        x, y = pos
        cv2.rectangle(img, (start_x + x * cell_size, start_y + y * cell_size),
                      (start_x + (x + 1) * cell_size, start_y + (y + 1) * cell_size), (0, 255, 0), -1)
    
    for i in range(11):
        cv2.line(img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), 1)
        cv2.line(img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), 1)
    
    text_x2 = start_x - 10
    text_x = start_x - 80
    text_y = start_y - 20
    text_y2 = start_y - 80
    cv2.putText(img, "0.2 Meter per square", (text_x,text_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.putText(img, f"X: {posisi_horizontal_piksel:.2f}px", (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img

def determine_direction(detection_grid, grid_size=10):
    if not np.any(detection_grid):
        return 'Maju', 0, "No detection, moving forward"
    
    mid_point = grid_size // 2
    left_count = np.sum(detection_grid[:, :mid_point])
    right_count = np.sum(detection_grid[:, mid_point:])
    manusia = "manusia terlalu jauh"
    
    if left_count > right_count:
        direction = 'Kanan'
        manusia = "Manusia di Kiri"
    
    elif right_count > left_count:
        direction = 'Kiri'
        manusia = "Manusia di Kanan"
            
    else:
        direction = 'Kanan'
        manusia = "Manusia di tengah"
        
    return direction, abs(right_count - left_count) * 0.2, manusia

class Direction(Enum):
    MAJU = 1
    BERHASIL_KE_KANAN = 2
    BERHASIL_KE_KIRI = 3

current_direction = Direction.MAJU
detection_grid = np.zeros((10, 10), dtype=bool)
camera_position = [(4, 0), (5, 0)]
newtex = None
last_detection_time = time.time()
counter = 0
moment_of_truth = False
while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    current_time = time.time()

    if current_time - start_time >= 1:
        fps = frame_count / (current_time - start_time)
        frame_count = 0
        start_time = current_time

    detection_grid.fill(False)
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_time_yolo = time.time()

    results = model.predict(img, stream=True)

    end_time_yolo = time.time()

    response_time_yolo = end_time_yolo - start_time_yolo
    print(f"Waktu respons YOLO: {response_time_yolo:.5f} detik")

    with open('log.csv', mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        for r in results:
            boxes = r.boxes

            for box in boxes:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                if confidence > 0.7:
                    cls = int(box.cls[0])
                    if cls == 0:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        color = (255, 0, 0)

                        tinggi_bounding_box = y2 - y1
                        lebar_bounding_box = x2 - x1
                        jarak_yolo = hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata)
                        
                        lebar_objek = hitung_lebar_objek(lebar_bounding_box, jarak_yolo, focal_length_pixel)
                        cv2.putText(img, f"X_min: {x1}px", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(img, f"BBox Width (Y)px : {lebar_objek:.2f} m", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(img, f"Yolo Distance: {jarak_yolo:.2f} m", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(img, f"Distance Yolo: {jarak_yolo:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        start_time_mediapipe = time.time()
                        current_time = time.time()

                        person_img = img[y1:y2, x1:x2]
                        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(person_img_rgb)

                        end_time_mediapipe = time.time()
                        response_time_mediapipe = end_time_mediapipe - start_time_mediapipe
                        print(f"Waktu respons MediaPipe: {response_time_mediapipe:.5f} detik")
                        if pose_results.pose_landmarks:
                            mp_drawing.draw_landmarks(img[y1:y2, x1:x2], pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                            lebar_img = person_img.shape[1]
                            siku_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                            pergelangan_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                            bahu_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                            bahu_kiri = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                            pusar = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                            jarak_pixbahu = hitung_jarak_euclidean(bahu_kanan, bahu_kiri, lebar_img)
                            jarak_pix = hitung_jarak_euclidean(siku_kanan, pergelangan_kanan, lebar_img)
                            lebar_img = img.shape[1]
                            tinggi_img = img.shape[0]
                            lebar_m = hitung_lebar_mediapipe(pose_results, lebar_img)
                            grid_size = 10
                            jarak_maksimum = 10 * 0.2
                            cv2.putText(img, f"Human Width: {lebar_m:.2f} m", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                            if jarak_pix > 0:
                                jarak_mediapipe = (k / jarak_pix) * 10
                                jarak_mediapipebahu = (k2 / jarak_pixbahu) * 10

                                posisi_horizontal_piksel = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * lebar_img
                                grid_x = int(((x1 + 350) / lebar_img) * 10)
                                pusar = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                                posisi_vertikal_piksel = pusar.y * tinggi_img
                                grid_y = int((jarak_maksimum - jarak_mediapipe) / 0.2)
                                grid_y = max(0, min(9 - grid_y, 9))
                                lebar_grid = max(1, int(lebar_m / 0.2))
                                grid_x = min(max(grid_x, 0), 9)
                                for i in range(max(0, grid_x - lebar_grid // 2), min(10, grid_x + lebar_grid // 2)):
                                    for j in range(max(0, grid_y), min(10, grid_y + lebar_grid // 2)):
                                        detection_grid[j][i] = True

                                cv2.putText(img, f"Counter : {counter}", (10,600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5 , (255,255,255), 2)
                                cv2.putText(img, f"MP Hand Distance: {jarak_mediapipe:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(img, f"MediaPipe Hand Distance: {jarak_mediapipe:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print(f"Jarak : {jarak_mediapipe:.2f}")
                                cv2.putText(img, f"MP Shoulder Distance: {jarak_mediapipebahu:.2f}", (x1,y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                cv2.putText(img, f"MediaPipe Shoulder Distance: {jarak_mediapipebahu:.2f}", (10 , 210), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,0),2)
                                cv2.putText(img, f"FPS: {fps:.2f}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                cv2.putText(img, f"(x,y)px Hand: {jarak_pix:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                cv2.putText(img, f"(x,y)px Shoulder : {jarak_pixbahu:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2 )
                                cv2.putText(img, f"X Pose Center: {posisi_horizontal_piksel:.2f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                
                        direction, distance, manusia = determine_direction(detection_grid)
                        if not np.any(detection_grid):
                            if current_time - last_detection_time > 1:
                                pesan = "MAJU"
                                print("Fallback: Mengirim perintah Maju")
                            last_detection_time = current_time
                        else:
                            if jarak_mediapipe < 1.0 or jarak_mediapipebahu < 1.0 or jarak_yolo < 1.0:
                                if direction == 'Kiri':
                                    pesan = "BELOK KIRI"
                                    color = (0, 0, 255)
                                    current_direction = Direction.BERHASIL_KE_KIRI
                                elif direction == 'Kanan':
                                    pesan = "BELOK KANAN"
                                    color = (0, 0, 255)
                                    current_direction = Direction.BERHASIL_KE_KANAN
                            else:
                                pesan = "MAJU"
                                color = (51, 255, 255)
                                current_direction = Direction.MAJU

                        if pesan != newtex:
                            delay(0.5)
                            arah = 'C\n'
                            s.send(arah.encode('utf-8'))
                            time.sleep(0.5)
                            date = datetime.datetime.now()
                            print(date)
                            agung = ("Stop")
                            data = {
                                "Direction": agung,
                                "Distance": distance,
                                "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            writer.writerow({"Direction": agung, "Distance": distance, "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})

                            arrow_start_x = img.shape[1] - 200  # Mengatur posisi panah di sebelah kiri grid
                            arrow_start_y = img.shape[0] - 50
                            if pesan == "BELOK KIRI":
                                arah = 'E\n'
                                date = datetime.datetime.now()
                                agung = ("Kiri")
                                start_point = (arrow_start_x, arrow_start_y)
                                end_point = (arrow_start_x - 100, arrow_start_y)
                                print(date)
                                print("kiri")
                                counter += 1
                            elif pesan == "BELOK KANAN":
                                arah = 'A\n'
                                date = datetime.datetime.now()
                                start_point = (arrow_start_x, arrow_start_y)
                                end_point = (arrow_start_x + 100, arrow_start_y)
                                agung = ("Kanan")
                                print(date)
                                print("kanan")
                                counter += 1
                            else:
                                arah = 'B\n'
                                date = datetime.datetime.now()
                                start_point = (arrow_start_x, arrow_start_y)
                                end_point = (arrow_start_x, arrow_start_y - 100)
                                agung = ("Maju")
                                print(date)

                            s.send(arah.encode('utf-8'))
                            newtex = pesan
                                
                            data = {
                                "Direction": agung,
                                "Distance": distance,
                                "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            writer.writerow({"Direction": agung, "Distance": distance, "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})
                                
                        print(f"FPS: {fps:.2f}")
                        cv2.arrowedLine(img, start_point, end_point, (0, 255, 255), 5)
                        direction, distance, manusia = determine_direction(detection_grid)
                        cv2.putText(img, f"{pesan} {distance:.2f}m", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(img, f"{pesan}",(500,400 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2 )
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        print(manusia)
                        img = draw_grid(img, detection_grid, camera_position)
                        

            if not any(boxes) or boxes == []:
                cv2.putText(img, f"Manusia Tidak Terdeteksi", (500,400 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                delay(0.5)
                if counter >= 2:
                    arah = 'B\n'
                    s.send(arah.encode('utf-8'))
                    current_direction = Direction.MAJU
                    counter -= counter
                    moment_of_truth = True

                if current_direction == Direction.MAJU:
                    arah = 'B\n'
                    date = datetime.datetime.now()
                    agung = ("Maju")
                    s.send(arah.encode('utf-8'))
                else :
                    print(date)
                    arah = 'B\n'
                    date = datetime.datetime.now()
                    agung = ("Maju")
                    print(date)
                    s.send(arah.encode('utf-8'))
                    delay(3)

                    arah = 'C\n'
                    s.send(arah.encode('utf-8'))
                    delay(1)
                    if current_direction == Direction.BERHASIL_KE_KIRI:
                        arah = 'A\n'
                        cv2.putText(img, f"Telah Berhasil Menghindar, Mengecek Posisi Manusia", (400,600 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 1)
                        start_point = (arrow_start_x, arrow_start_y)
                        end_point = (arrow_start_x - 100, arrow_start_y)
                    elif current_direction == Direction.BERHASIL_KE_KANAN:
                        arah = 'E\n'
                        cv2.putText(img, f"Telah Berhasil Menghindar, Mengecek Posisi Manusia", (400,600 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 1)
                        start_point = (arrow_start_x, arrow_start_y)
                        end_point = (arrow_start_x + 100, arrow_start_y)
                    else:
                        arah = 'B\n'
                        cv2.putText(img, f"Deteksi Error Perbaiki Kamera", (500,600 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 1)
                        start_point = (arrow_start_x, arrow_start_y)
                        end_point = (arrow_start_x, arrow_start_y - 100)
                        current_direction = Direction.MAJU
                    s.send(arah.encode('utf-8'))
                    time.sleep(5)

                    arah = 'C\n'
                    s.send(arah.encode('utf-8'))
                    delay(1)
                    current_direction = Direction.MAJU
                        

                    cv2.arrowedLine(img, start_point, end_point, (0, 255, 255), 5)
                    date = datetime.datetime.now()
                    print(date)
                    agung = ("Kembali ke Arah Utama")
                    print(f"Kembali ke arah: {agung}")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        counter = 0
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
