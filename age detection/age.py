import cv2
import numpy as np
from fer import FER
import os
import mediapipe as mp

# Tentukan path absolut ke file model usia
AGE_PROTO = r"E:\Document C\codingan\age detection\age_deploy.prototxt"
AGE_MODEL = r"E:\Document C\codingan\age detection\age_net.caffemodel"

# Cek apakah file model ada
if not os.path.isfile(AGE_PROTO) or not os.path.isfile(AGE_MODEL):
    print(f"File model tidak ditemukan! Pastikan {AGE_PROTO} dan {AGE_MODEL} ada.")
    exit()

# Muat model usia menggunakan OpenCV's dan module
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# Set perangkat keras untuk DNN
age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Gunakan CUDA sebagai backend
age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    # Target GPU

# Model untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kategori usia
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Buat detektor emosi menggunakan FER
emotion_detector = FER()

# Inisialisasi MediaPipe untuk pelacakan tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Mulai menangkap video dari kamera
cap = cv2.VideoCapture(0)

# Atur resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Lebar
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Tinggi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame secara horizontal agar tidak mirror
    frame = cv2.flip(frame, 1)

    # Ubah gambar menjadi grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Ambil area wajah (Region of Interest - ROI)
        face_img = frame[y:y+h, x:x+w].copy()

        # Pencerahan wajah dengan mengubah brightness dan contrast
        face_img = cv2.convertScaleAbs(face_img, alpha=1.3, beta=30)

        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (104, 117, 123), swapRB=False)

        # Prediksi usia
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]

        # Prediksi emosi
        emotion_result = emotion_detector.detect_emotions(face_img)
        emotion = 'Unknown'
        if emotion_result:
            emotion = max(emotion_result[0]['emotions'], key=emotion_result[0]['emotions'].get)

        # Gambar kotak di sekitar wajah dan tampilkan usia serta emosi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f'Age: {age}, Emotion: {emotion}'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Tempelkan wajah yang telah diproses kembali ke frame asli
        frame[y:y+h, x:x+w] = face_img

    # Deteksi tangan
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Jika tangan terdeteksi, gambar titik-titik dan garis pada tangan
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gambar garis dan titik untuk setiap landmark tangan
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan frame yang dihasilkan
    cv2.imshow('Age and Emotion Detector with Hand Tracking', frame)

    # Keluar dari loop jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan video capture dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
