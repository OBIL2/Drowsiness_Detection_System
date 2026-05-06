import os
import time

# 1. Silencing the technical noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'

import cv2
import numpy as np
from scipy.spatial import distance as dist
from keras.models import load_model
import pygame
import mediapipe as mp
from collections import deque
import threading

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("[INFO] Initializing audio alarm...")
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except Exception:
    print("[WARNING] alarm.wav not found! System will run muted.")
    alarm_sound = None

print("[INFO] Loading Autoencoder Models...")
eye_ae = load_model('eye_autoencoder.h5', compile=False)
yawn_ae = load_model('yawn_autoencoder.h5', compile=False)

# Warm up the models so first prediction isn't slow
dummy = np.zeros((1, 64, 64, 3))
eye_ae.predict(dummy, verbose=0)
yawn_ae.predict(dummy, verbose=0)
print("[INFO] Models warmed up!")

print("[INFO] Loading MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,  # Faster - we don't need iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def aspect_ratio(points, frame_w, frame_h):
    coords = np.array([(int(p.x * frame_w), int(p.y * frame_h)) for p in points])
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    return (A + B) / (2.0 * C), coords


def get_mar(landmarks, frame_w, frame_h):
    left  = (int(landmarks[78].x  * frame_w), int(landmarks[78].y  * frame_h))
    right = (int(landmarks[308].x * frame_w), int(landmarks[308].y * frame_h))
    top   = (int(landmarks[13].x  * frame_w), int(landmarks[13].y  * frame_h))
    bottom= (int(landmarks[14].x  * frame_w), int(landmarks[14].y  * frame_h))
    vertical = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)
    return vertical / horizontal, np.array([left, right, top, bottom])


def preprocess_for_ae(roi):
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_NEAREST)  # Fastest resize
    roi = roi.astype("float32") / 255.0  # float32 is faster than float64
    return np.expand_dims(roi, axis=0)


def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float32") - imageB.astype("float32")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err


def get_bounding_box(coords, pad=15):
    x_min, y_min = np.min(coords, axis=0) - pad
    x_max, y_max = np.max(coords, axis=0) + pad
    return int(x_min), int(y_min), int(x_max), int(y_max)


# ==========================================
# 4. BACKGROUND AUTOENCODER THREAD
# ==========================================
# This runs AE predictions in background so main loop never waits
ae_results = {'eye_mse': 0, 'yawn_mse': 0}
ae_input = {'eye_crop': None, 'mouth_crop': None, 'ready': False}
ae_lock = threading.Lock()

def ae_worker():
    while True:
        with ae_lock:
            if not ae_input['ready']:
                continue
            eye_crop = ae_input['eye_crop']
            mouth_crop = ae_input['mouth_crop']
            ae_input['ready'] = False

        try:
            if eye_crop is not None and eye_crop.size > 0:
                pre_eye = preprocess_for_ae(eye_crop)
                recon_eye = eye_ae.predict(pre_eye, verbose=0)
                ae_results['eye_mse'] = calculate_mse(pre_eye[0], recon_eye[0])

            if mouth_crop is not None and mouth_crop.size > 0:
                pre_mouth = preprocess_for_ae(mouth_crop)
                recon_mouth = yawn_ae.predict(pre_mouth, verbose=0)
                ae_results['yawn_mse'] = calculate_mse(pre_mouth[0], recon_mouth[0])
        except Exception:
            pass

ae_thread = threading.Thread(target=ae_worker, daemon=True)
ae_thread.start()

# ==========================================
# 5. THRESHOLDS & STATE
# ==========================================
EAR_THRESH = 0.22
MAR_THRESH = 0.5
MSE_EYE_THRESH = 0.015
MSE_YAWN_THRESH = 0.020

CLOSED_FRAMES_THRESHOLD = 15
closed_counter = 0
ALARM_ON = False
frame_count = 0

# FPS tracking
fps_buffer = deque(maxlen=30)
prev_time = time.time()

# ==========================================
# 6. MAIN LOOP
# ==========================================
print("[INFO] Starting Webcam... Press 'q' to exit.")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduces camera buffer lag

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # FPS Calculation
    curr_time = time.time()
    fps_buffer.append(1.0 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time
    fps = np.mean(fps_buffer)

    # Downscale frame for processing only (display stays full size)
    small_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            sh, sw = small_frame.shape[:2]
            fh, fw = frame.shape[:2]

            # Scale factor to map small frame coords back to full frame
            sx, sy = fw / sw, fh / sh

            # Geometry on small frame
            re_ear, re_coords_s = aspect_ratio([landmarks[i] for i in RIGHT_EYE], sw, sh)
            le_ear, le_coords_s = aspect_ratio([landmarks[i] for i in LEFT_EYE], sw, sh)
            ear = (re_ear + le_ear) / 2.0
            mar, m_coords_s = get_mar(landmarks, sw, sh)

            # Scale coords back to full frame for display
            re_coords = (re_coords_s * [sx, sy]).astype(int)
            le_coords = (le_coords_s * [sx, sy]).astype(int)
            m_coords  = (m_coords_s  * [sx, sy]).astype(int)

            # Send crops to background AE thread every 5th frame
            if frame_count % 5 == 0:
                all_eyes = np.vstack((re_coords, le_coords))
                ex1, ey1, ex2, ey2 = get_bounding_box(all_eyes)
                mx1, my1, mx2, my2 = get_bounding_box(m_coords, pad=20)

                eye_crop   = frame[max(0,ey1):ey2, max(0,ex1):ex2]
                mouth_crop = frame[max(0,my1):my2, max(0,mx1):mx2]

                with ae_lock:
                    ae_input['eye_crop']   = eye_crop.copy()
                    ae_input['mouth_crop'] = mouth_crop.copy()
                    ae_input['ready'] = True

            # Read latest AE results (never blocks)
            eye_mse  = ae_results['eye_mse']
            yawn_mse = ae_results['yawn_mse']

            # Hybrid Logic
            is_eyes_closed = (ear < EAR_THRESH) or (eye_mse > MSE_EYE_THRESH)
            is_yawning     = (mar > MAR_THRESH)  or (yawn_mse > MSE_YAWN_THRESH)

            status_text = "AWAKE"
            color = (0, 255, 0)

            if is_eyes_closed:
                closed_counter += 1
            else:
                closed_counter = max(0, closed_counter - 1)

            if is_yawning:
                status_text = "YAWNING DETECTED"
                color = (0, 255, 255)

            if closed_counter >= CLOSED_FRAMES_THRESHOLD:
                status_text = "!!! DROWSINESS ALERT !!!"
                color = (0, 0, 255)
                if not ALARM_ON and alarm_sound:
                    alarm_sound.play(-1)
                    ALARM_ON = True
            else:
                if ALARM_ON and not is_yawning:
                    alarm_sound.stop()
                    ALARM_ON = False

            # Visual Feedback
            cv2.putText(frame, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f} | Eye MSE: {eye_mse:.4f}", (10, fh - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f} | Yawn MSE: {yawn_mse:.4f}", (10, fh - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Landmarks
            for p in np.vstack((re_coords, le_coords, m_coords)):
                cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)

    # FPS display
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Drowsiness Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()