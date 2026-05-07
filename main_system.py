import os
import time
from threading import Thread

# Suppress warnings and optimize CPU threads for TensorFlow
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

# ==========================================
# 1. INITIALIZATION
# ==========================================
print("[INFO] Initializing audio alarm...")
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except Exception:
    print("[WARNING] alarm.wav not found! Muting alarm.")
    alarm_sound = None

print("[INFO] Loading Autoencoder Models...")
eye_ae = load_model('eye_autoencoder.h5', compile=False)
yawn_ae = load_model('yawn_autoencoder.h5', compile=False)

# Warm up models using the FAST direct-call method
dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
_ = eye_ae(dummy, training=False)
_ = yawn_ae(dummy, training=False)
print("[INFO] Models warmed up!")

print("[INFO] Loading MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


# ==========================================
# 2. HELPER FUNCTIONS & THREADING
# ==========================================
class WebcamStream:
    """Dedicated background thread for continuously pulling webcam frames at max speed."""

    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True


def aspect_ratio(points, frame_w, frame_h):
    coords = np.array([(int(p.x * frame_w), int(p.y * frame_h)) for p in points])
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    return (A + B) / (2.0 * C), coords


def get_mar(landmarks, frame_w, frame_h):
    left = (int(landmarks[78].x * frame_w), int(landmarks[78].y * frame_h))
    right = (int(landmarks[308].x * frame_w), int(landmarks[308].y * frame_h))
    top = (int(landmarks[13].x * frame_w), int(landmarks[13].y * frame_h))
    bottom = (int(landmarks[14].x * frame_w), int(landmarks[14].y * frame_h))
    return dist.euclidean(top, bottom) / dist.euclidean(left, right), \
        np.array([left, right, top, bottom])


def preprocess_for_ae(roi):
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_NEAREST)
    roi = roi.astype("float32") / 255.0
    return np.expand_dims(roi, axis=0)


def calculate_mse(a, b):
    err = np.sum((a - b) ** 2)
    return err / float(a.shape[0] * a.shape[1] * a.shape[2])


def get_bounding_box(coords, pad=15):
    x_min, y_min = np.min(coords, axis=0) - pad
    x_max, y_max = np.max(coords, axis=0) + pad
    return int(x_min), int(y_min), int(x_max), int(y_max)


def run_ae(eye_crop, mouth_crop):
    eye_mse = yawn_mse = 0
    try:
        if eye_crop is not None and eye_crop.size > 0:
            pre = preprocess_for_ae(eye_crop)
            recon = eye_ae(pre, training=False).numpy()
            eye_mse = calculate_mse(pre[0], recon[0])

        if mouth_crop is not None and mouth_crop.size > 0:
            pre = preprocess_for_ae(mouth_crop)
            recon = yawn_ae(pre, training=False).numpy()
            yawn_mse = calculate_mse(pre[0], recon[0])
    except Exception:
        pass
    return eye_mse, yawn_mse


# ==========================================
# 3. THRESHOLDS
# ==========================================
# --- Eye Thresholds ---
EAR_THRESH = 0.215
MSE_EYE_THRESH = 0.046

# --- Yawn Thresholds ---
MAR_THRESH = 0.35
MSE_YAWN_THRESH = 0.15

EAR_HISTORY_LEN = 3

# Sets the alarm to trigger after ~0.3 seconds of closed eyes
CLOSED_FRAMES_THRESHOLD = 10

# ==========================================
# 4. PER-FACE STATE
# ==========================================
face_states = {}


def get_face_state(face_id):
    if face_id not in face_states:
        face_states[face_id] = {
            'closed_counter': 0,
            'last_eye_mse': 0,
            'last_yawn_mse': 0,
            'last_seen': time.time(),
            'drowsy_frames': 0,
            'total_frames': 0,
            'ear_history': deque(maxlen=EAR_HISTORY_LEN),
            'mar_history': deque(maxlen=EAR_HISTORY_LEN),
            'status': 'OPEN EYES',
            'alert_cooldown': 0,
        }
    return face_states[face_id]


def cleanup_old_faces():
    now = time.time()
    for fid in [f for f, s in face_states.items() if now - s['last_seen'] > 2.0]:
        del face_states[fid]


# ==========================================
# 5. SIDE PANEL DRAWING
# ==========================================
PANEL_W = 280
STATUS_COLORS = {
    'OPEN EYES': (0, 220, 0),
    'EYES CLOSING': (0, 165, 255),  # Orange
    'DROWSINESS DETECTED': (0, 0, 255),  # Red
    'YAWNING': (0, 220, 220),  # Yellow
    'DROWSY & YAWNING': (0, 0, 255),  # Red
}


def draw_panel(panel, face_states, fps, total_faces):
    panel[:] = (25, 25, 35)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 50), (45, 45, 60), -1)
    cv2.putText(panel, "CLASSROOM MONITOR", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)
    cv2.putText(panel, f"FPS: {fps:.1f}   Faces: {total_faces}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 150), 1)

    y = 100
    for fid in sorted(face_states.keys()):
        s = face_states[fid]
        status = s['status']
        color = STATUS_COLORS.get(status, (200, 200, 200))
        drowsy_pct = (s['drowsy_frames'] / max(1, s['total_frames'])) * 100

        cv2.rectangle(panel, (8, y), (PANEL_W - 8, y + 110), (40, 40, 55), -1)
        cv2.rectangle(panel, (8, y), (PANEL_W - 8, y + 110), color, 1)
        cv2.putText(panel, f"Student {fid + 1}", (16, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.rectangle(panel, (14, y + 32), (PANEL_W - 14, y + 58), color, -1)

        # Handle text color dynamically based on background
        text_color = (0, 0, 0)
        if 'DROWSY' in status:
            text_color = (255, 255, 255)

        cv2.putText(panel, status, (18, y + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 2)

        ear_val = list(s['ear_history'])[-1] if s['ear_history'] else 0
        ear_bar = int(np.clip(ear_val / 0.4 * (PANEL_W - 40), 0, PANEL_W - 40))
        cv2.putText(panel, f"EAR: {ear_val:.2f}", (14, y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.rectangle(panel, (14, y + 80), (PANEL_W - 26, y + 88), (60, 60, 60), -1)
        cv2.rectangle(panel, (14, y + 80), (14 + ear_bar, y + 88), color, -1)
        cv2.putText(panel, f"Drowsy: {drowsy_pct:.1f}%", (14, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180),
                    1)
        y += 120

    cv2.putText(panel, "Press Q to quit", (10, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)


# ==========================================
# 6. MAIN LOOP
# ==========================================
ALARM_ON_GLOBAL = False
frame_count = 0
fps_buffer = deque(maxlen=30)
prev_time = time.time()

print("[INFO] Starting Webcam Thread... Press 'q' to exit.")
cap = WebcamStream(src=0).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    curr_time = time.time()
    fps_buffer.append(1.0 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time
    fps = np.mean(fps_buffer)

    small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    fh, fw = frame.shape[:2]
    sh, sw = small.shape[:2]
    sx, sy = fw / sw, fh / sh

    any_drowsy = False

    if result.multi_face_landmarks:
        cleanup_old_faces()

        for face_id, face_lm in enumerate(result.multi_face_landmarks):
            landmarks = face_lm.landmark
            state = get_face_state(face_id)
            state['last_seen'] = time.time()
            state['total_frames'] += 1

            re_ear, re_coords_s = aspect_ratio([landmarks[i] for i in RIGHT_EYE], sw, sh)
            le_ear, le_coords_s = aspect_ratio([landmarks[i] for i in LEFT_EYE], sw, sh)
            raw_ear = (re_ear + le_ear) / 2.0
            mar, m_coords_s = get_mar(landmarks, sw, sh)

            state['ear_history'].append(raw_ear)
            state['mar_history'].append(mar)
            ear = np.mean(state['ear_history'])
            smooth_mar = np.mean(state['mar_history'])

            re_coords = (re_coords_s * [sx, sy]).astype(int)
            le_coords = (le_coords_s * [sx, sy]).astype(int)
            m_coords = (m_coords_s * [sx, sy]).astype(int)

            if frame_count % 4 == face_id % 4:
                all_eyes = np.vstack((re_coords, le_coords))
                ex1, ey1, ex2, ey2 = get_bounding_box(all_eyes)
                mx1, my1, mx2, my2 = get_bounding_box(m_coords, pad=20)
                eye_crop = frame[max(0, ey1):ey2, max(0, ex1):ex2]
                mouth_crop = frame[max(0, my1):my2, max(0, mx1):mx2]
                eye_mse, yawn_mse = run_ae(eye_crop, mouth_crop)
                state['last_eye_mse'] = eye_mse
                state['last_yawn_mse'] = yawn_mse
            else:
                eye_mse = state['last_eye_mse']
                yawn_mse = state['last_yawn_mse']

            # Detection Logic
            ear_says_closed = ear < EAR_THRESH
            mse_says_closed = eye_mse > MSE_EYE_THRESH
            is_eyes_closed = ear_says_closed or mse_says_closed
            is_yawning = (smooth_mar > MAR_THRESH) or (yawn_mse > MSE_YAWN_THRESH)

            if is_eyes_closed:
                state['closed_counter'] += 1
                state['drowsy_frames'] += 1
            else:
                state['closed_counter'] = max(0, state['closed_counter'] - 2)

            if state['alert_cooldown'] > 0:
                state['alert_cooldown'] -= 1

            # ---> NEW RAPID-TRANSITION LOGIC <---
            is_drowsy_eyes = state['closed_counter'] >= CLOSED_FRAMES_THRESHOLD

            if is_drowsy_eyes and is_yawning:
                state['status'] = 'DROWSY & YAWNING'
                any_drowsy = True
            elif is_drowsy_eyes:
                state['status'] = 'DROWSINESS DETECTED'
                any_drowsy = True
            elif state['closed_counter'] >= 2:  # Instantly flashes EYES CLOSING before alarm
                state['status'] = 'EYES CLOSING'
            elif is_yawning:
                state['status'] = 'YAWNING'
            else:
                state['status'] = 'OPEN EYES'

            dot_color = STATUS_COLORS.get(state['status'], (0, 255, 0))
            for p in np.vstack((re_coords, le_coords, m_coords)):
                cv2.circle(frame, tuple(p), 2, dot_color, -1)

            head_y = int(min(re_coords_s[:, 1]) * sy) - 15
            head_x = int(np.mean(re_coords_s[:, 0]) * sx) - 20
            cv2.putText(frame, f"S{face_id + 1}", (max(0, head_x), max(15, head_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        dot_color, 2)

    panel = np.zeros((fh, PANEL_W, 3), dtype=np.uint8)
    total_faces = len(result.multi_face_landmarks) if result.multi_face_landmarks else 0
    draw_panel(panel, face_states, fps, total_faces)

    combined = np.hstack((frame, panel))

    if any_drowsy:
        if not ALARM_ON_GLOBAL and alarm_sound:
            alarm_sound.play(-1)
            ALARM_ON_GLOBAL = True
    else:
        if ALARM_ON_GLOBAL and alarm_sound:
            alarm_sound.stop()
            ALARM_ON_GLOBAL = False

    cv2.imshow("Classroom Drowsiness Detection", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
pygame.mixer.quit()
