# py -3.11 -m pip install numpy==1.26.4 keras==3.13.2 tensorflow==2.19.0 mediapipe==0.10.5 opencv-python==4.8.0.76 scipy pygame flask twilio --no-cache-dir
# py -3.11 main_system.py

import os
import time
from threading import Thread
from collections import deque
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from scipy.spatial import distance as dist
from keras.models import load_model
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session

# ==========================================
# 1. TWILIO WHATSAPP SETUP
# ==========================================
TWILIO_SID = 'pass'
TWILIO_AUTH_TOKEN = 'pass'
TWILIO_FROM = 'whatsapp:+14155238886'
TEACHER_PHONE = 'whatsapp:+9267676767'

try:
    from twilio.rest import Client

    twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    TWILIO_ENABLED = True
    print("[INFO] Twilio WhatsApp ready!")
except Exception as e:
    print(f"[WARNING] Twilio not available: {e}")
    TWILIO_ENABLED = False

app = Flask(__name__)
app.secret_key = 'secure_admin_key_123'

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("[INFO] Loading Models...")
eye_ae = load_model('eye_autoencoder.h5', compile=False)
yawn_ae = load_model('yawn_autoencoder.h5', compile=False)

dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
_ = eye_ae(dummy, training=False)
_ = yawn_ae(dummy, training=False)
print("[INFO] Models warmed up!")

print("[INFO] Loading MediaPipe...")
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
# 3. HELPER FUNCTIONS
# ==========================================
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
    return dist.euclidean(top, bottom) / dist.euclidean(left, right), np.array([left, right, top, bottom])


def preprocess_for_ae(roi):
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(roi.astype("float32") / 255.0, axis=0)


def calculate_mse(a, b):
    return np.sum((a - b) ** 2) / float(a.shape[0] * a.shape[1] * a.shape[2])


def get_bounding_box(coords, pad=15):
    x_min, y_min = np.min(coords, axis=0) - pad
    x_max, y_max = np.max(coords, axis=0) + pad
    return int(x_min), int(y_min), int(x_max), int(y_max)


def run_ae(eye_crop, mouth_crop):
    eye_mse = yawn_mse = 0
    try:
        if eye_crop is not None and eye_crop.size > 0:
            pre = preprocess_for_ae(eye_crop)
            eye_mse = calculate_mse(pre[0], eye_ae(pre, training=False).numpy()[0])
        if mouth_crop is not None and mouth_crop.size > 0:
            pre = preprocess_for_ae(mouth_crop)
            yawn_mse = calculate_mse(pre[0], yawn_ae(pre, training=False).numpy()[0])
    except Exception:
        pass
    return eye_mse, yawn_mse


def is_head_down(landmarks, frame_h):
    nose_y = landmarks[1].y * frame_h
    forehead_y = landmarks[10].y * frame_h
    chin_y = landmarks[152].y * frame_h
    if (chin_y - forehead_y) < 20 or landmarks[1].y > 0.80:
        return True
    return False


def is_head_tilted(landmarks, frame_w, frame_h):
    x1, y1 = landmarks[33].x * frame_w, landmarks[33].y * frame_h
    x2, y2 = landmarks[263].x * frame_w, landmarks[263].y * frame_h
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return abs(angle) > 20


def send_whatsapp_alert(message_body):
    if not TWILIO_ENABLED:
        print("[WHATSAPP BLOCKED] Twilio not enabled.")
        return False
    try:
        print("\n--- [ATTEMPTING WHATSAPP SEND] ---")
        msg = twilio_client.messages.create(
            from_=TWILIO_FROM,
            body=message_body,
            to=TEACHER_PHONE
        )
        print(f"[SUCCESS] WhatsApp sent! SID: {msg.sid}\n")
        return True
    except Exception as e:
        print(f"\n[CRITICAL TWILIO ERROR] {e}\n")
        return str(e)


# ==========================================
# 4. THRESHOLDS & STATE (Reverted to your working ones)
# ==========================================
EAR_THRESH = 0.18  # Reverted
MSE_EYE_THRESH = 0.046
MAR_THRESH = 0.40  # Reverted
MSE_YAWN_THRESH = 0.15
CLOSED_FRAMES_THRESHOLD = 20  # Reverted
EAR_HISTORY_LEN = 5
COOLDOWN_SECONDS = 15  # Reverted
FORGIVENESS_SECONDS = 900

face_states = {}
dashboard_data = {}


def get_face_state(face_id):
    if face_id not in face_states:
        face_states[face_id] = {
            'closed_counter': 0, 'last_seen': time.time(),
            'ear_history': deque(maxlen=EAR_HISTORY_LEN),
            'mar_history': deque(maxlen=EAR_HISTORY_LEN),
            'status': 'OPEN EYES', 'last_eye_mse': 0, 'last_yawn_mse': 0,
            'incident_count': 0, 'last_incident_time': 0,
            'drowsy_frames': 0, 'total_frames': 0, 'alert_sent': False
        }
    return face_states[face_id]


def cleanup_old_faces():
    now = time.time()
    for fid in [f for f, s in face_states.items() if now - s['last_seen'] > 3.0]:
        face_states.pop(fid, None)
        dashboard_data.pop(fid, None)


STATUS_COLORS = {
    'OPEN EYES': (0, 220, 0),
    'EYES CLOSING': (0, 165, 255),
    'DROWSINESS DETECTED': (0, 0, 255),
    'YAWNING': (0, 220, 220),
    'HEAD DOWN': (128, 0, 128),
    'HEAD TILTED': (192, 192, 192)
}

engagement_prompt = {'show': False, 'message': '', 'level': 0}


# ==========================================
# 5. VIDEO STREAM GENERATOR
# ==========================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: continue

        curr_time = time.time()
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        result = face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        fh, fw = frame.shape[:2]
        sh, sw = small.shape[:2]
        sx, sy = fw / sw, fh / sh

        if result.multi_face_landmarks:
            cleanup_old_faces()
            for face_id, face_lm in enumerate(result.multi_face_landmarks):
                landmarks = face_lm.landmark
                state = get_face_state(face_id)
                state['last_seen'] = curr_time
                state['total_frames'] += 1

                head_is_down = is_head_down(landmarks, fh)
                head_is_tilted = is_head_tilted(landmarks, sw, sh)

                re_ear, re_coords_s = aspect_ratio([landmarks[i] for i in RIGHT_EYE], sw, sh)
                le_ear, le_coords_s = aspect_ratio([landmarks[i] for i in LEFT_EYE], sw, sh)
                mar, m_coords_s = get_mar(landmarks, sw, sh)

                state['ear_history'].append((re_ear + le_ear) / 2.0)
                state['mar_history'].append(mar)
                ear = np.mean(state['ear_history'])
                smooth_mar = np.mean(state['mar_history'])

                re_coords = (re_coords_s * [sx, sy]).astype(int)
                le_coords = (le_coords_s * [sx, sy]).astype(int)
                m_coords = (m_coords_s * [sx, sy]).astype(int)

                if int(curr_time * 30) % 4 == face_id % 4:
                    ex1, ey1, ex2, ey2 = get_bounding_box(np.vstack((re_coords, le_coords)))
                    mx1, my1, mx2, my2 = get_bounding_box(m_coords, pad=20)
                    eye_mse, yawn_mse = run_ae(
                        frame[max(0, ey1):ey2, max(0, ex1):ex2],
                        frame[max(0, my1):my2, max(0, mx1):mx2]
                    )
                    state['last_eye_mse'], state['last_yawn_mse'] = eye_mse, yawn_mse
                else:
                    eye_mse, yawn_mse = state['last_eye_mse'], state['last_yawn_mse']

                is_eyes_closed = (ear < EAR_THRESH) or (eye_mse > MSE_EYE_THRESH)
                is_yawning = (smooth_mar > MAR_THRESH) or (yawn_mse > MSE_YAWN_THRESH)

                if is_eyes_closed and not head_is_down and not head_is_tilted:
                    state['closed_counter'] += 1
                    state['drowsy_frames'] += 1
                else:
                    state['closed_counter'] = max(0, state['closed_counter'] - 2)

                is_drowsy = state['closed_counter'] >= CLOSED_FRAMES_THRESHOLD

                if head_is_down:
                    state['status'] = 'HEAD DOWN'
                elif head_is_tilted:
                    state['status'] = 'HEAD TILTED'
                elif is_drowsy:
                    state['status'] = 'DROWSINESS DETECTED'
                elif state['closed_counter'] >= 5:
                    state['status'] = 'EYES CLOSING'
                elif is_yawning:
                    state['status'] = 'YAWNING'
                else:
                    state['status'] = 'OPEN EYES'

                # --- Exact Strike 3 Trigger Logic ---
                if is_drowsy and (curr_time - state['last_incident_time'] > COOLDOWN_SECONDS):
                    state['incident_count'] += 1
                    state['last_incident_time'] = curr_time

                    if state['incident_count'] == 3 and not state['alert_sent']:
                        state['alert_sent'] = True
                        msg_body = f"🚨 URGENT: Student {face_id + 1} has hit exactly 3 drowsiness strikes. Intervention needed."
                        Thread(target=send_whatsapp_alert, args=(msg_body,), daemon=True).start()

                # Dashboard Prompts
                max_strikes = max((s['incident_count'] for s in face_states.values()), default=0)
                if max_strikes >= 4:
                    engagement_prompt.update(
                        {'show': True, 'level': 3, 'message': "Consider a break — engagement is low."})
                elif max_strikes == 3:
                    engagement_prompt.update(
                        {'show': True, 'level': 3, 'message': "Strike 3! WhatsApp alert dispatched."})
                elif max_strikes == 2:
                    engagement_prompt.update({'show': True, 'level': 2, 'message': "Strike 2: Crack a joke!"})
                else:
                    engagement_prompt['show'] = False

                dot_color = STATUS_COLORS.get(state['status'], (0, 255, 0))
                for p in np.vstack((re_coords, le_coords, m_coords)):
                    cv2.circle(frame, tuple(p), 2, dot_color, -1)

                head_y = int(min(re_coords_s[:, 1]) * sy) - 15
                head_x = int(np.mean(re_coords_s[:, 0]) * sx) - 20
                cv2.putText(frame, f"S{face_id + 1} | {state['status']}",
                            (max(0, head_x), max(15, head_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2)

                dashboard_data[face_id] = {
                    'status': state['status'], 'strikes': state['incident_count'],
                    'drowsy_pct': round((state['drowsy_frames'] / max(1, state['total_frames'])) * 100, 1),
                    'ear': round(float(ear), 3), 'mar': round(float(smooth_mar), 3)
                }
        else:
            cleanup_old_faces()

        ret2, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ==========================================
# 6. FLASK ROUTES
# ==========================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = "Invalid Credentials. Please try again."
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'logged_in' not in session: return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if 'logged_in' not in session: return "Unauthorized", 401
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dashboard_data')
def get_dashboard_data():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'students': dashboard_data, 'engagement': engagement_prompt})


# --- THE TWILIO TEST ROUTE ---
@app.route('/test_whatsapp')
def test_whatsapp():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    result = send_whatsapp_alert("🛠️ MANUAL TEST: If you see this, your Twilio API is perfectly connected!")
    if result is True:
        return jsonify({'status': 'Success! Check your phone.'})
    else:
        return jsonify({'status': 'Failed.', 'error': result})


if __name__ == '__main__':
    print("[INFO] Starting server → http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, use_reloader=False)
