# py -3.11 -m pip install numpy==1.26.4 keras==3.13.2 tensorflow==2.19.0 mediapipe==0.10.14 protobuf==4.25.3 opencv-python==4.8.0.76 scipy pygame flask twilio --no-cache-dir
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

os.makedirs("known_students", exist_ok=True)

# ==========================================
# 1. CLASSLENS AUTHENTICATION & TWILIO
# ==========================================
TEACHERS = {'obil': 'obil123', 'inderias': 'inderias123', 'farhan': 'farhan123'}

TWILIO_SID = 'pass'
TWILIO_AUTH_TOKEN = 'pass'
TWILIO_FROM = 'whatsapp:+14155238886'
TEACHER_PHONE = 'whatsapp:+9212121212112'

try:
    from twilio.rest import Client

    twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    TWILIO_ENABLED = True
    print("[INFO] ClassLens: Twilio WhatsApp ready!")
except Exception as e:
    print(f"[WARNING] Twilio not available: {e}")
    TWILIO_ENABLED = False

app = Flask(__name__)
app.secret_key = 'secure_classlens_key_2026'

# --- MASTER PERSISTENT DATABASE ---
class_session = {'is_active': False, 'start_time': 0}
master_student_records = {}
face_states = {}
dashboard_data = {}
engagement_prompt = {'show': False, 'message': '', 'level': 0}


def get_master_record(name):
    if name not in master_student_records:
        master_student_records[name] = {
            'incident_count': 0, 'strike_times': [],
            'drowsy_frames': 0, 'total_frames': 0, 'alert_sent': False
        }
    return master_student_records[name]


# ==========================================
# 2. LOAD AI MODELS (DROWSINESS)
# ==========================================
print("[INFO] Loading ClassLens Drowsiness Models...")
eye_ae = load_model('eye_autoencoder.h5', compile=False)
yawn_ae = load_model('yawn_autoencoder.h5', compile=False)

dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
_ = eye_ae(dummy, training=False)
_ = yawn_ae(dummy, training=False)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.6,
                                  min_tracking_confidence=0.6)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================================
# 3. BUILD BIOMETRIC DATABASE (SIFT)
# ==========================================
print("[INFO] Booting OpenCV SIFT Biometric Engine...")
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

print("[INFO] Generating SIFT Constellations from known_students folder...")
known_descriptors = {}
for file in os.listdir("known_students"):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join("known_students", file)
        img = cv2.imread(img_path)
        if img is not None:
            res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                fh, fw = img.shape[:2]
                coords = np.array([(int(p.x * fw), int(p.y * fh)) for p in res.multi_face_landmarks[0].landmark])
                x_min, y_min = max(0, np.min(coords[:, 0]) - 10), max(0, np.min(coords[:, 1]) - 10)
                x_max, y_max = min(fw, np.max(coords[:, 0]) + 10), min(fh, np.max(coords[:, 1]) + 10)

                tight_crop = img[y_min:y_max, x_min:x_max]
                if tight_crop.size > 0:
                    gray_crop = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY)
                    # BUG FIX: Removed resizing so SIFT can analyze the natural face shape
                    kp, des = sift.detectAndCompute(gray_crop, None)
                    if des is not None:
                        name = file.split('.')[0].capitalize()
                        known_descriptors[name] = des

print(f"[INFO] Successfully loaded {len(known_descriptors)} student identities.")


# ==========================================
# 4. HELPER FUNCTIONS
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
    return (chin_y - forehead_y) < 20 or landmarks[1].y > 0.80


def is_head_tilted(landmarks, frame_w, frame_h):
    x1, y1 = landmarks[33].x * frame_w, landmarks[33].y * frame_h
    x2, y2 = landmarks[263].x * frame_w, landmarks[263].y * frame_h
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return abs(angle) > 20


def send_whatsapp_alert(message_body):
    if not TWILIO_ENABLED: return False
    try:
        twilio_client.messages.create(from_=TWILIO_FROM, body=message_body, to=TEACHER_PHONE)
        return True
    except Exception as e:
        print(f"\n[CRITICAL TWILIO ERROR] {e}\n")
        return str(e)


# ==========================================
# 5. THRESHOLDS & STATE
# ==========================================
EAR_THRESH = 0.18
MSE_EYE_THRESH = 0.046
MAR_THRESH = 0.40
MSE_YAWN_THRESH = 0.15
CLOSED_FRAMES_THRESHOLD = 20
EAR_HISTORY_LEN = 5
COOLDOWN_SECONDS = 15


def get_face_state(face_id):
    if face_id not in face_states:
        face_states[face_id] = {
            'name': None,
            'closed_counter': 0, 'last_seen': time.time(),
            'ear_history': deque(maxlen=EAR_HISTORY_LEN),
            'mar_history': deque(maxlen=EAR_HISTORY_LEN),
            'status': 'STANDBY', 'last_eye_mse': 0, 'last_yawn_mse': 0,
            'last_incident_time': 0, 'head_tilt_start': None,
            'last_x': 0, 'last_y': 0,
            'next_sift_scan': 0
        }
    return face_states[face_id]


def cleanup_old_faces():
    now = time.time()
    for fid in [f for f, s in face_states.items() if now - s['last_seen'] > 3.0]:
        face_states.pop(fid, None)
        dashboard_data.pop(fid, None)


STATUS_COLORS = {
    'STANDBY': (100, 100, 100), 'OPEN EYES': (0, 220, 0),
    'EYES CLOSING': (0, 165, 255), 'DROWSINESS DETECTED': (0, 0, 255),
    'YAWNING': (0, 220, 220), 'HEAD DOWN': (128, 0, 128),
    'HEAD TILTED': (192, 192, 192)
}


# ==========================================
# 6. VIDEO STREAM GENERATOR
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

                re_ear, re_coords_s = aspect_ratio([landmarks[i] for i in RIGHT_EYE], sw, sh)
                le_ear, le_coords_s = aspect_ratio([landmarks[i] for i in LEFT_EYE], sw, sh)
                mar, m_coords_s = get_mar(landmarks, sw, sh)

                re_coords = (re_coords_s * [sx, sy]).astype(int)
                le_coords = (le_coords_s * [sx, sy]).astype(int)
                m_coords = (m_coords_s * [sx, sy]).astype(int)

                all_pts = np.vstack((re_coords, le_coords, m_coords))
                ex1, ey1, ex2, ey2 = get_bounding_box(all_pts, pad=20)

                # --- ANTI-SPOOF: TELEPORT DETECTION ---
                center_x = int(np.mean(all_pts[:, 0]))
                center_y = int(np.mean(all_pts[:, 1]))

                # BUG FIX: Loosened distance from 80px to 200px to prevent accidental resets
                if state['last_x'] != 0:
                    if abs(center_x - state['last_x']) > 200 or abs(center_y - state['last_y']) > 200:
                        state['name'] = "Unknown"
                        state['closed_counter'] = 0

                state['last_x'] = center_x
                state['last_y'] = center_y

                # --- ANTI-LAG: THROTTLED SIFT MATCHER ---
                if state['name'] in [None, "Unknown"] and known_descriptors and curr_time > state['next_sift_scan']:
                    state['next_sift_scan'] = curr_time + 1.0
                    live_crop = frame[max(0, ey1):ey2, max(0, ex1):ex2]

                    if live_crop.shape[0] > 40 and live_crop.shape[1] > 40:
                        gray_live = cv2.cvtColor(live_crop, cv2.COLOR_BGR2GRAY)
                        # BUG FIX: Removed resizing here as well
                        kp_live, des_live = sift.detectAndCompute(gray_live, None)

                        if des_live is not None and len(des_live) > 5:
                            best_match = "Unknown"
                            max_good_matches = 0
                            for known_name, known_des in known_descriptors.items():
                                try:
                                    matches = bf.knnMatch(known_des, des_live, k=2)
                                    good_matches = sum(1 for m, n in matches if m.distance < 0.75 * n.distance)

                                    print(f"[DEBUG] SIFT Matches for {known_name}: {good_matches}")
                                    if good_matches > max_good_matches:
                                        max_good_matches = good_matches
                                        best_match = known_name
                                except Exception:
                                    pass

                            # TUNING
                            if max_good_matches >= 13:
                                state['name'] = best_match
                            else:
                                state['name'] = "Unknown"

                mx1, my1, mx2, my2 = get_bounding_box(m_coords, pad=20)
                head_is_down = is_head_down(landmarks, fh)
                head_is_tilted = is_head_tilted(landmarks, sw, sh)

                # --- HEAD TILT STOPWATCH ---
                if head_is_tilted:
                    if state['head_tilt_start'] is None: state['head_tilt_start'] = curr_time
                    tilt_duration = curr_time - state['head_tilt_start']
                else:
                    state['head_tilt_start'] = None
                    tilt_duration = 0

                state['ear_history'].append((re_ear + le_ear) / 2.0)
                state['mar_history'].append(mar)
                ear = np.mean(state['ear_history'])
                smooth_mar = np.mean(state['mar_history'])

                if int(curr_time * 30) % 4 == face_id % 4:
                    ex1_e, ey1_e, ex2_e, ey2_e = get_bounding_box(np.vstack((re_coords, le_coords)))
                    eye_mse, yawn_mse = run_ae(
                        frame[max(0, ey1_e):ey2_e, max(0, ex1_e):ex2_e],
                        frame[max(0, my1):my2, max(0, mx1):mx2]
                    )
                    state['last_eye_mse'], state['last_yawn_mse'] = eye_mse, yawn_mse
                else:
                    eye_mse, yawn_mse = state['last_eye_mse'], state['last_yawn_mse']

                is_eyes_closed = (ear < EAR_THRESH) or (eye_mse > MSE_EYE_THRESH)
                is_yawning = (smooth_mar > MAR_THRESH) or (yawn_mse > MSE_YAWN_THRESH)

                display_name = state['name'] if state['name'] and state[
                    'name'] != "Unknown" else f"Student {face_id + 1}"
                master_record = get_master_record(display_name)

                if class_session['is_active']:
                    master_record['total_frames'] += 1
                    is_tilt_drowsy = tilt_duration > 4.0

                    if (is_eyes_closed and not head_is_down and not head_is_tilted) or is_tilt_drowsy:
                        state['closed_counter'] += 1
                        master_record['drowsy_frames'] += 1
                    else:
                        state['closed_counter'] = max(0, state['closed_counter'] - 2)

                    is_drowsy = state['closed_counter'] >= CLOSED_FRAMES_THRESHOLD

                    if head_is_down:
                        state['status'] = 'HEAD DOWN'
                    elif is_drowsy:
                        state['status'] = 'DROWSINESS DETECTED'
                    elif head_is_tilted:
                        state['status'] = 'HEAD TILTED'
                    elif state['closed_counter'] >= 5:
                        state['status'] = 'EYES CLOSING'
                    elif is_yawning:
                        state['status'] = 'YAWNING'
                    else:
                        state['status'] = 'OPEN EYES'

                    if is_drowsy and (curr_time - state['last_incident_time'] > COOLDOWN_SECONDS):
                        master_record['incident_count'] += 1
                        state['last_incident_time'] = curr_time
                        strike_timestamp = time.strftime('%H:%M:%S')
                        master_record['strike_times'].append(strike_timestamp)

                        if master_record['incident_count'] == 3 and not master_record['alert_sent']:
                            master_record['alert_sent'] = True
                            msg_body = (
                                f"🚨 DROWSINESS ALERT\n"
                                f"{display_name} has fallen asleep 3 times.\n"
                                f"Time: {strike_timestamp}\n"
                                f"Recommended: Check on the student."
                            )
                            Thread(target=send_whatsapp_alert, args=(msg_body,), daemon=True).start()

                    max_strikes = max((r['incident_count'] for r in master_student_records.values()), default=0)
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
                else:
                    state['status'] = 'STANDBY'

                dot_color = STATUS_COLORS.get(state['status'], (0, 255, 0))
                for p in np.vstack((re_coords, le_coords, m_coords)):
                    cv2.circle(frame, tuple(p), 2, dot_color, -1)

                head_y = int(min(re_coords_s[:, 1]) * sy) - 15
                head_x = int(np.mean(re_coords_s[:, 0]) * sx) - 20

                cv2.putText(frame, f"{display_name} | {state['status']}", (max(0, head_x), max(15, head_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2)

                drowsy_pct = (master_record['drowsy_frames'] / max(1, master_record['total_frames'])) * 100 if \
                master_record['total_frames'] > 0 else 0
                dashboard_data[face_id] = {
                    'name': display_name,
                    'status': state['status'],
                    'strikes': master_record['incident_count'],
                    'strike_times': " | ".join(master_record['strike_times']) if master_record[
                        'strike_times'] else "None",
                    'drowsy_pct': round(drowsy_pct, 1),
                    'ear': round(float(ear), 3), 'mar': round(float(smooth_mar), 3)
                }
        else:
            cleanup_old_faces()

        ret2, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ==========================================
# 7. FLASK ROUTES & SECURITY
# ==========================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    lockout_timestamp = None

    # 1. Check if the user is in the 30-Second penalty box
    if 'lockout_time' in session:
        remaining = int(session['lockout_time'] - time.time())
        if remaining > 0:
            lockout_timestamp = session['lockout_time']
        else:
            session.pop('lockout_time', None)
            session['failed_attempts'] = 0

    if request.method == 'POST':
        # Block POSTs entirely if the timer is still running
        if lockout_timestamp:
            return render_template('login.html', error=error, lockout_timestamp=lockout_timestamp)

        username = request.form.get('username', '').lower().strip()
        password = request.form.get('password', '')

        if 'failed_attempts' not in session: session['failed_attempts'] = 0

        if username in TEACHERS and TEACHERS[username] == password:
            session.pop('failed_attempts', None)
            session.pop('lockout_time', None)
            session['logged_in'] = True
            session['teacher_name'] = username.capitalize()
            return redirect(url_for('index'))
        else:
            session['failed_attempts'] += 1
            attempts_left = 5 - session['failed_attempts']

            if session['failed_attempts'] >= 5:
                # Issue the 30-second penalty
                session['lockout_time'] = time.time() + 30
                lockout_timestamp = session['lockout_time']
            else:
                error = f"Invalid Credentials. You have {attempts_left} attempts remaining."

    return render_template('login.html', error=error, lockout_timestamp=lockout_timestamp)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'logged_in' not in session: return redirect(url_for('login'))
    return render_template('index.html', teacher_name=session.get('teacher_name', 'Instructor'))


@app.route('/video_feed')
def video_feed():
    if 'logged_in' not in session: return "Unauthorized", 401
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dashboard_data')
def get_dashboard_data():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({
        'students': dashboard_data,
        'engagement': engagement_prompt,
        'session_active': class_session['is_active']
    })


@app.route('/api/session/start', methods=['POST'])
def start_session():
    class_session['is_active'] = True
    class_session['start_time'] = time.time()
    master_student_records.clear()
    face_states.clear()
    dashboard_data.clear()
    engagement_prompt.update({'show': False, 'message': '', 'level': 0})
    return jsonify({'status': 'started'})


@app.route('/api/session/end', methods=['POST'])
def end_session():
    class_session['is_active'] = False
    mins, secs = divmod(int(time.time() - class_session['start_time']), 60)
    report = {'duration': f"{mins}m {secs}s", 'students': dashboard_data.copy()}
    return jsonify({'status': 'ended', 'report': report})


@app.route('/test_whatsapp')
def test_whatsapp():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    result = send_whatsapp_alert("🛠️ CLASSLENS SYSTEM TEST: API is perfectly connected!")
    if result is True:
        return jsonify({'status': 'Success! Check your phone.'})
    else:
        return jsonify({'status': 'Failed.', 'error': result})


if __name__ == '__main__':
    print("[INFO] Starting ClassLens Server → http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, use_reloader=False)
