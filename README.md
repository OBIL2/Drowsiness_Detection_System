# 🎓 ClassLens: The Classroom Drowsiness Detection System

**The Goal:** Keep students awake and engaged without embarrassing them! 
🤫 Instead of relying on a teacher's subjective observation, this system uses a standard webcam and advanced AI to act as a silent, digital monitor.

---

## ✨ System Features
* Includes a secure web page frontend with a teacher login and password system.
* Detects if a student puts their head down or tilts it on their shoulder for too long.
* Instantly recognizes student faces using lightweight math so the computer doesn't lag.
* Catches students immediately if they leave the frame and try to switch seats.
* Uses smart AI to accurately track closed eyes and yawning.
* Remembers a student's sleepiness score for the whole class, even if they walk off-camera.
* Gives teachers a private live dashboard to watch the focus levels of the entire room.
* Automatically saves a final CSV report with all the data when the class ends.
* Sends a direct WhatsApp text to the teacher if a student falls asleep three times.
* Suggests the teacher make an adjustment, like cracking a joke, if the class gets tired.
* Protects the system by locking the login screen for 30 seconds after five wrong guesses.

---

## 🧠 How the Magic Works

Here is how the magic works, broken down into two main pillars:

### 📐 Pillar 1: The Geometry (MediaPipe)
First, the system needs to "see" the face. We use Google's MediaPipe to map out 468 invisible dots on a student's face.

* **👁️ Eye Aspect Ratio (EAR):** By measuring the distance between the top and bottom eyelids, the system calculates a ratio. If the number drops below 0.215, the system knows the eye is physically closing.
* **🥱 Mouth Aspect Ratio (MAR):** It does the same for the mouth. If the lips stretch too far apart, it flags a yawn!

### 🤖 Pillar 2: The Deep Learning (Autoencoders)
Geometry alone isn't enough—what if a student is just wearing thick glasses or looking down? That's where our unique AI comes in.

* **🛡️ One-Class Classification:** Instead of training the AI on thousands of sleeping faces, we trained it only on awake faces. It is an expert at knowing what a normal, open eye looks like.
* **🚨 Anomaly Detection (MSE):** When a student closes their eyes, the AI gets confused because it has never seen a closed eye before! It fails to reconstruct the image. This "failure" is measured as a Mean Squared Error (MSE) score. A high error score mathematically proves the student is falling asleep.

### 🤝 The "AND" Logic (The Secret Sauce)
To prevent false alarms, the system requires both pillars to agree before sending an alert to the teacher:

* *Are the eyes geometrically closed? (Low EAR)* **AND...**
* *Is the AI failing to recognize the image? (High MSE)*

👉 **If YES to both = Drowsiness Detected!**

---

## 🚀 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ClassLens.git
cd ClassLens
```

---

## 📦 2. Install Dependencies

Run the following command in your terminal to install exactly what the system needs to run:

```bash
pip install numpy==1.26.4 keras==3.13.2 tensorflow==2.19.0 mediapipe==0.10.14 protobuf==4.25.3 opencv-python==4.8.0.76 scipy flask twilio
```

---

## 📂 3. File Structure Requirements

Make sure your project folder looks like this before running the code:

* `main_system.py` (The main code file)
* `eye_autoencoder.h5` (The AI model for eyes)
* `yawn_autoencoder.h5` (The AI model for yawning)
* `templates/` (Folder containing your `login.html` and `index.html`)
* `known_students/` (Folder containing clear `.jpg` pictures of your students' faces)

---

## 🔧 4. API Configuration (Optional)

If you want the WhatsApp alerts to work, open `main_system.py` and put your Twilio details in these lines:

```python
TWILIO_SID        = 'your_sid_here'
TWILIO_AUTH_TOKEN = 'your_token_here'
TWILIO_FROM       = 'whatsapp:+14155238886'
TEACHER_PHONE     = 'whatsapp:+your_number'
```

---

## 🎮 Usage Guide

1. Open your terminal and run the main file:

```bash
python main_system.py
```

2. Open your web browser and go to: `http://127.0.0.1:5000`
3. Log in using your teacher username and password.
4. Click **Start Session** on the dashboard.
5. The webcam will turn on, the AI will start scanning the room, and the live data will instantly show up on your screen!
