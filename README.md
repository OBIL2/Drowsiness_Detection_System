🎓 The Classroom Drowsiness Detection System

The Goal: Keep students awake and engaged without embarrassing them! 
🤫 Instead of relying on a teacher's subjective observation, this system uses a standard webcam and advanced AI to act as a silent, digital monitor.

Here is how the magic works, broken down into two main pillars:

📐 Pillar 1: The Geometry (MediaPipe)
First, the system needs to "see" the face. We use Google's MediaPipe to map out 468 invisible dots on a student's face.

👁️ Eye Aspect Ratio (EAR): By measuring the distance between the top and bottom eyelids, the system calculates a ratio. If the number drops below 0.215, the system knows the eye is physically closing.

🥱 Mouth Aspect Ratio (MAR): It does the same for the mouth. If the lips stretch too far apart, it flags a yawn!

🧠 Pillar 2: The Deep Learning (Autoencoders)
Geometry alone isn't enough—what if a student is just wearing thick glasses or looking down? That's where our unique AI comes in.

🛡️ One-Class Classification: Instead of training the AI on thousands of sleeping faces, we trained it only on awake faces. It is an expert at knowing what a normal, open eye looks like.

🚨 Anomaly Detection (MSE): When a student closes their eyes, the AI gets confused because it has never seen a closed eye before! It fails to reconstruct the image. This "failure" is measured as a Mean Squared Error (MSE) score. A high error score mathematically proves the student is falling asleep.

🤝 The "AND" Logic (The Secret Sauce)
To prevent false alarms, the system requires both pillars to agree before sending an alert to the teacher:

Are the eyes geometrically closed? (Low EAR) AND...

Is the AI failing to recognize the image? (High MSE)
👉 If YES to both = Drowsiness Detected!
