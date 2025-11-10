import cv2
import mediapipe as mp
import math
import numpy as np

# --- Constants ---
# --- CALIBRATE THESE! ---
EAR_THRESHOLD = 0.25 # Default: 0.25. Tune this by closing your eyes.
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.5  # Default: 0.5. Tune this by yawning.
MAR_CONSEC_FRAMES = 15

# --- State Variables ---
EAR_COUNTER = 0
MAR_COUNTER = 0
ATTENTION_STATUS = "FOCUSED"

# --- Landmark Indices ---
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# --- NEW ROBUST MOUTH LANDMARKS ---
# Using 8 points on the inner lip for a more stable MAR
MOUTH_LANDMARKS = [
    78, 308,  # Horizontal (Left/Right corners)
    13, 14,   # Vertical (Top/Bottom inner)
    81, 311,  # Vertical (Top/Bottom inner)
    82, 312   # Vertical (Top/Bottom inner)
]


# --- Helper Functions ---

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_eye_aspect_ratio(landmarks, eye_indices):
    try:
        p1 = landmarks[eye_indices[0]] # Horizontal
        p4 = landmarks[eye_indices[3]] # Horizontal
        p2 = landmarks[eye_indices[1]] # Vertical
        p6 = landmarks[eye_indices[4]] # Vertical
        p3 = landmarks[eye_indices[2]] # Vertical
        p5 = landmarks[eye_indices[5]] # Vertical

        ver_dist1 = euclidean_distance(p2, p6)
        ver_dist2 = euclidean_distance(p3, p5)
        hor_dist = euclidean_distance(p1, p4)

        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
        return ear
    except:
        return 0.0

def get_mouth_aspect_ratio(landmarks, mouth_indices):
    """
    Calculates a more robust Mouth Aspect Ratio (MAR)
    Averages 3 vertical distances and divides by the horizontal distance
    of the inner lip corners.
    """
    try:
        # Horizontal distance (p_left - p_right)
        p_left = landmarks[mouth_indices[0]]
        p_right = landmarks[mouth_indices[1]]
        hor_dist = euclidean_distance(p_left, p_right)
        if hor_dist == 0: return 0.0 # Avoid division by zero

        # Vertical distances
        p_top1 = landmarks[mouth_indices[2]]
        p_bottom1 = landmarks[mouth_indices[3]]
        ver_dist1 = euclidean_distance(p_top1, p_bottom1)
        
        p_top2 = landmarks[mouth_indices[4]]
        p_bottom2 = landmarks[mouth_indices[5]]
        ver_dist2 = euclidean_distance(p_top2, p_bottom2)
        
        p_top3 = landmarks[mouth_indices[6]]
        p_bottom3 = landmarks[mouth_indices[7]]
        ver_dist3 = euclidean_distance(p_top3, p_bottom3)
        
        # Average vertical distance
        avg_ver_dist = (ver_dist1 + ver_dist2 + ver_dist3) / 3.0
        
        mar = avg_ver_dist / hor_dist
        return mar
    except:
        return 0.0

# --- Main Program ---
# (The main loop from "while cap.isOpened():" onwards is IDENTICAL
#  to the previous file. You can just copy/paste the functions
#  and constants above into your existing file.)
# ---

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")
print("---")
print("CALIBRATION GUIDE:")
print("1. Watch the MAR value while talking. Set MAR_THRESHOLD higher than this.")
print("2. Watch the MAR value while yawning. Set MAR_THRESHOLD lower than this.")
print("3. Watch the EAR value while blinking. Set EAR_THRESHOLD higher than this.")
print("4. Watch the EAR value with eyes closed. Set EAR_THRESHOLD lower than this.")
print("---")


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    ATTENTION_STATUS = "FOCUSED"
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- 1. Eye Aspect Ratio (EAR) ---
        left_ear = get_eye_aspect_ratio(face_landmarks, LEFT_EYE_LANDMARKS)
        right_ear = get_eye_aspect_ratio(face_landmarks, RIGHT_EYE_LANDMARKS)
        avg_ear = (left_ear + right_ear) / 2.0

        # --- 2. Mouth Aspect Ratio (MAR) ---
        mar = get_mouth_aspect_ratio(face_landmarks, MOUTH_LANDMARKS)

        # --- 3. Heuristics Engine ---
        if avg_ear < EAR_THRESHOLD:
            EAR_COUNTER += 1
            if EAR_COUNTER >= EAR_CONSEC_FRAMES:
                ATTENTION_STATUS = "DROWSY"
        else:
            EAR_COUNTER = 0

        # Check for yawn *after* drowsiness, as a yawn can override
        if mar > MAR_THRESHOLD:
            MAR_COUNTER += 1
            if MAR_COUNTER >= MAR_CONSEC_FRAMES:
                # Only set to YAWNING if not already drowsy (drowsy is more critical)
                if ATTENTION_STATUS != "DROWSY":
                    ATTENTION_STATUS = "YAWNING"
        else:
            MAR_COUNTER = 0

        # --- Draw information ---
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
        
        # Print raw values for easier calibration
        print(f"EAR: {avg_ear:.4f} | MAR: {mar:.4f}")

        cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        ATTENTION_STATUS = "NO FACE DETECTED"

    # --- Display Final Status ---
    if ATTENTION_STATUS == "FOCUSED":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    
    cv2.putText(image, f"STATUS: {ATTENTION_STATUS}", (10, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Attention Detector Prototype', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
