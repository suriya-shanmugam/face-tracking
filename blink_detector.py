import cv2
import numpy as np
import mediapipe as mp
import argparse
from collections import deque

"""
Eye state & blink detection using EAR (Eye Aspect Ratio) with MediaPipe Face Mesh.

- Computes EAR for left & right eyes each frame.
- Uses hysteresis thresholds + short temporal smoothing to reduce flicker.
- Counts blinks on closed->open transitions within a reasonable duration.

Run:
  python ear_blink.py              # webcam
  python ear_blink.py --src video.mp4

Keys:
  q = quit
"""

# --- MediaPipe face mesh setup ---
mp_face_mesh = mp.solutions.face_mesh

# We’ll use 6 landmarks per eye to mimic the classic EAR definition.
# Indices chosen around the eyelids; these are common choices for blink detection.
# (Based on MediaPipe’s 468-point topology; they’re camera/view independent.)
RIGHT_EYE_IDS = [33, 160, 158, 133, 153, 144]  # p1 p2 p3 p4 p5 p6
LEFT_EYE_IDS  = [263, 387, 385, 362, 380, 373] # p1 p2 p3 p4 p5 p6

def euclidean(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))

def eye_aspect_ratio(pts):
    """
    pts: list of 6 (x, y) points in pixel coordinates ordered as:
         [p1, p2, p3, p4, p5, p6]
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    p1, p2, p3, p4, p5, p6 = pts
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4) + 1e-6  # avoid div by zero
    return (A + B) / (2.0 * C)

def landmarks_to_points(landmarks, ids, w, h):
    """Convert selected landmark indices to pixel (x,y)."""
    pts = []
    for i in ids:
        lm = landmarks[i]
        pts.append((lm.x * w, lm.y * h))
    return pts

def moving_median(queue):
    arr = np.array(queue, dtype=np.float32)
    return float(np.median(arr)) if len(arr) else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="0",
                        help="Video source. '0' for default webcam, or path to video file.")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Frames for median smoothing of EAR.")
    parser.add_argument("--ear_close", type=float, default=0.21,
                        help="EAR threshold to enter CLOSED state (hysteresis low).")
    parser.add_argument("--ear_open", type=float, default=0.25,
                        help="EAR threshold to exit to OPEN state (hysteresis high).")
    parser.add_argument("--min_closed", type=int, default=2,
                        help="Min consecutive CLOSED frames to consider a valid closed segment.")
    parser.add_argument("--max_blink", type=int, default=10,
                        help="Max consecutive CLOSED frames to still count as a blink (beyond this may be prolonged closure).")
    parser.add_argument("--no_draw", action="store_true", help="Disable drawing for max speed.")
    args = parser.parse_args()

    # OpenCV capture
    cap = cv2.VideoCapture(0 if args.src == "0" else args.src)
    if not cap.isOpened():
        print("Error: cannot open video source:", args.src)
        return

    # MediaPipe Face Mesh: choose a lightweight config for real-time
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True, # helpful around eyes/iris
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Temporal state
    ear_hist = deque(maxlen=args.smooth)
    state = "OPEN"               # current debounced eye state
    consec_closed = 0
    consec_open = 0
    blink_count = 0

    # For FPS
    fps_avg = deque(maxlen=30)
    t_prev = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # Mediapipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_val = None
        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0].landmark

            left_pts  = landmarks_to_points(face_lms, LEFT_EYE_IDS,  w, h)
            right_pts = landmarks_to_points(face_lms, RIGHT_EYE_IDS, w, h)

            ear_left  = eye_aspect_ratio(left_pts)
            ear_right = eye_aspect_ratio(right_pts)
            ear_val   = (ear_left + ear_right) / 2.0

            ear_hist.append(ear_val)
            ear_smooth = moving_median(ear_hist)
        else:
            # if face not detected, hold previous EAR (or set None)
            ear_smooth = ear_val

        # --- Hysteresis-based state machine ---
        if ear_smooth is not None:
            if state == "OPEN":
                if ear_smooth < args.ear_close:
                    consec_closed += 1
                    consec_open = 0
                    if consec_closed >= args.min_closed:
                        state = "CLOSED"
                else:
                    consec_open += 1
                    consec_closed = 0
            else:  # state == "CLOSED"
                if ear_smooth > args.ear_open:
                    consec_open += 1
                    consec_closed = 0
                    if consec_open >= 1:
                        # Count a blink only if the closed segment wasn't too long
                        if 0 < consec_open and 1 <= consec_closed <= args.max_blink:
                            pass  # consec_closed reset already, so we can’t use it here
                        state = "OPEN"
                        blink_count += 1
                        consec_open = 0
                else:
                    consec_closed += 1
                    consec_open = 0

        # --- Draw UI ---
        if not args.no_draw:
            # FPS
            t_now = cv2.getTickCount()
            dt = (t_now - t_prev) / tick_freq
            t_prev = t_now
            if dt > 0:
                fps_avg.append(1.0 / dt)
            fps = np.mean(fps_avg) if len(fps_avg) else 0.0

            # Top panel
            panel_h = 60
            cv2.rectangle(frame, (0, 0), (w, panel_h), (0, 0, 0), -1)

            # Status text
            txt_state = f"State: {state}"
            txt_blinks = f"Blinks: {blink_count}"
            txt_ear = f"EAR: {ear_smooth:.3f}" if ear_smooth is not None else "EAR: --"
            txt_fps = f"FPS: {fps:.1f}"

            cv2.putText(frame, txt_state, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if state=="OPEN" else (0, 200, 0), 2)
            cv2.putText(frame, txt_blinks, (220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, txt_ear, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, txt_fps, (220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Optional: draw the 6 eye points used for EAR
            if results.multi_face_landmarks:
                for (eye_pts, color) in [(left_pts, (255, 0, 0)), (right_pts, (0, 255, 0))]:
                    for (x, y) in eye_pts:
                        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

            cv2.imshow("EAR Blink Detector (MediaPipe)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

