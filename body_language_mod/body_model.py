import cv2
import mediapipe as mp
import numpy as np

#model pretrained


def init_model():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose_model




# --- Normalize utility ---
def normalize(value, min_val, max_val):
    return round(10 * np.clip((value - min_val) / (max_val - min_val), 0, 1), 1)


#### SCORING FUNCTIONS ####

# Leaning

def leaning_score(landmarks):
    visible = all(landmarks[i].visibility > 0.5 for i in [11, 12, 23, 24])
    if not visible:
        return None
    mid_shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
    return mid_shoulder_x - mid_hip_x

# Fidgeting

def fidgeting_score(landmark_sequence):
    total = 0
    valid_pairs = 0
    for i in range(1, len(landmark_sequence)):
        for idx in [15, 16]:
            if landmark_sequence[i][idx].visibility > 0.4 and landmark_sequence[i-1][idx].visibility > 0.4:
                dx = landmark_sequence[i][idx].x - landmark_sequence[i-1][idx].x
                dy = landmark_sequence[i][idx].y - landmark_sequence[i-1][idx].y
                dist = np.sqrt(dx**2 + dy**2)
                total += dist
                valid_pairs += 1
    return total if valid_pairs > 0 else None

# Hand gestures

def count_hand_gestures(landmark_sequence, threshold=0.05):
    count = 0
    for i in range(1, len(landmark_sequence)):
        for idx in [15, 16]:  # Left and right wrists
            if landmark_sequence[i][idx].visibility > 0.5 and landmark_sequence[i-1][idx].visibility > 0.5:
                dx = landmark_sequence[i][idx].x - landmark_sequence[i-1][idx].x
                dy = landmark_sequence[i][idx].y - landmark_sequence[i-1][idx].y
                dist = np.sqrt(dx**2 + dy**2)
                if dist > threshold:
                    count += 1
    return count

# Posture

def posture_score(landmarks):
    # Shoulders aligned and stacked over hips
    if not all(landmarks[i].visibility > 0.5 for i in [11, 12, 23, 24]):
        return None
    shoulder_diff_y = abs(landmarks[11].y - landmarks[12].y)
    torso_angle = abs(((landmarks[11].x + landmarks[12].x)/2) - ((landmarks[23].x + landmarks[24].x)/2))
    return (1 - (shoulder_diff_y + torso_angle))  # closer to 1 is better


# Space occupation

def space_occupation_score(landmark_sequence):
    total_motion = 0
    valid_frames = 0

    for i in range(1, len(landmark_sequence)):
        for idx in [15, 16]:  # Wrists
            lm1 = landmark_sequence[i][idx]
            lm0 = landmark_sequence[i - 1][idx]
            if lm1.visibility > 0.5 and lm0.visibility > 0.5:
                dx = lm1.x - lm0.x
                dy = lm1.y - lm0.y
                dist = np.sqrt(dx**2 + dy**2)
                total_motion += dist
                valid_frames += 1

    return total_motion / valid_frames if valid_frames > 0 else None

# --- Combine Scores with Confidence Filtering ---
######Function to interpret a posse with a landmark sequence => to give a score


def interpret_pose(landmark_sequence):
    if not landmark_sequence:
        return None

    current = landmark_sequence[-1]
    scores = {}
    valid_scores = []
    space_score = space_occupation_score(landmark_sequence)
    if space_score is not None:
        space_norm = normalize(space_score, 0.001, 0.02)  # Tune thresholds
        space_norm=10*space_norm
        scores["Space Occupation"] = space_norm
        valid_scores.append(space_norm)


    lean_score = leaning_score(current)
    if lean_score is not None:
        lean_norm = 10 - normalize(abs(lean_score), 0, 0.2)
        lean_norm=10*lean_norm
        scores["Leaning Stability"] = lean_norm

        valid_scores.append(lean_norm)

    fidg_score = fidgeting_score(landmark_sequence)
    if fidg_score is not None:
        fidg_norm = 10 - normalize(fidg_score, 0.005, 0.05)
        fidg_norm=10*fidg_norm
        scores["Composure Score"] = fidg_norm

        valid_scores.append(fidg_norm)

    gestures = count_hand_gestures(landmark_sequence)
    gesture_norm = normalize(gestures, 1, 20)  # higher is more expressive
    gesture_norm=10*gesture_norm
    scores["Hand Gesture Activity"] = gesture_norm
    if gesture_norm < 3:

        valid_scores.append(gesture_norm)
        valid_scores.append(gesture_norm)  # counts double
    else:
        valid_scores.append(gesture_norm)


    posture = posture_score(current)
    if posture is not None:
        posture_norm = normalize(posture, 0.7, 1.0)
        posture_norm=10*posture_norm
        scores["Posture Alignment"] = posture_norm

        valid_scores.append(posture_norm)

    if valid_scores:
        low_scores = [s for s in valid_scores if s <= 5 and s!=0]
    if len(low_scores) >= 1:
        max_low = max(low_scores)
        valid_scores.append(max_low)  # count max low score twice
    if valid_scores:
        scores["Overall Body Language Score"] = round(sum(valid_scores) / len(valid_scores), 1)

    return scores if scores else None

# --- Analyze a Specific Video File ---
def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    landmark_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(image_rgb)

        if results.pose_landmarks:
            landmark_sequence.append(results.pose_landmarks.landmark)

    cap.release()
    return interpret_pose(landmark_sequence)
video_path = "/Users/theaalfon/Desktop/TestVid.MOV"

print(analyze_video(video_path, init_model()))
