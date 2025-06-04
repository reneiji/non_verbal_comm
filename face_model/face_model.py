from deepface import DeepFace
import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, Grayscale
import torch.nn.functional as F
from PIL import Image
from collections import Counter

print("=== Starting face_model.py ===")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Recreate the model architecture
model_conf = models.resnet18(pretrained=False)
num_ftrs = model_conf.fc.in_features
model_conf.fc = nn.Linear(num_ftrs, 2)  # 2 classes: confident (0), non-confident (1)

# Load saved weights
model_conf.load_state_dict(torch.load('notebooks/best_conf_model.pth', map_location=device))
model_conf.to(device)
model_conf.eval()

# Recreate the model architecture
model_emot = models.resnet18(pretrained=False)
num_ftrs = model_emot.fc.in_features
model_emot.fc = torch.nn.Linear(num_ftrs, 7)  # 7 emotion classes

# Load saved weights
model_emot.load_state_dict(torch.load('notebooks/final_best_emot_model.pth', map_location=device))
model_emot.to(device)
model_emot.eval()

# Define transforms
transform_conf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale RGB
        transforms.Resize((224, 224)),  # Resizes pixels to 224x224
        transforms.ToTensor(),       # Conveert to pytorch Tensor in [0, 1] float
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalization for ImageNet dataset
])

transform_emot = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # general purpose normalization with mean=0.5 and std=0.5
])

# Function to analyze video for emotion and confidence detection, define video_path to upload a video file
# if not provided, it will use webcam
def analyze_video(model_conf=model_conf, model_emot=model_emot, emot_thresh = 0.68,
                conf_thresh = 0.6, use_deepface=False, video_path=None, device=device):

    is_live = video_path is None

    # Setup Mediapipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    # Video capture: webcam (0) or file path
    def get_working_camera():
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Using camera {i} for live detection.")
                return cap
        cap.release()
        raise RuntimeError("No working camera found.")
    cap = get_working_camera() if is_live else cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30

    # Counters
    total_confident = 0
    total_non_confident = 0
    emotion_counter = Counter()
    no_face_count = 0

    # FER2013 label map (if using model_emot)
    label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                 4: 'sad', 5: 'surprise', 6: 'neutral'}

    frame_count = 0
    ANALYZE_EVERY_N_FRAMES = int(fps)  # ~1 sec intervals

    # Keep last known box + label
    last_combined_label = None
    last_bbox = None

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            # Perform detection + predictions
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                detection = results.detections[0]  # take first face
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                forehead_offset = int(0.15 * (y2 - y1))
                y1 = max(0, y1 - forehead_offset)
                y2 = min(h, y2)

                face_img = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                pil_face_rgb = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                try:
                    # --- Emotion detection ---
                    if not use_deepface and model_emot is None:
                        raise ValueError("You must provide a model_emot when use_deepface is False.")
                    if use_deepface:
                        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = result[0]['dominant_emotion']
                    else:
                        input_tensor_emot = transform_emot(pil_face_rgb).unsqueeze(0).to(device).float()
                        output = model_emot(input_tensor_emot)
                        # Get probability of emotion labeling
                        probabilities = F.softmax(output, dim=1)
                        confidence, pred = torch.max(probabilities, 1)
                        if confidence < emot_thresh:
                            # If the confidence is below threshold label as neutral face
                            pred = torch.tensor([6], device=output.device)
                        dominant_emotion = label_map[pred.item()]
                    emotion_counter[dominant_emotion] += 1

                    # --- Confidence detection ---
                    pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    input_tensor_conf = transform_conf(pil_face).unsqueeze(0).to(device).float()
                    output_conf = model_conf(input_tensor_conf)
                    prob_conf = torch.softmax(output_conf, dim=1)
                    prob_class_1 = prob_conf[0,1]

                    if prob_class_1 > conf_thresh:
                        pred_conf = torch.tensor([1], device=output_conf.device)

                    else:
                        pred_conf = torch.tensor([0], device=output_conf.device)

                    if pred_conf.item() == 0:
                        total_confident += 1
                        confidence_label = "Confident"
                    else:
                        total_non_confident += 1
                        confidence_label = "Non-Confident"

                    last_combined_label = f"{dominant_emotion}, {confidence_label}"
                    last_bbox = (x1, y1, x2, y2)

                except Exception as e:
                    print(f"Error during analysis: {e}")

            else:
                no_face_count += 1

            frame_count = 0

        # Always draw the **last known box + label** (if any)
        if is_live and last_bbox and last_combined_label:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, last_combined_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if is_live:
            cv2.imshow('Live Emotion + Confidence Analyzer', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if is_live:
        cv2.destroyAllWindows()

    elapsed_time = time.time() - start_time
    print(f"\n--- Video Analysis Results ---")
    print(f"Processed in {elapsed_time:.2f} seconds")

    total_conf = total_confident + total_non_confident
    confidence_pct = (total_confident / total_conf) * 100 if total_conf > 0 else 0

    top_emotions = emotion_counter.most_common(3)
    emotion_summary = {emotion: (count / sum(emotion_counter.values())) * 100
                       for emotion, count in top_emotions}

    print(f"Confidence Level (Confident % over all faces): {confidence_pct:.2f}%")
    print(f"No face detected on {no_face_count} frame groups")
    print("\nTop 3 Dominant Emotions:")
    for emotion, pct in emotion_summary.items():
        print(f"{emotion}: {pct:.2f}%")
    return confidence_pct, emotion_summary


def predict_face_labels(frame, model_conf=model_conf, model_emot=model_emot, transform_conf=transform_conf, transform_emot=transform_emot, device=device):

    # Setup Mediapipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    label_map = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'sad', 5: 'surprise', 6: 'neutral'
    }

    # Convert image to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    output_frame = frame.copy()

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)

        # Adjust y1 to include forehead
        forehead_offset = int(0.15 * (y2 - y1))
        y1 = max(0, y1 - forehead_offset)
        x1 = max(0, x1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Extract and preprocess face
        face_img = frame[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        # Confidence prediction
        face_conf = transform_conf(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            conf_output = model_conf(face_conf)
            conf_pred = torch.argmax(conf_output, dim=1).item()
            conf_label = 'Confident' if conf_pred == 0 else 'Unconfident'

        # Emotion prediction
        face_emot = transform_emot(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emot_output = model_emot(face_emot)
            probs = F.softmax(emot_output, dim=1)
            emot_pred = torch.argmax(emot_output, dim=1).item()
            emot_label = label_map[emot_pred]
            emot_confidence = probs[0][emot_pred].item() * 100


        # Compute face height to scale font size
        face_height = y2 - y1
        font_scale = max(0.5, face_height / 200.0)       # scale text based on face height
        thickness = max(1, int(face_height / 100.0))     # line/text thickness
        # Split label into two lines
        line1 = conf_label
        line2 = f'{emot_label}' # Add ({emot_confidence:.1f}%) to the print statement to add emotion confidence score

        # Get text sizes
        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Choose max width and total height
        total_width = max(w1, w2)
        total_height = h1 + h2 + 10  # 10px spacing between lines

        # Starting position for top line
        text_x = x1
        text_y = max(20, y1 - total_height)

        # Adjust x if label would overflow the right side
        if text_x + total_width > frame.shape[1]:
            text_x = frame.shape[1] - total_width - 5

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw each line separately
        cv2.putText(output_frame, line1, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(output_frame, line2, (text_x, text_y + h2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    return output_frame

# === THIS GOES AT BOTTOM of face_model.py ===
if __name__ == "__main__":
    print("=== Starting live webcam detection ===")
    analyze_video()
    predict_face_labels()
