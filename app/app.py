import streamlit as st
import joblib
import cv2
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import parselmouth
from fer import FER
import os
import time

from model.model import extract_features


model = joblib.load('model/trained_model_combined.joblib')  # Load your pre-trained model
le = joblib.load('model/label_encoder.joblib')  # Load your label encoder


# Helper function for audio extraction from video
def extract_audio_from_video(video_path):
    """
    Extract audio from the video and save it as a .wav file with the same name as the video.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_audio_path = f"{base_name}_extracted_audio.wav"
    
    # Load the video file and extract audio
    video = VideoFileClip(video_path)
    audio = video.audio
    
    # Write the audio to a .wav file
    audio.write_audiofile(output_audio_path)
    
    return output_audio_path

# Helper function for extracting audio features (MFCC, pitch, jitter, shimmer)
def extract_audio_features(audio_path):
    """
    Extract MFCC features, Jitter, Pitch, and Shimmer from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)  # 'sr' is the sampling rate; None means keep the original
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']  
    
    jitter = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0
    shimmer = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0
    pitch_mean = np.mean(pitch_values)
    
    return np.hstack((mfcc_mean, pitch_mean, jitter, shimmer))

# Helper function for extracting visual features from video using FER
def extract_visual_features_from_video(video_path):
    """
    Extract visual emotion features from the video using FER.
    """
    detector = FER()
    video_capture = cv2.VideoCapture(video_path)
    
    visual_emotions = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Detect emotions from the frame
        emotions = detector.top_emotion(frame)  # Returns the top emotion (e.g., ('happy', 0.8))

        if emotions:
            emotion, confidence = emotions
            visual_emotions.append((emotion, confidence))
    
    video_capture.release()
    
    return visual_emotions

# Combine audio and visual features into a single feature vector
def combine_audio_visual_features(audio_features, visual_features):
    """
    Combine audio and visual features into a single feature vector.
    """
    audio_feature_vector = np.array(audio_features)  # Convert audio features to numpy array
    visual_feature_vector = np.array([confidence for emotion, confidence in visual_features])  # Convert visual features to numpy array
    
    # Concatenate the two feature vectors
    combined_features = np.concatenate((audio_feature_vector, visual_feature_vector))
        
    return combined_features

# Calculate the confidence score based on audio and visual features
def calculate_confidence_score(model, le, audio_file_path):
    """
    Calculate the confidence score based on the model's probabilities for 'happy', 'ps', 'angry', and 'neutral'.
    Higher scores indicate more confidence in the presentation.
    """
    # Extract features from the new audio file
    new_features = np.array(audio_features).reshape(1, -1)  # Reshape to match model input
    
    # Get the probabilities of each class (emotion)
    probas = model.predict_proba(new_features)
    
    # Define the class labels for emotions (focus on 'happy', 'ps', 'angry', 'neutral')
    emotion_labels = le.classes_  # e.g., ['angry', 'happy', 'nervous', ...]

    # Extract the relevant probabilities for 'happy', 'ps', 'angry', and 'neutral'
    happy_prob = probas[0][np.where(emotion_labels == 'happy')[0][0]]
    ps_prob = probas[0][np.where(emotion_labels == 'ps')[0][0]]
    angry_prob = probas[0][np.where(emotion_labels == 'angry')[0][0]]
    neutral_prob = probas[0][np.where(emotion_labels == 'neutral')[0][0]]
    
    # Print probabilities for each emotion of interest
    # print(f"Happy Probability: {happy_prob:.2f}")
    # print(f"PS Probability: {ps_prob:.2f}")
    # print(f"Angry Probability: {angry_prob:.2f}")
    # print(f"Neutral Probability: {neutral_prob:.2f}")
    
    # ("\nProbabilities for each emotion:")
    # for label, prob in zip(emotion_labels, probas[0]):
    #     (f"{label.capitalize()}: {prob:.2f}")
    
    # Confidence Score Calculation:
    # Increasing the weight for confidence emotions ('happy', 'ps', 'angry')
    confidence_emotions_prob = (happy_prob * 3 + ps_prob * 3 + angry_prob * 3)  # Increased weight for confidence emotions
    confidence_score = max(confidence_emotions_prob, 0.05) * 100  # Ensure a minimum contribution
    
    # Reduce the weight of neutral's penalty (dampen the effect)
    neutral_penalty = neutral_prob * 20  # Reduced penalty (previously 50)

    # Subtract the neutral penalty from the confidence score to get the final score
    final_confidence_score = confidence_score - neutral_penalty
    
    # Boost the confidence score by 20 if the probability of neutral is 0
    if neutral_prob <= 0.01:
        final_confidence_score += 20  # Boost confidence score if neutral is 0
    
    # Apply a minimum threshold to ensure the score doesn't drop below a certain level (e.g., 10)
    final_confidence_score = max(final_confidence_score, 10)
    
    # Clamp the score to a range of 0 to 100 for visualization
    final_confidence_score = max(0, min(100, final_confidence_score))
    
    print(f"Final Confidence Score: {final_confidence_score:.2f}")
    return final_confidence_score, probas, emotion_labels

# Streamlit UI setup for video upload
st.markdown("""
    <style>
        /* This targets the label of the file uploader specifically */
        div.stFileUploader label {
            font-size: 32px !important;  /* Make the font larger */
            font-weight: bold !important;  /* Make the font bold */
            color: red !important;  /* Change the font color to red */
        }
    </style>
""", unsafe_allow_html=True)


st.title("Confidence Score from Video")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4"])

if video_file:
    # Save the uploaded video to disk
    video_path = f"uploads/{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_path)  # Display uploaded video

    st.write("Processing video...")
    progress_bar = st.progress(0)
    
    if 'audio_features' not in st.session_state or 'visual_emotions' not in st.session_state:
        progress_bar.progress(10)
        audio_path = extract_audio_from_video(video_path)
        
        # Extract audio features
        progress_bar.progress(30)  # 30% progress after extracting audio features
        audio_features = extract_audio_features(audio_path)
        
        # Extract visual features from the video
        progress_bar.progress(50)  # 50% progress after extracting visual features
        visual_emotions = extract_visual_features_from_video(video_path)
        
        st.session_state.audio_features = audio_features
        st.session_state.visual_emotions = visual_emotions
        
    else:
        # Retrieve features from session state
        audio_features = st.session_state.audio_features
        visual_emotions = st.session_state.visual_emotions

    # If visual features are detected, use the first one (for simplicity)
    if visual_emotions:
        visual_emotions = visual_emotions[0]  # Use the first detected emotion

        # Combine audio and visual features into a single feature set
        # Visual feature contains emotion and confidence, so we need to extract confidence
        confidence = visual_emotions[1]
        
        # Now, call the calculate_confidence_score function with just the audio features
        progress_bar.progress(70)  # 70% progress before calculating confidence score
        confidence_score, probas, emotion_labels = calculate_confidence_score(model, le, audio_features)

        # Display the result
        progress_bar.progress(100)  # 100% progress when the score is ready
        st.markdown(f"<h3 style='color:red;'>Speech Confidence Score: {confidence_score:.2f}</h3>", unsafe_allow_html=True)
        
            # Add a button to show probabilities
        if st.button("Show Emotion Probabilities"):
            # Display the probabilities of each emotion
            st.write("\nProbabilities for each emotion:")
            for label, prob in zip(le.classes_, probas[0]):
                st.write(f"{label.capitalize()}: {prob:.2f}")
                
    else:
        st.write("No visual emotion detected in the video.")
        progress_bar.progress(100)  # 100% progress even if no visual emotion detected

    