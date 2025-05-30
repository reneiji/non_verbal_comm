import streamlit as st
import joblib
import cv2
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import parselmouth
from fer import FER
import os
import plotly.express as px
import pandas as pd
import subprocess
import sys


from speech_model.model import extract_features
from face_model.face_model import analyze_video
from body_language_mod.body_model import init_model as init_body_model, analyze_video as analyze_body_video

# Set page config
st.set_page_config(
    page_title="Nonverbal Communication Analyzer",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
    <style>

    /* ===== GLOBAL FONT ===== */
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
    }

    /* ===== FILE UPLOADER ===== */
    div.stFileUploader label div {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #d62828 !important;  /* deep red */
        padding-bottom: 10px;
    }

    /* ===== SECTION TITLES ===== */
    h3 {
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* ===== CUSTOM COLORS ===== */
    /* Speech Score */
    .speech-score h3 {
        color: #F25C05;  /* orange */
    }

    /* Face Score */
    .face-score h3 {
        color: #0077b6;  /* blue */
    }

    /* Body Score */
    .body-score h3 {
        color: #2a9d8f;  /* green */
    }

    /* ===== BUTTONS ===== */
    button[kind="secondary"] {
        font-size: 18px;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)


body_model = init_body_model()
model = joblib.load('speech_model/trained_model_combined.joblib')  # Load your pre-trained model
le = joblib.load('speech_model/label_encoder.joblib')  # Load your label encoder


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


# Streamlit frontend starts here:

# Upload video file
with st.sidebar:
    st.header("📂 Upload Video")
    video_file = st.file_uploader("Choose a video...", type=["mp4"])

    if video_file:
        st.video(video_file)

    # st.markdown("---")
    # st.header("🎥 Live Webcam")

    # if st.button("Launch Live Webcam"):
    #     subprocess.Popen(["open", "-a", "Terminal", "run_face_live.sh"])
    #     st.toast("Webcam launched!", icon="🎥")

    # if st.button("Stop Live Webcam"):
    #     open("close_webcam.flag", "w").close()
    #     st.toast("Sent stop signal!", icon="🛑")

st.title("🗣️ Communication Analyzer")
st.caption("Analyze **speech**, **facial expressions**, and **body language** to assess presentation confidence 🚀")
st.markdown("---")

if video_file:
    # Save the uploaded video to disk
    video_path = f"uploads/{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    
    # st.markdown(
    #     f"""
    #     <video controls width="500" style="border-radius:10px; margin-top:10px;">
    #         <source src="{video_path}" type="video/mp4">
    #         Your browser does not support the video tag.
    #     </video>
    #     """,
    #     unsafe_allow_html=True
    # )

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
        
    tab1, tab2, tab3 = st.tabs(["🎤 Speech Model", "😊 Face Model", "🕺 Body Language Model"])


    # If visual features are detected, use the first one (for simplicity)
    with tab1:
        st.header("🎤 Speech Confidence Score")

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
            st.markdown(f"<div class='speech-score'><h3>Speech Confidence Score: {confidence_score:.2f}</h3>", unsafe_allow_html=True)
            # Build DataFrame for pie chart
            speech_emotion_df = pd.DataFrame({
                'Emotion': emotion_labels,
                'Probability': probas[0]
            })
            #filter out disgust
            speech_emotion_df = speech_emotion_df[speech_emotion_df['Emotion'] != 'disgust']


            # Create pie chart
            fig_speech = px.pie(speech_emotion_df,
                                names='Emotion',
                                values='Probability',
                                color_discrete_sequence=px.colors.sequential.RdBu,
                                title="Detected Emotions (Speech Model)")

            fig_speech.update_traces(textposition='inside', textinfo='percent+label')
            fig_speech.update_layout(width=700, height=700)  # Adjust size here 🚀

            # Show in Streamlit
            st.plotly_chart(fig_speech, use_container_width=True)
            
            
            # Add a button to show probabilities
            # Initialize toggle state for speech model probabilities
            if "show_probs_speech" not in st.session_state:
                st.session_state.show_probs_speech = False

            # Toggle button
            if st.button("Show/Hide Details", key="speech_toggle"):
                st.session_state.show_probs_speech = not st.session_state.show_probs_speech

            # Conditionally display
            if st.session_state.show_probs_speech:
                st.write("\nProbabilities for each emotion:")
                for label, prob in zip(le.classes_, probas[0]):
                    st.write(f"{label.capitalize()}: {prob:.2f}")
            
        
        # Run Face Model Analysis
        with tab2: 
            st.header("😊 Face Confidence Score")

            with st.spinner("Analyzing facial expressions..."):
                face_conf_pct, face_emotion_summary = analyze_video(video_path=video_path)

            # Display face model results
            st.markdown(f"<div class='face-score'><h3>Facial Confidence Score: {face_conf_pct:.2f}</h3></div>", unsafe_allow_html=True)

            # Absolute path to your project
            project_path = os.path.abspath(".")
            face_model_script = os.path.join(project_path, "face_model/face_model.py")

                        
            # Add pie chart for top emotions
            # After face_emotion_summary is ready:
            if face_emotion_summary:
                labels = list(face_emotion_summary.keys())
                sizes = list(face_emotion_summary.values())

                # Create dataframe for Plotly
                import pandas as pd
                df_emotions = pd.DataFrame({
                    'Emotion': labels,
                    'Percentage': sizes
                })

                # Create pie chart with Plotly
                fig = px.pie(df_emotions,
                            names='Emotion',
                            values='Percentage',
                            color_discrete_sequence=px.colors.sequential.RdBu,
                            title="Top 3 Detected Emotions (Face Model)")

                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                fig.update_layout(width=700, height=700)  # Adjust size here 🚀

                # Show in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                    
        # Initialize toggle state for face model emotions
            if "show_probs_face" not in st.session_state:
                st.session_state.show_probs_face = False

            # Toggle button
            if st.button("Show/Hide Details", key="face_toggle"):
                st.session_state.show_probs_face = not st.session_state.show_probs_face

            # Conditionally display
            if st.session_state.show_probs_face:
                st.write("\nTop Emotions:")
                for emotion, pct in face_emotion_summary.items():
                    st.write(f"{emotion}: {pct:.2f}%")
                    
            # Path to venv
            # Auto detect venv activate path (pyenv safe)
            venv_activate = os.path.join(os.path.dirname(sys.executable), "activate")

            # Project + script path
            project_path = os.path.abspath(".")
            face_model_script = os.path.join(project_path, "face_model/face_model.py")

            # Build command
            command = f'source {venv_activate} && python {face_model_script}'

            # Build osascript AppleScript command
            osa_command = f'''osascript -e 'tell application "Terminal" to do script "cd {project_path} && {command}"' '''

            # Launch button
            if st.button("🎥 Try Live Webcam Face Detection"):
                st.info("Launching webcam in new Terminal window... Close webcam window to return to app.")
                subprocess.Popen(osa_command, shell=True)

        with tab3:
            st.header("🕺 Body Language Confidence Score")
               
                
            # Run Body Model Analysis
            with st.spinner("Analyzing body language..."):
                body_scores = analyze_body_video(video_path=video_path, model=body_model)

            # Display Body Language Score
            if body_scores and "Overall Body Language Score" in body_scores:
                st.markdown(f"<div class='body-score'><h3>Body Language Confidence Score: {body_scores['Overall Body Language Score']:.1f}</h3></div>", unsafe_allow_html=True)
            else:
                st.write("No valid body landmarks detected.")
            
            # Prepare data → exclude "Overall Body Language Score"
            body_scores_filtered = {k: v for k, v in body_scores.items() if k != "Overall Body Language Score"}

            # Convert to DataFrame
            body_scores_df = pd.DataFrame({
                'Metric': list(body_scores_filtered.keys()),
                'Score': list(body_scores_filtered.values())
            })

            # Create bar chart
            fig_body = px.bar(body_scores_df,
                            x='Metric',
                            y='Score',
                            color='Score',
                            color_continuous_scale='RdBu',
                            title="Detailed Body Language Scores",
                            text='Score')

            fig_body.update_layout(xaxis_title="", yaxis_range=[0, 100])  # Scores are on 0-10 scale

            # Show in Streamlit
            st.plotly_chart(fig_body, use_container_width=True)

            # Toggle for detailed body scores
            if "show_body_scores" not in st.session_state:
                st.session_state.show_body_scores = False

            if st.button("Show/Hide Detailed Body Language Scores"):
                st.session_state.show_body_scores = not st.session_state.show_body_scores

            if st.session_state.show_body_scores and body_scores:
                st.write("Body Language Scores:")
                for k, v in body_scores.items():
                    if k != "Overall Body Language Score":
                        st.write(f"{k}: {v:.1f}")
                        
       
            else:
                st.write("No visual emotion detected in the video.")
                progress_bar.progress(100)  # 100% progress even if no visual emotion detected

            