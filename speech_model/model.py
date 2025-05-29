from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import numpy as np
import librosa
import os
import parselmouth

import joblib  # Import joblib to save the model

# Step 2: Extract audio features (MFCC, pitch, jitter, shimmer)
def extract_features(audio_path):
    """
    Extract MFCC features, Jitter, Pitch, and Shimmer from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)  # 'sr' is the sampling rate; None means keep the original
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # We are using 13 MFCC coefficients
    mfcc_mean = np.mean(mfcc, axis=1)  # Average the MFCCs over time
    
    # Extract Jitter, Pitch, and Shimmer using parselmouth
    sound = parselmouth.Sound(audio_path)
    
    # Get pitch
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']  # Get pitch values
    
    # Jitter (variability in pitch)
    jitter = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0  # Jitter calculation
    
    # Shimmer (variability in amplitude)
    shimmer = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0  # Simplified shimmer calculation
    
    # Calculate mean of each feature
    pitch_mean = np.mean(pitch_values)
    
    # Combine all features into one array
    combined_features = np.hstack((mfcc_mean, pitch_mean, jitter, shimmer))
    
    return combined_features


# Step 3: Process all audio files in a directory
def process_audio_files(directory):
    """
    Process all the audio files in a given directory, extracting features and labels.
    """
    features = []
    labels = []

    # Loop over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            # Extract the label from the filename (e.g., OAF_back_angry.wav -> angry)
            label = filename.split('_')[-1].replace('.wav', '')  # Assuming emotion is the last part of the filename
            
            # Extract features
            audio_path = os.path.join(directory, filename)
            extracted_features = extract_features(audio_path)
            features.append(extracted_features)
            labels.append(label)

    # Convert to numpy array and return
    return np.array(features), np.array(labels)


def train_model(X, y):
    # Encode the labels (e.g., angry -> 0, sad -> 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=200)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Print classification report
    print("Model performance on the test set:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save the trained model and the label encoder using joblib
    joblib.dump(model, 'model/trained_model_combined.joblib')  # Save the model
    joblib.dump(le, 'model/label_encoder.joblib')  # Save the label encoder

    print("Model and label encoder saved successfully.")
    

directory = "notebooks/dataverse_files"  # Directory containing audio files
X, y = process_audio_files(directory)

train_model(X, y)
    
    