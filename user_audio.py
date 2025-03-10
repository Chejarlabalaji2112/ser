import sounddevice as sd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import librosa

def record_audio(duration=2.5, sr=22050):
    """Records audio for a given duration and sample rate."""
    print("Recording audio in memory...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio_data = audio_data.flatten()  # Flatten the array to 1D
    return audio_data, sr



def process_audio_in_memory(audio_array, sr, target_duration=2.5):
    """Ensures the audio is exactly target_duration seconds long (pads if necessary)."""
    required_length = int(target_duration * sr)
    if len(audio_array) < required_length:
        pad_length = required_length - len(audio_array)
        audio_array = np.pad(audio_array, (0, pad_length))
    return audio_array, sr

def zcr(data, frame_length, hop_length):
    zcr_feature = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_feature)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_feature = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_feature)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool=True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)  # n_mfcc can be adjusted
    # Transpose so that time is on axis 0
    mfcc_feature = mfcc_feature.T
    return np.ravel(mfcc_feature) if flatten else np.squeeze(mfcc_feature)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    """Extracts features by combining ZCR, RMSE, and MFCC."""
    features = np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return features




# Load the pre-trained model, scaler, and encoder.
loaded_model = load_model('/home/badri/mine/ser/saved_models/kaggle_saved_gpu3hrs/total_best_model.keras')

with open('/home/badri/mine/ser/saved_models/kaggle_saved_gpu3hrs/new_scaler2.pickle', 'rb') as f:
    new_scaler2 = pickle.load(f)

with open('/home/badri/mine/ser/saved_models/kaggle_saved_gpu3hrs/new_encoder2.pickle', 'rb') as f:
    new_encoder2 = pickle.load(f)

def get_predict_feat_from_array(audio_array, sr):
    # Ensure the audio is exactly 2.5 seconds long.
    audio_array, sr = process_audio_in_memory(audio_array, sr, target_duration=2.5)
    # Extract features
    features = extract_features(audio_array, sr=sr)
    result = np.array(features)
    # Reshape according to your model input (example shape: (1, 2376))
    result = np.reshape(result, newshape=(1, 2376))
    # Scale features
    scaled_result = new_scaler2.transform(result)
    # Expand dimensions if necessary (example adds a channel dimension)
    final_result = np.expand_dims(scaled_result, axis=2)
    return final_result

def prediction_from_array(audio_array, sr):
    features = get_predict_feat_from_array(audio_array, sr)
    predictions = loaded_model.predict(features)
    y_pred = new_encoder2.inverse_transform(predictions)
    print("Predicted Emotion:", y_pred[0][0])


