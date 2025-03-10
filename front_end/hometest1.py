from back_end import main
import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import sounddevice as sd
import wave
import time


# Load feedback values
json_path = '/home/badri/mine/ser/gnd/capstone_project/Back_end/back_end/data/feedback_interval.json'
audio_path = '/home/badri/mine/ser/gnd/capstone_project/Back_end/back_end/data/user_audio/'
with open(json_path, 'r') as f:
    values = json.load(f)

if "predicted" not in st.session_state:
    st.session_state.predicted = values['predicted']

# -------------------- Streamlit App Layout -------------------- #
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
st.title("🎙️ Speech Emotion Recognition System")
st.markdown(f"**Threshold:** {values['threshold']}")

st.subheader("Upload an Audio File")
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
if st.button("Record...."):
    print("Recording audio in memory...")
    audio_data = sd.rec(int(2.5 * 22050), samplerate=22050, channels=1, dtype='float32')
    sd.wait()
    print('recording completed')
    audio_data = (audio_data * 32767).astype(np.int16)
    current_audio = audio_path+'recorded_audio'+time.strftime("%H-%Y-%S")+'.wav'
    with wave.open(current_audio, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio_data.tobytes())
    prediction, conf = main.prediction(current_audio)  # Pass file path

    st.session_state.predicted += 1
    values['predicted'] = st.session_state.predicted
    st.session_state.prediction = prediction

    # Display Prediction Result
    st.success(f"Predicted Emotion: **{prediction}** 😊")
    st.metric(label="Confidence Score", value=f"{conf:.2f}%")
    st.markdown(f"**Predicted:** {st.session_state.predicted}")

    st.subheader("Audio Visualization")

    # Instead of reloading, use the same Librosa processing as backend
    y, sr = main.load_fixed_audio(current_audio)# Reuse backend function

    # Plot waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    st.session_state.path = current_audio
    
    

if uploaded_file :
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_wav_path = tmp_file.name
    prediction, conf = main.prediction(tmp_wav_path)  # Pass file path

    st.session_state.predicted += 1
    values['predicted'] = st.session_state.predicted

    # Display Prediction Result
    st.success(f"Predicted Emotion: **{prediction}** 😊")
    st.metric(label="Confidence Score", value=f"{conf:.2f}%")
    st.markdown(f"**Predicted:** {st.session_state.predicted}")

    st.subheader("Audio Visualization")

    # Instead of reloading, use the same Librosa processing as backend
    y, sr = main.load_fixed_audio(tmp_wav_path)  # Reuse backend function

    # Plot waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    st.session_state.prediction = prediction
    st.session_state.path = audio_path+uploaded_file.name
if st.session_state.predicted > values['threshold']:
    st.switch_page('pages/feedback.py')

if st.button("SAVE"):
        with open(json_path, 'w') as f:
            json.dump(values, f, indent=4)

    



