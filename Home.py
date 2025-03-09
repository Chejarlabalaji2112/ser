#main app file
from back_end import main
import streamlit as st
# -------------------- Streamlit App Layout -------------------- #
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
print("title set")
st.title("🎙️ Speech Emotion Recognition System")
st.write("Upload an audio file and get the predicted emotion!")

st.subheader("🔼 Upload an Audio File")
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav", "mp3"])

if uploaded_file:
    print("file_uploaded")
    st.audio(uploaded_file, format="audio/wav")
    prediction, conf = main.prediction(uploaded_file)
    # Simulate Audio Processing

    # Display Prediction Result
    st.success(f"Predicted Emotion: **{prediction}** 😊")
    st.metric(label="Confidence Score", value=f"{conf}%")
