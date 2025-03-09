import streamlit as st
import psutil
st.subheader("📢 Provide Feedback")
user_feedback = st.radio("Is the predicted emotion correct?", ["Yes", "No"])

if user_feedback == "No":
    correct_emotion = st.selectbox("Select the correct emotion", ["Calm", "Happy", "Sad", "Angry"])
    st.success(f"Thanks! We will use '{correct_emotion}' for model fine-tuning.")

# CPU Monitor
st.subheader("📊 System Performance")
cpu_usage = psutil.cpu_percent()
mem_usage = psutil.virtual_memory().percent
st.metric(label="CPU Usage", value=f"{cpu_usage}%")
st.metric(label="Memory Usage", value=f"{mem_usage}%")
