import streamlit as st
import requests
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

st.set_page_config(page_title="Raj Advanced AI Lab", layout="wide")
st.title("🚀 Raj Advanced AI Lab")

menu = st.sidebar.selectbox(
    "Choose Feature",
    ["🤖 AI Chatbot", "🎤 Voice Assistant", "📷 Custom Image Model"]
)

# ===============================
# 1️⃣ REAL AI CHATBOT (LLM API)
# ===============================

if menu == "🤖 AI Chatbot":
    st.header("Real AI Chatbot")

    api_key = st.text_input("Enter OpenAI API Key", type="password")
    user_input = st.text_area("Ask something")

    if st.button("Send") and api_key and user_input:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            st.success(reply)
        else:
            st.error("API Error")

# ===============================
# 2️⃣ REAL SPEECH ASSISTANT
# ===============================

elif menu == "🎤 Voice Assistant":
    st.header("Speech to Text + Text to Speech")

    api_key = st.text_input("Enter OpenAI API Key", type="password")
    audio_file = st.file_uploader("Upload Audio (wav/mp3)", type=["wav", "mp3"])

    if audio_file and api_key:
        st.info("Processing audio...")

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        files = {
            "file": audio_file,
            "model": (None, "whisper-1")
        }

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files
        )

        if response.status_code == 200:
            text = response.json()["text"]
            st.success("Transcribed Text:")
            st.write(text)

            # Text to Speech
            tts = gTTS(text)
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            tts.save(tmp_file.name)

            st.audio(tmp_file.name)

        else:
            st.error("Speech API Error")

# ===============================
# 3️⃣ CUSTOM TRAINED IMAGE MODEL
# ===============================

elif menu == "📷 Custom Image Model":
    st.header("Upload Your Trained Model (.h5)")

    model_file = st.file_uploader("Upload Keras Model (.h5)", type=["h5"])
    image_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if model_file and image_file:
        st.info("Loading model...")
        model = tf.keras.models.load_model(model_file)

        image = Image.open(image_file).resize((224, 224))
        st.image(image)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        st.success(f"Prediction Output: {prediction}")
