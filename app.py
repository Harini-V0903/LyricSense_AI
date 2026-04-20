import streamlit as st
import joblib
import re

# LOAD MODEL
model = joblib.load("model/song_mood_model.pkl")

# CLEAN FUNCTION
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# UI
st.title("🎵 Song Mood Classifier")

lyrics = st.text_area("Enter Song Lyrics")

if st.button("Predict Mood"):
    if lyrics.strip():
        cleaned = clean(lyrics)
        prediction = model.predict([cleaned])[0]

        st.success(f"Predicted Mood: {prediction}")
    else:
        st.warning("Please enter lyrics")