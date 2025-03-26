import streamlit as st
import requests
import tempfile
import os
from PIL import Image

API_URL = "http://127.0.0.1:8000/analyze"  # FastAPI backend URL

# Layout configuration
st.set_page_config(layout="wide")

# Add title at the top left and navigation link at the top right
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>Deepfake Detection</h1>
        <a href="another_page.py" style="font-size: 18px; text-decoration: none;">Go to Another Page</a>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Define columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="col1">', unsafe_allow_html=True)
    st.write("Please upload a video to analyze for deepfakes.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    
        
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name  # Stores path to temporary file

        st.video(video_path)  # Streamlit uses this file

        if st.button("Analyze Video"):
            with st.spinner("Processing video..."):
                with open(video_path, "rb") as f:  # Explicitly open and close the file
                    files = {"file": f}
                    response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()["deepfake_probability"]
                    st.success(f"Deepfake probability: {result * 100:.2f}%")
                else:
                    st.error("Error processing video. Please try again.")

        import time
        time.sleep(2)  # Give Streamlit time to release the file

        try:
            os.remove(video_path)  # Delete the file safely
        except PermissionError:
            st.warning("File is in use and cannot be deleted right now.")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Display the image
    image_path = r"D:\Btech_Project\Frontend\DALL·E-2025-03-26-17.15.40-A-visually-striking-deepfake-detection-concept.jpg"
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
