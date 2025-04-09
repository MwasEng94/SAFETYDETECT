import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import requests
from Utils.Image_utils import process_image_detection

def image_page():
    st.set_page_config(page_title="Image Detection", page_icon="📷")
    
    st.title("📷 Image Hazard Detection")
    st.markdown("Upload an image to detect safety hazards")
    
    with st.sidebar:
        st.header("Image Settings")
        with st.form(key='image_form'):
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"]
            )
            confidence = st.slider(
                "Confidence Threshold", 
                0.0, 1.0, 0.5, 
                key='img_conf'
            )
            submit = st.form_submit_button("Detect Hazards ⚠️")
    
    if submit and uploaded_file:
        process_image_detection(uploaded_file, confidence)

image_page()