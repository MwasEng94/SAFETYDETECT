import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import requests

def process_image_detection(uploaded_file, confidence_threshold):
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_container_width=True)
    
    with col2:
        with st.spinner("Detecting hazards..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                "http://localhost:8000/detect/image",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                detections = result["detections"]
                
                draw = ImageDraw.Draw(image)
                for det in detections:
                    if det["confidence"] >= confidence_threshold:
                        bbox = det["bbox"]
                        draw.rectangle(bbox, outline="red", width=3)
                        draw.text(
                            (bbox[0], bbox[1] - 10),
                            f"{det['class_name']} {det['confidence']:.2f}",
                            fill="red"
                        )
                
                st.image(image, caption='Processed Image', use_container_width=True)
                
                detection_data = [
                    {
                        "Class": det["class_name"],
                        "Confidence": f"{det['confidence']:.2f}",
                        "BBox": f"{det['bbox']}"
                    }
                    for det in detections if det["confidence"] >= confidence_threshold
                ]
                
                st.dataframe(
                    pd.DataFrame(detection_data),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.error(f"Error: {response.text}")