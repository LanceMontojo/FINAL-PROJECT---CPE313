import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile
import torch

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

# Load model with caching
@st.cache_resource
def load_model():
    return RTDETR("BESTO FRIENDO.pt")

model = load_model()

# Move model to CUDA if available for faster inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.model.to(device)

# Extinguisher recommendations dictionary
recommendations = {
    "Class A": {
        "safe": "Water mist, foam, or multipurpose dry chemicals extinguishers",
        "unsafe": "Carbon dioxide and clean agents extinguishers"
    },
    "Class B": {
        "safe": "Foam, carbon dioxide, or dry chemicals",
        "unsafe": "Water-based extinguishers"
    },
    "Class C": {
        "safe": "Carbon dioxide, clean agents, or dry chemicals",
        "unsafe": "Water-based extinguishers"
    },
    "Class D": {
        "safe": "Specialized dry powder extinguishers",
        "unsafe": None
    },
    "Class F": {
        "safe": "Wet chemical extinguishers to prevent splattering and reignition",
        "unsafe": None
    }
}

# Choose input mode
mode = st.radio("Select input type:", ["Image", "Video"])

# === IMAGE MODE ===
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model(image, conf=0.3)
            result = results[0]

            # Convert BGR plot to RGB to fix colors
            img_bgr = result.plot()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, channels="RGB")

            predicted_class_indices = result.boxes.cls.cpu().numpy().astype(int)
            detected_classes = result.names

            st.subheader("Extinguisher Recommendations")
            for class_id in predicted_class_indices:
                class_name = detected_classes[class_id]
                st.markdown(f"**{class_name}**")

                rec = recommendations.get(class_name)
                if rec:
                    st.markdown(f":green[✔ Safe: {rec['safe']}]")
                    if rec["unsafe"]:
                        st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                else:
                    st.warning("No extinguisher recommendation found for this class.")

# === VIDEO MODE ===
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save uploaded video to temp file to pass path to model
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()  # Important: close so model can access the file

        if st.button("Run Detection and Process Video"):
            # Run model prediction on saved video path and save output
            results = model.predict(
                source=tfile.name,
                conf=0.8,
                save=True,       # save annotated video
                save_txt=False,
                show=False,
                device=device
            )

            # The output video is saved by default in runs/detect/predict or similar folder.
            # Find the saved video file path
            output_dir = results[0].path.parent  # parent folder of result image/video
            # Find first mp4 file in output dir
            import os
            saved_video_path = None
            for file in os.listdir(output_dir):
                if file.endswith(".mp4"):
                    saved_video_path = os.path.join(output_dir, file)
                    break

            if saved_video_path:
                st.video(saved_video_path)
                with open(saved_video_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="processed_fire_detection.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("Could not find the processed video file.")

        # Optional: delete the temp input file after processing or app close

