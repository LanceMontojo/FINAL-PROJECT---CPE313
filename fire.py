import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile
from pathlib import Path
import torch
import shutil

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

# Load model with caching and CUDA support
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTDETR("BESTO FRIENDO.pt")
    model.model.to(device)
    return model, device

model, device = load_model()

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

            # Convert plot (BGR) to RGB to fix color tint
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
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        st.video(tfile.name)

        if st.button("Process Full Video"):
            with st.spinner("Processing video — this may take a while..."):
                # Run model on the whole video, save processed output
                results = model.predict(source=tfile.name, conf=0.3, save=True)

                # Find last saved processed video file
                last_dir = sorted(Path("runs/detect").glob("predict*"), key=lambda x: x.stat().st_mtime)[-1]
                output_video_path = list(last_dir.glob("*.mp4"))[0]

            st.success("Video processing complete!")

            # Show processed video
            st.video(str(output_video_path))

            # Button to download processed video
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="processed_fire_video.mp4",
                mime="video/mp4"
            )

            # OPTIONAL: Show extinguisher recommendations based on detected classes from video frames
            # For better accuracy, you might parse the detection results here
            # (not shown here to keep it simple)

