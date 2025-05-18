import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile
import torch
from pathlib import Path

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

# Load model with caching, move to CUDA if available
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RTDETR("BESTO FRIENDO.pt")
    model.model.to(device)
    return model, device

model, device = load_model()

# Recommendations dictionary
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

# --- VIDEO MODE WITH FULL PROCESSING AND DOWNLOAD ---
mode = st.radio("Select input type:", ["Image", "Video"])

if mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name

        if st.button("Process Video"):
            with st.spinner("Processing video, please wait..."):
                cap = cv2.VideoCapture(tfile_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                # Output file path
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                detected_class_names_all = set()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB (Ultralytics expects RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Predict with model on CUDA (model already on device)
                    results = model(frame_rgb, conf=0.3)
                    result = results[0]

                    detected_class_names_frame = set()

                    if result.boxes is not None and len(result.boxes) > 0:
                        predicted_class_indices = result.boxes.cls.cpu().numpy().astype(int)
                        class_names = result.names
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()

                        for (x1, y1, x2, y2), cls_id, conf in zip(boxes.astype(int), predicted_class_indices, confidences):
                            label = f"{class_names[cls_id]} {conf*100:.1f}%"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            detected_class_names_frame.add(class_names[cls_id])
                            detected_class_names_all.add(class_names[cls_id])

                    # Write annotated frame to output video
                    out.write(frame)

                cap.release()
                out.release()

            st.success("Video processing complete!")

            # Show processed video
            st.video(output_path)

            # Provide download button for processed video
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="processed_fire_detection.mp4",
                    mime="video/mp4"
                )

            # Show extinguisher recommendations for all detected classes in the video
            st.subheader("Summary of Detected Classes and Recommendations")
            if detected_class_names_all:
                for class_name in detected_class_names_all:
                    st.markdown(f"**{class_name}**")
                    rec = recommendations.get(class_name)
                    if rec:
                        st.markdown(f":green[✔ Safe: {rec['safe']}]")
                        if rec["unsafe"]:
                            st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                    else:
                        st.warning("No extinguisher recommendation found for this class.")
            else:
                st.info("No fire classes detected in the video.")
