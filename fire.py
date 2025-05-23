import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile
import torch
import os
from decord import VideoReader, cpu

st.set_page_config(layout="wide")

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

# Load model
@st.cache_resource
def load_model():
    return RTDETR("90maprtdetr.pt")

model = load_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.model.to(device)

# Recommendations
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

# Frame extraction function
def extract_frames(video_path, num_frames=16, size=(224, 224)):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = [Image.fromarray(vr[i].asnumpy()).resize(size) for i in indices]
    return frames

# Mode selection
mode = st.radio("Select input type:", ["Image", "Video"])

# === IMAGE MODE ===
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model(image, conf=0.7)
            result = results[0]
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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        action = st.radio("Choose action:", ["Run Detection", "Get Frames"])

        if action == "Run Detection":
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            col1, col2 = st.columns([2, 1])
            video_frame = col1.empty()
            rec_panel = col2.empty()

            all_detected_classes = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.7)
                result_frame = results[0].plot()
                out.write(result_frame)

                display_frame = cv2.resize(result_frame, (720, 400))
                video_frame.image(display_frame, channels="BGR")

                frame_detected_classes = set()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes.cls is not None else []
                class_names = results[0].names

                for cid in class_ids:
                    class_name = class_names[cid]
                    frame_detected_classes.add(class_name)
                    all_detected_classes.add(class_name)

                with rec_panel.container():
                    st.subheader("Extinguisher Recommendations")
                    if frame_detected_classes:
                        for class_name in sorted(frame_detected_classes):
                            rec = recommendations.get(class_name)
                            if rec:
                                st.markdown(f"**{class_name}**")
                                st.markdown(f":green[✔ Safe: {rec['safe']}]")
                                if rec["unsafe"]:
                                    st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                            else:
                                st.warning("No recommendation available.")
                    else:
                        st.info("No fire class detected in this frame.")

            cap.release()
            out.release()

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.success("Processing complete!")
            st.video(video_bytes)

            st.subheader("All Detected Classes in Video")
            if all_detected_classes:
                for class_name in sorted(all_detected_classes):
                    rec = recommendations.get(class_name)
                    st.markdown(f"**{class_name}**")
                    if rec:
                        st.markdown(f":green[✔ Safe: {rec['safe']}]")
                        if rec["unsafe"]:
                            st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                    else:
                        st.warning("No recommendation available.")

            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="processed_fire_detection.mp4",
                mime="video/mp4"
            )

        elif action == "Get Frames":
            num_frames = st.slider("Select number of frames to extract", min_value=4, max_value=64, value=16, step=4)
            frames = extract_frames(tfile.name, num_frames=num_frames, size=(224, 224))
            st.subheader("Detected Frames with Extinguisher Recommendations")
            frame_cols = st.columns(4)

            for idx, frame in enumerate(frames):
                with frame_cols[idx % 4]:
                    results = model(frame, conf=0.7)
                    result = results[0]
                    detected_frame = result.plot()
                    frame_array = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

                    st.image(frame_array, caption=f"Frame {idx + 1}", use_column_width=True)

                    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
                    class_names = result.names
                    detected_classes = {class_names[cid] for cid in class_ids}

                    if detected_classes:
                        for class_name in sorted(detected_classes):
                            st.markdown(f"**{class_name}**")
                            rec = recommendations.get(class_name)
                            if rec:
                                st.markdown(f":green[✔ Safe: {rec['safe']}]")
                                if rec["unsafe"]:
                                    st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                            else:
                                st.warning("No extinguisher recommendation found for this class.")
                    else:
                        st.info("No fire class detected in this frame.")
