import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile
import torch
import os

st.set_page_config(layout="wide")

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

@st.cache_resource
def load_model():
    return RTDETR("BESTO FRIENDO.pt")

model = load_model()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.model.to(device)

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

# === VIDEO MODE WITH RECOMMENDATIONS ===
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        if st.button("Run Detection"):
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            stframe_left, stframe_right = st.columns([2, 1])
            placeholder = stframe_left.empty()

            all_detected_classes = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.3)
                result_frame = results[0].plot()
                out.write(result_frame)

                # Resize for display
                display_frame = cv2.resize(result_frame, (720, 400))
                placeholder.image(display_frame, channels="BGR")

                # Update class list
                if results[0].boxes.cls is not None:
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    class_names = results[0].names
                    for cid in class_ids:
                        all_detected_classes.add(class_names[cid])

            cap.release()
            out.release()

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.success("Processing complete!")

            stframe_left.video(video_bytes)
            with stframe_right:
                st.subheader("Extinguisher Recommendations")
                if all_detected_classes:
                    for class_name in sorted(all_detected_classes):
                        st.markdown(f"**{class_name}**")
                        rec = recommendations.get(class_name)
                        if rec:
                            st.markdown(f":green[✔ Safe: {rec['safe']}]")
                            if rec["unsafe"]:
                                st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                        else:
                            st.warning("No recommendation available.")
                else:
                    st.info("No fire classes detected.")

            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="processed_fire_detection.mp4",
                mime="video/mp4"
            )
