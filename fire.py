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
            results = model(image, conf=0.3)  # higher confidence threshold
            result = results[0]

            # Convert plot (BGR) to RGB to fix blue tint
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
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        run_detection = st.button("Run Detection")
        detected_class_names_all = set()

        # Your existing realtime frame-by-frame detection with bounding boxes
        if run_detection:
            col1, col2 = st.columns([2, 1])
            stframe = col1.empty()
            rec_section = col2.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                stframe.image(frame, channels="BGR")

                if detected_class_names_frame:
                    rec_text = "### 🔥 Extinguisher Recommendations\n"
                    for class_name in detected_class_names_frame:
                        rec = recommendations.get(class_name)
                        rec_text += f"**{class_name}**\n"
                        if rec:
                            rec_text += f"- :green[✔ Safe: {rec['safe']}]\n"
                            if rec["unsafe"]:
                                rec_text += f"- :red[✘ Avoid: {rec['unsafe']}]\n"
                        else:
                            rec_text += "- ⚠️ No recommendation available\n"
                    rec_section.markdown(rec_text)
                else:
                    rec_section.markdown("### 🔥 Extinguisher Recommendations\nNo fire detected yet.")

            cap.release()

            st.subheader("Summary of Detected Classes and Recommendations")
            for class_name in detected_class_names_all:
                st.markdown(f"**{class_name}**")
                rec = recommendations.get(class_name)
                if rec:
                    st.markdown(f":green[✔ Safe: {rec['safe']}]")
                    if rec["unsafe"]:
                        st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                else:
                    st.warning("No extinguisher recommendation found for this class.")

        # New button for full video prediction + save + download
        if st.button("Process and Save Full Video for Download"):
            with st.spinner("Processing full video - please wait..."):
                results = model.predict(
                    source=tfile.name,
                    conf=0.3,
                    save=True,
                    save_txt=False,
                    device=device,
                    show=False
                )

                # Find latest saved video in runs/detect/predict*
                last_run_dir = sorted(Path("runs/detect").glob("predict*"), key=lambda x: x.stat().st_mtime)[-1]
                saved_video_path = list(last_run_dir.glob("*.mp4"))[0]

            st.success("Video processed and saved!")

            st.video(str(saved_video_path))

            with open(saved_video_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="processed_fire_video.mp4",
                mime="video/mp4"
            )
