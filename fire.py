import streamlit as st
from ultralytics import RTDETR
from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch  # add torch to check CUDA availability

# Load model with CUDA if available
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RTDETR("BESTO FRIENDO.pt")
    model.model.to(device)  # move model to device
    return model, device

model, device = load_model()

# Recommendations dictionary remains the same
recommendations = {
    "Class A": {"safe": "Water mist, foam, or dry chemicals", "unsafe": "CO2 and clean agents"},
    "Class B": {"safe": "Foam, CO2, dry chemicals", "unsafe": "Water-based extinguishers"},
    "Class C": {"safe": "CO2, clean agents, dry chemicals", "unsafe": "Water-based extinguishers"},
    "Class D": {"safe": "Specialized dry powder", "unsafe": None},
    "Class F": {"safe": "Wet chemical extinguishers", "unsafe": None}
}

# Video uploader
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.video(uploaded_video)

    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    with open(temp_path, 'wb') as f:
        f.write(uploaded_video.read())

    if st.button("Process Video"):
        with st.spinner("Running detection..."):
            # Pass device here - Ultralytics API doesn't explicitly require it, but model is already on CUDA.
            results = model.predict(source=temp_path, conf=0.5, save=True)

        # Get last saved video
        last_dir = sorted(Path("runs/detect").glob("predict*"), key=lambda x: x.stat().st_mtime)[-1]
        output_video = list(last_dir.glob("*.mp4"))[0]

        # Analyze frame-wise detections (optional)
        st.session_state["frame_classes"] = []
        cap = cv2.VideoCapture(str(output_video))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # If you want to run per-frame detection with CUDA:
            # Ultralytics handles numpy inputs on CUDA automatically when model is on CUDA
            result = model.predict(frame)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                frame_classes = set([result.names[int(c)] for c in result.boxes.cls.cpu().numpy()])
            else:
                frame_classes = set()
            st.session_state["frame_classes"].append(list(frame_classes))
        cap.release()
        st.session_state["video_path"] = str(output_video)
        st.success("Video processed!")

# Show processed video and real-time recommendations
if "video_path" in st.session_state:
    st.video(st.session_state["video_path"])
    st.markdown("### 🔥 Real-time Extinguisher Guidance")

    current_frame = st.slider("Video Progress", 0, len(st.session_state["frame_classes"]) - 1, 0)

    current_classes = st.session_state["frame_classes"][current_frame]
    for class_name in current_classes:
        st.markdown(f"**{class_name}**")
        rec = recommendations.get(class_name)
        if rec:
            st.markdown(f":green[✔ Safe: {rec['safe']}]")
            if rec["unsafe"]:
                st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
        else:
            st.warning("No recommendation for this class.")
