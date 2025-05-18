import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

# Load model
@st.cache_resource
def load_model():
    return RTDETR("BESTO FRIENDO.pt")

model = load_model()

# Choose mode
mode = st.radio("Select input type:", ["Image", "Video"])

# Extinguisher recommendations
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

# === IMAGE MODE ===
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model(image, conf=0.8)  # set confidence threshold to 0.8
            result = results[0]

            img_rgb = result.plot()  # already numpy RGB
            st.image(img_rgb)

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

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        run_detection = st.button("Run Detection")

        if run_detection:
            detected_class_names = set()
            frame_count = 0
            skip_frames = 4  # now process every 4th frame for faster speed
            display_every = 2  # only update Streamlit image every 2 processed frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue  # skip frames to speed up

                # Resize frame to smaller size for faster inference
                height, width = frame.shape[:2]
                scale_percent = 50  # reduce to 50% size
                new_width = int(width * scale_percent / 100)
                new_height = int(height * scale_percent / 100)
                frame_small = cv2.resize(frame, (new_width, new_height))

                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb, conf=0.8)
                result = results[0]

                predicted_class_indices = result.boxes.cls.cpu().numpy().astype(int)
                class_names = result.names
                for class_id in predicted_class_indices:
                    detected_class_names.add(class_names[class_id])

                # Only update display every 'display_every' processed frames to reduce UI lag
                if (frame_count // skip_frames) % display_every == 0:
                    frame_annotated = np.array(result.plot())  # RGB image
                    stframe.image(frame_annotated, channels="RGB")

            cap.release()

            st.subheader("Extinguisher Recommendations")
            for class_name in detected_class_names:
                st.markdown(f"**{class_name}**")
                rec = recommendations.get(class_name)
                if rec:
                    st.markdown(f":green[✔ Safe: {rec['safe']}]")
                    if rec["unsafe"]:
                        st.markdown(f":red[✘ Avoid: {rec['unsafe']}]")
                else:
                    st.warning("No extinguisher recommendation found for this class.")
