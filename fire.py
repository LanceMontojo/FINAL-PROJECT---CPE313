import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import tempfile

# Title
st.title("RTDETR Fire Classifier (Image/Video) with Extinguisher Recommendations")

# Load model with caching
@st.cache_resource
def load_model():
    return RTDETR("BESTO FRIENDO.pt")

model = load_model()

# Choose input mode
mode = st.radio("Select input type:", ["Image", "Video"])

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

# === IMAGE MODE ===
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model(image, conf=0.8)  # confidence threshold 0.8
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

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        run_detection = st.button("Run Detection")

        if run_detection:
            detected_class_names = set()
            frame_count = 0
            skip_frames = 4        # process every 4th frame to speed up

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue  # skip frames to speed up

                height, width = frame.shape[:2]

                # Resize frame to 50% for faster inference
                new_width = int(width * 0.5)
                new_height = int(height * 0.5)
                frame_small = cv2.resize(frame, (new_width, new_height))

                # Convert BGR to RGB for model input
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

                results = model(frame_rgb, conf=0.8)
                result = results[0]

                # Collect detected class names
                predicted_class_indices = result.boxes.cls.cpu().numpy().astype(int)
                class_names = result.names
                for class_id in predicted_class_indices:
                    detected_class_names.add(class_names[class_id])

                # Draw boxes on original frame
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    # scale boxes from small frame back to original frame size
                    scale_x = width / new_width
                    scale_y = height / new_height
                    boxes[:, [0, 2]] *= scale_x  # x1, x2
                    boxes[:, [1, 3]] *= scale_y  # y1, y2

                    confidences = result.boxes.conf.cpu().numpy()
                    classes = predicted_class_indices

                    for (x1, y1, x2, y2), cls_id, conf in zip(boxes.astype(int), classes, confidences):
                        label = f"{class_names[cls_id]} {conf*100:.1f}%"
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Put label text above box
                        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show the original frame with boxes in Streamlit (BGR, no color convert)
                stframe.image(frame, channels="BGR")

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
