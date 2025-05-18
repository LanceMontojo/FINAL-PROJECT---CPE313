import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2

# Title
st.title("RTDETR Fire Classifier with Extinguisher Recommendations")

# Load model
@st.cache_resource
def load_model():
    return RTDETR("BESTO FRIENDO.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Class extinguisher recommendation mapping
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

# Handle image and detection
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        results = model(image)
        result = results[0]  # Get the first result

        # Draw boxes on image
        img = np.array(result.plot())
        st.image(img, caption="Detected Image with Boxes", use_column_width=True)

        # Get detected classes
        detected_classes = result.names
        predicted_class_indices = result.boxes.cls.cpu().numpy().astype(int)

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
