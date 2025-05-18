import streamlit as st
from ultralytics import RTDETR
from PIL import Image

st.title("RTDETR Fire Classifier with Extinguisher Recommendations")

# Extinguisher recommendations dictionary
extinguisher_info = {
    "Class A": {
        "recommended": "Water mist, foam, or multipurpose dry chemical extinguishers",
        "not_recommended": "Carbon dioxide and clean agents extinguishers"
    },
    "Class B": {
        "recommended": "Foam, carbon dioxide, or dry chemicals extinguishers",
        "not_recommended": "Water-based extinguishers"
    },
    "Class C": {
        "recommended": "Carbon dioxide, clean agents, or dry chemicals extinguishers",
        "not_recommended": "Water-based extinguishers"
    },
    "Class D": {
        "recommended": "Specialized dry powder extinguishers",
        "not_recommended": "Other extinguisher types"
    },
    "Class F": {
        "recommended": "Wet chemical extinguishers to prevent splattering and reignition",
        "not_recommended": "Other extinguisher types"
    },
}

uploaded_file = st.file_uploader("Upload an image for fire class detection", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return RTDETR("BESTO FRIENDO.pt")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()

    if st.button("Run Detection"):
        results = model(image)
        st.image(results.orig_img, caption="Detected Image with Boxes", use_column_width=True)

        detected_classes = results.names  
        predicted_class_indices = results.box.cls.cpu().numpy().astype(int)
        detected_labels = set(detected_classes[i] for i in predicted_class_indices)

        st.write("Detected Fire Classes and Extinguisher Recommendations:")

        for cls in detected_labels:
            info = extinguisher_info.get(cls)
            if info:
                st.markdown(f"**{cls}**")
                st.markdown(f"<span style='color:green;'>Recommended: {info['recommended']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:red;'>Not Recommended: {info['not_recommended']}</span>", unsafe_allow_html=True)
            else:
                st.write(f"No extinguisher info available for class: {cls}")
