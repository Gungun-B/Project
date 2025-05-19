import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import os
from ultralytics import YOLO
import tempfile

# Load models
model = joblib.load("tuned_rf_color_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder_enhanced.pkl")
yolo_model = YOLO("my_model.pt")

# Predefined palettes per style
style_palette = {
    'modern': [(157, 138, 126), (132, 125, 121), (161, 142, 124)],
    'industrial': [(102, 87, 73), (205, 205, 205), (114, 107, 103)],
    'minimalist': [(172, 164, 151), (150, 126, 100), (78, 67, 52)],
    'scandinavian': [(139, 112, 88), (151, 137, 126), (222, 216, 206)],
    'boho': [(199, 195, 188), (185, 182, 174), (207, 201, 196)]
}

# Utility functions
def extract_color_palette(image_path, num_colors=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(image)

    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)

    sorted_idx = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_idx]

    dom_color = sorted_colors[0]
    palette = sorted_colors[1:]

    return dom_color, palette


def plot_palette(palette, title):
    st.markdown(f"**{title}**")
    fig, ax = plt.subplots(figsize=(len(palette), 1))
    for i, color in enumerate(palette):
        ax.fill_between([i, i+1], 0, 1, color=np.array(color)/255.0)
    ax.axis('off')
    st.pyplot(fig)

def recommend_style(dominant_color):
    rgb = np.array(dominant_color).reshape(1, -1)
    rgb_scaled = scaler.transform(rgb)
    pred = model.predict(rgb_scaled)
    style = label_encoder.inverse_transform(pred)[0]
    return style

def run_yolo(image_path):
    results = yolo_model(image_path)
    return results[0].plot()

# Sidebar layout like Hello Streamlit
st.sidebar.title("Smart Interior Design 💻")
page = st.sidebar.selectbox(
    "Choose a demo:",
    [
        "🏠 Welcome",
        " Color Extraction",
        " Color Prediction",
        " Detect Objects"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image to try features.")

# Welcome Page
if page == "🏠 Welcome":
    st.title(" Welcome to Smart Interior Design Assistant ")
    st.markdown("""
    This app helps you:
    
    - 🎨 Extract dominant colors from interior room photos
    - 🖌️ Predict ideal wall paint colors using ML
    - 🔍 Detect furniture and items in the room using YOLOv8
    
    Use the sidebar to try out each demo!
    """)
    st.image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c", caption=" Interior Design ", use_column_width=True)

# Color Extraction Page
elif page == " Color Extraction":
    st.title(" Extract Colors")
    uploaded_file = st.file_uploader("Upload a Room Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        dom_color, palette = extract_color_palette(image_path)
        st.markdown(f"**Dominant Color:** `{dom_color}`")
        st.color_picker("Dominant", value=f'#{dom_color[0]:02x}{dom_color[1]:02x}{dom_color[2]:02x}')
        st.markdown("**Palette:**")
        for i, color in enumerate(palette):
            st.color_picker(f"Palette {i+1}", value=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')

# Wall Color Prediction Page
elif page == " Color Prediction":
    st.title(" Predict Colors ")
    uploaded_file = st.file_uploader("Upload a Room Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        dom_color, _ = extract_color_palette(image_path)

        dom_color, palette = extract_color_palette(image_path)
        rf_pred = plot_palette([dom_color] + list(palette), model)
        
        st.success(f"Color : {rf_pred}")

# Object Detection Page
elif page == " Detect Objects":
    st.title(" Object Detection ")
    uploaded_file = st.file_uploader("Upload a Room Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        result_image = run_yolo(image_path)
        st.image(result_image, caption="Detected Objects", use_column_width=True)