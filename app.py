import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Image Processing Studio", layout="wide")

# ----------- MENU BAR -----------
menu = st.sidebar.selectbox("📂 File Menu", ["Open Image", "Save Processed", "Exit"])

# ----------- IMAGE UPLOAD -----------
if menu == "Open Image":
    uploaded_file = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.session_state["image"] = image

# Exit Button
if menu == "Exit":
    st.stop()

# If no image, show info
if "image" not in st.session_state:
    st.info("👆 Upload an image to get started")
    st.stop()

image = st.session_state["image"]
processed_img = image.copy()

# ----------- SIDEBAR OPERATIONS -----------
st.sidebar.header("🛠 Operations")

operation = st.sidebar.radio(
    "Select Operation",
    [
        "Image Info",
        "Color Conversion",
        "Transformation",
        "Filtering & Morphology",
        "Enhancement",
        "Edge Detection",
        "Compression",
    ],
)

# ----------- OPERATIONS -----------
if operation == "Image Info":
    h, w, c = image.shape
    st.sidebar.write(f"Resolution: {w}x{h}")
    st.sidebar.write(f"Channels: {c}")
    st.sidebar.write(f"File Size: ~{image.nbytes / 1024:.2f} KB")

elif operation == "Color Conversion":
    choice = st.sidebar.selectbox(
        "Choose Conversion", ["RGB→Gray", "RGB→HSV", "RGB→YCbCr"]
    )
    if choice == "RGB→Gray":
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif choice == "RGB→HSV":
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif choice == "RGB→YCbCr":
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

elif operation == "Transformation":
    choice = st.sidebar.selectbox(
        "Choose Transformation", ["Rotate", "Scale", "Translate"]
    )
    if choice == "Rotate":
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 45)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        processed_img = cv2.warpAffine(image, M, (w, h))
    elif choice == "Scale":
        scale = st.sidebar.slider("Scale Factor", 0.1, 2.0, 1.0)
        processed_img = cv2.resize(image, None, fx=scale, fy=scale)
    elif choice == "Translate":
        tx = st.sidebar.slider("Shift X", -100, 100, 0)
        ty = st.sidebar.slider("Shift Y", -100, 100, 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        processed_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

elif operation == "Filtering & Morphology":
    choice = st.sidebar.selectbox(
        "Choose Filter", ["Gaussian", "Median", "Sobel", "Laplacian"]
    )
    if choice == "Gaussian":
        processed_img = cv2.GaussianBlur(image, (5, 5), 0)
    elif choice == "Median":
        processed_img = cv2.medianBlur(image, 5)
    elif choice == "Sobel":
        processed_img = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    elif choice == "Laplacian":
        processed_img = cv2.Laplacian(image, cv2.CV_64F)

elif operation == "Enhancement":
    choice = st.sidebar.selectbox(
        "Choose Enhancement", ["Histogram Equalization", "Sharpen"]
    )
    if choice == "Histogram Equalization":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.equalizeHist(gray)
    elif choice == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(image, -1, kernel)

elif operation == "Edge Detection":
    choice = st.sidebar.selectbox(
        "Choose Edge Detector", ["Canny", "Sobel", "Laplacian"]
    )
    if choice == "Canny":
        processed_img = cv2.Canny(image, 100, 200)
    elif choice == "Sobel":
        processed_img = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    elif choice == "Laplacian":
        processed_img = cv2.Laplacian(image, cv2.CV_64F)

elif operation == "Compression":
    choice = st.sidebar.selectbox("Save Format", ["JPG", "PNG", "BMP"])
    _, buffer = cv2.imencode(f".{choice.lower()}", image)
    st.sidebar.download_button(
        "💾 Download", buffer.tobytes(), file_name=f"processed.{choice.lower()}"
    )

# ----------- DISPLAY AREA -----------
col1, col2 = st.columns(2)
with col1:
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Original",
        use_column_width=True,
    )
with col2:
    if processed_img.ndim == 2:
        st.image(
            processed_img, caption="Processed", use_column_width=True, channels="GRAY"
        )
    else:
        st.image(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            caption="Processed",
            use_column_width=True,
        )

# ----------- STATUS BAR -----------
st.markdown("---")
st.write(
    f"**Status:** Image {image.shape[1]}x{image.shape[0]}, Channels: {image.shape[2]}"
)
