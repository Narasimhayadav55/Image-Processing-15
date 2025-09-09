# 🖼️ Streamlit Image Processing Studio

## 📌 Abstract
This project is a **Streamlit-based GUI application** designed for **image processing and analysis**.  
It provides an intuitive interface to perform operations like **color conversion, filtering, transformation, enhancement, and edge detection** on images.  
The application is built as part of a **B.Tech Major Project (2025)** to demonstrate **AI/ML integration, user interface design, and dataset preparation** for cultural and visual computing research.  

---

## 🎯 Objectives
- Build an **interactive GUI** for image processing using Streamlit.  
- Provide **real-time transformations** and filtering options.  
- Design a **user-friendly layout** with menu bar, sidebar, display area, and status bar.  
- Support **image compression and export** for lightweight datasets.  
- Enable students and researchers to **analyze images easily** without coding.  

---

## 🚀 Features
### 1. **Menu Bar**
- 📂 Open → Upload image  
- 💾 Save → Download processed image  
- ❌ Exit → Quit the app  

### 2. **Sidebar Operations**
- **Image Info** → Resolution, channels, file size  
- **Color Conversions** → RGB ↔ Gray, HSV, YCbCr  
- **Transformations** → Rotation, Scaling, Translation  
- **Filtering & Morphology** → Gaussian, Median, Sobel, Laplacian  
- **Enhancement** → Histogram Equalization, Sharpening  
- **Edge Detection** → Canny, Sobel, Laplacian  
- **Compression** → Save in JPG, PNG, BMP  

### 3. **Display Area**
- Side-by-side view of **Original** and **Processed** images  

### 4. **Status Bar**
- Displays image properties like dimensions, channels, and file size  

---

## 🛠️ Requirements
- Python 3.9+  
- Libraries:  
  ```bash
  pip install streamlit opencv-python-headless numpy pillow
