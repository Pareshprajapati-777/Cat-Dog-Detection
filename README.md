# 🐾 Cat vs Dog AI Detector

A modern, interactive Deep Learning Web Application built with **PyTorch** and **Streamlit** to classify images of Cats 🐱 and Dogs 🐶 with real-time confidence scores and probability metrics.

---

## 🌟 Key Features

- **🤖 PyTorch AI Engine**: Powered by **MobileNetV2** architecture fine-tuned for high-accuracy binary image classification.
- **🎨 Glassmorphism UI**: Custom Dark-themed Streamlit interface with dynamic visual feedback and smooth styling.
- **📊 Detailed Analysis**: Displays predicted class, percentage confidence, and probability distribution for both Cat and Dog classes.
- **🖼️ Flexible Input**:
  - Upload your own local images (`.jpg`, `.jpeg`, `.png`).
  - Use 1-click sample images provided in the dataset.
- **⚡ Fallback Mechanism**: Dynamically loads fine-tuned model weights (`cat_dog_model.pth`) or falls back to pre-trained ImageNet weights seamlessly.

---

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Deep Learning**: PyTorch (`torch`, `torchvision`)
- **Image Processing**: Pillow (`PIL`)
- **Language**: Python 3.8+

---

## 📂 Project Structure

```text
Cat_Dog/
├── app.py                # Main Streamlit web app interface & layout
├── model_utils.py        # PyTorch model loader, pre-processing & inference logic
├── cat_dog_model.pth     # Fine-tuned PyTorch model weights
├── archive/              # Dataset folder containing sample Cat & Dog images
└── README.md             # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Install the required packages using `pip`:

```bash
pip install streamlit torch torchvision pillow
```

### 3. Run the Application
Start the Streamlit application by running:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 📸 How to Use
...
1. Launch the web app using `streamlit run app.py`.
2. Select a sample image from the sidebar or upload your own image.
3. View instant predictions, confidence scores, and probability distribution charts.
...
---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
