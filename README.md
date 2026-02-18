# Pneumonia Diagnostic Hub: Multi-Model AI Detector

## Overview

This project is an end-to-end medical imaging application designed to assist in the detection of Pneumonia from Chest X-Ray images. The system evaluates and compares four distinct Deep Learning architectures to provide highly accurate diagnostic predictions.

## Features

* Multi-Model Selection: Compare results from VGG19, ResNet50, MobileNetV2, and EfficientNetB0.
* Fine-Tuned Precision: Includes specialized models with "unfrozen" layers for better detection of medical textures.
* Live Web Interface: A Flask-based platform for image uploads and real-time analysis.
* Confidence Scoring: Provides a probability percentage for every diagnosis.

## Technical Methodology

The project employs Transfer Learning by leveraging models pre-trained on the ImageNet dataset. The training pipeline consists of:

1. Feature Extraction: Freezing base layers to train a custom classification head.
2. Fine-Tuning: Unfreezing the final convolutional blocks to adapt to specific radiographic features.
3. Preprocessing: Converting images to grayscale, resizing to 128x128, and normalizing pixel values to a 0-1 range.

## Directory Structure

```text
project-root/
├── app.py                   # Main Flask application
├── static/
│   └── uploads/             # Directory for user-uploaded images
├── templates/
│   └── index.html           # Web interface template
├── model_weights/           # Directory for model storage
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Shahab-khan396/Pneumonia-Diagnostic-Multi-Model-AI-Detector.git
cd Pneumonia-Diagnostic-Multi-Model-AI-Detector

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the Application

```bash
python app.py

```

Access the application at `http://127.0.0.1:5000/`.

## Model Downloads

Due to GitHub's file size limitations, pre-trained model files are hosted externally. Download the required files and place them in the root directory:

| Model Filename | Architecture | Recommended For | Download Link |
| --- | --- | --- | --- |
| VGG19_model.h5 | VGG19 | High-detail analysis | [Link](https://drive.google.com/file/d/1StoeboiUFOCr903wNBwS6yl5dBBruCav/view?usp=drive_link) |
| resnet50_model.h5 | ResNet50 | Complex feature patterns | [Link](https://drive.google.com/file/d/1rE9Bj4h81CNxFfAuywLNxLSocl29tVCu/view?usp=drive_link) |
| mobilenet_model.h5 | MobileNetV2 | High-speed inference | [Link](https://drive.google.com/file/d/1fSpyvv54v3dFbQ3buHW5HlSr61fXBCh8/view?usp=drive_link) |
| efficientnet_model.h5 | EfficientNetB0 | Balanced performance | [Link](https://drive.google.com/file/d/1f8G8AduJYqA0AO4tCyrClWD1yWcYODrk/view?usp=drive_link) |

## Disclaimer

This application is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
