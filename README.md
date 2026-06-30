# ♻️ Smart Drinking Waste Detection and Classification

An end-to-end computer vision system that detects and classifies drinking waste (Aluminum Cans, Glass, PET, and HDPEM plastics) using **YOLOv8** for object detection and **MobileNetV2** for material classification, with **Grad-CAM** explainability and an interactive **Gradio** web app for real-time recycling recommendations.

## 📌 Overview

This project tackles automated waste sorting by combining object detection with transfer-learning-based classification:

1. **YOLOv8** detects waste objects in an image and generates bounding boxes.
2. Each detected object is cropped and passed to a **MobileNetV2** classifier.
3. The classifier predicts the material type: AluCan, Glass, PET, or HDPEM.
4. **Grad-CAM** generates a heatmap explaining which regions influenced the prediction.
5. A rule-based recommendation module provides recycling guidance based on the predicted material.
6. Results, confidence scores, and Grad-CAM overlays are displayed through an interactive Gradio interface.

## 🧠 Model Architecture

- **Detection:** YOLOv8 trained on a custom drinking-waste dataset with advanced data augmentation.
- **Classification:** MobileNetV2 (pretrained on ImageNet) with a Global Average Pooling layer, Dense(256, ReLU), Dropout(0.4), and a Softmax output layer for 4-class classification.
- **Training:** Adam optimizer, categorical cross-entropy loss, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
- **Explainability:** Grad-CAM activation heatmaps highlighting the regions driving each prediction.

## 📊 Dataset

A stratified 70% / 15% / 15% train-validation-test split was used to preserve class balance across four categories:

- AluCan (Aluminum Cans)
- Glass
- PET
- HDPEM

Images were resized to 224×224 and augmented with random rotation, translation, zoom, brightness adjustment, and horizontal flipping.

## 📈 Results

| Metric | Value |
|---|---|
| Classification Accuracy | 96.55% |
| Precision | 96.63% |
| Recall | 96.55% |
| F1-Score | 96.56% |
| YOLO mAP@0.5 | 0.9915 |
| YOLO mAP@0.5:0.95 | 0.8318 |
| YOLO F1-Score | 0.9871 |

Evaluation included accuracy/loss curves, a confusion matrix, per-class mAP analysis, classification reports, and Grad-CAM visualizations confirming the classifier focused on the waste object rather than background regions.

## 🌐 Web Application

The Gradio app integrates YOLOv8 detection, MobileNetV2 classification, and Grad-CAM into a single interface. Users upload an image and the system:

- Detects waste objects and draws bounding boxes
- Classifies each object with confidence scores
- Overlays a Grad-CAM heatmap
- Displays a recycling recommendation for the predicted material

## 🛠️ Tech Stack

`Python` · `YOLOv8 (Ultralytics)` · `TensorFlow / Keras` · `MobileNetV2` · `OpenCV` · `Grad-CAM` · `Gradio` · `Scikit-learn` · `Matplotlib / Seaborn`

## 🚀 Future Work

- Expand the dataset with additional waste categories
- Enable real-time webcam deployment
- Optimize the pipeline for embedded/edge devices

## 👥 Team

- **Ufaq Hafeez** (2023-CS-75)
- **Hira Sohail** (2023-CS-76)

**Supervised by:** Dr. Muhammad Waseem
Department of Computer Science, University of Engineering and Technology (UET), Lahore

## 🙏 Acknowledgements

We sincerely thank our respected instructors, Dr. Muhammad Usman Ghani Khan and Dr. Muhammad Waseem, as well as the Department of Computer Science, UET Lahore, for their guidance, support, and encouragement throughout this project.
