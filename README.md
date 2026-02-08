# 👁 Eye Disease Detection using Deep Learning
## 📌 Project Overview

This project is a Flask-based web application that detects eye diseases from retinal images using a Convolutional Neural Network (CNN) with MobileNetV2 transfer learning.

The system classifies images into four categories:

Cataract

Diabetic Retinopathy

Glaucoma

Normal

Users can upload an eye image through a web interface and receive an instant prediction with confidence score.

## 🧠 Model Details

Architecture: MobileNetV2 (Transfer Learning)

Input Size: 224 × 224 × 3

Classes: 4

Training Accuracy: ~95%

Validation Accuracy: ~80%

Framework: TensorFlow / Keras

A confidence-threshold rejection mechanism is implemented to avoid predictions on non-eye images.


📂 Dataset Structure

Dataset/
 ├── cataract/
 ├── diabetic_retinopathy/
 ├── glaucoma/
 └── normal/