# 🦋 Butterfly Recognition AI

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/IljaTech/Butterfly-Recognizer)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app/)

A Deep Learning application capable of classifying **100 butterfly species** based on images. The model is built using a Convolutional Neural Network (CNN) and deployed as a web app.

## 🎥 Live Demo
Try the model yourself without installation:
👉 **[Click here to open the App on Hugging Face](https://huggingface.co/spaces/IljaTech/Butterfly-Recognizer)**

### How it works
![Demo Animation](/public/butterfly.gif)
*(Upload an image of a butterfly, and the model predicts the species with a confidence score)*

## 📊 Interface Screenshot
![App Screenshot](/public/butterfly.png)

## 🧠 Project Overview

### The Goal
To create an AI model that can accurately identify butterfly species from photographs, helping nature enthusiasts and researchers categorize insects.

### The Model
* **Architecture:** Custom Convolutional Neural Network (CNN).
* **Accuracy:** Achieved **94.44% accuracy** on the test dataset.
* **Dataset:** [Butterfly Image Classification](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species) (100 classes).
* **Training:** Performed in Google Colab (GPU accelerated).

## 🛠 Tech Stack
* **Deep Learning:** TensorFlow, Keras.
* **Data Processing:** NumPy, Pandas, PIL.
* **Interface:** Gradio (for the web GUI).
* **Deployment:** Hugging Face Spaces.

## 📂 Repository Structure
```text
.
├── model/                  # Trained model files
│   ├── butterfly_model_94_44acc.keras
│   └── class_names.pkl
├── notebooks/              # Jupyter Notebooks used for training
│   └── training.ipynb
├── app.py                  # Main application script
├── public/                 # Demo assets (gif, png)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
