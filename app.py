import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
import os

MODEL_PATH = "butterfly_model_94acc.keras" 
CLASSES_PATH = "class_names.pkl"

print(f"Текущая папка: {os.getcwd()}")
print(f"Файлы в папке: {os.listdir()}")

if os.path.exists(MODEL_FILE):
    print("Файл модели найден. Загружаю...")
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print(" Модель успешно загружена в память!")
    except Exception as e:
        print(f"❌ Ошибка при чтении файла модели: {e}")
        model = None
else:
    print(f"❌ ОШИБКА: Файл '{MODEL_FILE}' НЕ НАЙДЕН в папке приложения.")
    model = None

if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "rb") as f:
        class_names = pickle.load(f)
else:
    print(f"Файл классов '{CLASSES_FILE}' не найден. Использую заглушку.")
    class_names = [f"Class {i}" for i in range(75)]

def classify_butterfly(image):
    if image is None: return None
    
    image = tf.image.resize(image, (224, 224))
    img_array = tf.expand_dims(image, 0)
    
    prediction = model.predict(img_array)
    scores = tf.nn.softmax(prediction[0])
    
    results = {}
    for i in range(len(class_names)):
        results[class_names[i]] = float(scores[i])
    return results


with gr.Blocks(title="Butterfly AI") as demo:
    gr.Markdown("#Butterfly Recognition AI")
    gr.Markdown("Upload a photo and the neural network will identify the butterfly species.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Фото", type="numpy", height=350)
            btn = gr.Button("Define type", variant="primary")
        
        with gr.Column():
            output = gr.Label(num_top_classes=3, label="Result")
    
    btn.click(classify_butterfly, inputs=input_img, outputs=output)

# Запуск
demo.launch()