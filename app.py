import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

st.title("🧠 Brain Tumor Classification with Grad-CAM")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    image_resized = cv2.resize(image, (224, 224))
    image_array = image_resized / 255.0
    input_array = np.expand_dims(image_array, axis=0)

    # 🔁 Load the trained model (Change the path if needed)
    model = load_model("model/brain_tumor_model.h5")

    # 🔍 Make prediction
    prediction = model.predict(input_array)
    st.write("Prediction:", prediction)

    # 🔥 Grad-CAM explainability
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("conv2d").output, model.output]  # Replace 'conv2d' with your last conv layer name
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_array)
        loss = predictions[:, np.argmax(prediction[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    st.image(superimposed_img, caption="Grad-CAM Explanation", use_column_width=True)
