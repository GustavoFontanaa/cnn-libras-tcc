# ✋ Libras Sign Recognition — Computer Science TCC

Welcome to my **Undergraduate Thesis (TCC)** project for the **Computer Science** course! 🚀  
This repository brings together two interconnected solutions focused on recognizing **Brazilian Sign Language (LIBRAS)** alphabet signs through the power of **Computer Vision** and **Machine Learning**.

---

## 🗂️ Repository Structure

### 📸 1. Web Application – Real-Time Sign Recognition

**Description:**  
An interactive **web application** that captures hand gestures through the webcam, detects the hand using **MediaPipe Hands**, and classifies the corresponding LIBRAS letter using a trained **TensorFlow.js** model.

**Technologies:**
- HTML5 & JavaScript
- TensorFlow.js
- MediaPipe Hands

**How it works:**
- The webcam captures the user's hand.
- A **Region of Interest (ROI)** is extracted and resized to **64x64 pixels**.
- The **TensorFlow.js** model predicts the corresponding letter directly in the browser — **no server needed**!

---

### 🧠 2. Machine Learning - Training the CNN Model

**Description:**  
A powerful **Python script** that trains a **Convolutional Neural Network (CNN)** to classify **21 LIBRAS hand signs**.

**Technologies:**
- Python 3
- TensorFlow / Keras
- OpenCV
- Matplotlib
- Scikit-learn

**How it works:**
- A dataset of hand sign images is used to train the model.
- After training, the model is saved in `.h5` format.
- Training history plots (accuracy and loss) and the network architecture diagram are also generated automatically.

## 🎯 Project Objective

The aim of this project is to **promote accessibility and inclusion** by leveraging modern technologies to recognize and translate **LIBRAS signs** automatically.

> Empowering communication with **technology and empathy**. 🤝✨

## 👨‍💻 Developed by

**Gustavo Antonio dos Santos Fontana**  
Undergraduate Thesis (TCC) — Computer Science
