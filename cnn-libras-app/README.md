# 🤚 Hand Gesture Recognition for Brazilian Sign Language (Libras)

This is a **web application** that uses **TensorFlow.js** and **MediaPipe Hands** to recognize static hand gestures representing letters from the Brazilian Sign Language (**Libras**) alphabet via your webcam.

---

## 📸 How It Works

- The application accesses your **device's camera** and shows a **live video**.
- A **colored rectangle** (green/red) in the center indicates the **ROI (Region of Interest)** where your hand should be positioned.
- When a hand is detected inside the ROI, a snapshot is taken, resized to **64x64 pixels**, and sent to a **pre-trained model** for prediction.
- The app displays the predicted **letter** and the **confidence percentage** on the screen.

---

## 🛠️ Project Structure

```
/
├── index.html          # Main HTML page
├── tfjs_model/         # Folder containing the trained TensorFlow.js model
│   ├── model.json
│   └── group1-shard1of1.bin
└── README.md           # Project documentation
```

---

## 🧠 About the Model

- **Type**: Convolutional Neural Network (CNN)
- **Input shape**: `(64, 64, 3)` (RGB image)
- **Output classes**: 21 static hand gestures corresponding to the following letters:

  ```
  A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y
  ```

- **Training**: Model trained with Keras and converted to TensorFlow.js format.

---

## 🚀 Technologies Used

- [TensorFlow.js](https://www.tensorflow.org/js) — Machine learning in the browser.
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) — Real-time hand detection and tracking.
- **HTML5 + JavaScript** — Web development.

---

## 🌐 How to Run

### Running Locally (for development)

Browsers require a server to access the webcam for security reasons. You need to run a local server.

Example with Python:

```bash
# If you have Python 3 installed:
python -m http.server
```

Then open your browser and go to:

```
http://localhost:8000
```

## 📂 Model Loading

In the `index.html`, the model is loaded like this:

```javascript
async function loadModel() {
  model = await tf.loadLayersModel("tfjs_model/model.json");
  resultEl.innerText = "Model loaded!";
}
```

Make sure the path to `model.json` is correct relative to your `index.html` file.
