# CNN-Libras

This project aims to create a sign language recognition system for Libras (Brazilian Sign Language) using convolutional neural networks (CNNs).

## Requirements

- Python version 3.7.9
- TensorFlow.js for converting the `.h5` model to `.json`

## Setting Up the Environment

1. **Install Python 3.7.9**:
   Make sure you are using Python 3.7.9. You can check your installed version with:

   ```bash
   python --version
   ```

2. **Install dependencies**:
   Run the following command to install all project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you need to install manually, use:

   ```bash
   pip install tensorflow tensorflowjs
   ```

   In addition to these dependencies, you must also install CUDA 11.2 for GPU support.

## Model Conversion

To convert the saved `.h5` model to a JSON format compatible with TensorFlow.js, use the following command:

```bash
tensorflowjs_converter --input_format keras modelo.h5 pasta_destino
```

Replace `modelo.h5` with the path to your model and `pasta_destino` with the directory where you want to save the converted files.

## Running the Project

Make sure all dependencies are installed.

- To **train the model**, run the following command inside the `main` folder:

  ```bash
  python main/train.py
  ```

- To **run the application**, use:

  ```bash
  python main/app_64x64x3.py
  ```