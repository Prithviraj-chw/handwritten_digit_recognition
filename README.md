# 🔢 Handwritten Digit Recognition

A machine learning project that recognizes handwritten digits using a trained deep learning model. This project uses the famous MNIST dataset to classify digits from **0–9** with high accuracy. It demonstrates the fundamentals of image preprocessing, neural networks, and digit classification in Python.

---

# ✨ Features

- Handwritten digit recognition using Deep Learning
- Trained on the MNIST dataset
- Image preprocessing and normalization
- Predicts digits from custom handwritten input
- Beginner-friendly machine learning project
- Simple and clean implementation

---

# 🛠️ Tech Stack

- Python: Core programming language.

- PyTorch: Deep learning framework used for the CNN architecture.

- Torchvision: Used for loading the MNIST dataset and image transformations.

- NumPy: Used for matrix operations and data manipulation.

- Matplotlib: Used for visualizing model predictions and performance.

- Streamlit: Framework used for the web-based deployment.

---

# 📚 Dataset

This project uses the **MNIST Handwritten Digit Dataset**, which contains:

- 60,000 training images
- 10,000 testing images
- Grayscale images of size 28×28 pixels

The MNIST dataset is one of the most widely used datasets for image classification and handwritten digit recognition.

---

# 📁 Project Structure

```bash
handwritten_digit_recognition/
│── model/                 # Saved trained models
│── dataset/               # Dataset files (if included)
│── images/                # Sample test images
│── train.py               # Model training script
│── predict.py             # Prediction script
│── app.py                 # GUI/Web app (if available)
│── requirements.txt       # Required dependencies
│── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Prithviraj-chw/handwritten_digit_recognition.git
cd handwritten_digit_recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 🧠 Train the Model

```bash
python train.py
```

## 🔍 Run Prediction

```bash
python predict.py
```

## 🌐 Run the App 

```bash
python app.py
```

---

# 🧩 How It Works

1. The MNIST dataset is loaded and preprocessed.
2. Images are normalized for better training performance.
3. A neural network model is trained on handwritten digit images.
4. The trained model predicts unseen handwritten digits.
5. The output digit is displayed with prediction confidence.

---

# 📈 Model Accuracy

The model achieves high accuracy on the MNIST test dataset, making it effective for handwritten digit recognition tasks.

---

# 🚧 Future Improvements

- Add real-time drawing canvas
- Improve prediction accuracy
- Deploy as a web application
- Add support for multiple handwritten digits
- Optimize model performance

---

# 🎓 Learning Outcomes

Through this project, you can learn:

- Basics of Machine Learning
- Neural Networks and CNNs
- Image preprocessing
- Model training and evaluation
- Working with TensorFlow/Keras

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

# 👨‍💻 Author

Created by **Prithviraj Chowdhury**  
🌐 GitHub: https://github.com/Prithviraj-chw
