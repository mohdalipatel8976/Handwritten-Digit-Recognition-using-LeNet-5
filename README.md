# 📝 Handwritten Digit Recognition using LeNet-5 CNN  

## 📌 Objective  
The goal of this project is to build and evaluate a **Convolutional Neural Network (CNN)** based on the **LeNet-5 architecture** to classify handwritten digits (0–9) from the **MNIST dataset**.  

This project demonstrates:  
- Implementation of a classical CNN (LeNet-5).  
- Achieving high classification accuracy on MNIST.  
- Understanding CNN layers (convolution, pooling, fully connected).  
- Evaluating performance with metrics like accuracy, precision, recall, and F1-score.  
- Practical experience with **TensorFlow/Keras** for model development.  

## 📊 Dataset: MNIST  
- **Training Set:** 60,000 grayscale images  
- **Test Set:** 10,000 grayscale images  
- **Image Size:** 28x28 pixels (1 channel)  
- **Classes:** Digits 0–9  

## 🏗️ LeNet-5 Architecture  
1. **Input Layer:** 28×28 grayscale image  
2. **Conv1 (C1):** 6 filters, 5×5 kernel, ReLU → output 24×24×6  
3. **Pooling (S2):** Average Pooling 2×2 → output 12×12×6  
4. **Conv2 (C3):** 16 filters, 5×5 kernel, ReLU → output 8×8×16  
5. **Pooling (S4):** Average Pooling 2×2 → output 4×4×16  
6. **FC1 (C5):** 120 neurons, ReLU  
7. **FC2 (F6):** 84 neurons, ReLU  
8. **Output Layer:** 10 neurons (Softmax)  

## ⚙️ Implementation  

### 🔹 Install Dependencies  
```bash
pip install tensorflow matplotlib numpy
````

### 🔹 Model Training (TensorFlow/Keras)

```python
# Load MNIST dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# Build Model
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(6, (5,5), activation="relu", input_shape=(28,28,1)),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(16, (5,5), activation="relu"),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
```

## 📈 Results

* **Final Test Accuracy:** \~94.48%
* **Validation Accuracy:** \~95.25%
* Model converged within **5 epochs** using the **Adam optimizer**.

## 🔮 Future Improvements

* Experiment with deeper CNNs (AlexNet, ResNet).
* Apply data augmentation for robustness.
* Deploy using **Streamlit/Flask**.
* Compare with SVM, Random Forest, or modern CNN architectures.

## 🛠️ Tech Stack

* Python, NumPy, Matplotlib
* TensorFlow / Keras
* Jupyter Notebook


## ✨ Author

**Mohammedali Patel**
