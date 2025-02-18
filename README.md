# Breast Cancer Classification using Machine Learning

This project implements a **breast cancer classification** model using **Machine Learning** algorithms like **Logistic Regression** . The project is built in Python using **Scikit-Learn** for model building and **Pandas** and **NumPy** for data processing. The model is evaluated on a dataset to predict whether a given tumor is **malignant** or **benign**.

---
## 📌 Features
- Implements **Logistic Regression** algorithm for classification.
- Achieved **92.9% accuracy** on the test set.
- Includes **data preprocessing** steps such as handling missing values and feature scaling.
- The model can be tested with any new data after training.

---
## 📂 Dataset
The dataset used in this project is the **Breast Cancer Wisconsin Dataset**, which is available in `sklearn.datasets`.
- Features: Various cell nucleus measurements.
- Target: **1 (Benign) or 0 (Malignant)**.
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---
## 📂 Files
- `Breast_Cancer_Classification_using_Machine_Learning.ipynb` → Jupyter Notebook containing data loading, preprocessing, model training, evaluation, and testing.
- `model.pkl` → Saved machine learning model (Pickle format) for future predictions.
- `requirements.txt` → List of required Python packages.

---
## 🚀 How to Run

### 📌 Prerequisites:
- **Python 3.x** installed.
- **Jupyter Notebook** or **Google Colab**.

### 📌 Installation:
1. **Clone this repository:**
   ```bash
   git clone https://github.com/nandinisharma166/Breast-Cancer-Classification-ML.git
   cd Breast-Cancer-Classification-ML
   ```
2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `Breast_Cancer_Classification_using_Machine_Learning.ipynb` and execute the cells.

---
## 🎯 Example Usage
Once the model is trained, you can use it to make predictions:
```python
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example input (random feature values)
sample_input = np.array([[14.1, 15.5, 89.4, 611.3, 0.1, 0.09, 0.05, 0.06, 0.18, 0.07]])

# Predict the class
prediction = model.predict(sample_input)
print("Predicted Class:", "Malignant" if prediction[0] == 1 else "Benign")
```

---
## 📜 License
This project is open-source and free to use for educational purposes.

---
## ⭐ Contributing
Feel free to fork this repository, improve the code, and submit pull requests!

---
## 📬 Contact
For any questions or discussions, reach out via GitHub Issues!

---

