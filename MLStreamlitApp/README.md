# 📊 Classification Machine Learning Streamlit App

## 🚀 Project Overview

This Streamlit app allows users to interactively explore and train **classification** machine learning models on tabular datasets. Users can upload their own CSV file or choose a built-in dataset (Titanic), preprocess the data, select features, and evaluate models with visual metrics like ROC curves and accuracy scores.

## 🛠️ Instructions

### ✅ How to Run Locally:

1. **Clone this repo**:
   ```bash
   git clone https://github.com/ArisC123/Cardoso-Data-Science-Portfolio.git
   cd Cardoso-Data-Science-Portfolio/MLStreamlitApp
   
2. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   
3.  **Run the App**
   ```bash
   streamlit run main.py
   ```

### 🌐 Deployed App:
👉 Try it here: [Machine Learning App](https://cardoso-data-science-portfolio-mzzmppfiwiggftbj8u4ytj.streamlit.app/)

## ✨ App Features

### 📁 Dataset Options
- Upload your own CSV
- Choose the Titanic dataset

### 🧹 Preprocessing Tools
- Remove duplicates
- Handle missing values
- Encode categorical variables

### 📈 Model Selection
- Decision Tree: A non-linear model that splits the data based on feature thresholds, offering flexibility and interpretability.
- Logistic Regression: A linear classification model useful for binary classification problems
  

### ⚙️ Hyperparameters
- **Decision Tree**: Users can interactively select values for **Criterion**, **Max Depth**, **Min Samples Split**, **Min Samples Leaf** to control model complexity and performance
- **Logistic Regression**: None for this app

### 📊 Model Evaluation
- Accuracy
- ROC Curve and AUC Score

## 📈 Visual Examples
Here is an example of a graph that provides insight into the dataset:

![ROC Curve](./Assets/img/ROC-Curve.png)

## 📚 References
Below are the resources used to build this project. Feel free to explore them for your own work.

- [Streamlit API](https://docs.streamlit.io/develop/api-reference)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Matplotlib](https://matplotlib.org/stable/api/pyplot_summary.html)

