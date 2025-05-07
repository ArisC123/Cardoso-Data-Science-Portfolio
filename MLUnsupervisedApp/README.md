# 📊 Unsupervised Machine Learning App — PCA & K-Means Clustering

## 🔍 Project Overview

This interactive Streamlit app allows users to explore **unsupervised machine learning techniques** using either a built-in breast cancer dataset or a custom CSV upload. The app implements **Principal Component Analysis (PCA)** and **K-Means Clustering**, providing intuitive controls to tune parameters and visualize results.

### ✅ Users can:
- Upload their own datasets or use the built-in breast cancer data
- Choose which columns to analyze
- Adjust key hyperparameters (number of PCA components or clusters)
- Visualize PCA variance and clustering results with clear, annotated plots
- For **K-Means Clustering**, users can:
  - Select the number of clusters (`k`)
  - View the silhouette score to evaluate clustering quality
  - See clusters displayed in 2D space (via PCA reduction)

  ## 🛠️ Instructions

### ✅ How to Run Locally:

1. **Clone this repo**:
   ```bash
   git clone https://github.com/ArisC123/Cardoso-Data-Science-Portfolio.git
   cd Cardoso-Data-Science-Portfolio/MLUnsupervisedApp
   
2. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   
3.  **Run the App**
   ```bash
   streamlit run main.py
   ```