import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- Main UI Header ---
st.title("🔍 PCA & K-Means Clustering Explorer")
st.write("Use this app to explore clustering and dimensionality reduction using your own dataset or a sample one.")


# --- Sidebar: Dataset Selection ---
st.sidebar.title("📂 Data Input")
use_sample = st.sidebar.checkbox("Use Breast Cancer Dataset (built-in)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type=["csv"])

# --- Load Data ---
if use_sample:
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data  # Feature matrix
    y = breast_cancer.target  # Target variable (diagnosis)
    feature_names = breast_cancer.feature_names
    target_names = breast_cancer.target_names
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    # Display the dataset
    st.dataframe(X.head())
    st.write("**Target Classes:**", target_names)



else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df = pd.get_dummies(df, drop_first=False)  # Convert categorical variables to dummy variables
        target_col = st.selectbox("Select the target column", df.columns)
        x_features = df.drop(columns=[target_col])
        X = st.multiselect("Select features for the model", x_features.columns) # Get this list of features
        X = df[X] # Turn that list into a dataframe
        y = df[target_col]
        col1, col2 = st.columns(2)
        with col1:
                st.write("Preview of X:")
                st.dataframe(X.head())
        with col2: 
                st.write("Preview of y:")
                st.dataframe(y.head())
    else:
        st.warning("Please upload a dataset or check the sample dataset box.")
        st.stop()



# --- Data Preparation ---

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# --- Sidebar: Model Selection ---
st.sidebar.title("⚙️ Model Settings")
model_type = st.sidebar.radio("Choose a model", ["PCA", "K-Means Clustering"])

# --- PCA Section ---
if model_type == "PCA":
    st.header("📉 Principal Component Analysis")
    n_components = st.slider("Number of components", 2, len(feature_names), 2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    explained_variance = pca.explained_variance_ratio_

    st.write(f"**Explained variance ratio:** {np.round(np.cumsum(explained_variance),2)}")
    st.caption("Note: The explained variance ratio indicates how much information (variance) is captured by each principal component.")

    st.subheader("📊 PCA Bar Plot")
    # 3d. Bar Plot: Variance Explained by Each Component
    fig, ax = plt.subplots(figsize=(5, 4))
    components = range(1, len(pca.explained_variance_ratio_) + 1)

    ax.bar(components, pca.explained_variance_ratio_, alpha=0.7, color='green')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('Variance Explained by Each Principal Component')
    ax.set_xticks(components)
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.subheader("📊 PCA Scree Plot")
    pca_full = PCA(n_components = len(feature_names)).fit(X_std)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Variance Explained')
    ax.set_xticks(range(1, len(cumulative_variance) + 1))
    ax.grid(True)
    st.pyplot(fig)
    
# --- K-Means Section ---
elif model_type == "K-Means Clustering":
    st.subheader("📈 K-Means Clustering")
    k = st.slider("Number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_std)