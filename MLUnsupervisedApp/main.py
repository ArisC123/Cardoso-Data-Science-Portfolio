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
        feature_names = df.drop(columns=[target_col])
        X = st.multiselect("Select features for the model", feature_names.columns) # Get this list of features
        X = df[X] # Turn that list into a dataframe
        y = df[target_col]
        feature_names = X.columns.tolist()  # Get the feature names
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
    st.caption("PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms high-dimensional data into a smaller set of uncorrelated variables called principal components. These components capture the directions of maximum variance in the data, helping to simplify analysis while retaining the most important information.")
    n_components = st.slider("Number of components", 2, len(feature_names), 2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    explained_variance = pca.explained_variance_ratio_

    st.write(f"**Explained variance ratio:** {np.round(np.cumsum(explained_variance),2)}")
    st.caption("Note: The explained variance ratio indicates how much information (variance) is captured by each principal component.")

    st.subheader("📊 PCA Bar Plot")
    # Assuming `pca` has already been fitted
    explained = pca.explained_variance_ratio_ * 100  # Convert to percentage
    components = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Bar plot for individual variance
    bar_color = 'steelblue'
    ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])

    # Add labels above bars
    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

    # Line plot for cumulative variance on secondary axis
    ax2 = ax1.twinx()
    line_color = 'crimson'
    ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    ax1.grid(False)
    ax2.grid(False)

    plt.title('PCA: Variance Explained', pad=20)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)
    with st.expander("📘 What does this graph show?"):
        st.write("""
        This graph displays how much variance each principal component explains (bars), along with the cumulative variance captured as more components are included (line).  
        It helps you decide how many components are needed to retain most of the data's information.
        """)

    st.divider()
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    # Split the standardized (original) data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
    # Split the PCA-reduced data into training and test sets (using the same random state)
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # 4a. Logistic Regression on Original Data
    lg_model = LogisticRegression()
    lg_model.fit(X_train, y_train)
    y_pred = lg_model.predict(X_test)
    accuracy_lg = accuracy_score(y_test, y_pred)
 

    # 4a. Logistic Regression on PCA Data
    lg_model_pca = LogisticRegression()
    lg_model_pca.fit(X_train_pca, y_train)
    y_pred_pca = lg_model_pca.predict(X_test_pca)
    accuracy_lg_pca = accuracy_score(y_test, y_pred_pca)
    
    # Display accruacy comparison reports
    col1, col2 = st.columns(2)
    col1.metric(label = "Logistic Regression Accuracy with Original Data", value = f"{accuracy_lg:.3f}")
    col2.metric(label = "Logistic Regression Accuracy with PCA Data", value = f"{accuracy_lg_pca:.3f}")
   

    
# --- K-Means Section ---
elif model_type == "K-Means Clustering":
    st.subheader("📈 K-Means Clustering")
    st.caption("K-Means is an unsupervised learning algorithm that groups data into k clusters based on similarity. It assigns each data point to the nearest cluster center (centroid), then updates the centroids iteratively to minimize within-cluster variance. The goal is to find natural groupings in the data without using labeled outcomes")
    k = st.slider("Number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluserts = kmeans.fit_predict(X_std)