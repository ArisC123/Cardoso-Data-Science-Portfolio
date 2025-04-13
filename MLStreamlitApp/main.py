import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import graphviz
from sklearn import tree 
from sklearn.metrics import roc_curve, roc_auc_score

st.title("Welcome to This Machine Learning App")
st.divider()

# Step 1: File upload
st.write("Upload a CSV file to start")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Step 2: Initialize session state
if uploaded_file and "df" not in st.session_state:
    st.session_state.original_df = pd.read_csv(uploaded_file)
    st.session_state.df = st.session_state.original_df.copy()

# Step 3: Only proceed if the file has been uploaded and stored
if "df" in st.session_state:
    df = st.session_state.df 

    st.write("Here's a sneak peek of your dataset:")
    st.dataframe(df.head())
    st.write("Dataset shape:", df.shape)
    st.divider()

    # Sidebar Preprocessing
    st.sidebar.header("Preprocessing Steps")
    preprocess = st.sidebar.radio("Select the preprocessing step", 
                                   ["None", "Remove Duplicates", "Remove Missing Values", "Encode Categorical Variables"])
    
    tab1, tab2 = st.tabs(["🧹 Data Preprocessing", "🤖 Model Training"])

    ###################### Data Processing Tab Code ############################################

    with tab1:
        # Remove Duplicates
        if preprocess == "Remove Duplicates":
            before = df.shape[0]
            df.drop_duplicates(inplace=True) #Drop duplicates 
            after = df.shape[0]
            st.success(f"Removed {before - after} duplicate rows.")
            st.session_state.df = df # update session state with new df
            st.dataframe(df.head())
            st.write("New shape:", df.shape)

        # Remove Missing Values
        elif preprocess == "Remove Missing Values":
            st.write("Missing values per column:")
            st.dataframe(df.isnull().sum())
            drop_col = st.selectbox("Select the column to drop rows with missing values", ['None'] + list(df.columns))
            if drop_col == 'None':
                pass
            else: 
                if st.radio(f'Drop the rows with missing columns of {drop_col}:', ['No', 'Yes']) == 'Yes':
                    before = df.shape[0]
                    df.dropna(subset=[drop_col], inplace=True) # Drop rows with missing values in the selected column
                    after = df.shape[0]
                    st.success(f"Dropped {before - after} rows with missing values in '{drop_col}'")
                    st.session_state.df = df
                    st.dataframe(df.head())
                    st.write("New shape:", df.shape)

        # Get Dummy Variables
        elif preprocess == "Encode Categorical Variables":
            encode = st.selectbox("Select the column to encode", ['None'] + list(df.select_dtypes(exclude = ['int64', 'float64', 'bool']).columns))
            if encode == 'None':
                pass
            else:
                if st.radio(f'Encode {encode}?:', ['No', 'Yes']) == 'Yes':
                    st.write(f"Encoding column: {encode}")
                    # Create dummy variables
                    df = pd.get_dummies(df, columns=[encode], drop_first=True) 
                    st.session_state.df = df # update session state with new df
            st.dataframe(df.head())

        # Option to reset
        if st.button("Reset to Original Dataset"):
            st.session_state.df = st.session_state.original_df.copy()
            st.success("Dataset reset to original version.")
            
###################### Modeling Tab Code ############################################

        st.divider()
    with tab2 :
        # Model Selection
        st.header('Model Selection')
        model_type = st.radio("Choose the model to train", ["Decision Tree", "Logistic Regression"])
        # Setting the x and y
        if model_type == "Decision Tree":
            st.success("You selected Decision Tree")
            target_col = st.selectbox("Select the target column", df.columns)
            x_features = df.drop(columns=[target_col])
            X = st.multiselect("Select features for the model", x_features.columns) # Get this list of features
            X = df[X] # Turn that list into a dataframe
            y = df[target_col]
            st.subheader(f'Here are the features:\n **{list(X.columns)}**')
            st.subheader(f'Here is the target:\n **{target_col}**')
            st.divider()

            # Displaying the chosen features and target variable
            col1, col2 = st.columns(2)
            with col1:
                st.write("Preview of X:")
                st.dataframe(X.head())
            with col2: 
                st.write("Preview of y:")
                st.dataframe(y.head())
            
            st.sidebar.divider()
            st.sidebar.subheader("Hyperparameters for Decision Tree")
            criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
            max_depth = st.sidebar.number_input("Max Depth", min_value=1, max_value=20, value=5, step=1)
            min_samples_split = st.sidebar.slider("Min Samples Split", min_value= 1, max_value=10, 
                                                        value=2, step=1)
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=10,
                                                        value=1, step=1)
            if st.button("Train Model"):
                st.divider()
                # Model Training
                # Split dataset into training and testing subsets
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    random_state=42)
                # Initialize and train tree classification model
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42)
                
                model.fit(X_train, y_train)

                # Predict on test data
                y_pred = model.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy:.2f}")


        
        elif model_type == "Logistic Regression":
            st.success("You selected Logistic Regression")
            target_col = st.selectbox("Select the target column", df.columns)
            x_features = df.drop(columns=[target_col])
            X = st.multiselect("Select features for the model", x_features.columns) # Get this list of features
            X = df[X] # Turn that list into a dataframe
            y = df[target_col]
            st.subheader(f'Here are the features:\n **{list(X.columns)}**')
            st.subheader(f'Here is the target:\n **{target_col}**')
            st.sidebar.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.write("Preview of X:")
                st.dataframe(X.head())
            with col2: 
                st.write("Preview of y:")
                st.dataframe(y.head())
        
  
else:
    st.info("Please upload a CSV file to proceed.")




