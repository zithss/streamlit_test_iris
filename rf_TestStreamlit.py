import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Create a DataFrame for preview
    iris_df = pd.DataFrame(data=X, columns=feature_names)
    iris_df['target'] = y

    # Split the dataset (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    
    return iris_df, X_train, X_test, y_train, y_test, rf_classifier, accuracy, conf_matrix, class_report, feature_names

def save_model(model, filename='rf_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return filename

def load_model(filename='rf_model.pkl'):
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def main():
    st.title("Iris Dataset Classification with Random Forest")
    st.write("This app trains a Random Forest classifier on the Iris dataset, evaluates its performance, "
             "and saves/loads the model using pickle.")

    if st.button("Train Model"):
        # Train the model and get outputs
        (iris_df, X_train, X_test, y_train, y_test, model,
         accuracy, conf_matrix, class_report, feature_names) = train_model()
        
        st.subheader("Dataset Preview")
        st.write(iris_df.head())
        
        st.subheader("Model Evaluation")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(conf_matrix)
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(class_report).transpose())
        
        # Plot and display feature importances
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, palette="viridis", ax=ax)
        ax.set_title("Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
        
        # Save the trained model
        model_filename = save_model(model)
        st.success(f"Model saved as **{model_filename}**")
        
        # Load the model and verify predictions
        loaded_model = load_model(model_filename)
        y_loaded_pred = loaded_model.predict(X_test)
        loaded_accuracy = accuracy_score(y_test, y_loaded_pred)
        st.write(f"**Loaded model accuracy:** {loaded_accuracy:.2f}")

if __name__ == "__main__":
    main()
