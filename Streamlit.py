# Streamlit App for Machine Learning Model Deployment
import streamlit as st
import numpy as np

# Placeholder function to simulate model prediction
def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Replace this with actual model prediction logic
    # For now, it randomly predicts one of the Iris species
    species = np.random.choice(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    return species

def main():
    st.title("Machine Learning Model Deployment")

    # Input fields for features
    st.header("Input Features")
    sepal_length = st.number_input("sepal_length", value=3.28)
    sepal_width = st.number_input("sepal_width", value=4.72)
    petal_length = st.number_input("petal_length", value=3.05)
    petal_width = st.number_input("petal_width", value=2.12)

    # Make Prediction button
    if st.button("Make Prediction"):
        # Prepare input data
        input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        # Get prediction
        prediction = predict(sepal_length, sepal_width, petal_length, petal_width)
        
        # Display prediction
        st.success(f"The prediction is: {prediction}")

if __name__ == '__main__':
    main()
