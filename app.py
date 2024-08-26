import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler  # Ensure you're using the right scaler

# Load the model
model = load_model('ckd_classifier_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the correct feature names (replace with the actual names used during training)
feature_names = ['bp (Diastolic)', 'bp limit', 'sg', 'al', 'rbc', 'su', 'pc',
                 'pcc', 'ba', 'bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc',
                 'wbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'grf', 'stage',
                 'affected', 'age']  # Add or modify feature names as needed

# Streamlit app
st.title("Risk Factor Prediction of Chronic Kidney Disease")
st.write("Input the data to classify kidney disease:")

# Create input fields for user inputs
st.header("Input Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input data to DataFrame for processing
input_df = pd.DataFrame([input_data])

# Ensure the DataFrame has the correct columns and order
input_df = input_df[feature_names]

# Button to classify
if st.button("Classify"):
    # Scale the input data
    input_scaled = scaler.transform(input_df)  # Ensure scaler has the transform method

    # Predict the class
    prediction = model.predict(input_scaled)
    predicted_class = (prediction > 0.5).astype(int).flatten()[0]  # Adjust if needed

    # Display the result
    st.subheader("Prediction")
    st.markdown(
        f"""
    <div style="background-color: #121212; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: #4CAF50; font-family: Arial, sans-serif;">Prediction Result</h3>
        <p style="font-size: 20px; color: #ffffff; font-family: Arial, sans-serif;">The predicted class is:</p>
        <h1 style="color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold;">{"ckd" if predicted_class==0 else "Not ckd"}</h1>
    </div>
    """,
        unsafe_allow_html=True
    )
