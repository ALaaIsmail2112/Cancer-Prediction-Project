import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import os

def load_Model():
    with open('model.pkl', 'rb') as model_pkl:
        model = pickle.load(model_pkl)
    return model

model = load_Model()

def show_Predict_page():
    st.title("Cancer Prediction App")

    # List of input features
    features = [
        'radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean', 'radius_se', 'perimeter_se',
        'area_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst'
    ]
    
    # Input fields for user data
    inputs = {}
    for feature in features:
        inputs[feature] = st.number_input(f"{feature.replace('_', ' ')}")
    
    if st.button("Predict"):
        # Prepare input data for prediction
        input_data = np.array([inputs[feature] for feature in features])
        input_data = input_data.reshape(1, -1)

        # Set up the Gemini API
        os.environ["GENAI_API_KEY"] = "AIzaSyC5RqfKHABR9k9DCQHhrcPlf_VstJjhXNA"
        genai.configure(api_key=os.environ["GENAI_API_KEY"])
        
        # Make the prediction
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            # If the user is diagnosed with cancer, generate advice using the Gemini API
            model_gemini = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model_gemini.generate_content(
                "Give concise advice to a person diagnosed with cancer, offering support and guidance."
            )
            
            # Format and truncate the Gemini advice
            advice = response.text
            advice_lines = advice.split('. ')
            condensed_advice = '. '.join(advice_lines[:3])  # Get the first 3 sentences

            # Display the Gemini advice
            st.write("You have been diagnosed with cancer.")
            st.markdown("### Here is some advice:")
            st.markdown(f"> {condensed_advice}")
        else:
            # If the user is not diagnosed with cancer
            st.write("Good news, you don't have cancer!")
