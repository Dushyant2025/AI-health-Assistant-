import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

def newfea():
    # 🔹 Load environment variables
    GEMINI_API_KEY = "AIzaSyAfXomua6OD94ntm9K-bHv5ZSRUrBp1JnQ"

    # 🔹 Initialize Gemini LLM with LangChain
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

    # 🔹 Load the trained model and tokenizer
    model = tf.keras.models.load_model('model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # 🔹 Load disease dataset
    df = pd.read_csv('disease_dataset.csv')
    diseases = df['Sub-Disease'].unique()

    # Set max length based on your model's input requirements
    max_length = 8  
    label_encoder = LabelEncoder()
    df['Sub-Disease'] = label_encoder.fit_transform(df['Sub-Disease'])

    # 🔹 Streamlit App UI
    st.title("🩺 Symtoms Analyser")
    st.write("Enter your symptoms separated by commas (e.g., fever, cough, headache)")

    unique_symptoms = set()
    for symptoms in df['Symptoms']:
        symptom_list = symptoms.split(', ')
        unique_symptoms.update(symptom_list)

    unique_symptoms = sorted(list(unique_symptoms))
    symptoms_input = st.multiselect("Choose Symptoms:", unique_symptoms)

    # 🔹 Prediction Logic
    def predict_disease(symptoms):
        symptoms_seq = tokenizer.texts_to_sequences([symptoms])
        symptoms_padded = pad_sequences(symptoms_seq, maxlen=max_length, padding='post')
        prediction = model.predict(symptoms_padded)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    

    # 🔹 LangChain Prompt Templates
    treatment_prompt = PromptTemplate(
        input_variables=["disease"],
        template="""You are an expert doctor. Provide a personalized treatment plan for {disease}.
        Include:
        1. Possible causes
        2. Common Test 
        3. Home remedies
        4. When to see a doctor
        5. Recommended lifestyle changes
        """
    )

    hospital_prompt = PromptTemplate(
        input_variables=["disease"],
        template="""You are a healthcare assistant. Suggest top 5 best hospitals for treating {disease} in only indore, Madhya pradesh.
        Include:
        1. Recommended hospitals
        2. Hospital specialties
        3. Location details
        4. Contact information
        """
    )

    # 🔹 Get Disease Information using Gemini LLM
    @st.cache_data
    def get_disease_treatment(disease_name):
        response = llm.invoke(treatment_prompt.format(disease=disease_name))
        return response.content

    @st.cache_data
    def get_hospitals_for_disease(disease_name):
        response = llm.invoke(hospital_prompt.format(disease=disease_name))
        return response.content

    # 🔹 Predict Disease and Display Information
    if st.button("Predict Disease"):
        if not symptoms_input:
            st.warning("Please select some symptoms!")
        else:
            result = predict_disease(', '.join(symptoms_input))  
            st.success(f"🚑 Predicted Disease: **{result}**")

            with st.spinner("Fetching disease treatment details..."):
                disease_treatment = get_disease_treatment(result)
                st.write(f"### 🩺 Treatment Plan for {result}:")
                st.write(disease_treatment)
            
            with st.spinner("Finding hospitals..."):
                hospital_suggestions = get_hospitals_for_disease(result)
                st.write(f"### 🏥 Suggested Hospitals for {result}:")
                st.write(hospital_suggestions)


