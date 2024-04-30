# ---------------------------------------------------------

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_extras.add_vertical_space import add_vertical_space
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# -----------------------------------------------------------------------
# Load the trained model
with open('water_quality_model.pkl', 'rb') as file:
    model = pickle.load(file)
scaler = StandardScaler()
with st.sidebar:
    st.title('''       😎 KP Prediction App 🤖        ''')
    st.markdown("------------------")
    option = "About"
    st.markdown(f"<h3 style='text-align: center;'>{option}</h3>",unsafe_allow_html=True)
    st.markdown('''
     Kundeshwar V. Pundalik 
    - [Source](<https://github.com/kundeshwar/>)
    
    💡 Note: Add your data here!
    ''')
    add_vertical_space(4)
    st.write('Made with ❤️ by Kundeshwar Pundalik 😍')
#-----------------------------------------------------------------------
# Function to predict water quality index
def predict_water_quality(inputs):
    # Preprocess the input data
    inputs_scaled = scaler.fit_transform(inputs)
    # Predict the water quality index
    prediction = model.predict(inputs_scaled)
    return prediction[0][0]

# Streamlit app
st.title('Water Quality Prediction')

# Input fields
temperature = st.number_input('Temperature (⁰C)', value=25)
conductivity = st.number_input('Conductivity (μmhos/cm)', value=200)
bod = st.number_input('BOD (mg/L)', value=5)
nitrate_n = st.number_input('Nitrate N (mg/L)', value=2)
faecal_coliform = st.number_input('Faecal Coliform (MPN/100ml)', value=100)
total_coliform = st.number_input('Total Coliform (MPN/100ml)', value=150)
total_dissolved_solids = st.number_input('Total Dissolved Solids (mg/L)', value=250)
fluoride = st.number_input('Fluoride (mg/L)', value=0.5)
arsenic = st.number_input('Arsenic (mg/L)', value=0.1)
ph = st.number_input('pH', value=7.2)

# Predict button
if st.button('Predict Water Quality Index'):
    # Gather input data
    inputs = [temperature, conductivity, bod, nitrate_n, faecal_coliform, total_coliform,
              total_dissolved_solids, fluoride, arsenic, ph]
    # Predict water quality index
    prediction = predict_water_quality(inputs)
    # Display the prediction
    st.write(f'Predicted Water Quality Index: {prediction:.2f}')
