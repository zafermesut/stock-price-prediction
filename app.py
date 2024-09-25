import streamlit as st
import pandas as pd
import numpy as np
from joblib import load


model = load('model.joblib')

#features = [Open, High, Low, Adj Close, Volume]
def predict_price(open, high, low, volume):
    features = np.array([open, high, low, volume]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

st.title('Apple Stock Price Prediction')  
st.write('This is a simple web app that predicts the closing price of Apple stock.')

open = st.number_input('Open', value=0.0)
high = st.number_input('High', value=0.0)
low = st.number_input('Low', value=0.0)
volume = st.number_input('Volume', value=0.0)

if st.button('Predict'):
    prediction = predict_price(open, high, low, volume)
    st.write('The predicted closing price is:', prediction)

    

