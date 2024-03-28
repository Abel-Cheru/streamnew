# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:43:55 2024

@author: Xabi
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
#load saved 
scaler = StandardScaler()

df = pd.read_csv('C:/Users/kabbe/Desktop/Hibuna/diabetes.csv')
x = df.drop(columns='Outcome', axis=1)
y = df['Outcome']
scaler.fit(x)
standardized_data = scaler.transform(x)
#"C:\Users\kabbe\Desktop\DiabStreamlit\Diab_cnnlstm66.keras"
diabetes_model = load_model('C:/Users/kabbe/Desktop/DiabStreamlit/Diab_cnnlstm66.keras')
#diabetes_model = pickle.load(open('C:/Users/kabbe/Desktop/DiabStreamlit/diabetes_model.sav','rb'))


#side bar

with st.sidebar:
    selected = option_menu('DIABETE Disease Pred',
                    ['Diabetes Prediction', 'GFR','About the Diab'],
                    default_index = 0)
    
    
    
if (selected == 'Diabetes Prediction' ):
    
    st.title('Diabetes Prediction with ML')
    Pregnancies = st.text_input('Preg')
    Glucose = st.text_input('Gluco')
    BloodPressure = st.text_input('Blood Pres')
    SkinThickness = st.text_input('SkinThick')
    Insulin = st.text_input('Insul')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('DiabPedi')
    Age = st.text_input('Age')
    
    # code for pred
    
    diab_dignosis = ''
    
    #creatint button for prediction
    
    if st.button('Diabetes Test Result'):
        input_data = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        #scaler.fit(input_data)
        #standardized_data = scaler.transform(input_data)
#input_data = (6,148,72,35,0,33.6,0.627,50)
#input_data = (8,183,64,0,0,23.3,0.672,32)
#input_data = (1,103,30,38,83,43.3,0.183,33)
#input_data = (4,142,86,0,0,44,0.645,22)
#change input to array
        input_data_array = np.asarray(input_data)

        #reshape
        input_data_reshaped = input_data_array.reshape(1,-1)

        #standerdizing
       # scaler.fit_transform(input_data_reshaped)
        std_data = scaler.transform(input_data_reshaped)
        #print(std_data)
        
        
        #diab_prediction = diabetes_model.predict([[Pregnancies,Glucose, BloodPressure,SkinThickness,
                                                  #Insulin,BMI,DiabetesPedigreeFunction,Age]])
        diab_prediction = diabetes_model.predict(std_data)
        
        if(diab_prediction >= 0.5):
            diab_dignosis = ' Diabetic'
        else:
            diab_dignosis = 'not Diabetic'
    st.success(diab_dignosis)