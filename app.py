from flask import Flask, render_template, request
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
#load the model
model= pickle.load(open('gb_model.pkl','rb'))
#title for app
st.title('Health Insurance Price Prediction')
# define inputs to take onputs from user
age=st.number_input('Age',min_value=1,max_value=100,value=25)
gender=st.selectbox('Gender',('male','female'))
bmi=st.number_input('BMI',min_value=10,max_value=80,value=25)
smoker=st.selectbox('Smoker',('yes','no'))
children=st.number_input('Children',min_value=0,max_value=10,value=0)
region=st.selectbox('Region',('northeast','northwest','southeast','southwest'))
#Encoding techniques 
#smoker
Smoker= 1 if smoker=='yes' else 0
#gender
sex_female=1 if gender=='female' else 0
sex_male=1 if gender=='male' else 0
#region
region_dict={'southeast':3,'northeast':2,'northwest':1,'southwest':0}
Region=region_dict[region]
#create a dataframe
input_data=pd.DataFrame({
    'age':[age],
    'bmi':[bmi],
    'children':[children],
    'Smoker':[Smoker],
    'sex_female':[sex_female],
    'sex_male':[sex_male],
    'Region':[Region]
})

scaler=StandardScaler()
input_data[['age','bmi']]=scaler.fit_transform(input_data[['age','bmi']])

if st.button('Predict'):
  predictions=model.predict(input_data)
  output=round(np.exp(predictions[0]),2)
  st.success(f'The insurance price is: {output}')

