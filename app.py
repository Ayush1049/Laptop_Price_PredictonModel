import streamlit as st
import pickle
import pandas as pd
import numpy as np
#To hit api we need requests module
import requests

st.title('Laptop Price Predictor')

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

#Showing dataframe
#st.dataframe(df)
#Company selection
company = st.selectbox('Brand',df['Company'].unique())

#Type of Laptop
type = st.selectbox('Type', df['TypeName'].unique())

#RAM selection
ram = st.selectbox('RAM',df['Ram'].unique())

#weight
weight = st.selectbox('Weight', df['Weight'].unique())

#touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

#IPS(in plane switch)
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.selectbox('Screen Size(Inch)', [11.6,12,13.3,14,15.6,17])
# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

#HDD
hdd = st.selectbox('HDD(in GB)',[0,8,128,256,512,1024])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

#Prediction part
if st.button('Predict'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips == 'Yes':
        ips=1
    else:
        ips=0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2))/ screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    predicted_price = pipe.predict(query)
    predicted_price = int(np.exp(predicted_price[0]))

    st.title("The predicted price of this configuration is :-" + str(predicted_price))