import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from io import BytesIO
import base64
import logging
import time


@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_Bengaluru_House_Data.csv', index_col=0)
    return data

@st.cache_data
def train_model(X_train, y_train):
    column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['location']), remainder='passthrough')
    scaler = StandardScaler()
    lr = LinearRegression()
    pipe_lr = make_pipeline(column_trans, scaler, lr)
    pipe_lr.fit(X_train, y_train)

    lasso = Lasso()
    pipe_lasso = make_pipeline(column_trans, scaler, lasso)
    pipe_lasso.fit(X_train, y_train)

    ridge = Ridge()
    pipe_ridge = make_pipeline(column_trans, scaler, ridge)
    pipe_ridge.fit(X_train, y_train)

    return pipe_lr, pipe_lasso, pipe_ridge

# Function to generate downloadable Excel file
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="prediction.xlsx">Download Excel File</a>'

# Load the data
data = load_data()

X = data.drop(columns=['price'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train models
pipe_lr, pipe_lasso, pipe_ridge = train_model(X_train, y_train)


st.title("Bengaluru House Price Prediction")

# Data Exploration
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(data)

# Model Selection
model_choice = st.selectbox('Choose Model', ['Linear Regression', 'Lasso Regression', 'Ridge Regression'])


# Data Filtering
min_price = st.slider('Minimum Price', min_value=0, max_value=int(data['price'].max()), step=1)
max_price = st.slider('Maximum Price', min_value=0, max_value=int(data['price'].max()), step=1)
filtered_data = data[(data['price'] >= min_price) & (data['price'] <= max_price)]
st.write("Filtered Data", filtered_data)

# Visualizations
fig = px.scatter(data, x="total_sqft", y="price", color="location")
st.plotly_chart(fig)

# User input
location = st.selectbox('Location', data['location'].unique())
total_sqft = st.slider("Total Square Feet", int(X['total_sqft'].min()), int(X['total_sqft'].max()), step=1)
bath = st.slider("Number of Bathrooms", int(X['bath'].min()), int(X['bath'].max()), step=1)
bhk = st.slider("Number of Bedrooms", int(X['bhk'].min()), int(X['bhk'].max()), step=1)

user_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

# Button 
if st.button("Predict"):
    
    try:
        with st.spinner('Predicting...'):
            model = {'Linear Regression': pipe_lr, 'Lasso Regression': pipe_lasso, 'Ridge Regression': pipe_ridge}[model_choice]
            price_per_sqft = model.predict(user_data)[0]
            total_price = price_per_sqft

        st.success('Done!')
        st.write(f"{model_choice} Total Price: {total_price*100000} ")
    
        st.markdown(get_table_download_link(pd.DataFrame([[model_choice, total_price]], columns=["Model", "Total Price"])), unsafe_allow_html=True)
    except Exception as e:
        st.write(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

# User Feedback
with st.form("my_form"):
    feedback = st.text_area("Feedback")
    submit_button = st.form_submit_button("Submit")
