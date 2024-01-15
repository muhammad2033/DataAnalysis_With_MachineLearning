import streamlit as st
import numpy as np
import pickle

# Load the machine learning model from the pickle file
with open("./Logistic.pkl", "rb") as newfile:
    model = pickle.load(newfile)    

# Streamlit app
st.title("Titanic Using Logistic Regression ")
Pclass = st.number_input("Enter the Pclass value", min_value=0)
Age = st.number_input("Enter the Age", min_value=0)
Parch = st.number_input("Enter the value for Parch", min_value=0)
SibSp = st.number_input("Enter the number of siblings/spouses aboard the Titanic", min_value=0)
Fare = st.number_input("Enter the Fare", min_value=0)
Male = st.number_input("Enter the Male category", min_value=0, max_value=1, step=1)
Q = st.number_input("Enter the Embarked value1", min_value=0, max_value=1, step=1)
S = st.number_input("Enter the Embarked value", min_value=0, max_value=1, step=1)

data = np.array([Pclass, Age, SibSp, Parch, Fare, Male, Q, S])
data = data.reshape(1, -1)

if st.button("Predict"):
    pred = model.predict(data)

    if pred == 0:
        st.write("You survived!")
    else:
        st.write("You did not survive!")
