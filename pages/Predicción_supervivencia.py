import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import streamlit as st

# Load the trained model
filename = "models/LogisticRegressionClModel.sav"
# filename = "models/titanic_model.pkl"
model = pickle.load(open(filename, "rb"))


# Function to predict survival based on the model
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(features)
    return prediction[0]


# Encode categorical variables using label encoder
def encode_sex(sex):
    le = LabelEncoder()
    le.fit(["male", "female"])
    return le.transform([sex])[0]


def encode_embarked(embarked):
    le = LabelEncoder()
    le.fit(["C", "Q", "S"])
    return le.transform([embarked])[0]


# App interface
st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3])
Sex = st.selectbox("Sex", options=["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25, step=1)
SibSp = st.number_input(
    "Number of Siblings/Spouses Aboard (SibSp)",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)
Parch = st.number_input(
    "Number of Parents/Children Aboard (Parch)",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)
Fare = st.number_input(
    "Fare Paid (Fare)", min_value=0.0, max_value=600.0, value=10.0, step=0.1
)
Embarked = st.selectbox("Port of Embarkation", options=["C", "Q", "S"])

Sex_encoded = encode_sex(Sex)
Embarked_encoded = encode_embarked(Embarked)

if st.button("Predict"):
    prediction = predict_survival(
        Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded
    )
    if prediction == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")
