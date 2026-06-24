import streamlit as st
import numpy as np
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    path = kagglehub.dataset_download("laotse/credit-risk-dataset")
    df = pd.read_csv(path + "/credit_risk_dataset.csv")
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)
    le = LabelEncoder()
    for col in ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]:
        df[col] = le.fit_transform(df[col])
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

st.title("🏦 Loan Default Predictor")
st.write("Applicant ki details bharo — hum batayenge loan safe hai ya nahi!")
st.divider()

age = st.slider("Umar (Age)", 18, 100, 30)
income = st.number_input("Salary (Annual Income)", min_value=0, value=50000)
home = st.selectbox("Ghar", [0,1,2,3], format_func=lambda x: ["RENT","OWN","MORTGAGE","OTHER"][x])
emp_length = st.slider("Job Experience (Years)", 0, 40, 5)
intent = st.selectbox("Loan Purpose", [0,1,2,3,4,5], format_func=lambda x: ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"][x])
grade = st.selectbox("Loan Grade", [0,1,2,3,4,5,6], format_func=lambda x: ["A","B","C","D","E","F","G"][x])
amount = st.number_input("Loan Amount", min_value=0, value=10000)
int_rate = st.slider("Interest Rate", 5.0, 25.0, 12.0)
percent_income = st.slider("Loan Percent of Income", 0.0, 1.0, 0.2)
default_hist = st.selectbox("Pehle Default Kiya?", [0,1], format_func=lambda x: ["No","Yes"][x])
cred_hist = st.slider("Credit History (Years)", 0, 30, 5)

st.divider()

if st.button("Predict"):
    input_data = np.array([[age, income, home, emp_length, intent,
                            grade, amount, int_rate, percent_income,
                            default_hist, cred_hist]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 0:
        st.success("LOW RISK - Loan Safe Hai! Default probability: " + str(round(probability*100, 1)) + "%")
    else:
        st.error("HIGH RISK - Default Ho Sakta Hai! Default probability: " + str(round(probability*100, 1)) + "%")
