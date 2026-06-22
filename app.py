import streamlit as st
import pickle
import numpy as np

# Model load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏦 Loan Default Predictor")
st.write("Applicant ki details bharo — hum batayenge loan safe hai ya nahi!")

st.divider()

# Input fields
age = st.slider("👤 Umar (Age)", 18, 100, 30)
income = st.number_input("💰 Salary (Annual Income)", min_value=0, value=50000)
home = st.selectbox("🏠 Ghar", [0, 1, 2, 3], format_func=lambda x: ["RENT", "OWN", "MORTGAGE", "OTHER"][x])
emp_length = st.slider("💼 Job Experience (Years)", 0, 40, 5)
intent = st.selectbox("🎯 Loan Purpose", [0,1,2,3,4,5], format_func=lambda x: ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"][x])
grade = st.selectbox("📊 Loan Grade", [0,1,2,3,4,5,6], format_func=lambda x: ["A","B","C","D","E","F","G"][x])
amount = st.number_input("💵 Loan Amount", min_value=0, value=10000)
int_rate = st.slider("📈 Interest Rate (%)", 5.0, 25.0, 12.0)
percent_income = st.slider("📉 Loan % of Income", 0.0, 1.0, 0.2)
default_hist = st.selectbox("⚠️ Pehle Default Kiya?", [0, 1], format_func=lambda x: ["No", "Yes"][x])
cred_hist = st.slider("📅 Credit History (Years)", 0, 30, 5)

st.divider()

if st.button("🔍 Predict"):
    input_data = np.array([[age, income, home, emp_length, intent,
                            grade, amount, int_rate, 0, percent_income,
                            default_hist, cred_hist]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 0:
        st.success(f"✅ LOW RISK — Loan Safe Hai! (Default probability: {round(probability*100, 1)}%)")
    else:
        st.error(f"❌ HIGH RISK — Default Ho Sakta Hai! (Default probability: {round(probability*100, 1)}%)")