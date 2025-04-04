import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data
from preprocessing import preprocess_data
from model import train_model
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="💰")
st.title("🏦 Loan Eligibility Predictor")

# ========== Load and preprocess data ==========
try:
    raw_df = load_data()
    df = preprocess_data(raw_df)

    X = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    expected_columns = X_train.columns.tolist()

except Exception as e:
    st.error(f"❌ Failed to load or preprocess data: {e}")
    st.stop()

# ========== Clean User Input Form ==========

def user_inputs(expected_columns):
    st.subheader("Enter Applicant Information")

    # 👤 Basic info
    gender = st.radio("Gender", ["Male", "Female"])
    married = st.radio("Married", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.radio("Education", ["Graduate", "Not Graduate"])
    self_employed = st.radio("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    # 💰 Financials
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.selectbox("Loan Amount Term", options=[360.0, 120.0, 180.0, 300.0])
    credit_history = st.selectbox("Credit History", options=[1.0, 0.0])

    # 🔄 Convert to one-hot format
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Married_Yes': 1 if married == 'Yes' else 0,
        'Dependents_1.0': 1 if dependents == '1' else 0,
        'Dependents_2.0': 1 if dependents == '2' else 0,
        'Dependents_3+': 1 if dependents == '3+' else 0,
        'Education_Not_Graduate': 1 if education == 'Not Graduate' else 0,
        'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
        'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if property_area == 'Urban' else 0
    }

    # Add any other expected columns with 0s
    for col in expected_columns:
        if col not in input_data:
            input_data[col] = 0

    return pd.DataFrame([input_data])

# ========== Prediction ==========
input_df = user_inputs(expected_columns)
input_df = input_df[expected_columns]  # Reorder columns

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Denied"
        st.success(result)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ========== Evaluation ==========
try:
    st.subheader("📊 Model Accuracy & Confusion Matrix")
    acc, cm = evaluate_model(model, X_test, y_test)
    st.write(f"**Accuracy:** {acc:.2%}")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not display evaluation results: {e}")
