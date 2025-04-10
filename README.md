# Loan Eligibility Predictor

This app has been built using Streamlit and deployed with Streamlit Community Cloud.

🔗 **Visit the app here**: [https://loaneligibilitymodelsolution-mvod549ftbu3ep35hs9rtf.streamlit.app/]  
 **Password** (if needed): `streamlit`

---

## Description

This app predicts whether a user is eligible for a loan using classification techniques based on the UCI Credit Approval dataset.

---

## Dataset

The model is trained on a structured dataset with features such as:
- Applicant income
- Coapplicant income
- Loan amount
- Loan term
- Credit history
- Marital status
- Education level
- Property area

---

## Technologies Used

- **Streamlit** – For building the interactive web application  
- **Scikit-learn** – For model training and evaluation  
- **Pandas & NumPy** – For data preprocessing and manipulation  
- **Matplotlib & Seaborn** – For data visualization and exploration (optional)

---

##  Model Summary

Trained using Logistic Regression. Includes preprocessing with one-hot encoding, scaling, and stratified train/test splitting.

---

## Future Enhancements

- Add support for additional datasets  
- Incorporate explainability tools like SHAP or LIME  
- Add charts to visualize user inputs and predictions

---

## Local Installation

```bash
git clone https://github.com/zoezoe513/Loan_Eligibility_Model_Solution
cd loan_eligibility_predictor
python -m venv env
source env/bin/activate  # On Windows use `env\\Scripts\\activate`
pip install -r requirements.txt
streamlit run app.py
```

---

Thanks for using **Loan Eligibility Predictor**! 🙌  
Feel free to contribute or share your feedback.
