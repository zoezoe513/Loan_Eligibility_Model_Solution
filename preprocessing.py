import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Drop ID
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    # Fill missing values
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    # Convert to categorical
    df['Credit_History'] = df['Credit_History'].astype('object')
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')

    # Encode categoricals
    df = pd.get_dummies(df, columns=[
        'Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'Property_Area'
    ], drop_first=True)

    # Map target
    df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

    return df
