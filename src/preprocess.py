import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
    df = pd.read_csv(url)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Map target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # One-hot encode categorical columns
    categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    # Split features and target
    X = df.drop(columns=['Loan_Status', 'Loan_ID'])
    y = df['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
