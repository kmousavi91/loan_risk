from preprocess import load_and_preprocess_data
from utils import save_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, scaler, features,
               model_path="models/model.joblib",
               scaler_path="models/scaler.joblib",
               columns_path="models/columns.joblib")
