import joblib

def save_model(model, scaler, columns, model_path, scaler_path, columns_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(columns, columns_path)

def load_model(model_path, scaler_path, columns_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)
    return model, scaler, columns
