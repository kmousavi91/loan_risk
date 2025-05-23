# 🌐 Loan Approval Prediction API

A complete machine learning project to predict whether a loan application is likely to be approved. Designed for real-world applicability in banking and fintech, with a REST API built for scalable integration.

## 📖 Overview

This project builds and serves a loan approval model using:

* ✅ A clean dataset of real-world loan applications
* ✅ Preprocessing with encoding and scaling
* ✅ Random Forest Classifier for prediction
* ✅ FastAPI for serving predictions via REST
* ✅ Modular structure for reuse and deployment

---

## 📁 Project Structure

```
loan_risk_project/
├── app/
│   └── app.py               # FastAPI service
├── src/
│   ├── preprocess.py        # Data loading and preprocessing
│   ├── train.py             # Model training & evaluation
│   ├── utils.py             # Model save/load helpers
│   └── __init__.py
├── models/                    # Saved model, scaler, columns
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

* Loads the dataset
* Preprocesses it (dropna, encode, scale)
* Trains a Random Forest Classifier
* Saves the model, scaler, and column order to `models/`

### 3. Run the API

```bash
uvicorn app.app:app --reload
```

* Go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Predict

Send a JSON payload with the `features` list matching the model’s expected input:

```json
{
  "features": [4583, 1508, 128, 360, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
}
```

This will return:

```json
{ "approved": false }
```

```json
{
  "features": [6000, 2000, 100, 360, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
}
```

This will return:

```json
{ "approved": true }
```

---

## 📊 Model

* **Algorithm**: Random Forest Classifier
* **Target**: `Loan_Status` (Approved or Not Approved)
* **Features**: Personal, financial, and property-related fields
* **Preprocessing**:

  * One-hot encoding of categorical features
  * Standard scaling of numerical ones

---

## 🚀 Tech Stack

* Python 3.10+
* scikit-learn
* pandas & numpy
* FastAPI
* Uvicorn
* Joblib

---

## 🔹 Use Cases

* Loan Pre-approval Automation
* Risk Evaluation for Lending
* Fintech Credit Decision APIs

---

## 👨‍💻 Author

**Mohammad Mousavi**
Data Scientist | ML Engineer
[LinkedIn](https://www.linkedin.com/in/mohammad-mousavi-895763113/) | [GitHub](https://github.com/kmousavi91)

---

## 📄 License

MIT License

