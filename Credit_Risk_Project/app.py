import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Step 1: Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML Based Credit Risk Assessment System",
    version="1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 2: Load trained model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")


# Step 3: Define input schema
class CustomerData(BaseModel):
    person_age: int = Field(..., gt=0, description="Age of the customer in years")
    person_income: float = Field(..., gt=0, description="Annual income of the customer")
    person_emp_length: float = Field(..., ge=0, description="Employment length in years")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., gt=0, description="Interest rate on loan")
    loan_percent_income: float = Field(..., gt=0, description="Loan amount as a percentage of annual income (e.g., 0.20 for 20%)")
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Length of credit history in years")
    person_home_ownership: str = Field(..., description="Home ownership status: RENT, OWN, MORTGAGE, OTHER")
    loan_intent: str = Field(..., description="Loan intent: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Loan grade: A, B, C, D, E, F, G")
    cb_person_default_on_file: str = Field(..., description="Historical default on file: Y or N")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 25,
                "person_income": 50000.0,
                "person_emp_length": 3.0,
                "loan_amnt": 10000.0,
                "loan_int_rate": 12.5,
                "loan_percent_income": 0.2,
                "cb_person_cred_hist_length": 4.0,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "cb_person_default_on_file": "N"
            }
        }
    }


# Step 4: Home & Health endpoints
@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }


# Step 5: Prediction endpoint
@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please verify credit_risk_model.pkl exists."
        )

    try:
        # Convert input into DataFrame
        input_data = pd.DataFrame([data.model_dump()])

        # Predict probability for class 1 (Default / High Risk)
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        # Classification threshold
        threshold = 0.40
        prediction = int(probability >= threshold)

        return {
            "default_probability": round(probability, 4),
            "risk_percentage": round(probability * 100, 2),
            "threshold": threshold,
            "prediction": prediction,
            "risk_status": "HIGH RISK" if prediction == 1 else "LOW RISK"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

