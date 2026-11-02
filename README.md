# Credit Risk Prediction System 💳

An end-to-end Machine Learning based Credit Risk Assessment and Loan Default Prediction platform with a high-performance **FastAPI** backend and an intuitive, interactive **Streamlit** user interface.

---

## 📁 Project Structure

```
Credit_Risk_Project/
│
├── app.py                     # FastAPI backend application
├── credit_risk_model.pkl      # Pre-trained ML model pipeline
├── requirements.txt           # Project dependencies
│
├── frontend/
│   └── streamlit_app.py       # Streamlit UI dashboard
│
└── README.md                  # Project documentation
```

---

## 🚀 Key Features

- **Machine Learning Inference**: Predicts customer loan default probability using pre-trained model pipeline.
- **RESTful API with FastAPI**: Clean, validated endpoints with Pydantic request models and auto-generated Swagger UI.
- **Interactive Streamlit Web Dashboard**:
  - Organized inputs for Borrower Profile, Loan Details, and Credit History.
  - Automatic Loan-to-Income percentage calculator.
  - "Quick Load Sample Profile" one-click button.
  - Visual Risk Status alerts (`LOW RISK` vs `HIGH RISK`), risk progress meters, and metric summaries.
  - Direct local model fallback when running standalone without the API.
  - Expandable JSON payload & response inspector.
- **Configurable Risk Threshold**: Policy cutoff set to **0.40 (40%)**. Applications with default probability $\ge 0.40$ are classified as **HIGH RISK**.

---

## 🛠️ Installation & Setup

### 1. Clone / Navigate to Project Directory
```powershell
cd Credit_Risk_Project
```

### 2. (Optional) Create and Activate a Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Step 1: Start the FastAPI Backend
In your terminal, run:
```powershell
uvicorn app:app --reload --port 8000
```
- API root: `http://127.0.0.1:8000/`
- Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Alternative API Docs (ReDoc): `http://127.0.0.1:8000/redoc`

### Step 2: Start the Streamlit Frontend
Open another terminal in the project directory and run:
```powershell
streamlit run frontend/streamlit_app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 📡 API Reference

### `POST /predict`
Evaluates a loan applicant and returns default probability and classification.

#### Request Body (JSON)
```json
{
  "person_age": 25,
  "person_income": 50000,
  "person_emp_length": 3,
  "loan_amnt": 10000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.2,
  "cb_person_cred_hist_length": 4,
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "cb_person_default_on_file": "N"
}
```

#### Example Response (JSON)
```json
{
  "default_probability": 0.1245,
  "risk_percentage": 12.45,
  "threshold": 0.4,
  "prediction": 0,
  "risk_status": "LOW RISK"
}
```

#### Field Descriptions:
| Field | Type | Description | Example Values |
|---|---|---|---|
| `person_age` | Integer | Age in years | `25` |
| `person_income` | Float | Annual income in USD | `50000.0` |
| `person_emp_length` | Float | Employment duration in years | `3.0` |
| `loan_amnt` | Float | Requested loan amount in USD | `10000.0` |
| `loan_int_rate` | Float | Annual interest rate (%) | `12.5` |
| `loan_percent_income` | Float | Ratio of loan amount to annual income | `0.20` |
| `cb_person_cred_hist_length` | Float | Length of credit bureau history in years | `4.0` |
| `person_home_ownership` | String | Housing situation | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `loan_intent` | String | Purpose of loan | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | String | Risk grade | `A`, `B`, `C`, `D`, `E`, `F`, `G` |
| `cb_person_default_on_file` | String | Past default record on credit file | `Y` (Yes) or `N` (No) |

---

## 🧪 Testing with cURL / PowerShell

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"person_age": 25, "person_income": 50000, "person_emp_length": 3, "loan_amnt": 10000, "loan_int_rate": 12.5, "loan_percent_income": 0.2, "cb_person_cred_hist_length": 4, "person_home_ownership": "RENT", "loan_intent": "PERSONAL", "loan_grade": "B", "cb_person_default_on_file": "N"}'
```

