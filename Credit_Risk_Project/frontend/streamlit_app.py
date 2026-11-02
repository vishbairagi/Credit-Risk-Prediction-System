import os
import streamlit as st
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 9999px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 9999px;
        display: inline-block;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Helper: Direct Model Fallback (in case API is offline)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_local_model():
    model_paths = [
        os.path.join(os.path.dirname(__file__), "..", "credit_risk_model.pkl"),
        os.path.join(os.getcwd(), "credit_risk_model.pkl"),
        "credit_risk_model.pkl"
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                import joblib
                return joblib.load(p)
            except Exception:
                pass
    return None

# -----------------------------------------------------------------------------
# Sidebar: Settings & Preloaded Samples
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-cards.png", width=64)
    st.title("Settings & Tools")

    api_url = st.text_input(
        "FastAPI Backend URL",
        value="http://127.0.0.1:8000",
        help="Address of the running FastAPI server"
    )

    # API Connection Check
    api_online = False
    try:
        res = requests.get(f"{api_url.rstrip('/')}/", timeout=1.5)
        if res.status_code == 200:
            api_online = True
            st.success(" Backend API: Online", icon="🟢")
        else:
            st.warning(" Backend API: Responding with error", icon="🟡")
    except Exception:
        st.info(" Backend API: Offline (Direct model fallback available)", icon="ℹ️")

    st.markdown("---")
    st.subheader("⚡ Quick Load Preset")
    if st.button("Load Sample Profile", use_container_width=True):
        st.session_state["age"] = 25
        st.session_state["income"] = 50000.0
        st.session_state["emp_length"] = 3.0
        st.session_state["loan_amnt"] = 10000.0
        st.session_state["int_rate"] = 12.5
        st.session_state["cred_hist"] = 4.0
        st.session_state["home_ownership"] = "RENT"
        st.session_state["intent"] = "PERSONAL"
        st.session_state["grade"] = "B"
        st.session_state["default_on_file"] = "N"
        st.rerun()

    st.markdown("---")
    st.markdown("### 📌 About System")
    st.caption(
        "This system uses a machine learning classifier to evaluate "
        "loan application risk by computing default probability. "
        "Default classification threshold is set to **0.40 (40%)**."
    )

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Credit Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluate borrower creditworthiness and loan default probability in real-time.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
with st.form("credit_risk_form"):
    col1, col2, col3 = st.columns([1.1, 1.1, 1.1])

    with col1:
        st.markdown("#### 👤 Borrower Information")
        person_age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=st.session_state.get("age", 25),
            step=1
        )
        person_income = st.number_input(
            "Annual Income ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=float(st.session_state.get("income", 50000.0)),
            step=5000.0,
            format="%.2f"
        )
        person_emp_length = st.number_input(
            "Employment Length (years)",
            min_value=0.0,
            max_value=60.0,
            value=float(st.session_state.get("emp_length", 3.0)),
            step=0.5
        )
        home_ownership_options = ["RENT", "OWN", "MORTGAGE", "OTHER"]
        default_home_idx = home_ownership_options.index(
            st.session_state.get("home_ownership", "RENT")
        ) if st.session_state.get("home_ownership", "RENT") in home_ownership_options else 0
        person_home_ownership = st.selectbox(
            "Home Ownership Status",
            options=home_ownership_options,
            index=default_home_idx
        )

    with col2:
        st.markdown("#### 💳 Loan Details")
        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=500.0,
            max_value=1000000.0,
            value=float(st.session_state.get("loan_amnt", 10000.0)),
            step=500.0,
            format="%.2f"
        )
        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.1,
            max_value=40.0,
            value=float(st.session_state.get("int_rate", 12.5)),
            step=0.1,
            format="%.2f"
        )
        intent_options = [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
        default_intent_idx = intent_options.index(
            st.session_state.get("intent", "PERSONAL")
        ) if st.session_state.get("intent", "PERSONAL") in intent_options else 0
        loan_intent = st.selectbox(
            "Loan Purpose / Intent",
            options=intent_options,
            index=default_intent_idx
        )
        grade_options = ["A", "B", "C", "D", "E", "F", "G"]
        default_grade_idx = grade_options.index(
            st.session_state.get("grade", "B")
        ) if st.session_state.get("grade", "B") in grade_options else 1
        loan_grade = st.selectbox(
            "Loan Grade",
            options=grade_options,
            index=default_grade_idx
        )

    with col3:
        st.markdown("#### 📜 Credit History & Metrics")
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (years)",
            min_value=0.0,
            max_value=60.0,
            value=float(st.session_state.get("cred_hist", 4.0)),
            step=1.0
        )
        default_options = ["N", "Y"]
        default_def_idx = default_options.index(
            st.session_state.get("default_on_file", "N")
        ) if st.session_state.get("default_on_file", "N") in default_options else 0
        cb_person_default_on_file = st.selectbox(
            "Prior Historical Default on File?",
            options=default_options,
            index=default_def_idx,
            help="Has the person previously defaulted on a loan? (Y = Yes, N = No)"
        )

        # Computed loan percent income
        computed_pct = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
        st.markdown(f"**Calculated Loan % of Income:** `{computed_pct * 100:.2f}%` (`{computed_pct}`)")
        loan_percent_income = st.number_input(
            "Loan % of Income (Fraction)",
            min_value=0.01,
            max_value=2.0,
            value=float(computed_pct if 0.01 <= computed_pct <= 2.0 else 0.20),
            step=0.01,
            format="%.4f",
            help="Calculated as Loan Amount ÷ Annual Income. Adjust if needed."
        )

    submit_button = st.form_submit_button("🔍 Evaluate Credit Risk", use_container_width=True, type="primary")

# -----------------------------------------------------------------------------
# Prediction Logic & Output Display
# -----------------------------------------------------------------------------
if submit_button:
    payload = {
        "person_age": int(person_age),
        "person_income": float(person_income),
        "person_emp_length": float(person_emp_length),
        "loan_amnt": float(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file
    }

    result = None
    source = None
    error_message = None

    # Step 1: Try FastAPI backend
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json=payload,
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            source = "FastAPI Backend"
        else:
            error_message = f"API returned status {response.status_code}: {response.text}"
    except Exception as e:
        error_message = f"Unable to reach API at {api_url}: {e}"

    # Step 2: Fallback to direct model if API is offline
    if result is None:
        local_model = load_local_model()
        if local_model is not None:
            try:
                df_input = pd.DataFrame([payload])
                prob = float(local_model.predict_proba(df_input)[0][1])
                thresh = 0.40
                pred = int(prob >= thresh)
                result = {
                    "default_probability": round(prob, 4),
                    "risk_percentage": round(prob * 100, 2),
                    "threshold": thresh,
                    "prediction": pred,
                    "risk_status": "HIGH RISK" if pred == 1 else "LOW RISK"
                }
                source = "Direct Local Model (Fallback)"
            except Exception as model_err:
                st.error(f"Failed to run local model fallback: {model_err}")
        else:
            st.error(f"Prediction failed: {error_message}")

    # Display results
    if result:
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")

        is_high_risk = result["prediction"] == 1
        probability = result["default_probability"]
        risk_pct = result["risk_percentage"]
        threshold = result.get("threshold", 0.40)

        # Status Banner
        if is_high_risk:
            st.error(
                f"### ⚠️ Status: HIGH RISK (Likely to Default)\n"
                f"Default Probability (**{risk_pct:.2f}%**) exceeds policy threshold (**{threshold * 100:.1f}%**). "
                f"Loan approval is **Not Recommended** or requires secondary risk review & collateral.",
                icon="🚫"
            )
        else:
            st.success(
                f"### ✅ Status: LOW RISK (Likely to Repay)\n"
                f"Default Probability (**{risk_pct:.2f}%**) is within policy threshold (**{threshold * 100:.1f}%**). "
                f"Loan application meets primary credit risk standards.",
                icon="🎉"
            )

        # Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Default Probability", f"{risk_pct:.2f}%")
        with m_col2:
            st.metric("Decision Threshold", f"{threshold * 100:.1f}%")
        with m_col3:
            st.metric("Risk Classification", result["risk_status"])
        with m_col4:
            st.metric("Evaluation Source", source)

        # Visual Probability Bar
        st.markdown("##### Default Risk Meter")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Risk score: {probability:.4f} / 1.0000 (Cutoff: {threshold})")

        # Summary Breakdown
        with st.expander("🔍 Risk Profile Breakdown & Details", expanded=True):
            r1, r2 = st.columns(2)
            with r1:
                st.write("**Key Credit Indicators:**")
                st.write(f"- **Loan to Income Ratio:** {loan_percent_income * 100:.1f}% {'(⚠️ High)' if loan_percent_income > 0.35 else '(✅ Healthy)'}")
                st.write(f"- **Prior Default Record:** {'⚠️ Yes (Higher Risk)' if cb_person_default_on_file == 'Y' else '✅ No'}")
                st.write(f"- **Loan Grade:** {loan_grade} {'(Subprime tier)' if loan_grade in ['D', 'E', 'F', 'G'] else '(Prime tier)'}")
                st.write(f"- **Interest Rate:** {loan_int_rate:.2f}%")
            with r2:
                st.write("**Raw Payload & Response:**")
                st.json({"request": payload, "response": result})

