import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the Model, Scaler, and Columns
# We use @st.cache_resource so it only loads once, making the app faster
@st.cache_resource
def load_assets():
    # Ensure these three files are in your GitHub repo
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl') 
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# 2. App Layout - Sidebar for Inputs
st.sidebar.header("Customer Details")

# --- Numeric Inputs ---
tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

# --- Categorical Inputs ---
# These options must match your original dataset values exactly
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# 3. Main Dashboard Area
st.title("📉 The Money Saver: Churn Risk & ROI")

# --- DASHBOARD METRICS (HARDCODED FROM YOUR TEST DATA) ---
st.markdown("### 📊 Model Performance (Test Data)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Recall (Accuracy on Risk)", value="79%", delta="High Sensitivity")
with col2:
    st.metric(label="Potential Revenue Saved", value="$18,195", delta="Test Set Only")
with col3:
    st.metric(label="Net ROI", value="$9,100", delta="+100% Return")
st.divider()

st.markdown("Adjust the threshold to balance **Risk vs. Retention Cost**.")

# --- The "Logic" Slider ---
threshold = st.slider("⚠️ Risk Threshold (Sensitivity)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.caption(f"Customers with a Churn Probability > {threshold:.2f} will be flagged as High Risk.")

# 4. Processing the Input
if st.sidebar.button("Predict Customer Risk"):
    
    # Create a dict of the inputs
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Gender': gender,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
        # NOTE: If your model used other columns, add them here!
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Generate dummies (One-Hot Encoding)
    input_df = pd.get_dummies(input_df)
    
    # Align Columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # --- NEW SCALING LOGIC (Matches your Notebook) ---
    # We only scale the specific columns you trained on
    cols_to_scale = ['MonthlyCharges', 'TotalCharges']
    
    # We copy the dataframe to avoid warnings
    input_df_ready = input_df.copy()
    
    # Scale ONLY the 2 numeric columns
    input_df_ready[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    # 5. Prediction (Use the partially scaled dataframe)
    churn_prob = model.predict_proba(input_df_ready)[0][1]
    
    # Apply the Slider Threshold
    prediction = 1 if churn_prob > threshold else 0
    
    # 6. Display Results
    st.divider()
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.subheader("Churn Probability")
        st.metric(label="Probability", value=f"{churn_prob:.2%}")
        
    with col_res2:
        st.subheader("Risk Status")
        if prediction == 1:
            st.error("🚨 HIGH RISK")
        else:
            st.success("✅ SAFE")

    # 7. The "Money Saver" ROI Logic
    st.divider()
    st.subheader("💰 Single Customer ROI Analysis")
    
    if prediction == 1:
        st.write(f"This customer generates **${monthly_charges}** per month.")
        
        # Scenario: We offer a 20% discount to keep them
        discount_cost = monthly_charges * 0.20
        retained_value = monthly_charges * 0.80 * 12 # Value over next year if saved
        
        st.warning(f"Strategy: Offer 20% Discount (Cost: ${discount_cost:.2f}/mo)")
        st.success(f"📉 Potential Revenue Saved (1 Year): **${retained_value:.2f}**")
        st.info("💡 Recommendation: **INTERVENE NOW**")
        
    else:
        st.write("Customer is low risk. No intervention needed.")
        st.write(f"Projected Revenue (1 Year): **${monthly_charges * 12:.2f}**")

else:
    st.info("👈 Use the sidebar to enter customer details and click 'Predict'.")