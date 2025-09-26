import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
load_model = joblib.load("ai4i2020_rfc_M2.pkl")
model = load_model["model"]

st.title("🔧 Predictive Maintenance - Model 2 (Enhanced Features)")
st.write("Enter machine stats to predict failure risk.")

# User inputs (raw features)
type_choice = st.selectbox("Product Type", ["L", "M", "H"])
air_temp = st.number_input("Air temperature [K]", value=300.0)
process_temp = st.number_input("Process temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=100)

# Compute engineered features (must match training pipeline)
temp_diff = process_temp - air_temp
power = torque * rot_speed * (2 * np.pi / 60)   # watts
wear_torque = tool_wear * torque

# TWF risk rule
twf_risk = 1 if (200 <= tool_wear <= 240) else 0

# OSF risk rule depends on Type
osf_risk = 0
if type_choice == "L" and wear_torque > 11000:
    osf_risk = 1
elif type_choice == "M" and wear_torque > 12000:
    osf_risk = 1
elif type_choice == "H" and wear_torque > 13000:
    osf_risk = 1

# ===== Derived failure features (approximation rules) =====
TWF = 1 if tool_wear >= 220 else 0
HDF = 1 if temp_diff > 25 else 0
PWF = 1 if power > 1.5e6 else 0
OSF = osf_risk  # reuse OSF risk
RNF = 0         # keep stable, could randomize if desired


# Prepare input dataframe (must match training feature order)
data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "TWF": TWF,
    "HDF": HDF,
    "PWF": PWF,
    "OSF": OSF,
    "RNF": RNF,
    "Temp_diff": temp_diff,
    "Power [W]": power,
    "Wear_Torque": wear_torque,
    "TWF_risk": twf_risk,
    "OSF_risk": osf_risk,
    "Type_H": 1 if type_choice == "H" else 0,
    "Type_L": 1 if type_choice == "L" else 0,
    "Type_M": 1 if type_choice == "M" else 0
}])

# Predict
if st.button("Predict"):
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]  # probability of failure
    if pred == 1:
        st.error(f"⚠️ Machine Status: FAILURE RISK (p={proba:.2f})")
    else:
        st.success(f"✅ Machine Status: HEALTHY (p={proba:.2f})")
