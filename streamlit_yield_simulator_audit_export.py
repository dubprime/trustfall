
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Yield Simulator with Audit Trail & Export", layout="wide")
st.title("📊 Capital Yield Simulator with Assumptions Summary & Export")

# Setup for scenario directory
SCENARIO_DIR = "scenarios"
os.makedirs(SCENARIO_DIR, exist_ok=True)

# Load scenario if selected
scenario_files = [f[:-5] for f in os.listdir(SCENARIO_DIR) if f.endswith(".json")]
selected_scenario = st.sidebar.selectbox("📂 Load Scenario", [""] + scenario_files)

if selected_scenario and st.sidebar.button("🔁 Load Selected Scenario"):
    with open(f"{SCENARIO_DIR}/{selected_scenario}.json", "r") as f:
        s = json.load(f)
    for key in s:
        st.session_state[key] = s[key]
    st.rerun()

# Sidebar inputs
st.sidebar.header("Loan Mix & Pricing")
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, st.session_state.get("interest_1m", 0.03), 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, st.session_state.get("interest_2m", 0.06), 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, st.session_state.get("interest_3m", 0.12), 0.005)

share_1m = st.sidebar.slider("Share of 1M Loans", 0.0, 1.0, st.session_state.get("share_1m", 0.3), 0.01)
share_2m = st.sidebar.slider("Share of 2M Loans", 0.0, 1.0 - share_1m, st.session_state.get("share_2m", 0.4), 0.01)
share_3m = max(0.0, 1.0 - share_1m - share_2m)

st.sidebar.header("Capital & Risk")
cost_of_capital = st.sidebar.slider("Cost of Capital", 0.01, 0.25, st.session_state.get("cost_of_capital", 0.08), 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, st.session_state.get("default_rate", 0.02), 0.005)

# Scenario saving
scenario_name = st.sidebar.text_input("📄 Scenario Name")
if st.sidebar.button("💾 Save Scenario") and scenario_name:
    scenario = {
        "interest_1m": interest_1m,
        "interest_2m": interest_2m,
        "interest_3m": interest_3m,
        "share_1m": share_1m,
        "share_2m": share_2m,
        "cost_of_capital": cost_of_capital,
        "default_rate": default_rate
    }
    with open(f"{SCENARIO_DIR}/{scenario_name}.json", "w") as f:
        json.dump(scenario, f)
    st.success(f"Scenario '{scenario_name}' saved.")

# Assumptions
loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
shares = [share_1m, share_2m, share_3m]
cycles_per_year = [12 / d for d in loan_durations]

# Summary header
st.subheader("🧾 Assumptions Summary")
st.write(pd.DataFrame({
    "Loan Type": ["1-Month", "2-Month", "3-Month"],
    "Share": shares,
    "Interest per Loan": interest_rates,
    "Duration (Months)": loan_durations,
    "Cycles/Year": cycles_per_year
}))

# Core calculations
weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
capital_turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
gross_yield = interest_per_loan * capital_turnover
loss_from_defaults = gross_yield * default_rate
net_after_defaults = gross_yield - loss_from_defaults
net_yield = net_after_defaults - cost_of_capital

# Audit Trail
st.subheader("🧮 Audit Trail")
audit_df = pd.DataFrame({
    "Metric": [
        "Weighted Duration",
        "Capital Turnover",
        "Interest per Loan (Blended)",
        "Gross Yield",
        "Loss from Defaults",
        "After Defaults",
        "Cost of Capital Deduction",
        "Net Yield"
    ],
    "Value": [
        weighted_duration,
        capital_turnover,
        interest_per_loan,
        gross_yield,
        loss_from_defaults,
        net_after_defaults,
        cost_of_capital,
        net_yield
    ]
})
st.table(audit_df)

# Export download
cohort_df = pd.DataFrame({
    "Loan Type": ["1-Month", "2-Month", "3-Month"],
    "Share": shares,
    "Interest per Loan": interest_rates,
    "Duration (Months)": loan_durations,
    "Cycles/Year": cycles_per_year,
    "Annualized Yield": [interest_rates[i] * cycles_per_year[i] for i in range(3)],
    "Weighted Yield Contribution": [shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)]
})

csv = cohort_df.to_csv(index=False)
st.download_button("📤 Download Cohort Table (CSV)", data=csv, file_name="cohort_yield_table.csv", mime="text/csv")
