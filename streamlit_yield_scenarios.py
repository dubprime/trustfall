
import streamlit as st
import numpy as np
import json
import os

st.set_page_config(page_title="Scenario Saving - Yield Simulator", layout="wide")

st.title("💾 Yield Simulator with Scenario Saving")

# Scenario directory
SCENARIO_DIR = "scenarios"
os.makedirs(SCENARIO_DIR, exist_ok=True)

# Sidebar
st.sidebar.header("Loan Mix & Pricing")
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, 0.12, 0.005)

share_1m = st.sidebar.slider("Share of 1M Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2M Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = max(0.0, 1.0 - share_1m - share_2m)

st.sidebar.header("Capital & Risk")
cost_of_capital = st.sidebar.slider("Cost of Capital", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

# Saving and loading scenarios
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

scenario_files = [f[:-5] for f in os.listdir(SCENARIO_DIR) if f.endswith(".json")]
selected_scenario = st.sidebar.selectbox("📂 Load Scenario", [""] + scenario_files)

if selected_scenario and st.sidebar.button("🔁 Load Selected Scenario"):
    with open(f"{SCENARIO_DIR}/{selected_scenario}.json", "r") as f:
        s = json.load(f)
    st.session_state["interest_1m"] = s["interest_1m"]
    st.session_state["interest_2m"] = s["interest_2m"]
    st.session_state["interest_3m"] = s["interest_3m"]
    st.session_state["share_1m"] = s["share_1m"]
    st.session_state["share_2m"] = s["share_2m"]
    st.session_state["cost_of_capital"] = s["cost_of_capital"]
    st.session_state["default_rate"] = s["default_rate"]
    st.rerun()

st.subheader("🧾 Current Configuration")
st.json({
    "Interest Rates": {
        "1M": interest_1m,
        "2M": interest_2m,
        "3M": interest_3m
    },
    "Shares": {
        "1M": share_1m,
        "2M": share_2m,
        "3M": share_3m
    },
    "Capital Assumptions": {
        "Cost of Capital": cost_of_capital,
        "Default Rate": default_rate
    }
})
