
import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Scenario Comparison - Yield Simulator", layout="wide")
st.title("📊 Scenario Comparison — Capital Yield Simulator")

# Setup
SCENARIO_DIR = "scenarios"
os.makedirs(SCENARIO_DIR, exist_ok=True)

scenario_files = [f[:-5] for f in os.listdir(SCENARIO_DIR) if f.endswith(".json")]
if not scenario_files:
    st.warning("Save at least two scenarios to use comparison view.")
    st.stop()

# Scenario selection
col1, col2 = st.columns(2)
with col1:
    scenario_a = st.selectbox("📁 Select Scenario A", scenario_files, key="scenario_a")
with col2:
    scenario_b = st.selectbox("📁 Select Scenario B", scenario_files, key="scenario_b")

if scenario_a == scenario_b:
    st.warning("Please select two different scenarios for comparison.")
    st.stop()

# Load scenarios
def load_scenario(name):
    with open(f"{SCENARIO_DIR}/{name}.json", "r") as f:
        return json.load(f)

sA = load_scenario(scenario_a)
sB = load_scenario(scenario_b)

# Helper calc
def compute_summary(s):
    shares = [s["share_1m"], s["share_2m"], max(0, 1 - s["share_1m"] - s["share_2m"])]
    rates = [s["interest_1m"], s["interest_2m"], s["interest_3m"]]
    durations = [1, 2, 3]
    wd = sum([shares[i] * durations[i] for i in range(3)])
    turn = 12 / wd
    int_per_loan = sum([shares[i] * rates[i] for i in range(3)])
    gross = int_per_loan * turn
    loss = gross * s["default_rate"]
    after_loss = gross - loss
    net = after_loss - s["cost_of_capital"]
    return {
        "shares": shares,
        "rates": rates,
        "net_yield": net,
        "gross_yield": gross,
        "weighted_duration": wd,
        "capital_turnover": turn
    }

summary_A = compute_summary(sA)
summary_B = compute_summary(sB)

# Pie Charts
st.subheader("🥧 Loan Mix Comparison")
pc = st.columns(2)
with pc[0]:
    st.markdown(f"**Scenario A: {scenario_a}**")
    pieA = go.Figure(go.Pie(
        labels=["1M", "2M", "3M"],
        values=summary_A["shares"],
        hole=0.4
    ))
    st.plotly_chart(pieA, use_container_width=True)
with pc[1]:
    st.markdown(f"**Scenario B: {scenario_b}**")
    pieB = go.Figure(go.Pie(
        labels=["1M", "2M", "3M"],
        values=summary_B["shares"],
        hole=0.4
    ))
    st.plotly_chart(pieB, use_container_width=True)

# Bar Comparison
st.subheader("📈 Net Yield Comparison")
bar_fig = go.Figure()
bar_fig.add_trace(go.Bar(name=f"{scenario_a}", x=["Net Yield"], y=[summary_A["net_yield"]]))
bar_fig.add_trace(go.Bar(name=f"{scenario_b}", x=["Net Yield"], y=[summary_B["net_yield"]]))
bar_fig.update_layout(barmode='group', yaxis_title="Annual Net Yield", height=400)
st.plotly_chart(bar_fig, use_container_width=True)

# Table of deltas
st.subheader("🧮 Key Differences")
delta_table = pd.DataFrame({
    "Metric": [
        "Net Yield", "Gross Yield", "Weighted Duration", "Capital Turnover"
    ],
    f"{scenario_a}": [
        summary_A["net_yield"], summary_A["gross_yield"],
        summary_A["weighted_duration"], summary_A["capital_turnover"]
    ],
    f"{scenario_b}": [
        summary_B["net_yield"], summary_B["gross_yield"],
        summary_B["weighted_duration"], summary_B["capital_turnover"]
    ],
    "Delta": [
        summary_A["net_yield"] - summary_B["net_yield"],
        summary_A["gross_yield"] - summary_B["gross_yield"],
        summary_A["weighted_duration"] - summary_B["weighted_duration"],
        summary_A["capital_turnover"] - summary_B["capital_turnover"]
    ]
})
st.dataframe(delta_table.set_index("Metric").style.format("{:.2%}"), use_container_width=True)
