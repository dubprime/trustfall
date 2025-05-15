
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Yield Breakdown Waterfall (Labeled)", layout="wide")

st.title("📉 Yield Breakdown Waterfall with Labels")
st.markdown("See numeric breakdown from Gross → Defaults → Cost of Capital → Net Yield.")

# Inputs
st.sidebar.header("Inputs")
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, 0.12, 0.005)

share_1m = st.sidebar.slider("Share of 1M Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2M Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = max(0, 1.0 - share_1m - share_2m)

cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

# Core calculations
shares = [share_1m, share_2m, share_3m]
rates = [interest_1m, interest_2m, interest_3m]
durations = [1, 2, 3]

weighted_duration = sum([shares[i] * durations[i] for i in range(3)])
turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * rates[i] for i in range(3)])
gross_yield = interest_per_loan * turnover
loss_from_defaults = gross_yield * default_rate
net_after_defaults = gross_yield - loss_from_defaults
net_yield = net_after_defaults - cost_of_capital

# Values and labels
x_vals = ["Gross Yield", "Defaults", "Cost of Capital", "Net Yield"]
y_vals = [gross_yield, -loss_from_defaults, -cost_of_capital, net_yield]
label_texts = [f"{v:.2%}" for v in y_vals]

fig = go.Figure(go.Waterfall(
    x=x_vals,
    y=y_vals,
    measure=["relative", "relative", "relative", "total"],
    text=label_texts,
    textposition="outside",
    connector={"line": {"color": "gray"}},
    decreasing={"marker": {"color": "indianred"}},
    increasing={"marker": {"color": "lightgreen"}},
    totals={"marker": {"color": "steelblue"}}
))

fig.update_layout(
    title="📉 Yield Breakdown Waterfall (Labeled)",
    yaxis_title="Annual Yield",
    height=500
)

st.plotly_chart(fig, use_container_width=True)
