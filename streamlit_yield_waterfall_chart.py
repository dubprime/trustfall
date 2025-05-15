
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Yield Breakdown Waterfall", layout="wide")

st.title("📉 Yield Breakdown Waterfall Chart")
st.markdown("Visualize the sequential impact of capital cost and defaults on portfolio yield.")

st.sidebar.header("🔧 Inputs")

# Inputs
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, 0.12, 0.005)

share_1m = st.sidebar.slider("Share of 1-Month Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2-Month Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = max(0, 1.0 - share_1m - share_2m)

cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
shares = [share_1m, share_2m, share_3m]

# Core calculations
weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
capital_turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
gross_yield = interest_per_loan * capital_turnover
loss_due_to_defaults = gross_yield * default_rate
net_after_defaults = gross_yield - loss_due_to_defaults
net_yield = net_after_defaults - cost_of_capital

# Waterfall data
x_labels = ["Gross Yield", "Defaults", "Cost of Capital", "Net Yield"]
y_values = [gross_yield, -loss_due_to_defaults, -cost_of_capital, net_yield]

fig = go.Figure(go.Waterfall(
    name="Yield",
    orientation="v",
    measure=["relative", "relative", "relative", "total"],
    x=x_labels,
    y=y_values,
    text=[f"{v:.2%}" for v in y_values],
    connector={"line":{"color":"rgb(63, 63, 63)"}},
    decreasing={"marker":{"color":"indianred"}},
    increasing={"marker":{"color":"lightgreen"}},
    totals={"marker":{"color":"darkblue"}}
))

fig.update_layout(
    title="📉 Yield Breakdown: From Gross to Net",
    yaxis_title="Annual Yield",
    waterfallgroupgap=0.5,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
