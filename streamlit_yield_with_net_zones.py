
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Yield Sensitivity Model", layout="wide")

st.title("📈 Capital Yield Sensitivity Explorer")
st.markdown("Adjust loan shares, interest rates, capital costs, and defaults to analyze portfolio vs. cohort-normalized annual yield.")

st.sidebar.header("🔧 Input Parameters")

# Portfolio structure
share_1m = st.sidebar.slider("Share of 1-Month Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2-Month Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = 1.0 - share_1m - share_2m
st.sidebar.markdown(f"**Calculated 3-Month Share:** {share_3m:.2f}")

# Interest per loan
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.005, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.005, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.005, 0.25, 0.12, 0.005)

# Capital settings
st.sidebar.header("💰 Capital Assumptions")
cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

# Setup
loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
shares = [share_1m, share_2m, share_3m]
cycles_per_year = [12 / d for d in loan_durations]

# Blended portfolio metrics
weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
capital_turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])

# Gross yields
simple_gross_yield = interest_per_loan * capital_turnover
advanced_gross_yield = sum([shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)])

# Net yield (after default and capital cost)
simple_net_yield = (1 - default_rate) * simple_gross_yield - cost_of_capital
advanced_net_yield = (1 - default_rate) * advanced_gross_yield - cost_of_capital
delta_yield = advanced_net_yield - simple_net_yield

# 📊 Summary Table
st.subheader("📊 Yield Metrics Summary")
summary_df = pd.DataFrame({
    "Metric": [
        "Weighted Loan Duration (months)",
        "Capital Turnover (12 / Duration)",
        "Blended Interest per Loan",
        "Simple Gross Yield",
        "Advanced Gross Yield",
        "Net Yield (Simple - Cost - Defaults)",
        "Net Yield (Advanced - Cost - Defaults)",
        "Net Yield Delta (Adv - Sim)"
    ],
    "Value": [
        f"{weighted_duration:.2f}",
        f"{capital_turnover:.2f}",
        f"{interest_per_loan:.4%}",
        f"{simple_gross_yield:.2%}",
        f"{advanced_gross_yield:.2%}",
        f"{simple_net_yield:.2%}",
        f"{advanced_net_yield:.2%}",
        f"{delta_yield:.2%}"
    ]
})
st.dataframe(summary_df, use_container_width=True)

# 📈 Yield Comparison Plot
fig = go.Figure()
fig.add_trace(go.Bar(x=["Net Simple Yield"], y=[simple_net_yield], name="Simple", marker_color="blue"))
fig.add_trace(go.Bar(x=["Net Advanced Yield"], y=[advanced_net_yield], name="Advanced", marker_color="green"))
fig.add_hrect(y0=0.00, y1=0.05, fillcolor="red", opacity=0.2, layer="below", line_width=0)
fig.add_hrect(y0=0.05, y1=0.15, fillcolor="orange", opacity=0.2, layer="below", line_width=0)
fig.add_hrect(y0=0.15, y1=0.30, fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
fig.update_layout(title="Net Yield Comparison with Risk Zones", yaxis_title="Net Yield (Annual)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# 🔍 Cohort Table
st.subheader("🔍 Cohort Detail Breakdown")
cohort_df = pd.DataFrame({
    "Loan Type": ["1-Month", "2-Month", "3-Month"],
    "Share": shares,
    "Interest per Loan": interest_rates,
    "Cycles/Year": cycles_per_year,
    "Annualized Cohort Yield": [r * c for r, c in zip(interest_rates, cycles_per_year)],
    "Weighted Yield Contribution": [shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)]
})
st.dataframe(cohort_df, use_container_width=True)
