
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Yield Sensitivity Model", layout="wide")

st.title("📈 Capital Yield Sensitivity Explorer")
st.markdown("Adjust loan shares, interest rates, and durations to analyze portfolio vs. cohort-normalized annual yield.")

st.sidebar.header("🔧 Input Parameters")

# Inputs
share_1m = st.sidebar.slider("Share of 1-Month Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2-Month Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = 1.0 - share_1m - share_2m
st.sidebar.markdown(f"**Calculated 3-Month Share:** {share_3m:.2f}")

st.sidebar.markdown("---")

interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.005, 0.10, 0.025, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.005, 0.15, 0.055, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.005, 0.25, 0.12, 0.005)

loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
shares = [share_1m, share_2m, share_3m]
cycles_per_year = [12 / d for d in loan_durations]

# Derived values
weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
capital_turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
simple_annual_yield = interest_per_loan * capital_turnover
advanced_annual_yield = sum([shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)])
delta = advanced_annual_yield - simple_annual_yield

# Results
st.subheader("📊 Yield Metrics Summary")
st.markdown("Compare the two modeling approaches side-by-side:")

summary_df = pd.DataFrame({
    "Metric": [
        "Weighted Loan Duration (months)",
        "Capital Turnover (12 / Duration)",
        "Blended Interest per Loan",
        "Simple Annual Yield (Portfolio-Based)",
        "Advanced Annual Yield (Cohort-Normalized)",
        "Delta (Advanced - Simple)"
    ],
    "Value": [
        f"{weighted_duration:.2f}",
        f"{capital_turnover:.2f}",
        f"{interest_per_loan:.4%}",
        f"{simple_annual_yield:.2%}",
        f"{advanced_annual_yield:.2%}",
        f"{delta:.2%}"
    ]
})

st.dataframe(summary_df, use_container_width=True)

# Chart
fig = go.Figure()
fig.add_trace(go.Bar(x=["Simple Yield"], y=[simple_annual_yield], name="Simple", marker_color="blue"))
fig.add_trace(go.Bar(x=["Advanced Yield"], y=[advanced_annual_yield], name="Advanced", marker_color="green"))
fig.update_layout(
    title="Annual Yield Comparison",
    yaxis_title="Annual Yield",
    template="plotly_white",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Table of components
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
