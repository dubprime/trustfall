
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Yield Sensitivity Model", layout="wide")

st.title("📈 Capital Yield Sensitivity Explorer")
st.markdown("Adjust loan shares, interest rates, and durations to analyze portfolio vs. cohort-normalized annual yield.")

st.sidebar.header("🔧 Input Parameters")

# Main sliders
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

# Yield Calculations
weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
capital_turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
simple_annual_yield = interest_per_loan * capital_turnover
advanced_annual_yield = sum([shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)])
delta = advanced_annual_yield - simple_annual_yield

# Results
st.subheader("📊 Yield Metrics Summary")
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

# Bar chart
fig = go.Figure()
fig.add_trace(go.Bar(x=["Simple Yield"], y=[simple_annual_yield], name="Simple", marker_color="blue"))
fig.add_trace(go.Bar(x=["Advanced Yield"], y=[advanced_annual_yield], name="Advanced", marker_color="green"))
fig.update_layout(title="Annual Yield Comparison", yaxis_title="Annual Yield", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Cohort table
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

# Sensitivity analysis
st.subheader("📈 Sensitivity Analysis")

variable = st.selectbox("Sweep Variable", ["Share of 1-Month Loans", "Interest per 1M Loan", "Interest per 3M Loan"])
sweep = np.linspace(0.05, 0.95, 50)
results = []

for x in sweep:
    s1, s2, s3 = share_1m, share_2m, share_3m
    i1, i2, i3 = interest_1m, interest_2m, interest_3m

    if variable == "Share of 1-Month Loans":
        s1 = x
        s2 = min(0.9, (1 - s1) / 2)
        s3 = 1 - s1 - s2
    elif variable == "Interest per 1M Loan":
        i1 = x
    elif variable == "Interest per 3M Loan":
        i3 = x

    shares_tmp = [s1, s2, s3]
    rates_tmp = [i1, i2, i3]
    durations_tmp = [1, 2, 3]
    cycles_tmp = [12 / d for d in durations_tmp]

    dur = sum([shares_tmp[i] * durations_tmp[i] for i in range(3)])
    turnover = 12 / max(dur, 0.01)
    loan_interest = sum([shares_tmp[i] * rates_tmp[i] for i in range(3)])
    simple = loan_interest * turnover
    advanced = sum([shares_tmp[i] * rates_tmp[i] * cycles_tmp[i] for i in range(3)])

    results.append({
        "Sweep Value": x,
        "Simple Yield": simple,
        "Advanced Yield": advanced
    })

df_sens = pd.DataFrame(results)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_sens["Sweep Value"], y=df_sens["Simple Yield"], mode="lines", name="Simple Yield"))
fig2.add_trace(go.Scatter(x=df_sens["Sweep Value"], y=df_sens["Advanced Yield"], mode="lines", name="Advanced Yield"))
fig2.update_layout(title=f"Sensitivity: {variable}", xaxis_title=variable, yaxis_title="Annual Yield", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)
