
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Loan Yield Optimizer", layout="wide")

st.title("📈 Capital Yield Sensitivity Explorer with Optimization")
st.markdown("Adjust assumptions or let the optimizer find the portfolio mix that maximizes net yield.")

st.sidebar.header("🔧 Interest Rate Inputs")

# Interest per loan
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.005, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.005, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.005, 0.25, 0.12, 0.005)

st.sidebar.header("💰 Capital Assumptions")
cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
cycles_per_year = [12 / d for d in loan_durations]

# Optimization function
def net_yield(shares):
    if np.any(np.array(shares) < 0) or np.sum(shares) > 1.0:
        return -1e6  # penalize invalid allocations
    s1, s2, s3 = shares
    shares = [s1, s2, s3]
    dur = sum([shares[i] * loan_durations[i] for i in range(3)])
    turnover = 12 / max(dur, 0.01)
    blended_interest = sum([shares[i] * interest_rates[i] for i in range(3)])
    simple_yield = blended_interest * turnover
    net = (1 - default_rate) * simple_yield - cost_of_capital
    return -net  # minimize negative of yield

# Run optimizer
result = minimize(
    net_yield,
    x0=[0.3, 0.4, 0.3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints=[{"type": "eq", "fun": lambda x: sum(x) - 1}]
)

opt_shares = result.x
opt_net_yield = -result.fun

# Display results
st.subheader("📈 Optimized Portfolio Allocation")
opt_df = pd.DataFrame({
    "Loan Term": ["1-Month", "2-Month", "3-Month"],
    "Optimal Share": [round(x, 4) for x in opt_shares],
    "Interest per Loan": interest_rates,
    "Cycles per Year": cycles_per_year,
    "Annualized Yield": [r * c for r, c in zip(interest_rates, cycles_per_year)],
    "Weighted Contribution": [opt_shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)]
})

dur_opt = sum([opt_shares[i] * loan_durations[i] for i in range(3)])
turn_opt = 12 / dur_opt
int_opt = sum([opt_shares[i] * interest_rates[i] for i in range(3)])
gross_opt = int_opt * turn_opt
net_opt = (1 - default_rate) * gross_opt - cost_of_capital

summary_df = pd.DataFrame({
    "Metric": [
        "Weighted Duration",
        "Capital Turnover",
        "Interest per Loan",
        "Gross Yield",
        "Net Yield (Optimized)"
    ],
    "Value": [
        f"{dur_opt:.2f}",
        f"{turn_opt:.2f}",
        f"{int_opt:.4%}",
        f"{gross_opt:.2%}",
        f"{net_opt:.2%}"
    ]
})

st.dataframe(summary_df, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Pie(labels=["1M", "2M", "3M"], values=opt_shares, hole=0.4))
fig.update_layout(title="Optimized Loan Mix")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Cohort Detail (Optimized)")
st.dataframe(opt_df, use_container_width=True)
