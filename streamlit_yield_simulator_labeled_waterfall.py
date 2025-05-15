
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Unified Yield Simulator with Labeled Waterfall", layout="wide")

st.title("🧠 Unified Yield Simulator")
st.markdown("Explore loan pricing, duration tradeoffs, capital cost, and default risk — all in one place.")

# Sidebar
st.sidebar.header("Loan Mix & Pricing")
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, 0.12, 0.005)

share_1m = st.sidebar.slider("Share of 1M Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2M Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = max(0.0, 1.0 - share_1m - share_2m)
st.sidebar.markdown(f"Calculated 3M Share: **{share_3m:.2f}**")

st.sidebar.header("Capital & Risk")
cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

# Constants
loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
shares = [share_1m, share_2m, share_3m]

# Pie Chart
st.subheader("🥧 Loan Mix")
pie_fig = go.Figure(go.Pie(
    labels=["1M", "2M", "3M"],
    values=shares,
    hole=0.4
))
st.plotly_chart(pie_fig, use_container_width=True)

# Contour Plot
st.subheader("📈 Net Yield Contour: 1M vs 3M Share")
s1_vals = np.linspace(0.01, 0.99, 30)
s3_vals = np.linspace(0.01, 0.99, 30)
z_matrix = []
for s1 in s1_vals:
    row = []
    for s3 in s3_vals:
        s2 = 1 - s1 - s3
        if s2 < 0 or s2 > 1:
            row.append(None)
            continue
        temp_shares = [s1, s2, s3]
        duration = sum([temp_shares[i] * loan_durations[i] for i in range(3)])
        turnover = 12 / duration
        interest = sum([temp_shares[i] * interest_rates[i] for i in range(3)])
        gross_yield = interest * turnover
        net_yield = (1 - default_rate) * gross_yield - cost_of_capital
        row.append(net_yield)
    z_matrix.append(row)
contour_fig = go.Figure(data=go.Contour(
    z=z_matrix,
    x=s3_vals,
    y=s1_vals,
    colorscale='Viridis',
    colorbar_title="Net Yield"
))
contour_fig.update_layout(
    xaxis_title="3M Share",
    yaxis_title="1M Share",
    height=600
)
st.plotly_chart(contour_fig, use_container_width=True)

# Waterfall Chart (Labeled)
st.subheader("📉 Yield Breakdown Waterfall (Labeled)")
weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
turnover = 12 / weighted_duration
interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
gross_yield = interest_per_loan * turnover
loss_from_defaults = gross_yield * default_rate
net_after_defaults = gross_yield - loss_from_defaults
net_yield = net_after_defaults - cost_of_capital

x_vals = ["Gross Yield", "Defaults", "Capital Cost", "Net Yield"]
y_vals = [gross_yield, -loss_from_defaults, -cost_of_capital, net_yield]
label_texts = [f"{v:.2%}" for v in y_vals]

wf_fig = go.Figure(go.Waterfall(
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
wf_fig.update_layout(
    yaxis_title="Annual Yield",
    height=500
)
st.plotly_chart(wf_fig, use_container_width=True)

# 3D Surface
st.subheader("🧠 Yield Surface: Interest × Duration → Annual Return")
interest_grid = np.linspace(0.01, 0.25, 50)
duration_grid = np.linspace(1, 12, 50)
I, D = np.meshgrid(interest_grid, duration_grid)
Z = I * (12 / D)
surf_fig = go.Figure(data=[go.Surface(
    z=Z, x=interest_grid, y=duration_grid,
    colorscale='Viridis'
)])
surf_fig.update_layout(
    scene=dict(
        xaxis_title='Interest per Loan',
        yaxis_title='Duration (Months)',
        zaxis_title='Annualized Yield'
    ),
    height=700
)
st.plotly_chart(surf_fig, use_container_width=True)
