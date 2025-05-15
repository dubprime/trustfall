
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Contour Yield Visualizer", layout="wide")

st.title("📈 Portfolio Yield Contour Explorer")
st.markdown("Explore how different loan mix allocations affect portfolio net yield given your capital and pricing assumptions.")

st.sidebar.header("🔧 Assumptions")

# Interest per loan
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, 0.12, 0.005)

cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
cycles_per_year = [12 / d for d in loan_durations]

# Sweep over 1M and 3M shares
share_1m_range = np.linspace(0.01, 0.99, 30)
share_3m_range = np.linspace(0.01, 0.99, 30)

z_matrix = []

for s1 in share_1m_range:
    z_row = []
    for s3 in share_3m_range:
        s2 = 1 - s1 - s3
        if s2 < 0 or s2 > 1:
            z_row.append(None)
            continue

        shares = [s1, s2, s3]
        duration = sum([shares[i] * loan_durations[i] for i in range(3)])
        turnover = 12 / duration
        interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
        gross_yield = interest_per_loan * turnover
        net_yield = (1 - default_rate) * gross_yield - cost_of_capital
        z_row.append(net_yield)

    z_matrix.append(z_row)

fig = go.Figure(data=go.Contour(
    z=z_matrix,
    x=share_3m_range,
    y=share_1m_range,
    colorscale='Viridis',
    contours_coloring='heatmap',
    colorbar_title="Net Yield"
))

fig.update_layout(
    title="📊 Net Yield Contour by 1M vs 3M Loan Share",
    xaxis_title="3-Month Loan Share",
    yaxis_title="1-Month Loan Share",
    height=700
)

st.plotly_chart(fig, use_container_width=True)
