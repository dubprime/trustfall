
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Unified Capital Yield Simulator", layout="wide")

st.title("🧠 Unified Capital Yield Simulator")
st.markdown("Explore loan mix, capital assumptions, and pricing sensitivities in one integrated view.")

# Sidebar inputs
st.sidebar.header("Loan Mix & Pricing")
interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, 0.03, 0.005)
interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, 0.06, 0.005)
interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, 0.12, 0.005)

share_1m = st.sidebar.slider("Share of 1M Loans", 0.0, 1.0, 0.3, 0.01)
share_2m = st.sidebar.slider("Share of 2M Loans", 0.0, 1.0 - share_1m, 0.4, 0.01)
share_3m = max(0.0, 1.0 - share_1m - share_2m)
st.sidebar.markdown(f"Calculated 3M Share: **{share_3m:.2f}**")

st.sidebar.header("Capital & Risk Assumptions")
cost_of_capital = st.sidebar.slider("Cost of Capital (Annual)", 0.01, 0.25, 0.08, 0.005)
default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, 0.02, 0.005)

# Derived constants
loan_durations = [1, 2, 3]
interest_rates = [interest_1m, interest_2m, interest_3m]
shares = [share_1m, share_2m, share_3m]

# ========== Pie Chart ==========
st.subheader("🥧 Loan Mix Pie Chart")
try:
    pie_fig = go.Figure(data=[go.Pie(
        labels=["1M", "2M", "3M"],
        values=shares,
        hole=0.4
    )])
    pie_fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering pie chart: {e}")

# ========== Contour Plot ==========
st.subheader("📈 Net Yield Contour: 1M vs 3M Share")
try:
    share_1m_range = np.linspace(0.01, 0.99, 30)
    share_3m_range = np.linspace(0.01, 0.99, 30)
    z_matrix = []
    for s1 in share_1m_range:
        row = []
        for s3 in share_3m_range:
            s2 = 1 - s1 - s3
            if s2 < 0 or s2 > 1:
                row.append(None)
                continue
            temp_shares = [s1, s2, s3]
            durations = [1, 2, 3]
            turnover = 12 / sum([temp_shares[i] * durations[i] for i in range(3)])
            int_per_loan = sum([temp_shares[i] * interest_rates[i] for i in range(3)])
            gross_yield = int_per_loan * turnover
            net_yield = (1 - default_rate) * gross_yield - cost_of_capital
            row.append(net_yield)
        z_matrix.append(row)

    contour_fig = go.Figure(data=go.Contour(
        z=z_matrix,
        x=share_3m_range,
        y=share_1m_range,
        colorscale='Viridis',
        colorbar_title="Net Yield"
    ))
    contour_fig.update_layout(
        xaxis_title="3M Loan Share",
        yaxis_title="1M Loan Share",
        height=600
    )
    st.plotly_chart(contour_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering contour plot: {e}")

# ========== Waterfall Chart ==========
st.subheader("📉 Yield Breakdown Waterfall")
try:
    weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
    turnover = 12 / weighted_duration
    interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
    gross_yield = interest_per_loan * turnover
    loss_from_defaults = gross_yield * default_rate
    after_defaults = gross_yield - loss_from_defaults
    net_yield = after_defaults - cost_of_capital

    wf_fig = go.Figure(go.Waterfall(
        x=["Gross Yield", "Defaults", "Capital Cost", "Net Yield"],
        y=[gross_yield, -loss_from_defaults, -cost_of_capital, net_yield],
        measure=["relative", "relative", "relative", "total"],
        connector={"line": {"color": "gray"}}
    ))
    wf_fig.update_layout(yaxis_title="Annual Yield", height=500)
    st.plotly_chart(wf_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering waterfall chart: {e}")

# ========== 3D Surface Plot ==========
st.subheader("🧠 3D Surface: Interest × Duration → Annualized Return")
try:
    interest_range = np.linspace(0.01, 0.25, 50)
    duration_range = np.linspace(1, 12, 50)
    I, D = np.meshgrid(interest_range, duration_range)
    annualized_yield = I * (12 / D)

    surface_fig = go.Figure(data=[go.Surface(
        z=annualized_yield,
        x=interest_range,
        y=duration_range,
        colorscale='Viridis'
    )])
    surface_fig.update_layout(
        scene=dict(
            xaxis_title='Interest per Loan',
            yaxis_title='Duration (Months)',
            zaxis_title='Annual Yield'
        ),
        height=700
    )
    st.plotly_chart(surface_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering 3D surface: {e}")
