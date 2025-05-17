# NOTE: This script requires Streamlit. Run with `streamlit run this_file.py`

try:
    import streamlit as st
    from streamlit.components.v1 import html
except ModuleNotFoundError:
    st = None
    import warnings
    warnings.warn("Streamlit is not installed. Please install it using `pip install streamlit` to run this dashboard.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if st:
    # Inject CSS to override max-width limitation
    st.markdown("""
        <style>
            .main .block-container {
                max-width: 100% !important;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- App title ---
    st.title("Loan Portfolio Simulator with Defaults, Reinvestment & Investor Metrics")

    # --- Sidebar Inputs ---
    st.sidebar.header("Loan Allocation Settings")
    alloc_1 = st.sidebar.slider("1-Month Loan Allocation (%)", 0, 100, 50, key='alloc_1')
    alloc_2 = st.sidebar.slider("2-Month Loan Allocation (%)", 0, 100 - alloc_1, 25, key='alloc_2')
    alloc_3 = 100 - alloc_1 - alloc_2
    st.sidebar.markdown(f"**3-Month Loan Allocation (%):** {alloc_3}")

    st.sidebar.header("Portfolio Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, key='initial_capital')
    monthly_interest = st.sidebar.number_input("Monthly Simple Interest Rate (%)", value=3.0, key='monthly_interest')
    num_months = st.sidebar.slider("Simulation Duration (Months)", 1, 60, 12, key='num_months')

    st.sidebar.header("Default Rates (Per Bucket at Maturity)")
    default_1 = st.sidebar.slider("1-Month Loan Default Rate (%)", 0.0, 400.0, 10.0, step=1.0, key='default_1')
    default_2 = st.sidebar.slider("2-Month Loan Default Rate (%)", 0.0, 400.0, 10.0, step=1.0, key='default_2')
    default_3 = st.sidebar.slider("3-Month Loan Default Rate (%)", 0.0, 400.0, 10.0, step=1.0, key='default_3')

    # --- Convert Inputs ---
    p = np.array([alloc_1, alloc_2, alloc_3]) / 100
    r = monthly_interest / 100
    d = np.array([default_1, default_2, default_3]) / 100

    # --- Transition Matrix for shifting buckets ---
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]])

    # --- Initialize Portfolio State ---
    x = np.zeros((3, num_months + 1))
    x[:, 0] = [initial_capital, 0, 0]

    # --- Traces for Metrics ---
    defaults_trace = []
    interest_trace = []
    net_reinvested_trace = []

    # --- Simulation Loop (corrected) ---
    for t in range(num_months):
        # accumulate this month’s cash flows across all buckets
        total_default_loss    = 0.0
        total_interest_earned = 0.0
        total_reinvestment    = 0.0

        # check each bucket i = 0,1,2 (1-, 2-, 3-month loans)
        for i in range(3):
            if t - i >= 0:
                matured = x[i, t - i]
                loss    = d[i] * matured
                recovered = matured - loss
                interest  = r * recovered

                total_default_loss    += loss
                total_interest_earned += interest
                total_reinvestment     += recovered + interest

        # record monthly metrics
        defaults_trace.append(total_default_loss)
        interest_trace.append(total_interest_earned)
        net_reinvested_trace.append(total_reinvestment)

        # shift surviving loans forward one period
        carried = A @ x[:, t]

        # allocate the pooled reinvestment once across the three new buckets
        new_loans = total_reinvestment * p
        x[:, t + 1] = carried + new_loans

    # --- Portfolio DataFrame ---
    df = pd.DataFrame({
        "Month": range(num_months + 1),
        "1-Month Loans": x[0],
        "2-Month Loans": x[1],
        "3-Month Loans": x[2],
        "Total Outstanding": x.sum(axis=0)
    })

    # --- Metrics DataFrame ---
    metrics = pd.DataFrame({
        "Month": range(1, num_months + 1),
        "Defaults": defaults_trace,
        "Interest Earned": interest_trace,
        "Net Reinvested": net_reinvested_trace
    })

    # --- Outputs ---
    st.subheader("Loan Portfolio Over Time")
    st.dataframe(df.style.format("{:,.2f}"))

    st.subheader("Defaults and Cash Flow Metrics")
    st.dataframe(metrics.style.format("{:,.2f}"))

    # --- Plotting ---
    st.subheader("Portfolio Growth and Risk")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Month"], df["1-Month Loans"], label="1-Month")
    ax.plot(df["Month"], df["2-Month Loans"], label="2-Month")
    ax.plot(df["Month"], df["3-Month Loans"], label="3-Month")
    ax.plot(df["Month"], df["Total Outstanding"], label="Total Outstanding", linestyle='--', linewidth=2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Outstanding Principal ($)")
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax.set_title("Loan Book Evolution with Defaults")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- METRICS DASHBOARD ---
    st.subheader("Private Credit Metrics Dashboard")

    total_interest = np.sum(interest_trace)
    total_defaults = np.sum(defaults_trace)
    total_reinvested = np.sum(net_reinvested_trace)
    avg_assets = np.mean(df["Total Outstanding"][1:])

    # ROA
    gross_roa = total_interest / avg_assets
    net_roa   = (total_interest - total_defaults) / avg_assets

    # MOIC
    gross_moic = (total_interest + initial_capital) / initial_capital
    net_moic   = (total_interest + initial_capital - total_defaults) / initial_capital

    # Annualized yield
    gross_yield_ann = (1 + gross_roa) ** 12 - 1
    net_yield_ann   = (1 + net_roa) ** 12 - 1

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Interest Earned", f"${total_interest:,.0f}")
    col2.metric("Total Principal Defaults", f"${total_defaults:,.0f}")
    col3.metric("Total Net Reinvested", f"${total_reinvested:,.0f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Gross ROA", f"{gross_roa:.2%}")
    col5.metric("Net ROA", f"{net_roa:.2%}")
    col6.metric("Avg Outstanding", f"${avg_assets:,.0f}")

    col7, col8, col9 = st.columns(3)
    col7.metric("Gross MOIC", f"{gross_moic:.2f}x")
    col8.metric("Net MOIC", f"{net_moic:.2f}x")
    col9.metric("Initial Capital", f"${initial_capital:,.0f}")

    col10, col11 = st.columns(2)
    col10.metric("Gross Yield (Annualized)", f"{gross_yield_ann:.2%}")
    col11.metric("Net Yield (Annualized)", f"{net_yield_ann:.2%}")
else:
    print("Streamlit is not available in this environment. Please run the script using `streamlit run this_file.py` after installing Streamlit with `pip install streamlit`.")
