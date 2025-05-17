# NOTE: This script requires Streamlit. Run with `streamlit run the_matrix_simulation.py`

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None
    import warnings
    warnings.warn("Streamlit is not installed. Please install it using `pip install streamlit` to run this dashboard.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if st:
    # --- App title ---
    st.title("DubPrime Loan Portfolio Simulator with Defaults, Reinvestment & Investor Metrics")

    # --- Sidebar Inputs ---
    st.sidebar.header("Loan Allocation Settings")
    alloc_1 = st.sidebar.slider("1-Month Loan Allocation (%)", 0, 100, 50)
    alloc_2 = st.sidebar.slider("2-Month Loan Allocation (%)", 0, 100 - alloc_1, 25)
    alloc_3 = 100 - alloc_1 - alloc_2
    st.sidebar.markdown(f"**3-Month Loan Allocation (%):** {alloc_3}")

    st.sidebar.header("Portfolio Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000)
    monthly_interest = st.sidebar.number_input("Monthly Simple Interest Rate (%)", value=10.0)
    num_months = st.sidebar.slider("Simulation Duration (Months)", 1, 60, 24)

    st.sidebar.header("Default Rates (Per Bucket at Maturity)")
    default_1 = st.sidebar.slider("1-Month Loan Default Rate (%)", 0.0, 100.0, 2.0)
    default_2 = st.sidebar.slider("2-Month Loan Default Rate (%)", 0.0, 100.0, 4.0)
    default_3 = st.sidebar.slider("3-Month Loan Default Rate (%)", 0.0, 100.0, 6.0)

    # --- Convert Inputs ---
    p = np.array([alloc_1, alloc_2, alloc_3]) / 100
    r = monthly_interest / 100
    d = np.array([default_1, default_2, default_3]) / 100

    # --- Transition Matrix ---
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]])
    e1T = np.array([[1, 0, 0]])

    # --- Initialize Portfolio ---
    x = np.zeros((3, num_months + 1))
    x[:, 0] = [initial_capital, 0, 0]

    # --- Traces for Metrics ---
    defaults_trace = []
    interest_trace = []
    net_reinvested_trace = []

    # --- Simulation Loop ---
    for t in range(num_months):
        matured = x[0, t]
        default_loss = d[0] * matured
        recovered = matured - default_loss
        interest_earned = r * recovered
        reinvestment = recovered + interest_earned

        defaults_trace.append(default_loss)
        interest_trace.append(interest_earned)
        net_reinvested_trace.append(reinvestment)

        new_loans = reinvestment * p
        next_x = A @ x[:, t] + new_loans

        if t >= 1:
            prev_2m = x[1, t - 1]
            loss_2m = d[1] * prev_2m
            recovered_2m = prev_2m - loss_2m
            interest_2m = r * recovered_2m
            reinvestment += recovered_2m + interest_2m
            new_loans = reinvestment * p
            next_x += new_loans
            defaults_trace[-1] += loss_2m
            interest_trace[-1] += interest_2m
            net_reinvested_trace[-1] += recovered_2m + interest_2m

        if t >= 2:
            prev_3m = x[2, t - 2]
            loss_3m = d[2] * prev_3m
            recovered_3m = prev_3m - loss_3m
            interest_3m = r * recovered_3m
            reinvestment += recovered_3m + interest_3m
            new_loans = reinvestment * p
            next_x += new_loans
            defaults_trace[-1] += loss_3m
            interest_trace[-1] += interest_3m
            net_reinvested_trace[-1] += recovered_3m + interest_3m

        x[:, t + 1] = next_x

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
    ax.set_title("Loan Book Evolution with Defaults")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- METRICS DASHBOARD ---
    st.subheader("Private Credit Metrics Dashboard")

    # Cumulative values
    total_interest = np.sum(interest_trace)
    total_defaults = np.sum(defaults_trace)
    total_reinvested = np.sum(net_reinvested_trace)
    total_principal_invested = initial_capital
    avg_assets = np.mean(df["Total Outstanding"][1:])

    # ROA
    gross_roa = total_interest / avg_assets
    net_roa = (total_interest - total_defaults) / avg_assets

    # MOIC
    gross_moic = (total_interest + total_principal_invested) / total_principal_invested
    net_moic = (total_interest + total_principal_invested - total_defaults) / total_principal_invested

    # Annualized yield
    gross_yield_ann = (1 + gross_roa) ** 12 - 1
    net_yield_ann = (1 + net_roa) ** 12 - 1

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
