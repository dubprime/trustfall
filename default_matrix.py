# NOTE: This script requires Streamlit. Run with `streamlit run streamlit_app.py`

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
    # Inject CSS for full-width layout
    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 100% !important;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # --- App Title ---
    st.title("Loan Portfolio Simulator with Amortizing Loans & Breakouts")

    # --- Sidebar Inputs ---
    st.sidebar.header("Loan Allocation Settings")
    alloc_1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    alloc_2 = st.sidebar.slider("2-Month Allocation (%)", 0, 100 - alloc_1, 25)
    alloc_3 = 100 - alloc_1 - alloc_2
    st.sidebar.markdown(f"**3-Month Allocation:** {alloc_3}%")

    st.sidebar.header("Portfolio Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000)
    monthly_interest = st.sidebar.number_input("Monthly Interest (%)", value=3.0)
    num_months = st.sidebar.slider("Duration (Months)", 1, 60, 12)

    st.sidebar.header("Default Rates at Maturity")
    default_1 = st.sidebar.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
    default_2 = st.sidebar.slider("2-Month Default (%)", 0.0, 100.0, 10.0)
    default_3 = st.sidebar.slider("3-Month Default (%)", 0.0, 100.0, 10.0)

    # --- Convert inputs ---
    p = np.array([alloc_1, alloc_2, alloc_3]) / 100
    r = monthly_interest / 100
    d = np.array([default_1, default_2, default_3]) / 100

    # --- Initialize schedule & traces ---
    payment_schedule = np.zeros(num_months + 4)
    payment_schedule[0] = initial_capital

    new_by_t = np.zeros((num_months, 3))
    defaults_trace = []
    interest_trace = []
    net_reinvest_trace = []

    # --- Simulation Loop with amortization ---
    for t in range(num_months):
        cash = payment_schedule[t]
        new = cash * p
        new_by_t[t] = new

        md = mi = mr = 0.0
        # 1-month bullet
        P1 = new[0]
        loss1 = d[0] * P1
        rec1 = P1 - loss1
        int1 = r * rec1
        payment_schedule[t + 1] += rec1 + int1
        md += loss1
        mi += int1
        mr += rec1 + int1

        # 2-month amortizing
        P2 = new[1]
        half = P2 / 2
        # first payment at t+1
        int2a = r * P2
        payment_schedule[t + 1] += half + int2a
        # second payment at t+2 (with default on remaining)
        out2 = P2 - half
        loss2 = d[1] * out2
        rec2 = out2 - loss2
        int2b = r * rec2
        payment_schedule[t + 2] += rec2 + int2b
        md += loss2
        mi += int2a + int2b
        mr += half + int2a + rec2 + int2b

        # 3-month amortizing
        P3 = new[2]
        third = P3 / 3
        # payment 1 at t+1
        int3a = r * P3
        payment_schedule[t + 1] += third + int3a
        # payment 2 at t+2
        out3_2 = P3 - third
        int3b = r * out3_2
        payment_schedule[t + 2] += third + int3b
        # payment 3 at t+3 (with default on remaining)
        out3_3 = P3 - 2 * third
        loss3 = d[2] * out3_3
        rec3 = out3_3 - loss3
        int3c = r * rec3
        payment_schedule[t + 3] += rec3 + int3c
        md += loss3
        mi += int3a + int3b + int3c
        mr += third + int3a + third + int3b + rec3 + int3c

        defaults_trace.append(md)
        interest_trace.append(mi)
        net_reinvest_trace.append(mr)

        # --- Build outstanding by bucket for the active simulation window (0…num_months-1) ---
    months = np.arange(num_months)
    out_1 = np.zeros(num_months, float)
    out_2 = np.zeros(num_months, float)
    out_3 = np.zeros(num_months, float)

    for t in months:
        # 1-month outstanding only at origination
        out_1[t] = new_by_t[t, 0]
        # 2-month amortizing outstanding
        for j in range(max(0, t - 1), t + 1):
            age = t - j
            if age == 0:
                out_2[t] += new_by_t[j, 1]
            elif age == 1:
                out_2[t] += new_by_t[j, 1] * 0.5
        # 3-month amortizing outstanding
        for j in range(max(0, t - 2), t + 1):
            age = t - j
            if age == 0:
                out_3[t] += new_by_t[j, 2]
            elif age == 1:
                out_3[t] += new_by_t[j, 2] * 2/3
            elif age == 2:
                out_3[t] += new_by_t[j, 2] * 1/3

    out_total = out_1 + out_2 + out_3

    # --- DataFrames ---
    df_breakout = pd.DataFrame({
        "Month": months,
        "1-Month": out_1,
        "2-Month": out_2,
        "3-Month": out_3,
        "Total": out_total
    })
    # metrics months align with cash flows at t=1…num_months
    metrics_months = np.arange(1, num_months + 1)
    df_metrics = pd.DataFrame({
        "Month": metrics_months,
        "Defaults": defaults_trace,
        "Interest": interest_trace,
        "Reinvested": net_reinvest_trace
    })

    # --- Display breakouts & metrics ---
    st.subheader("Outstanding by Tenor (active window)")
    st.line_chart(df_breakout.set_index("Month"))

    st.subheader("Defaults & Cash-Flows")
    st.dataframe(df_metrics.style.format("{:.2f}"))

    # --- Private Credit Dashboard (MOIC focus) ---
    total_int = sum(interest_trace)
    total_def = sum(defaults_trace)
    total_reinv = sum(net_reinvest_trace)
    gross_moic = (total_int + initial_capital) / initial_capital
    net_moic = (total_int + initial_capital - total_def) / initial_capital

    st.subheader("Key Metrics (MOIC)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gross MOIC", f"{gross_moic:.2f}x")
    c2.metric("Net MOIC", f"{net_moic:.2f}x")
    c3.metric("Initial Capital", f"${initial_capital:,.0f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Total Interest", f"${total_int:,.0f}")
    c5.metric("Total Defaults", f"${total_def:,.0f}")
    c6.metric("Total Reinvested", f"${total_reinv:,.0f}")

else:
    print("Streamlit is not available. Run with `streamlit run streamlit_app.py`.")