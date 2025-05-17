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
    # Full-width layout
    st.markdown(
        """
        <style>
            .main .block-container { max-width:100% !important; padding-left:2rem; padding-right:2rem; }
        </style>
        """, unsafe_allow_html=True
    )

    # Title
    st.title("Loan Portfolio Simulator with Amortizing Loans & Breakouts")

    # Sidebar: Allocation Settings
    st.sidebar.header("Loan Allocation Settings")
    alloc_1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    max_alloc_2 = max(0, 100 - alloc_1)
    alloc_2 = st.sidebar.slider("2-Month Allocation (%)", 0, max_alloc_2, min(25, max_alloc_2)) if max_alloc_2>0 else 0
    alloc_3 = 100 - alloc_1 - alloc_2

    # Pie chart for mix
    import plotly.express as px
    df_alloc = pd.DataFrame({
        "Tenor": ["1-Month","2-Month","3-Month"],
        "Allocation": [alloc_1, alloc_2, alloc_3]
    })
    fig_alloc = px.pie(
        df_alloc, names="Tenor", values="Allocation", title="Allocation Mix",
        color_discrete_sequence=["#e45756","#4c78a8","#f58518"]
    )
    st.sidebar.plotly_chart(fig_alloc, use_container_width=True)

    # Sidebar: Portfolio Params
    st.sidebar.header("Portfolio Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000)
    monthly_interest = st.sidebar.number_input("Monthly Interest (%)", value=3.0)
    num_months = st.sidebar.slider("Duration (Months)", 1, 60, 12)

    # Sidebar: Defaults
    st.sidebar.header("Default Rates at Maturity")
    default_1 = st.sidebar.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
    default_2 = st.sidebar.slider("2-Month Default (%)", 0.0, 100.0, 10.0)
    default_3 = st.sidebar.slider("3-Month Default (%)", 0.0, 100.0, 10.0)

    # Convert inputs
    p = np.array([alloc_1, alloc_2, alloc_3]) / 100
    r = monthly_interest / 100
    d = np.array([default_1, default_2, default_3]) / 100

    # Initialize cashflow schedule
    payment_schedule = np.zeros(num_months + 4)
    payment_schedule[0] = initial_capital
    new_by_t = np.zeros((num_months, 3))
    defaults, interest, reinvest = [], [], []

    # Simulate
    for t in range(num_months):
        cash = payment_schedule[t]
        new = cash * p
        new_by_t[t] = new
        md = mi = mr = 0.0
        # 1-month bullet
        P1 = new[0]; loss1 = d[0]*P1; rec1 = P1-loss1; int1 = r*rec1
        payment_schedule[t+1] += rec1 + int1; md += loss1; mi += int1; mr += rec1 + int1
        # 2-month amortizing
        P2 = new[1]; half=P2/2; int2a=r*P2; payment_schedule[t+1]+=half+int2a
        out2=P2-half; loss2=d[1]*out2; rec2=out2-loss2; int2b=r*rec2
        payment_schedule[t+2]+=rec2+int2b; md+=loss2; mi+=int2a+int2b; mr+=half+int2a+rec2+int2b
        # 3-month amortizing
        P3=new[2]; third=P3/3; int3a=r*P3; payment_schedule[t+1]+=third+int3a
        out3_2=P3-third; int3b=r*out3_2; payment_schedule[t+2]+=third+int3b
        out3_3=P3-2*third; loss3=d[2]*out3_3; rec3=out3_3-loss3; int3c=r*rec3
        payment_schedule[t+3]+=rec3+int3c; md+=loss3; mi+=int3a+int3b+int3c; mr+=third+int3a+third+int3b+rec3+int3c
        defaults.append(md); interest.append(mi); reinvest.append(mr)

    # Compute outstanding by tenor
    months = np.arange(num_months)
    out1 = np.zeros(num_months); out2 = np.zeros(num_months); out3 = np.zeros(num_months)
    for t in months:
        out1[t] = new_by_t[t,0]
        for j in range(max(0,t-1), t+1): age=t-j; out2[t]+=new_by_t[j,1] * (1 if age==0 else 0.5)
        for j in range(max(0,t-2), t+1): age=t-j; factor={0:1,1:2/3,2:1/3}[age]; out3[t]+=new_by_t[j,2]*factor
    total = out1 + out2 + out3

    # New widget: Stacked bar + total line at the top
    st.subheader("Outstanding by Tenor (Stacked) + Total")
    fig_s, ax_s = plt.subplots(figsize=(10,6))
    ax_s.bar(months, out1, label="1-Month")
    ax_s.bar(months, out2, bottom=out1, label="2-Month")
    ax_s.bar(months, out3, bottom=out1+out2, label="3-Month")
    ax_s.plot(months, total, color="black", marker="o", linewidth=2, label="Total")
    ax_s.set_xlabel("Month"); ax_s.set_ylabel("Outstanding Principal ($)")
    ax_s.legend(); st.pyplot(fig_s)

    # Original line chart widget
    st.subheader("Portfolio Growth and Risk by Tenor")
    fig_l, ax_l = plt.subplots(figsize=(10,6))
    ax_l.plot(months, out1, label="1-Month")
    ax_l.plot(months, out2, label="2-Month")
    ax_l.plot(months, out3, label="3-Month")
    ax_l.plot(months, total, label="Total", linestyle="--", linewidth=2)
    ax_l.set_xlabel("Month"); ax_l.set_ylabel("Outstanding Principal ($)")
    ax_l.legend(); ax_l.grid(True);
    st.pyplot(fig_l)

    # Then continue with breakouts, metrics, etc.
else:
    print("Run with `streamlit run streamlit_app.py` after installing Streamlit.")
