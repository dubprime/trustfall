# NOTE: This script requires Streamlit. Run with `streamlit run streamlit_app.py`

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None
    import warnings
    warnings.warn("Streamlit is not installed. Please install it using `pip install streamlit` to run this dashboard.")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    st.title("DubPrime.com Loan Portfolio Simulator with Amortizing Loans & Guided Breakouts")

    # Sidebar: Allocation Settings
    st.sidebar.header("Loan Allocation Settings")
    alloc_1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    max_alloc_2 = max(0, 100 - alloc_1)
    alloc_2 = st.sidebar.slider("2-Month Allocation (%)", 0, max_alloc_2, min(25, max_alloc_2)) if max_alloc_2 > 0 else 0
    alloc_3 = 100 - alloc_1 - alloc_2

    # Sidebar: Portfolio Parameters & Defaults
    st.sidebar.header("Portfolio Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000)
    monthly_interest = st.sidebar.number_input("Monthly Interest (%)", value=12.0)
    num_months = st.sidebar.slider("Duration (Months)", 1, 60, 12)
    st.sidebar.header("Default Rates at Maturity (Term-Rate)")
    default_1 = st.sidebar.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
    default_2 = st.sidebar.slider("2-Month Default (%) (term-cumulated)", 0.0, 100.0, 10.0)
    default_3 = st.sidebar.slider("3-Month Default (%) (term-cumulated)", 0.0, 100.0, 10.0)

    # Toggles: net vs gross, and defaults overlay
    show_def_overlay = st.sidebar.checkbox("Overlay defaults within gross view", value=False)

    # Pie chart for allocation mix
    df_alloc = pd.DataFrame({"Tenor": ["1-Month", "2-Month", "3-Month"],
                             "Allocation": [alloc_1, alloc_2, alloc_3]})
    fig_alloc = px.pie(df_alloc, names="Tenor", values="Allocation", title="Allocation Mix",
                       color_discrete_sequence=["#e45756", "#4c78a8", "#f58518"])
    st.sidebar.plotly_chart(fig_alloc, use_container_width=True)

    # Convert inputs to fractions
    p = np.array([alloc_1, alloc_2, alloc_3]) / 100
    r = monthly_interest / 100
    d = np.array([default_1, default_2, default_3]) / 100

    # Precompute level-payment factors and hazard rates
    terms = [2, 3]
    pmt_factor = {n: r / (1 - (1 + r) ** (-n)) for n in terms}
    hazard = {n: 1 - (1 - d[i+1]) ** (1 / n) for i, n in enumerate(terms)}

    # Precompute gross & net outstanding weight schedules
    weights_gross = {1: [1, 0]}
    weights_net = {1: [1 - d[0], 0]}
    for n in terms:
        wg = [(1 + r) ** k - (((1 + r) ** k - 1) * pmt_factor[n] / r) for k in range(n + 1)]
        wn = [wg[k] * (1 - hazard[n]) ** k for k in range(n + 1)]
        weights_gross[n] = wg
        weights_net[n] = wn

    # Initialize cashflow schedule & trackers
    payment_schedule = np.zeros(num_months + max(terms) + 1)
    payment_schedule[0] = initial_capital
    new_by_t = np.zeros((num_months, 3))
    defaults, interest, reinvest = [], [], []

    # Simulation loop
    for t in range(num_months):
        cash = payment_schedule[t]
        new = cash * p
        new_by_t[t] = new
        md = mi = mr = 0.0

        # 1-month bullet (default at issuance)
        P1 = new[0]
        if P1 > 0:
            loss1 = d[0] * P1
            rec1 = P1 - loss1
            int1 = r * rec1
            payment_schedule[t + 1] += rec1 + int1
            md += loss1; mi += int1; mr += rec1 + int1

        # 2- & 3-month level-payment loans (with hazard)
        for i, n in enumerate(terms, start=1):
            P = new[i]
            if P <= 0:
                continue
            pf = pmt_factor[n]
            pmt = pf * P
            remaining = P
            h = hazard[n]
            for k in range(1, n + 1):
                # apply hazard
                loss_k = h * remaining
                remaining -= loss_k
                # interest + principal
                int_k = r * remaining
                prin_k = pmt - int_k
                remaining -= prin_k
                idx = t + k
                payment_schedule[idx] += prin_k + int_k
                md += loss_k; mi += int_k; mr += prin_k + int_k

        defaults.append(md)
        interest.append(mi)
        reinvest.append(mr)

    # Build outstanding series by tenor
    months = np.arange(num_months)
    out_g = {
        1: new_by_t[:, 0],
        2: np.array([sum(new_by_t[j,1] * weights_gross[2][min(t-j,2)] for j in range(t+1)) for t in months]),
        3: np.array([sum(new_by_t[j,2] * weights_gross[3][min(t-j,3)] for j in range(t+1)) for t in months])
    }
    out_n = {
        1: new_by_t[:,0] * weights_net[1][0],
        2: np.array([sum(new_by_t[j,1] * weights_net[2][min(t-j,2)] for j in range(t+1)) for t in months]),
        3: np.array([sum(new_by_t[j,2] * weights_net[3][min(t-j,3)] for j in range(t+1)) for t in months])
    }
    total_g = sum(out_g.values())
    total_n = sum(out_n.values())
    def_out = {k: out_g[k] - out_n[k] for k in out_g}

    # Choose view
    out = out_g
    total = total_g

    # Stacked bar + total line
    st.subheader("Outstanding by Tenor (Stacked) + Total (Gross)")
    fig1 = go.Figure()
    for k, color in zip([1,2,3], ['#e45756','#4c78a8','#f58518']):
        if show_def_overlay:
            fig1.add_trace(go.Bar(x=months, y=out_n[k], name=f'{k}-Month Surviving', marker_color=color))
            fig1.add_trace(go.Bar(x=months, y=def_out[k], name=f'{k}-Month Defaults', marker_color=color, opacity=0.5))
        else:
            fig1.add_trace(go.Bar(x=months, y=out[k], name=f'{k}-Month', marker_color=color))
    fig1.add_trace(go.Scatter(x=months, y=total, name='Total', mode='lines+markers', line=dict(color='black', width=2)))
    fig1.update_layout(barmode='stack', xaxis_title='Month', yaxis_title='Outstanding Principal ($)')
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------------------
    # Dedicated "Defaults by Tenor" stacked bar
    # ----------------------------------------
    st.subheader("Defaults by Tenor (Stacked)")

    # build a DataFrame so you can use either px.bar or go.Bar
    df_def = pd.DataFrame({
        "Month": months,
        "1-Month Default": def_out[1],
        "2-Month Default": def_out[2],
        "3-Month Default": def_out[3],
    })

    # using Plotly Express
    fig_def = px.bar(
        df_def,
        x="Month",
        y=["1-Month Default", "2-Month Default", "3-Month Default"],
        title="Defaults by Tenor",
        barmode="stack",
        labels={"value": "Defaults ($)", "variable": "Tenor"}
    )
    st.plotly_chart(fig_def, use_container_width=True)

    # Portfolio Growth and Risk by Tenor
    st.subheader("Portfolio Growth and Risk by Tenor")
    fig2 = go.Figure()
    for k, color in zip([1,2,3], ['#e45756','#4c78a8','#f58518']):
        fig2.add_trace(go.Scatter(x=months, y=out[k], name=f'{k}-Month', mode='lines+markers', line=dict(color=color), marker=dict(color=color)))
    fig2.add_trace(go.Scatter(x=months, y=total, name='Total', mode='lines+markers', line=dict(color='black', dash='dash'), marker=dict(color='black')))
    fig2.update_layout(xaxis_title='Month', yaxis_title='Outstanding Principal ($)', legend=dict(title='', traceorder='normal'))
    st.plotly_chart(fig2, use_container_width=True, key="portfolio_growth_risk_1")
    
    # Original line chart (Plotly) with matching colors and legend order
    st.subheader("Portfolio Growth and Risk by Tenor")
    fig2 = go.Figure()
    # 1-Month
    fig2.add_trace(go.Scatter(
        x=months, y=out[1], name='1-Month', mode='lines+markers',
        line=dict(color='#e45756'), marker=dict(color='#e45756')
    ))
    # 2-Month
    fig2.add_trace(go.Scatter(
        x=months, y=out[2], name='2-Month', mode='lines+markers',
        line=dict(color='#4c78a8'), marker=dict(color='#4c78a8')
    ))
    # 3-Month
    fig2.add_trace(go.Scatter(
        x=months, y=out[3], name='3-Month', mode='lines+markers',
        line=dict(color='#f58518'), marker=dict(color='#f58518')
    ))
    # Total
    fig2.add_trace(go.Scatter(
        x=months, y=total, name='Total', mode='lines+markers',
        line=dict(color='black', dash='dash'), marker=dict(color='black')
    ))
    fig2.update_layout(
        xaxis_title='Month',
        yaxis_title='Outstanding Principal ($)',
        legend=dict(title='', traceorder='normal')
    )
    st.plotly_chart(fig2, use_container_width=True, key="portfolio_growth_risk_2")

    # Guided Cashflow Breakouts
    df_origin = pd.DataFrame(new_by_t, columns=["1-Month Orig","2-Month Orig","3-Month Orig"])  
    df_origin['Month'] = months
    df_inflow = pd.DataFrame({'Month': np.arange(len(payment_schedule)), 'Inflow': payment_schedule})
    df_metrics = pd.DataFrame({'Month': months+1, 'Defaults': defaults, 'Interest': interest, 'Reinvested': reinvest})
    df_cum = df_metrics.copy()
    df_cum[['Cum Defaults','Cum Interest','Cum Reinvested']] = df_cum[['Defaults','Interest','Reinvested']].cumsum()

    st.subheader("Step 1: New Originations by Tenor")
    with st.expander("Details and chart", expanded=False):
        st.dataframe(df_origin.set_index('Month').style.format('{:,.2f}'))
        fig_o = px.bar(df_origin, x='Month', y=['1-Month Orig','2-Month Orig','3-Month Orig'])
        st.plotly_chart(fig_o, use_container_width=True)

    st.subheader("Step 2: Scheduled Cash Inflows")
    with st.expander("Details and trend", expanded=False):
        st.dataframe(df_inflow.set_index('Month').style.format('{:,.2f}'))
        fig_i = px.line(df_inflow, x='Month', y='Inflow', markers=True)
        fig_i.update_layout(yaxis_title='Cumulative Inflow',
                            yaxis=dict(tickformat='$,') )
        st.plotly_chart(fig_i, use_container_width=True)

    st.subheader("Step 3: Monthly Cashflow Components")
    with st.expander("Breakdown of defaults, interest, reinvestment", expanded=False):
        st.dataframe(df_metrics.set_index('Month').style.format('{:,.2f}'))
        fig_c = px.area(df_metrics, x='Month', y=['Defaults','Interest','Reinvested'])
        st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("Step 4: Cumulative Cashflow Tally")
    with st.expander("See cumulative build-up", expanded=False):
        st.dataframe(df_cum.set_index('Month').style.format('{:,.2f}'))
        fig_cc = px.area(df_cum, x='Month', y=['Cum Defaults','Cum Interest','Cum Reinvested'])
        st.plotly_chart(fig_cc, use_container_width=True)

    # MOIC Dashboard
    total_int = sum(interest)
    total_def = sum(defaults)
    total_reinv = sum(reinvest)
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
    print("Run with `streamlit run streamlit_app.py` after installing Streamlit.")
