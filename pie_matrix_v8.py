import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Try importing Streamlit ─────────────────────────────────────────────────
try:
    import streamlit as st
except ModuleNotFoundError:
    st = None
    import warnings
    warnings.warn(
        "Streamlit is not installed. Please install it with `pip install streamlit`."
    )

# ─── Constants ────────────────────────────────────────────────────────────────
CSS = """
<style>
    .main .block-container { 
        max-width:100% !important; 
        padding-left:2rem; 
        padding-right:2rem; 
    }
</style>
"""

# color palette by tenor
COLORS: Dict[int, str] = {1: "#e45756", 2: "#4c78a8", 3: "#f58518"}
TERMS = [1, 2, 3]  # in months

# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class InputParams:
    initial_capital: float
    monthly_interest: float     # e.g. 0.12 for 12%
    num_months: int
    default_rates: Dict[int, float]
    allocation: Dict[int, float]
    show_def_overlay: bool

@dataclass
class SimulationResults:
    months: np.ndarray
    new_by_t: np.ndarray
    payment_schedule: np.ndarray
    out_gross: Dict[int, np.ndarray]
    out_net: Dict[int, np.ndarray]
    def_out: Dict[int, np.ndarray]
    per_period_defaults: np.ndarray
    per_period_interest: np.ndarray
    per_period_principal: np.ndarray
    payment_by_tenor: Dict[int, np.ndarray]
    interest_by_tenor: Dict[int, np.ndarray]
    principal_by_tenor: Dict[int, np.ndarray]
    default_by_tenor: Dict[int, np.ndarray]
    pmt_factors: Dict[int, float]

# ─── Input Loading ────────────────────────────────────────────────────────────
def load_inputs() -> InputParams:
    if st is None:
        return None

    st.markdown(CSS, unsafe_allow_html=True)
    st.title("DubPrime.com Loan Portfolio Simulator")

    # Sidebar: Allocation Mix Pie at top
    st.sidebar.subheader("Allocation Mix")
    a1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    max2 = max(0, 100 - a1)
    a2 = st.sidebar.slider("2-Month Allocation (%)", 0, max2, min(25, max2)) if max2 else 0
    a3 = 100 - a1 - a2
    allocation = {1: a1/100, 2: a2/100, 3: a3/100}
    df_alloc = pd.DataFrame({
        "Tenor": [f"{n}-Month" for n in TERMS],
        "Allocation": [allocation[n]*100 for n in TERMS]
    })
    fig_alloc = px.pie(
        df_alloc, names="Tenor", values="Allocation",
        title=None,
        color_discrete_sequence=list(COLORS.values())
    )
    st.sidebar.plotly_chart(fig_alloc, use_container_width=True)

    # Sidebar Controls
    with st.sidebar.expander("Settings", expanded=True):
        st.header("Loan Allocation Settings")
        # reuse a1,a2,a3
        st.header("Portfolio Parameters")
        init_cap = st.number_input("Initial Capital ($)", value=100_000)
        mon_int = st.number_input("Monthly Interest (%)", value=12.0)/100
        num_m = st.slider("Duration (Months)", 1, 60, 12)

        st.header("Default Rates at Maturity")
        d1 = st.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
        d2 = st.slider("2-Month Default (%)", 0.0, 100.0, 10.0)
        d3 = st.slider("3-Month Default (%)", 0.0, 100.0, 10.0)
        default_rates = {1: d1/100, 2: d2/100, 3: d3/100}

        overlay = st.checkbox("Overlay defaults within gross view", value=True)

    return InputParams(
        initial_capital=init_cap,
        monthly_interest=mon_int,
        num_months=num_m,
        default_rates=default_rates,
        allocation=allocation,
        show_def_overlay=overlay,
    )

# ─── Simulation Logic ────────────────────────────────────────────────────────
@st.cache_data
def run_simulation(params: InputParams) -> SimulationResults:
    r, d, p, T = params.monthly_interest, params.default_rates, params.allocation, params.num_months
    payment_schedule = np.zeros(T + max(TERMS) + 1)
    payment_schedule[0] = params.initial_capital
    new_by_t = np.zeros((T, len(TERMS)))

    payment_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    interest_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    principal_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    default_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}

    pmt_factors = {n: r/(1-(1+r)**(-n)) for n in TERMS if n>1}
    hazard = {n: 1-(1-d[n])**(1/n) for n in TERMS if n>1}

    for t in range(T):
        cash = payment_schedule[t]
        deployed = np.array([p[n]*cash for n in TERMS])
        new_by_t[t] = deployed

        # 1-Month bullet
        P1 = deployed[0]
        loss1 = d[1]*P1
        default_by_tenor[1][t] += loss1
        recov1 = P1 - loss1
        int1 = r*recov1
        payment_schedule[t+1] += recov1 + int1
        payment_by_tenor[1][t+1] += recov1 + int1
        interest_by_tenor[1][t+1] += int1
        principal_by_tenor[1][t+1] += recov1

        # Multi-month level-payment with hazard
        for i, n in enumerate(TERMS[1:], start=1):
            Pn = deployed[i]
            if Pn <= 0: continue
            pf, hz = pmt_factors[n], hazard[n]
            remaining = Pn
            pmt = pf*Pn
            for k in range(1, n+1):
                # default event
                loss_k = hz*remaining
                default_by_tenor[n][t+k] += loss_k
                remaining -= loss_k
                # if fully defaulted, stop further payments
                if remaining <= 0:
                    remaining = 0
                    break
                # interest on surviving balance
                int_k = r*remaining
                # principal portion is schedule minus interest, but cannot exceed remaining
                prin_k = min(pmt - int_k, remaining)
                remaining -= prin_k
                # schedule inflow
                payment_schedule[t+k] += prin_k + int_k
                payment_by_tenor[n][t+k] += prin_k + int_k
                interest_by_tenor[n][t+k] += int_k
                principal_by_tenor[n][t+k] += prin_k

    months = np.arange(T)

    # Outstanding calculation
    weights_gross = {1:[1,0]}
    weights_net   = {1:[1-d[1],0]}
    for n in TERMS[1:]:
        wg = [(1+r)**k - (((1+r)**k-1)*pmt_factors[n]/r) for k in range(n+1)]
        wn = [wg[k]*(1-hazard[n])**k for k in range(n+1)]
        weights_gross[n], weights_net[n] = wg, wn

    out_gross, out_net, def_out = {}, {}, {}
    for idx, n in enumerate(TERMS):
        out_gross[n] = np.array([
            sum(new_by_t[j,idx]*weights_gross[n][min(t-j,n)] for j in range(t+1))
            for t in months
        ])
        out_net[n] = np.array([
            sum(new_by_t[j,idx]*weights_net[n][min(t-j,n)] for j in range(t+1))
            for t in months
        ])
        def_out[n] = out_gross[n] - out_net[n]

    # True per-period cashflow components
    per_defaults = np.array([sum(default_by_tenor[n][t] for n in TERMS) for t in months])
    per_interest = np.array([sum(interest_by_tenor[n][t] for n in TERMS) for t in months])
    per_principal= np.array([sum(principal_by_tenor[n][t] for n in TERMS) for t in months])

    return SimulationResults(
        months=months,
        new_by_t=new_by_t,
        payment_schedule=payment_schedule,
        out_gross=out_gross,
        out_net=out_net,
        def_out=def_out,
        per_period_defaults=per_defaults,
        per_period_interest=per_interest,
        per_period_principal=per_principal,
        payment_by_tenor=payment_by_tenor,
        interest_by_tenor=interest_by_tenor,
        principal_by_tenor=principal_by_tenor,
        default_by_tenor=default_by_tenor,
        pmt_factors=pmt_factors
    )

# ─── Figure Building & Rendering ───────────────────────────────────────────────
def build_and_render(params: InputParams, sim: SimulationResults):
    # Main charts
    st.subheader("Outstanding by Tenor (Stacked) + Total")
    total = sum(sim.out_gross.values())
    fig_ost = go.Figure()
    for n in TERMS:
        if params.show_def_overlay:
            fig_ost.add_trace(go.Bar(x=sim.months, y=sim.out_net[n], name=f"{n}-Month Surviving", marker_color=COLORS[n]))
            fig_ost.add_trace(go.Bar(x=sim.months, y=sim.def_out[n],    name=f"{n}-Month Defaults",  marker_color=COLORS[n], opacity=0.5))
        else:
            fig_ost.add_trace(go.Bar(x=sim.months, y=sim.out_gross[n], name=f"{n}-Month", marker_color=COLORS[n]))
    fig_ost.add_trace(go.Scatter(x=sim.months, y=total, name="Total", mode="lines+markers", line=dict(color="black", width=2)))
    fig_ost.update_layout(barmode="stack", xaxis_title="Month", yaxis_title="Outstanding Principal ($)")
    st.plotly_chart(fig_ost, use_container_width=True)

    # Defaults chart
    df_def = pd.DataFrame({"Month": sim.months, **{f"{n}-Month Default": sim.default_by_tenor[n][sim.months] for n in TERMS}})
    fig_def = px.bar(df_def, x="Month", y=[f"{n}-Month Default" for n in TERMS], barmode="stack", labels={"value":"Defaults ($)"}, color_discrete_sequence=list(COLORS.values()))
    st.plotly_chart(fig_def, use_container_width=True)

    # Other charts reuse existing color logic...
    df_int = pd.DataFrame({"Month": sim.months, **{f"{n}-Month Interest": sim.interest_by_tenor[n][sim.months] for n in TERMS}})
    fig_int = px.bar(df_int, x="Month", y=[f"{n}-Month Interest" for n in TERMS], barmode="stack", color_discrete_sequence=list(COLORS.values()))
    st.plotly_chart(fig_int, use_container_width=True)

    df_prin = pd.DataFrame({"Month": sim.months, **{f"{n}-Month Principal": sim.principal_by_tenor[n][sim.months] for n in TERMS}})
    fig_prin = px.bar(df_prin, x="Month", y=[f"{n}-Month Principal" for n in TERMS], barmode="stack", color_discrete_sequence=list(COLORS.values()))
    st.plotly_chart(fig_prin, use_container_width=True)

    fig_net = go.Figure()
    for n in TERMS:
        fig_net.add_trace(go.Scatter(x=sim.months, y=sim.out_net[n], name=f"{n}-Month Net", mode="lines+markers", line=dict(color=COLORS[n])))
    fig_net.update_layout(xaxis_title="Month", yaxis_title="Net Outstanding ($)")
    st.plotly_chart(fig_net, use_container_width=True)

    df_yld = pd.DataFrame({"Tenor": [f"{n}-Month" for n in TERMS],
                           "Gross MOIC": [(1+params.monthly_interest)**n for n in TERMS],
                           "Net MOIC": [sim.payment_by_tenor[n].sum()/sim.new_by_t[:,i].sum() if sim.new_by_t[:,i].sum()>0 else 0 for i,n in enumerate(TERMS)]})
    fig_yld = px.bar(df_yld, x="Tenor", y=["Gross MOIC","Net MOIC"], barmode="group", color_discrete_sequence=list(COLORS.values()))
    st.plotly_chart(fig_yld, use_container_width=True)

    # Guided Cashflow Breakouts unchanged...
    df_origin=pd.DataFrame(sim.new_by_t,columns=[f"{n}-Month Orig" for n in TERMS]).assign(Month=sim.months)
    st.subheader("Step 1: New Originations by Tenor")
    with st.expander("Details and chart",expanded=False):
        st.dataframe(df_origin.set_index('Month').style.format("{:,.2f}"))
        st.plotly_chart(px.bar(df_origin,x="Month",y=[f"{n}-Month Orig" for n in TERMS]),use_container_width=True)

    df_inflow=pd.DataFrame({'Month':np.arange(len(sim.payment_schedule)),'Inflow':sim.payment_schedule})
    st.subheader("Step 2: Scheduled Cash Inflows")
    with st.expander("Details and trend",expanded=False):
        st.dataframe(df_inflow.set_index('Month').style.format("{:,.2f}"))
        fig_i=px.line(df_inflow,x="Month",y="Inflow",markers=True)
        fig_i.update_layout(yaxis_title="Cumulative Inflow",yaxis=dict(tickformat="$,"))
        st.plotly_chart(fig_i,use_container_width=True)

    df_metrics=pd.DataFrame({'Month':sim.months+1,'Defaults':sim.per_period_defaults,'Interest':sim.per_period_interest,'Reinvested':sim.per_period_principal})
    st.subheader("Step 3: Monthly Cashflow Components")
    with st.expander("Breakdown of defaults, interest, reinvestment",expanded=False):
        st.dataframe(df_metrics.set_index('Month').style.format("{:,.2f}"))
        st.plotly_chart(px.area(df_metrics,x="Month",y=['Defaults','Interest','Reinvested'],color_discrete_sequence=list(COLORS.values())),use_container_width=True)

    df_cum=df_metrics.copy()
    df_cum[['Cum Defaults','Cum Interest','Cum Reinvested']]=df_cum[['Defaults','Interest','Reinvested']].cumsum()
    st.subheader("Step 4: Cumulative Cashflow Tally")
    with st.expander("See cumulative build-up",expanded=False):
        st.dataframe(df_cum.set_index('Month').style.format("{:,.2f}"))
        st.plotly_chart(px.area(df_cum,x="Month",y=['Cum Defaults','Cum Interest','Cum Reinvested'],color_discrete_sequence=list(COLORS.values())),use_container_width=True)

def main():
    if st is None:
        print("Run this app with `streamlit run streamlit_app.py`")
        sys.exit(1)
    params = load_inputs()
    sim = run_simulation(params)
    build_and_render(params, sim)

if __name__ == "__main__":
    main()
