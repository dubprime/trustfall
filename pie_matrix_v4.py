# streamlit_app.py

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None
    import warnings
    warnings.warn(
        "Streamlit is not installed. Please install it with `pip install streamlit`."
    )

import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    defaults: List[float]
    interest: List[float]
    reinvest: List[float]
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

    # allocation sliders
    st.sidebar.header("Loan Allocation Settings")
    a1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    max2 = max(0, 100 - a1)
    a2 = st.sidebar.slider("2-Month Allocation (%)", 0, max2, min(25, max2)) if max2 else 0
    a3 = 100 - a1 - a2
    allocation = {1: a1 / 100, 2: a2 / 100, 3: a3 / 100}

    # portfolio parameters
    st.sidebar.header("Portfolio Parameters")
    init_cap = st.sidebar.number_input("Initial Capital ($)", value=100_000)
    mon_int = st.sidebar.number_input("Monthly Interest (%)", value=12.0) / 100
    num_m = st.sidebar.slider("Duration (Months)", 1, 60, 12)

    # default rates at maturity
    st.sidebar.header("Default Rates at Maturity")
    d1 = st.sidebar.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
    d2 = st.sidebar.slider("2-Month Default (%)", 0.0, 100.0, 10.0)
    d3 = st.sidebar.slider("3-Month Default (%)", 0.0, 100.0, 10.0)
    default_rates = {1: d1 / 100, 2: d2 / 100, 3: d3 / 100}

    # view toggle
    overlay = st.sidebar.checkbox("Overlay defaults within gross view", value=False)

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
    r = params.monthly_interest
    d = params.default_rates
    p = params.allocation
    T = params.num_months

    # schedule arrays
    payment_schedule = np.zeros(T + max(TERMS) + 1)
    payment_schedule[0] = params.initial_capital
    new_by_t = np.zeros((T, len(TERMS)))

    # per-tenor cashflow and loss trackers (length = schedule length)
    payment_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    interest_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    principal_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    default_by_tenor   = {n: np.zeros_like(payment_schedule) for n in TERMS}

    # factors
    pmt_factors = {n: r / (1 - (1 + r)**(-n)) for n in TERMS if n > 1}
    hazard = {n: 1 - (1 - d[n])**(1 / n) for n in TERMS if n > 1}

    defaults, interest, reinvest = [], [], []

    for t in range(T):
        cash = payment_schedule[t]
        deployed = np.array([p[n] * cash for n in TERMS])
        new_by_t[t, :] = deployed

        loss_acc = int_acc = reinv_acc = 0.0

        # bullet (1-month)
        P1 = deployed[0]
        if P1 > 0:
            loss1 = d[1] * P1
            default_by_tenor[1][t] += loss1
            recov1 = P1 - loss1
            int1 = r * recov1
            idx = t + 1
            payment_schedule[idx] += recov1 + int1
            payment_by_tenor[1][idx] += recov1 + int1
            interest_by_tenor[1][idx] += int1
            principal_by_tenor[1][idx] += recov1
            loss_acc += loss1
            int_acc += int1
            reinv_acc += recov1 + int1

        # level-payment (multi-month)
        for i, n in enumerate(TERMS[1:], start=1):
            Pn = deployed[i]
            if Pn <= 0:
                continue
            pf = pmt_factors[n]
            hz = hazard[n]
            remaining = Pn
            pmt = pf * Pn
            for k in range(1, n + 1):
                loss_k = hz * remaining
                remaining -= loss_k
                default_by_tenor[n][t + k] += loss_k
                int_k = r * remaining
                prin_k = pmt - int_k
                remaining -= prin_k
                idx2 = t + k
                payment_schedule[idx2] += prin_k + int_k
                payment_by_tenor[n][idx2] += prin_k + int_k
                interest_by_tenor[n][idx2] += int_k
                principal_by_tenor[n][idx2] += prin_k
                loss_acc += loss_k
                int_acc += int_k
                reinv_acc += prin_k + int_k

        defaults.append(loss_acc)
        interest.append(int_acc)
        reinvest.append(reinv_acc)

    months = np.arange(T)

    # outstanding schedules
    weights_gross = {1: [1, 0]}
    weights_net   = {1: [1 - d[1], 0]}
    for n in TERMS[1:]:
        wg = [(1 + r)**k - (((1 + r)**k - 1) * pmt_factors[n] / r) for k in range(n+1)]
        wn = [wg[k] * (1 - hazard[n])**k for k in range(n+1)]
        weights_gross[n], weights_net[n] = wg, wn

    out_gross, out_net, def_out = {}, {}, {}
    for idx, n in enumerate(TERMS):
        # gross outstanding
        out_gross[n] = np.array([
            sum(new_by_t[j, idx] * weights_gross[n][min(t-j, n)] for j in range(t+1))
            for t in months
        ])
        # net surviving
        out_net[n] = np.array([
            sum(new_by_t[j, idx] * weights_net[n][min(t-j, n)] for j in range(t+1))
            for t in months
        ])
        def_out[n] = out_gross[n] - out_net[n]

    return SimulationResults(
        months=months,
        new_by_t=new_by_t,
        payment_schedule=payment_schedule,
        out_gross=out_gross,
        out_net=out_net,
        def_out=def_out,
        defaults=defaults,
        interest=interest,
        reinvest=reinvest,
        payment_by_tenor=payment_by_tenor,
        interest_by_tenor=interest_by_tenor,
        principal_by_tenor=principal_by_tenor,
        default_by_tenor=default_by_tenor,
        pmt_factors=pmt_factors,
    )

# ─── Figure Building ─────────────────────────────────────────────────────────

def build_figures(params: InputParams, sim: SimulationResults) -> Dict[str, go.Figure]:
    figs: Dict[str, go.Figure] = {}

    # pie
    df_alloc = pd.DataFrame({
        "Tenor": [f"{n}-Month" for n in TERMS],
        "Allocation": [params.allocation[n] * 100 for n in TERMS]
    })
    figs["alloc"] = px.pie(
        df_alloc, names="Tenor", values="Allocation",
        title="Allocation Mix",
        color_discrete_sequence=list(COLORS.values())
    )

    # outstanding
    total_gross = sum(sim.out_gross.values())
    fig_ost = go.Figure()
    for n in TERMS:
        if params.show_def_overlay:
            fig_ost.add_trace(go.Bar(
                x=sim.months, y=sim.out_net[n], name=f"{n}-Month Surviving", marker_color=COLORS[n]
            ))
            fig_ost.add_trace(go.Bar(
                x=sim.months, y=sim.def_out[n], name=f"{n}-Month Defaults", marker_color=COLORS[n], opacity=0.5
            ))
        else:
            fig_ost.add_trace(go.Bar(
                x=sim.months, y=sim.out_gross[n], name=f"{n}-Month", marker_color=COLORS[n]
            ))
    fig_ost.add_trace(go.Scatter(
        x=sim.months, y=total_gross, name="Total", mode="lines+markers",
        line=dict(color="black", width=2)
    ))
    fig_ost.update_layout(
        barmode="stack",
        xaxis_title="Month",
        yaxis_title="Outstanding Principal ($)"
    )
    figs["outstanding"] = fig_ost

    # defaults
    df_def = pd.DataFrame({
        "Month": sim.months,
        **{f"{n}-Month Default": sim.default_by_tenor[n][sim.months] for n in TERMS}
    })
    figs["defaults"] = px.bar(
        df_def, x="Month",
        y=[f"{n}-Month Default" for n in TERMS],
        title="Defaults by Tenor", barmode="stack",
        labels={"value": "Defaults ($)", "variable": "Tenor"}
    )

    # interest cashflows
    df_int = pd.DataFrame({
        "Month": sim.months,
        **{f"{n}-Month Interest": sim.interest_by_tenor[n][sim.months] for n in TERMS}
    })
    figs["interest"] = px.bar(
        df_int, x="Month",
        y=[f"{n}-Month Interest" for n in TERMS],
        title="Monthly Interest Cashflows by Tenor", barmode="stack"
    )

    # principal repaid
    df_prin = pd.DataFrame({
        "Month": sim.months,
        **{f"{n}-Month Principal": sim.principal_by_tenor[n][sim.months] for n in TERMS}
    })
    figs["principal"] = px.bar(
        df_prin, x="Month",
        y=[f"{n}-Month Principal" for n in TERMS],
        title="Monthly Principal Repaid by Tenor", barmode="stack"
    )

    # net outstanding
    fig_net = go.Figure()
    for n in TERMS:
        fig_net.add_trace(go.Scatter(
            x=sim.months, y=sim.out_net[n],
            name=f"{n}-Month Net", mode="lines+markers", line=dict(color=COLORS[n])
        ))
    fig_net.update_layout(
        xaxis_title="Month",
        yaxis_title="Net Outstanding ($)"
    )
    figs["net_out"] = fig_net

    # yield curve
    deployed = {n: sim.new_by_t[:, i].sum() for i, n in enumerate(TERMS)}
    gross_moic = {}
    net_moic   = {}
    for n in TERMS:
        # gross: no-default cashflow per issuance
        if n == 1:
            gross_moic[n] = 1 + params.monthly_interest
        else:
            gross_moic[n] = sim.pmt_factors[n] * n
        # net: actual cash returned / deployed
        total_return_n = sim.payment_by_tenor[n].sum()
        net_moic[n] = total_return_n / deployed[n] if deployed[n] > 0 else 0

    df_yield = pd.DataFrame({
        "Tenor": [f"{n}-Month" for n in TERMS],
        "Gross MOIC": [gross_moic[n] for n in TERMS],
        "Net MOIC":   [net_moic[n]   for n in TERMS],
    })
    figs["yield_curve"] = px.bar(
        df_yield, x="Tenor", y=["Gross MOIC", "Net MOIC"],
        title="Yield Curve (Gross vs Net MOIC)", barmode="group"
    )

    return figs

# ─── Rendering ────────────────────────────────────────────────────────────────

def render_dashboard(params: InputParams, sim: SimulationResults, figs: Dict[str, go.Figure]) -> None:
    if st is None:
        print("Run this app with `streamlit run streamlit_app.py`")
        return

    st.sidebar.plotly_chart(figs["alloc"], use_container_width=True)

    st.subheader("Outstanding by Tenor (Stacked) + Total")
    st.plotly_chart(figs["outstanding"], use_container_width=True)

    st.subheader("Defaults by Tenor (Stacked)")
    st.plotly_chart(figs["defaults"], use_container_width=True)

    st.subheader("Monthly Interest Cashflows by Tenor")
    st.plotly_chart(figs["interest"], use_container_width=True)

    st.subheader("Monthly Principal Repaid by Tenor")
    st.plotly_chart(figs["principal"], use_container_width=True)

    st.subheader("Net Outstanding by Tenor")
    st.plotly_chart(figs["net_out"], use_container_width=True)

    st.subheader("Yield Curve (Gross vs Net MOIC)")
    st.plotly_chart(figs["yield_curve"], use_container_width=True)

# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    if st is None:
        print("Run this app with `streamlit run streamlit_app.py`")
        sys.exit(1)
    params = load_inputs()
    sim    = run_simulation(params)
    figs   = build_figures(params, sim)
    render_dashboard(params, sim, figs)

if __name__ == "__main__":
    main()
