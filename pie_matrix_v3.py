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

# Color map for tenors
COLORS: Dict[int, str] = {
    1: "#e45756",
    2: "#4c78a8",
    3: "#f58518",
}

# Loan tenors (in months)
TERMS = [1, 2, 3]


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class InputParams:
    initial_capital: float
    monthly_interest: float       # as fraction, e.g. 0.12 for 12%
    num_months: int
    default_rates: Dict[int, float]   # e.g. {1:0.10,2:0.10,3:0.10}
    allocation: Dict[int, float]      # e.g. {1:0.50,2:0.25,3:0.25}
    show_def_overlay: bool


@dataclass
class SimulationResults:
    months: np.ndarray                    # shape (T,)
    new_by_t: np.ndarray                  # shape (T, 3)
    payment_schedule: np.ndarray          # shape (T + max_term + 1,)
    out_gross: Dict[int, np.ndarray]      # gross outstanding by tenor
    out_net: Dict[int, np.ndarray]        # net surviving by tenor
    def_out: Dict[int, np.ndarray]        # defaults by tenor
    defaults: List[float]                 # per-month default cashflows
    interest: List[float]                 # per-month interest cashflows
    reinvest: List[float]                 # per-month reinvested cashflows


# ─── Input Loading ────────────────────────────────────────────────────────────

def load_inputs() -> InputParams:
    """Render all sidebar widgets and return a populated InputParams."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("DubPrime.com Loan Portfolio Simulator")

    # Allocation sliders
    st.sidebar.header("Loan Allocation Settings")
    a1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    max2 = max(0, 100 - a1)
    a2 = st.sidebar.slider("2-Month Allocation (%)", 0, max2, min(25, max2)) if max2 else 0
    a3 = 100 - a1 - a2
    allocation = {1: a1 / 100, 2: a2 / 100, 3: a3 / 100}

    # Portfolio parameters
    st.sidebar.header("Portfolio Parameters")
    init_cap = st.sidebar.number_input("Initial Capital ($)", value=100_000)
    mon_int = st.sidebar.number_input("Monthly Interest (%)", value=12.0) / 100.0
    num_m = st.sidebar.slider("Duration (Months)", 1, 60, 12)

    # Default rates
    st.sidebar.header("Default Rates at Maturity")
    d1 = st.sidebar.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
    d2 = st.sidebar.slider("2-Month Default (%)", 0.0, 100.0, 10.0)
    d3 = st.sidebar.slider("3-Month Default (%)", 0.0, 100.0, 10.0)
    default_rates = {1: d1 / 100, 2: d2 / 100, 3: d3 / 100}

    # Overlay toggle
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
    """
    Given input parameters, simulate monthly cashflows,
    defaults, outstanding balances (gross & net).
    """
    r = params.monthly_interest
    d = params.default_rates
    p = params.allocation
    T = params.num_months

    # Level-payment factors for n>1
    pmt_factors = {n: r / (1 - (1 + r) ** (-n)) for n in TERMS if n > 1}
    # Periodic hazard rates from term-cumulative defaults
    hazard = {n: 1 - (1 - d[n]) ** (1 / n) for n in TERMS if n > 1}

    payment_schedule = np.zeros(T + max(TERMS) + 1)
    payment_schedule[0] = params.initial_capital
    new_by_t = np.zeros((T, len(TERMS)))
    defaults, interest, reinvest = [], [], []

    for t in range(T):
        cash = payment_schedule[t]
        new = cash * np.array([p[n] for n in TERMS])
        new_by_t[t, :] = new

        loss_acc = 0.0
        int_acc = 0.0
        reinv_acc = 0.0

        # TENOR = 1 (bullet)
        P1 = new[0]
        if P1 > 0:
            loss1 = d[1] * P1
            recov1 = P1 - loss1
            int1 = r * recov1
            payment_schedule[t + 1] += recov1 + int1
            loss_acc += loss1
            int_acc += int1
            reinv_acc += recov1 + int1

        # TENOR > 1 (level-payment with hazard)
        for idx, n in enumerate(TERMS[1:], start=1):
            Pn = new[idx]
            if Pn <= 0:
                continue
            pf = pmt_factors[n]
            hz = hazard[n]
            remaining = Pn
            pmt = pf * Pn
            for k in range(1, n + 1):
                loss_k = hz * remaining
                remaining -= loss_k
                int_k = r * remaining
                prin_k = pmt - int_k
                remaining -= prin_k
                payment_schedule[t + k] += prin_k + int_k
                loss_acc += loss_k
                int_acc += int_k
                reinv_acc += prin_k + int_k

        defaults.append(loss_acc)
        interest.append(int_acc)
        reinvest.append(reinv_acc)

    months = np.arange(T)
    # Precompute survival weights
    weights_gross: Dict[int, List[float]] = {1: [1, 0]}
    weights_net:   Dict[int, List[float]] = {1: [1 - d[1], 0]}
    for n in TERMS[1:]:
        wg = [(1 + r) ** k - (((1 + r) ** k - 1) * pmt_factors[n] / r) for k in range(n + 1)]
        wn = [wg[k] * (1 - hazard[n]) ** k for k in range(n + 1)]
        weights_gross[n] = wg
        weights_net[n] = wn

    # Build outstanding schedules
    out_gross, out_net, def_out = {}, {}, {}
    for idx, n in enumerate(TERMS):
        out_gross[n] = np.array([
            sum(new_by_t[j, idx] * weights_gross[n][min(t - j, n)] for j in range(t + 1))
            for t in months
        ])
        out_net[n] = np.array([
            sum(new_by_t[j, idx] * weights_net[n][min(t - j, n)] for j in range(t + 1))
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
    )


# ─── Figure Building ─────────────────────────────────────────────────────────

def build_figures(params: InputParams, sim: SimulationResults) -> Dict[str, go.Figure]:
    """Generate all Plotly figures needed in the dashboard."""
    figs: Dict[str, go.Figure] = {}

    # Allocation Pie
    df_alloc = pd.DataFrame({
        "Tenor": [f"{n}-Month" for n in TERMS],
        "Allocation": [params.allocation[n] * 100 for n in TERMS]
    })
    figs["alloc"] = px.pie(
        df_alloc, names="Tenor", values="Allocation", title="Allocation Mix",
        color_discrete_sequence=list(COLORS.values())
    )

    # Outstanding (stacked) + Total
    fig_ost = go.Figure()
    total = sum(sim.out_gross.values())
    for n in TERMS:
        if params.show_def_overlay:
            fig_ost.add_trace(go.Bar(
                x=sim.months, y=sim.out_net[n],
                name=f"{n}-Month Surviving", marker_color=COLORS[n]
            ))
            fig_ost.add_trace(go.Bar(
                x=sim.months, y=sim.def_out[n],
                name=f"{n}-Month Defaults", marker_color=COLORS[n], opacity=0.5
            ))
        else:
            fig_ost.add_trace(go.Bar(
                x=sim.months, y=sim.out_gross[n],
                name=f"{n}-Month", marker_color=COLORS[n]
            ))
    fig_ost.add_trace(go.Scatter(
        x=sim.months, y=total,
        name="Total", mode="lines+markers",
        line=dict(color="black", width=2)
    ))
    fig_ost.update_layout(
        barmode="stack",
        xaxis_title="Month",
        yaxis_title="Outstanding Principal ($)"
    )
    figs["outstanding"] = fig_ost

    # Defaults by Tenor
    df_def = pd.DataFrame({
        "Month": sim.months,
        **{f"{n}-Month Default": sim.def_out[n] for n in TERMS}
    })
    figs["defaults"] = px.bar(
        df_def, x="Month",
        y=[f"{n}-Month Default" for n in TERMS],
        title="Defaults by Tenor",
        barmode="stack",
        labels={"value": "Defaults ($)", "variable": "Tenor"}
    )

    # Growth & Risk Lines
    fig_growth = go.Figure()
    for n in TERMS:
        fig_growth.add_trace(go.Scatter(
            x=sim.months, y=sim.out_gross[n],
            name=f"{n}-Month", mode="lines+markers",
            line=dict(color=COLORS[n]), marker=dict(color=COLORS[n])
        ))
    fig_growth.add_trace(go.Scatter(
        x=sim.months, y=total,
        name="Total", mode="lines+markers",
        line=dict(color="black", dash="dash"), marker=dict(color="black")
    ))
    fig_growth.update_layout(
        xaxis_title="Month",
        yaxis_title="Outstanding Principal ($)",
        legend=dict(title="", traceorder="normal")
    )
    figs["growth"] = fig_growth

    return figs


# ─── Rendering ────────────────────────────────────────────────────────────────

def render_dashboard(params: InputParams, sim: SimulationResults, figs: Dict[str, go.Figure]) -> None:
    """Lay out all charts, tables, and key metrics in Streamlit."""
    # Sidebar pie
    st.sidebar.plotly_chart(figs["alloc"], use_container_width=True)

    # Main charts
    st.subheader("Outstanding by Tenor (Stacked) + Total")
    st.plotly_chart(figs["outstanding"], use_container_width=True)

    st.subheader("Defaults by Tenor (Stacked)")
    st.plotly_chart(figs["defaults"], use_container_width=True)

    st.subheader("Portfolio Growth and Risk by Tenor")
    st.plotly_chart(figs["growth"], use_container_width=True)

    # Guided Breakouts
    df_orig = pd.DataFrame(
        sim.new_by_t,
        columns=[f"{n}-Month Orig" for n in TERMS]
    ).assign(Month=sim.months)
    df_inf = pd.DataFrame({
        "Month": np.arange(len(sim.payment_schedule)),
        "Inflow": sim.payment_schedule
    })
    df_met = pd.DataFrame({
        "Month": sim.months + 1,
        "Defaults": sim.defaults,
        "Interest": sim.interest,
        "Reinvested": sim.reinvest
    })
    df_cum = df_met.copy()
    df_cum[["Cum Defaults", "Cum Interest", "Cum Reinvested"]] = df_met[
        ["Defaults", "Interest", "Reinvested"]
    ].cumsum()

    # Step 1
    st.subheader("Step 1: New Originations by Tenor")
    with st.expander("Details and chart", expanded=False):
        st.dataframe(df_orig.set_index("Month").style.format("{:,.2f}"))
        st.plotly_chart(px.bar(df_orig, x="Month", y=[f"{n}-Month Orig" for n in TERMS]),
                        use_container_width=True)

    # Step 2
    st.subheader("Step 2: Scheduled Cash Inflows")
    with st.expander("Details and trend", expanded=False):
        st.dataframe(df_inf.set_index("Month").style.format("{:,.2f}"))
        fig_i = px.line(df_inf, x="Month", y="Inflow", markers=True)
        fig_i.update_layout(
            yaxis_title="Cumulative Inflow",
            yaxis=dict(tickformat="$,")
        )
        st.plotly_chart(fig_i, use_container_width=True)

    # Step 3
    st.subheader("Step 3: Monthly Cashflow Components")
    with st.expander("Breakdown of defaults, interest, reinvestment", expanded=False):
        st.dataframe(df_met.set_index("Month").style.format("{:,.2f}"))
        st.plotly_chart(px.area(df_met, x="Month", y=["Defaults", "Interest", "Reinvested"]),
                        use_container_width=True)

    # Step 4
    st.subheader("Step 4: Cumulative Cashflow Tally")
    with st.expander("See cumulative build-up", expanded=False):
        st.dataframe(df_cum.set_index("Month").style.format("{:,.2f}"))
        st.plotly_chart(px.area(df_cum, x="Month", y=["Cum Defaults", "Cum Interest", "Cum Reinvested"]),
                        use_container_width=True)

    # Key Metrics (MOIC)
    total_int = sum(sim.interest)
    total_def = sum(sim.defaults)
    total_reinv = sum(sim.reinvest)
    gross_moic = (total_int + params.initial_capital) / params.initial_capital
    net_moic = (total_int + params.initial_capital - total_def) / params.initial_capital

    st.subheader("Key Metrics (MOIC)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gross MOIC", f"{gross_moic:.2f}x")
    c2.metric("Net MOIC",   f"{net_moic:.2f}x")
    c3.metric("Initial Capital", f"${params.initial_capital:,.0f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Total Interest",   f"${total_int:,.0f}")
    c5.metric("Total Defaults",   f"${total_def:,.0f}")
    c6.metric("Total Reinvested", f"${total_reinv:,.0f}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    if st is None:
        print("Run this app via `streamlit run streamlit_app.py` after installing Streamlit.")
        sys.exit(1)

    params = load_inputs()
    sim = run_simulation(params)
    figs = build_figures(params, sim)
    render_dashboard(params, sim, figs)


if __name__ == "__main__":
    main()
