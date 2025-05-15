
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Unified Yield Simulator", layout="wide")
st.title("🧠 Unified Yield Simulator")

SCENARIO_DIR = "scenarios"
os.makedirs(SCENARIO_DIR, exist_ok=True)

view = st.sidebar.radio("🧭 Navigation", ["📊 Simulator", "📈 Scenario Comparison"])

def load_scenario(name):
    with open(f"{SCENARIO_DIR}/{name}.json", "r") as f:
        return json.load(f)

if view == "📊 Simulator":
    st.header("📊 Capital Yield Simulator")

    scenario_files = [f[:-5] for f in os.listdir(SCENARIO_DIR) if f.endswith(".json")]
    selected_scenario = st.sidebar.selectbox("📂 Load Scenario", [""] + scenario_files)
    if selected_scenario and st.sidebar.button("🔁 Load"):
        s = load_scenario(selected_scenario)
        for key in s: st.session_state[key] = s[key]
        st.rerun()

    interest_1m = st.sidebar.slider("Interest per 1M Loan", 0.01, 0.10, st.session_state.get("interest_1m", 0.03), 0.005)
    interest_2m = st.sidebar.slider("Interest per 2M Loan", 0.01, 0.15, st.session_state.get("interest_2m", 0.06), 0.005)
    interest_3m = st.sidebar.slider("Interest per 3M Loan", 0.01, 0.25, st.session_state.get("interest_3m", 0.12), 0.005)

    share_1m = st.sidebar.slider("Share of 1M Loans", 0.0, 1.0, st.session_state.get("share_1m", 0.3), 0.01)
    share_2m = st.sidebar.slider("Share of 2M Loans", 0.0, 1.0 - share_1m, st.session_state.get("share_2m", 0.4), 0.01)
    share_3m = max(0.0, 1.0 - share_1m - share_2m)

    cost_of_capital = st.sidebar.slider("Cost of Capital", 0.01, 0.25, st.session_state.get("cost_of_capital", 0.08), 0.005)
    default_rate = st.sidebar.slider("Expected Default Rate", 0.00, 0.20, st.session_state.get("default_rate", 0.02), 0.005)

    scenario_name = st.sidebar.text_input("📄 Scenario Name")
    if st.sidebar.button("💾 Save Scenario") and scenario_name:
        scenario = {
            "interest_1m": interest_1m, "interest_2m": interest_2m, "interest_3m": interest_3m,
            "share_1m": share_1m, "share_2m": share_2m,
            "cost_of_capital": cost_of_capital, "default_rate": default_rate
        }
        with open(f"{SCENARIO_DIR}/{scenario_name}.json", "w") as f: json.dump(scenario, f)
        st.success(f"Scenario '{scenario_name}' saved.")

    loan_durations = [1, 2, 3]
    interest_rates = [interest_1m, interest_2m, interest_3m]
    shares = [share_1m, share_2m, share_3m]
    cycles_per_year = [12 / d for d in loan_durations]

    st.subheader("🧾 Assumptions Summary")
    st.write(pd.DataFrame({
        "Loan Type": ["1M", "2M", "3M"],
        "Share": shares,
        "Interest": interest_rates,
        "Duration": loan_durations,
        "Cycles/Year": cycles_per_year
    }))

    # Audit trail
    wd = sum([shares[i] * loan_durations[i] for i in range(3)])
    turnover = 12 / wd
    ip_loan = sum([shares[i] * interest_rates[i] for i in range(3)])
    gross = ip_loan * turnover
    loss = gross * default_rate
    after_loss = gross - loss
    net = after_loss - cost_of_capital

    audit_df = pd.DataFrame({
        "Metric": ["Weighted Duration", "Turnover", "Interest per Loan", "Gross Yield", "Loss", "After Defaults", "Capital Cost", "Net Yield"],
        "Value": [wd, turnover, ip_loan, gross, loss, after_loss, cost_of_capital, net]
    })
    st.subheader("📋 Audit Trail")
    st.table(audit_df)

    # Export CSV
    cohort_df = pd.DataFrame({
        "Loan Type": ["1M", "2M", "3M"],
        "Share": shares,
        "Interest": interest_rates,
        "Duration": loan_durations,
        "Cycles/Year": cycles_per_year,
        "Annualized Yield": [interest_rates[i] * cycles_per_year[i] for i in range(3)],
        "Weighted Yield Contribution": [shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)]
    })
    csv = cohort_df.to_csv(index=False)
    st.download_button("📤 Download Cohort Table (CSV)", data=csv, file_name="cohort_yield_table.csv", mime="text/csv")

    # Pie chart
    st.subheader("🥧 Loan Mix")
    st.plotly_chart(go.Figure(go.Pie(labels=["1M", "2M", "3M"], values=shares, hole=0.4)), use_container_width=True)

    # Contour plot
    st.subheader("📈 Net Yield Contour")
    s1_vals, s3_vals = np.linspace(0.01, 0.99, 30), np.linspace(0.01, 0.99, 30)
    z_matrix = []
    for s1 in s1_vals:
        row = []
        for s3 in s3_vals:
            s2 = 1 - s1 - s3
            if s2 < 0 or s2 > 1: row.append(None); continue
            sh = [s1, s2, s3]
            d = sum([sh[i] * loan_durations[i] for i in range(3)])
            t = 12 / d
            i = sum([sh[i] * interest_rates[i] for i in range(3)])
            g = i * t
            ny = (1 - default_rate) * g - cost_of_capital
            row.append(ny)
        z_matrix.append(row)
    st.plotly_chart(go.Figure(go.Contour(z=z_matrix, x=s3_vals, y=s1_vals, colorscale='Viridis', colorbar_title="Net Yield")), use_container_width=True)

    # Waterfall
    st.subheader("📉 Yield Breakdown")
    y_vals = [gross, -loss, -cost_of_capital, net]
    label_texts = [f"{v:.2%}" for v in y_vals]
    wf_fig = go.Figure(go.Waterfall(
        x=["Gross", "Defaults", "Cost", "Net"],
        y=y_vals,
        measure=["relative", "relative", "relative", "total"],
        text=label_texts, textposition="outside"
    ))
    st.plotly_chart(wf_fig, use_container_width=True)

    # 3D Surface
    st.subheader("🧠 Yield Surface")
    I, D = np.meshgrid(np.linspace(0.01, 0.25, 50), np.linspace(1, 12, 50))
    Z = I * (12 / D)
    st.plotly_chart(go.Figure(go.Surface(z=Z, x=I[0], y=D[:,0], colorscale='Viridis')), use_container_width=True)

elif view == "📈 Scenario Comparison":
    scenario_files = [f[:-5] for f in os.listdir(SCENARIO_DIR) if f.endswith(".json")]
    if len(scenario_files) < 2:
        st.warning("Save at least two scenarios to use comparison view.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        scenario_a = st.selectbox("Scenario A", scenario_files, key="a")
    with col2:
        scenario_b = st.selectbox("Scenario B", scenario_files, key="b")
    if scenario_a == scenario_b:
        st.warning("Please select different scenarios.")
        st.stop()

    def summary(s):
        sh = [s["share_1m"], s["share_2m"], 1 - s["share_1m"] - s["share_2m"]]
        rt = [s["interest_1m"], s["interest_2m"], s["interest_3m"]]
        d = [1, 2, 3]
        wd = sum([sh[i] * d[i] for i in range(3)])
        t = 12 / wd
        i = sum([sh[i] * rt[i] for i in range(3)])
        g = i * t
        l = g * s["default_rate"]
        a = g - l
        n = a - s["cost_of_capital"]
        return {"shares": sh, "rates": rt, "net": n, "gross": g, "wd": wd, "t": t}

    A, B = load_scenario(scenario_a), load_scenario(scenario_b)
    sA, sB = summary(A), summary(B)

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"**{scenario_a}**")
        st.plotly_chart(go.Figure(go.Pie(labels=["1M", "2M", "3M"], values=sA["shares"], hole=0.4)), use_container_width=True)
    with colB:
        st.markdown(f"**{scenario_b}**")
        st.plotly_chart(go.Figure(go.Pie(labels=["1M", "2M", "3M"], values=sB["shares"], hole=0.4)), use_container_width=True)

    bar = go.Figure()
    bar.add_trace(go.Bar(name=scenario_a, x=["Net Yield"], y=[sA["net"]]))
    bar.add_trace(go.Bar(name=scenario_b, x=["Net Yield"], y=[sB["net"]]))
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("📋 Delta Comparison")
    st.write(pd.DataFrame({
        "Metric": ["Net Yield", "Gross Yield", "Duration", "Turnover"],
        scenario_a: [sA["net"], sA["gross"], sA["wd"], sA["t"]],
        scenario_b: [sB["net"], sB["gross"], sB["wd"], sB["t"]],
        "Delta": [sA["net"] - sB["net"], sA["gross"] - sB["gross"], sA["wd"] - sB["wd"], sA["t"] - sB["t"]]
    }).set_index("Metric").style.format("{:.2%}"))
