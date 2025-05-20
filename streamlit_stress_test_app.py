import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

from pie_matrix import InputParams, SimulationResults, run_simulation
from financial_metrics import compute_extended_metrics, format_metrics_for_display, ExtendedFinancialMetrics
from stress_testing import (
    StressTestScenario, run_stress_test, run_multiple_stress_tests,
    get_predefined_scenarios, generate_stress_test_report,
    plot_stress_test_impacts, run_sensitivity_analysis,
    plot_sensitivity_analysis
)

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DubPrime Stress Test Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container { 
        max-width:100% !important; 
        padding-left:2rem; 
        padding-right:2rem; 
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: bold;
        color: #555;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .negative {
        color: #ff4b4b;
    }
    .positive {
        color: #00cc96;
    }
    .neutral {
        color: #636efa;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helper functions ──────────────────────────────────────────────────────
def display_scenario_metrics(metrics: ExtendedFinancialMetrics, title: str):
    """Display key financial metrics in a nice format."""
    formatted = format_metrics_for_display(metrics)
    
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio IRR</div>
            <div class="metric-value">{formatted["Portfolio IRR"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weighted Average Life</div>
            <div class="metric-value">{formatted["Portfolio WAL"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Concentration Risk</div>
            <div class="metric-value">{formatted["Concentration Risk"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cash Flow Volatility</div>
            <div class="metric-value">{formatted["Cash Flow Volatility"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recovery_vals = formatted["Recovery Rates (%)"]
        avg_recovery = sum([float(v.strip('%')) for v in recovery_vals.values()]) / len(recovery_vals)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Recovery Rate</div>
            <div class="metric-value">{avg_recovery:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        irrs = formatted["IRR (%)"]
        tenor_irrs = "<br>".join([f"{k}: {v}" for k, v in irrs.items()])
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IRR by Tenor</div>
            <div style="font-size: 1rem; font-weight: bold;">{tenor_irrs}</div>
        </div>
        """, unsafe_allow_html=True)

def display_impact_summary(impact_summary: Dict[str, float]):
    """Display the impact summary with color coding."""
    st.subheader("Impact Summary")
    
    cols = st.columns(len(impact_summary))
    
    for i, (metric, value) in enumerate(impact_summary.items()):
        # Determine color based on value and metric
        if metric == "Default Impact (%)" or metric == "WAL Change (%)":
            # For these metrics, negative is bad
            color_class = "negative" if value < 0 else "positive"
        else:
            # For IRR and other metrics, positive is good
            color_class = "positive" if value > 0 else "negative"
        
        # If close to zero, use neutral
        if abs(value) < 1.0:
            color_class = "neutral"
        
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{metric}</div>
                <div class="metric-value {color_class}">{value:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

def create_custom_scenario() -> StressTestScenario:
    """Create a custom stress test scenario based on user inputs."""
    st.subheader("Define Custom Stress Scenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Scenario Name", value="Custom Scenario")
        description = st.text_area("Description", value="User-defined stress scenario")
        default_multiplier = st.slider("Default Rate Multiplier", 0.5, 5.0, 1.0, 0.1)
    
    with col2:
        interest_delta = st.slider("Interest Rate Change (percentage points)", -5.0, 5.0, 0.0, 0.5) / 100
        
        st.write("Allocation Shifts")
        shift_1m = st.slider("1-Month Allocation Shift", -0.5, 0.5, 0.0, 0.05)
        shift_3m = st.slider("3-Month Allocation Shift", -0.5, 0.5, 0.0, 0.05)
        
        # Balance the shifts - if user increases 1m and 3m, reduce 2m accordingly
        shift_2m = -(shift_1m + shift_3m)
        st.write(f"2-Month Allocation Shift: {shift_2m:.2f}")
    
    allocation_shift = {1: shift_1m, 2: shift_2m, 3: shift_3m}
    
    return StressTestScenario(
        name=name,
        description=description,
        default_rate_multiplier=default_multiplier,
        interest_rate_delta=interest_delta,
        allocation_shift=allocation_shift
    )

def plot_tenor_metrics(metrics: Dict[str, ExtendedFinancialMetrics], metric_name: str):
    """Plot a specific metric across tenors for different scenarios."""
    # Extract data for plotting
    data = []
    
    for scenario_name, scenario_metrics in metrics.items():
        if metric_name == "IRR":
            values = scenario_metrics.irr
        elif metric_name == "Recovery Rates":
            values = scenario_metrics.recovery_rates
        elif metric_name == "WAL":
            values = scenario_metrics.weighted_average_life
        elif metric_name == "Duration":
            values = scenario_metrics.duration
        elif metric_name == "Convexity":
            values = scenario_metrics.convexity
        else:
            return None
        
        for tenor, value in values.items():
            data.append({
                "Scenario": scenario_name,
                "Tenor": f"{tenor}-Month",
                "Value": value
            })
    
    df = pd.DataFrame(data)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x="Tenor", 
        y="Value", 
        color="Scenario",
        barmode="group",
        title=f"{metric_name} by Tenor Across Scenarios"
    )
    
    return fig

# ─── Main app ──────────────────────────────────────────────────────────────
def main():
    st.title("DubPrime Stress Test Simulator")
    st.write("Analyze portfolio performance under various stress scenarios")
    
    # ─── Sidebar: Portfolio Parameters ───────────────────────────────────────
    st.sidebar.header("Portfolio Parameters")
    
    init_cap = st.sidebar.number_input("Initial Capital ($)", value=100_000)
    mon_int = st.sidebar.number_input("Monthly Interest (%)", value=12.0)/100
    num_m = st.sidebar.slider("Duration (Months)", 1, 60, 12)
    
    st.sidebar.subheader("Default Rates")
    d1 = st.sidebar.slider("1-Month Default (%)", 0.0, 100.0, 10.0)
    d2 = st.sidebar.slider("2-Month Default (%)", 0.0, 100.0, 10.0)
    d3 = st.sidebar.slider("3-Month Default (%)", 0.0, 100.0, 10.0)
    default_rates = {1: d1/100, 2: d2/100, 3: d3/100}
    
    st.sidebar.subheader("Allocation Mix")
    a1 = st.sidebar.slider("1-Month Allocation (%)", 0, 100, 50)
    max2 = max(0, 100 - a1)
    a2 = st.sidebar.slider("2-Month Allocation (%)", 0, max2, min(25, max2)) if max2 else 0
    a3 = 100 - a1 - a2
    allocation = {1: a1/100, 2: a2/100, 3: a3/100}
    
    # Create pie chart of allocation
    df_alloc = pd.DataFrame({
        "Tenor": [f"{n}-Month" for n in [1, 2, 3]],
        "Allocation": [allocation[n]*100 for n in [1, 2, 3]]
    })
    fig_alloc = px.pie(
        df_alloc, names="Tenor", values="Allocation",
        title="Portfolio Allocation",
        color_discrete_sequence=["#e45756", "#4c78a8", "#f58518"]
    )
    st.sidebar.plotly_chart(fig_alloc, use_container_width=True)
    
    # Input params object
    base_params = InputParams(
        initial_capital=init_cap,
        monthly_interest=mon_int,
        num_months=num_m,
        default_rates=default_rates,
        allocation=allocation,
        show_def_overlay=True
    )
    
    # ─── Main Content: Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Base Portfolio", 
        "Predefined Scenarios",
        "Custom Scenario",
        "Sensitivity Analysis"
    ])
    
    # Run base simulation and compute metrics (only once)
    with st.spinner("Running base simulation..."):
        base_sim = run_simulation(base_params)
        base_metrics = compute_extended_metrics(base_params, base_sim)
    
    # Tab 1: Base Portfolio
    with tab1:
        display_scenario_metrics(base_metrics, "Base Portfolio Metrics")
        
        # Add explanation about IRR differences
        st.markdown("""
        <details>
        <summary style="font-weight: bold; cursor: pointer; color: #4c78a8;">Why is portfolio IRR higher than tenor-specific IRRs? (click to expand)</summary>
        <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-top: 0.5rem;">
        <p>The discrepancy between portfolio IRR and tenor-specific IRRs occurs due to several financial mechanics:</p>
        <ul>
            <li><strong>Compounding effect:</strong> The portfolio IRR captures the compounding effect of continuously reinvesting returns over the full simulation period. Each individual tenor IRR only measures the return for a single lending cycle.</li>
            <li><strong>Reinvestment at higher rates:</strong> When principal is repaid from shorter tenors, it gets reinvested across all tenors again at the full interest rate, creating a powerful compounding effect.</li>
            <li><strong>Time horizon difference:</strong> The tenor IRRs represent individual investment cycles, while the portfolio IRR represents the entire simulation period with multiple reinvestment cycles.</li>
            <li><strong>Cash flow structure:</strong> The portfolio IRR calculation uses the complete cash flow series from the entire portfolio lifecycle, including the initial outlay and all subsequent inflows.</li>
        </ul>
        <p>This is actually a realistic reflection of how private credit portfolios work - the yield on the overall portfolio over time can be substantially higher than what you might calculate by looking at the yields of individual investments, due to reinvestment and compounding over multiple cycles.</p>
        </div>
        </details>
        """, unsafe_allow_html=True)
        
        st.subheader("Portfolio Cashflow Breakdown")
        # Create dataframe for cashflow visualization
        cf_data = pd.DataFrame({
            "Month": base_sim.months,
            "Interest": base_sim.per_period_interest,
            "Principal": base_sim.per_period_principal,
            "Defaults": base_sim.per_period_defaults
        })
        
        # Display cashflow chart
        fig_cf = px.area(
            cf_data, x="Month", 
            y=["Interest", "Principal", "Defaults"],
            title="Monthly Cashflow Components",
            color_discrete_sequence=["#4c78a8", "#f58518", "#e45756"]
        )
        st.plotly_chart(fig_cf, use_container_width=True)
        
        # Create dataframe for outstanding balance
        total_out = np.zeros_like(base_sim.months, dtype=float)
        for n in [1, 2, 3]:
            total_out += base_sim.out_gross[n]
        
        out_data = pd.DataFrame({
            "Month": base_sim.months,
            "1-Month": base_sim.out_gross[1],
            "2-Month": base_sim.out_gross[2],
            "3-Month": base_sim.out_gross[3],
            "Total": total_out
        })
        
        # Display outstanding balance chart
        fig_out = go.Figure()
        for n in [1, 2, 3]:
            fig_out.add_trace(go.Bar(
                x=out_data["Month"], 
                y=out_data[f"{n}-Month"],
                name=f"{n}-Month",
                marker_color=["#e45756", "#4c78a8", "#f58518"][n-1]
            ))
        fig_out.add_trace(go.Scatter(
            x=out_data["Month"], 
            y=out_data["Total"],
            name="Total",
            mode="lines+markers",
            line=dict(color="black", width=2)
        ))
        fig_out.update_layout(
            title="Outstanding Balance by Tenor",
            barmode="stack",
            xaxis_title="Month",
            yaxis_title="Outstanding Balance ($)"
        )
        st.plotly_chart(fig_out, use_container_width=True)
    
    # Tab 2: Predefined Scenarios
    with tab2:
        st.subheader("Predefined Stress Scenarios")
        
        # Get predefined scenarios
        predefined_scenarios = get_predefined_scenarios()
        
        # Allow user to select scenarios
        scenario_options = [scenario.name for scenario in predefined_scenarios]
        selected_scenarios = st.multiselect(
            "Select scenarios to analyze",
            options=scenario_options,
            default=["Base Case", "High Default", "Severe Default", "Credit Crisis"]
        )
        
        if not selected_scenarios:
            st.warning("Please select at least one scenario to analyze.")
        else:
            # Filter scenarios based on selection
            selected_scenario_objects = [
                scenario for scenario in predefined_scenarios
                if scenario.name in selected_scenarios
            ]
            
            # Run stress tests
            with st.spinner("Running stress tests..."):
                scenario_results = run_multiple_stress_tests(
                    base_params, selected_scenario_objects
                )
            
            # Generate report
            report = generate_stress_test_report(base_params, scenario_results)
            st.dataframe(report, use_container_width=True)
            
            # Collect metrics for comparison
            scenario_metrics = {
                "Base Case": base_metrics
            }
            for name, result in scenario_results.items():
                scenario_metrics[name] = result.stressed_metrics
            
            # Plot impact comparison
            st.subheader("Scenario Impact Comparison")
            
            # Convert matplotlib figure to plotly for better interactivity
            impact_data = []
            for scenario_name, result in scenario_results.items():
                for metric, value in result.impact_summary.items():
                    impact_data.append({
                        "Scenario": scenario_name,
                        "Metric": metric,
                        "Impact (%)": value
                    })
            
            impact_df = pd.DataFrame(impact_data)
            fig_impact = px.bar(
                impact_df,
                x="Scenario",
                y="Impact (%)",
                color="Metric",
                barmode="group",
                title="Impact of Stress Scenarios on Key Metrics"
            )
            st.plotly_chart(fig_impact, use_container_width=True)
            
            # Plot tenor-specific metrics comparison
            metric_to_plot = st.selectbox(
                "Select metric to compare across tenors:",
                ["IRR", "Recovery Rates", "WAL", "Duration", "Convexity"]
            )
            
            fig_tenor = plot_tenor_metrics(scenario_metrics, metric_to_plot)
            if fig_tenor:
                st.plotly_chart(fig_tenor, use_container_width=True)
    
    # Tab 3: Custom Scenario
    with tab3:
        st.subheader("Define and Run Custom Stress Scenario")
        
        custom_scenario = create_custom_scenario()
        
        if st.button("Run Custom Scenario"):
            with st.spinner("Running custom stress test..."):
                custom_result = run_stress_test(base_params, custom_scenario)
            
            col1, col2 = st.columns(2)
            
            with col1:
                display_scenario_metrics(base_metrics, "Base Portfolio Metrics")
            
            with col2:
                display_scenario_metrics(custom_result.stressed_metrics, "Stressed Portfolio Metrics")
            
            display_impact_summary(custom_result.impact_summary)
            
            # Create detailed comparison
            st.subheader("Impact on Cash Flows")
            
            # Run base simulation for comparison
            base_sim = run_simulation(base_params)
            
            # Run stressed simulation
            stressed_params = base_params
            stressed_params.default_rates = {
                n: min(0.999, rate * custom_scenario.default_rate_multiplier)
                for n, rate in base_params.default_rates.items()
            }
            stressed_params.monthly_interest += custom_scenario.interest_rate_delta
            # Note: allocation shifts would require deeper modification
            
            stressed_sim = run_simulation(stressed_params)
            
            # Compare cash flows
            cf_compare = pd.DataFrame({
                "Month": base_sim.months,
                "Base Interest": base_sim.per_period_interest,
                "Base Principal": base_sim.per_period_principal,
                "Base Defaults": base_sim.per_period_defaults,
                "Stressed Interest": stressed_sim.per_period_interest,
                "Stressed Principal": stressed_sim.per_period_principal,
                "Stressed Defaults": stressed_sim.per_period_defaults
            })
            
            # Plot cash flow comparison
            fig_cf_compare = go.Figure()
            
            # Base case
            fig_cf_compare.add_trace(go.Bar(
                x=cf_compare["Month"],
                y=cf_compare["Base Interest"],
                name="Base Interest",
                marker_color="rgba(76, 120, 168, 0.7)"
            ))
            fig_cf_compare.add_trace(go.Bar(
                x=cf_compare["Month"],
                y=cf_compare["Base Principal"],
                name="Base Principal",
                marker_color="rgba(245, 133, 24, 0.7)"
            ))
            fig_cf_compare.add_trace(go.Bar(
                x=cf_compare["Month"],
                y=cf_compare["Base Defaults"],
                name="Base Defaults",
                marker_color="rgba(228, 87, 86, 0.7)"
            ))
            
            # Stressed case
            fig_cf_compare.add_trace(go.Bar(
                x=cf_compare["Month"],
                y=cf_compare["Stressed Interest"],
                name="Stressed Interest",
                marker_color="rgba(76, 120, 168, 1.0)"
            ))
            fig_cf_compare.add_trace(go.Bar(
                x=cf_compare["Month"],
                y=cf_compare["Stressed Principal"],
                name="Stressed Principal",
                marker_color="rgba(245, 133, 24, 1.0)"
            ))
            fig_cf_compare.add_trace(go.Bar(
                x=cf_compare["Month"],
                y=cf_compare["Stressed Defaults"],
                name="Stressed Defaults",
                marker_color="rgba(228, 87, 86, 1.0)"
            ))
            
            fig_cf_compare.update_layout(
                title="Cash Flow Comparison: Base vs Stressed",
                barmode="group",
                xaxis_title="Month",
                yaxis_title="Amount ($)"
            )
            
            st.plotly_chart(fig_cf_compare, use_container_width=True)
    
    # Tab 4: Sensitivity Analysis
    with tab4:
        st.subheader("Sensitivity Analysis")
        st.write("Analyze how changes in individual parameters affect portfolio performance")
        
        # Allow user to select parameters to analyze
        st.write("Select parameters for sensitivity analysis:")
        
        param_selections = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Monthly Interest Rate", value=True):
                interest_min = max(0.01, mon_int - 0.05)
                interest_max = mon_int + 0.05
                interest_range = np.linspace(interest_min, interest_max, 10)
                param_selections["monthly_interest"] = interest_range
            
            for n in [1, 2, 3]:
                if st.checkbox(f"{n}-Month Default Rate"):
                    default_min = max(0.01, default_rates[n] - 0.1)
                    default_max = min(0.99, default_rates[n] + 0.2)
                    default_range = np.linspace(default_min, default_max, 10)
                    param_selections[f"default_rate_{n}"] = default_range
        
        with col2:
            for n in [1, 2, 3]:
                if st.checkbox(f"{n}-Month Allocation"):
                    alloc_min = max(0.1, allocation[n] - 0.2)
                    alloc_max = min(0.9, allocation[n] + 0.2)
                    alloc_range = np.linspace(alloc_min, alloc_max, 10)
                    param_selections[f"allocation_{n}"] = alloc_range
        
        if not param_selections:
            st.warning("Please select at least one parameter for sensitivity analysis.")
        else:
            if st.button("Run Sensitivity Analysis"):
                with st.spinner("Running sensitivity analysis..."):
                    sensitivity_results = run_sensitivity_analysis(base_params, param_selections)
                
                # Convert matplotlib figure to plotly for better interactivity
                for param_name, results in sensitivity_results.items():
                    x_values, y_values = zip(*results)
                    
                    # Determine appropriate labels
                    if param_name == "monthly_interest":
                        param_label = "Monthly Interest Rate"
                        x_format = "{:.2%}"
                    elif param_name.startswith("default_rate_"):
                        tenor = param_name.split("_")[-1]
                        param_label = f"{tenor}-Month Default Rate"
                        x_format = "{:.2%}"
                    elif param_name.startswith("allocation_"):
                        tenor = param_name.split("_")[-1]
                        param_label = f"{tenor}-Month Allocation"
                        x_format = "{:.2%}"
                    else:
                        param_label = param_name
                        x_format = "{:.2f}"
                    
                    # Create plotly figure
                    fig_sensitivity = go.Figure()
                    fig_sensitivity.add_trace(go.Scatter(
                        x=x_values,
                        y=[y * 100 for y in y_values],  # Convert to percentage
                        mode="lines+markers",
                        name="Portfolio IRR",
                        line=dict(color="#4c78a8", width=2)
                    ))
                    
                    # Add vertical line at base value
                    if param_name == "monthly_interest":
                        base_value = mon_int
                    elif param_name.startswith("default_rate_"):
                        tenor = int(param_name.split("_")[-1])
                        base_value = default_rates[tenor]
                    elif param_name.startswith("allocation_"):
                        tenor = int(param_name.split("_")[-1])
                        base_value = allocation[tenor]
                    else:
                        base_value = None
                    
                    if base_value is not None:
                        fig_sensitivity.add_vline(
                            x=base_value,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Base Value",
                            annotation_position="top right"
                        )
                    
                    fig_sensitivity.update_layout(
                        title=f"Sensitivity of Portfolio IRR to {param_label}",
                        xaxis_title=param_label,
                        yaxis_title="Portfolio IRR (%)",
                        xaxis_tickformat=x_format
                    )
                    
                    st.plotly_chart(fig_sensitivity, use_container_width=True)

if __name__ == "__main__":
    main() 