import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from pie_matrix import InputParams, SimulationResults, run_simulation
from financial_metrics import compute_extended_metrics, ExtendedFinancialMetrics

@dataclass
class StressTestScenario:
    """Defines a stress test scenario with modified parameters."""
    name: str
    description: str
    default_rate_multiplier: float = 1.0  # Multiplier for default rates
    interest_rate_delta: float = 0.0      # Absolute change in interest rate (e.g., +0.02 for +2%)
    allocation_shift: Dict[int, float] = None  # Shift in allocation percentages

@dataclass
class StressTestResult:
    """Results from a stress test."""
    scenario: StressTestScenario
    base_metrics: ExtendedFinancialMetrics
    stressed_metrics: ExtendedFinancialMetrics
    impact_summary: Dict[str, float]  # Key metrics and their percentage change

def apply_stress_scenario(base_params: InputParams, scenario: StressTestScenario) -> InputParams:
    """
    Apply a stress scenario to base parameters.
    
    Args:
        base_params: Original input parameters
        scenario: Stress test scenario to apply
        
    Returns:
        New input parameters with stress scenario applied
    """
    # Create a copy of base parameters
    stressed_params = InputParams(
        initial_capital=base_params.initial_capital,
        monthly_interest=base_params.monthly_interest + scenario.interest_rate_delta,
        num_months=base_params.num_months,
        default_rates={n: min(0.999, rate * scenario.default_rate_multiplier) 
                     for n, rate in base_params.default_rates.items()},
        allocation=base_params.allocation.copy(),
        show_def_overlay=base_params.show_def_overlay
    )
    
    # Apply allocation shifts if specified
    if scenario.allocation_shift:
        for tenor, shift in scenario.allocation_shift.items():
            if tenor in stressed_params.allocation:
                stressed_params.allocation[tenor] += shift
        
        # Normalize allocations to ensure they sum to 1
        total = sum(stressed_params.allocation.values())
        if total > 0:
            for tenor in stressed_params.allocation:
                stressed_params.allocation[tenor] /= total
    
    return stressed_params

def run_stress_test(base_params: InputParams, scenario: StressTestScenario) -> StressTestResult:
    """
    Run a stress test comparing base case with stressed scenario.
    
    Args:
        base_params: Original input parameters
        scenario: Stress test scenario to apply
        
    Returns:
        Stress test results with comparison metrics
    """
    print(f"\n=== Running stress test for scenario: {scenario.name} ===")
    print(f"Scenario details: {scenario}")
    
    # Run base case simulation
    print("Running base case simulation...")
    base_sim = run_simulation(base_params)
    base_metrics = compute_extended_metrics(base_params, base_sim)
    
    # Apply stress scenario and run stressed simulation
    print("Applying stress scenario and running stressed simulation...")
    stressed_params = apply_stress_scenario(base_params, scenario)
    print(f"Stressed parameters: {stressed_params}")
    stressed_sim = run_simulation(stressed_params)
    stressed_metrics = compute_extended_metrics(stressed_params, stressed_sim)
    
    # Calculate impact on key metrics
    print("Calculating impact on key metrics...")
    impact_summary = {
        "IRR Change (%)": calculate_percentage_change(
            stressed_metrics.portfolio_irr, base_metrics.portfolio_irr),
        "WAL Change (%)": calculate_percentage_change(
            stressed_metrics.portfolio_wal, base_metrics.portfolio_wal),
        "Duration Change (%)": calculate_average_percentage_change(
            stressed_metrics.duration, base_metrics.duration),
        "Default Impact (%)": calculate_average_percentage_change(
            stressed_metrics.recovery_rates, base_metrics.recovery_rates, invert=True),
        "Cash Flow Volatility Change (%)": calculate_percentage_change(
            stressed_metrics.volatility, base_metrics.volatility)
    }
    
    print(f"Impact summary: {impact_summary}")
    
    return StressTestResult(
        scenario=scenario,
        base_metrics=base_metrics,
        stressed_metrics=stressed_metrics,
        impact_summary=impact_summary
    )

def run_multiple_stress_tests(base_params: InputParams, 
                            scenarios: List[StressTestScenario]) -> Dict[str, StressTestResult]:
    """
    Run multiple stress tests and compile results.
    
    Args:
        base_params: Original input parameters
        scenarios: List of stress test scenarios to apply
        
    Returns:
        Dictionary mapping scenario names to stress test results
    """
    results = {}
    for scenario in scenarios:
        results[scenario.name] = run_stress_test(base_params, scenario)
    return results

def calculate_percentage_change(new_value: float, old_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        new_value: The new value after change
        old_value: The original value before change
        
    Returns:
        Percentage change (as a percentage, not decimal)
    """
    # Debug info
    print(f"Calculating percentage change: new={new_value}, old={old_value}")
    
    # Handle NaN inputs
    if np.isnan(new_value) or np.isnan(old_value):
        print("ERROR: NaN value detected in percentage change calculation")
        return np.nan
    
    # Handle infinity inputs
    if np.isinf(new_value) or np.isinf(old_value):
        if np.isinf(old_value) and np.isinf(new_value):
            # Both infinity
            if (old_value > 0 and new_value > 0) or (old_value < 0 and new_value < 0):
                # Same sign, no change
                return 0.0
            else:
                # Different signs, extreme change
                return 1000.0 if new_value > 0 else -1000.0
        elif np.isinf(new_value):
            # New value is infinity
            return 1000.0 if new_value > 0 else -1000.0
        else:
            # Old value is infinity
            return -1000.0 if old_value > 0 else 1000.0
    
    # Handle zero or near-zero old value
    if abs(old_value) < 0.0001:
        if abs(new_value) < 0.0001:
            # Both values are effectively zero
            print("Both values effectively zero, returning 0% change")
            return 0.0
        else:
            # From zero to non-zero - calculate sign and cap magnitude
            print(f"Base value near zero, returning capped value for {new_value}")
            direction = 1.0 if new_value > 0 else -1.0
            magnitude = min(1000.0, 100.0 * abs(new_value) / 0.0001)
            return direction * magnitude
    
    # Standard percentage change calculation with caps
    raw_pct_change = 100.0 * (new_value - old_value) / abs(old_value)
    
    # Cap extreme values for better display and stability
    if abs(raw_pct_change) > 1000.0:
        print(f"Capping extreme percentage change: {raw_pct_change}")
        capped_pct_change = 1000.0 if raw_pct_change > 0 else -1000.0
        print(f"Capped percentage change: {capped_pct_change}%")
        return capped_pct_change
    
    print(f"Calculated percentage change: {raw_pct_change}%")
    return raw_pct_change

def calculate_average_percentage_change(new_dict: Dict[int, float], 
                                      old_dict: Dict[int, float],
                                      invert: bool = False) -> float:
    """Calculate average percentage change across dictionary values."""
    print(f"Calculating average percentage change between: {old_dict} and {new_dict}")
    changes = []
    
    for key in new_dict:
        if key in old_dict:
            # Skip NaN values
            if np.isnan(new_dict[key]) or np.isnan(old_dict[key]):
                print(f"Skipping NaN value for key {key}")
                continue
                
            if invert:
                # For metrics where decrease is negative impact (like recovery rates)
                changes.append(calculate_percentage_change(old_dict[key], new_dict[key]))
            else:
                changes.append(calculate_percentage_change(new_dict[key], old_dict[key]))
    
    if not changes:
        print("No valid changes to average, returning NaN")
        return np.nan
        
    avg_change = sum(changes) / len(changes)
    print(f"Average percentage change: {avg_change}%")
    return avg_change

def get_predefined_scenarios() -> List[StressTestScenario]:
    """
    Get a list of predefined stress test scenarios.
    
    Returns:
        List of stress test scenarios
    """
    return [
        StressTestScenario(
            name="Base Case",
            description="No stress applied - baseline scenario",
            default_rate_multiplier=1.0,
            interest_rate_delta=0.0,
            allocation_shift=None
        ),
        StressTestScenario(
            name="High Default",
            description="Significant increase in default rates",
            default_rate_multiplier=2.0,
            interest_rate_delta=0.0,
            allocation_shift=None
        ),
        StressTestScenario(
            name="Severe Default",
            description="Severe market downturn with very high defaults",
            default_rate_multiplier=3.0,
            interest_rate_delta=0.0,
            allocation_shift=None
        ),
        StressTestScenario(
            name="Interest Rate Hike",
            description="Central bank increases interest rates",
            default_rate_multiplier=1.0,
            interest_rate_delta=0.02,
            allocation_shift=None
        ),
        StressTestScenario(
            name="Interest Rate Drop",
            description="Central bank decreases interest rates",
            default_rate_multiplier=1.0,
            interest_rate_delta=-0.02,
            allocation_shift=None
        ),
        StressTestScenario(
            name="Short-Term Shift",
            description="Portfolio shifts toward shorter-term loans",
            default_rate_multiplier=1.0,
            interest_rate_delta=0.0,
            allocation_shift={1: 0.2, 3: -0.2}
        ),
        StressTestScenario(
            name="Long-Term Shift",
            description="Portfolio shifts toward longer-term loans",
            default_rate_multiplier=1.0,
            interest_rate_delta=0.0,
            allocation_shift={1: -0.2, 3: 0.2}
        ),
        StressTestScenario(
            name="Combined Stress",
            description="Combined stress: higher defaults, rate hike, and long-term shift",
            default_rate_multiplier=1.5,
            interest_rate_delta=0.01,
            allocation_shift={1: -0.1, 3: 0.1}
        ),
        StressTestScenario(
            name="Credit Crisis",
            description="Severe credit crisis with liquidity issues",
            default_rate_multiplier=2.5,
            interest_rate_delta=0.03,
            allocation_shift={1: 0.3, 3: -0.3}  # Flight to shorter-term loans
        )
    ]

def generate_stress_test_report(base_params: InputParams, 
                              results: Dict[str, StressTestResult]) -> pd.DataFrame:
    """
    Generate a summary report of stress test results.
    
    Args:
        base_params: Original input parameters
        results: Dictionary of stress test results
        
    Returns:
        DataFrame with stress test results
    """
    print("Generating stress test report...")
    report_data = []
    
    for scenario_name, result in results.items():
        print(f"Formatting results for scenario: {scenario_name}")
        row = {
            "Scenario": scenario_name,
            "Description": result.scenario.description,
            "Default Multiplier": result.scenario.default_rate_multiplier,
            "Interest Rate Delta": f"{result.scenario.interest_rate_delta * 100:+.2f}%",
        }
        
        # Format portfolio IRR with better error handling and more meaningful values for credit analysis
        if np.isnan(result.stressed_metrics.portfolio_irr):
            # In a real credit scenario, this would likely be a deeply distressed portfolio
            row["Portfolio IRR"] = "Not calculable"
        elif np.isinf(result.stressed_metrics.portfolio_irr):
            if result.stressed_metrics.portfolio_irr > 0:
                row["Portfolio IRR"] = "∞"
            else:
                row["Portfolio IRR"] = "Total Loss"  # More meaningful than -∞
        elif result.stressed_metrics.portfolio_irr < -0.75:
            # For catastrophic negative IRRs (worse than -75% monthly)
            row["Portfolio IRR"] = f"Catastrophic ({result.stressed_metrics.portfolio_irr*100:.0f}%)"
        elif result.stressed_metrics.portfolio_irr < -0.40:
            # For very negative IRRs
            row["Portfolio IRR"] = f"Severe Loss ({result.stressed_metrics.portfolio_irr*100:.0f}%)"
        elif result.stressed_metrics.portfolio_irr < -0.15:
            # For significantly negative IRRs
            row["Portfolio IRR"] = f"Major Loss ({result.stressed_metrics.portfolio_irr*100:.1f}%)"
        elif result.stressed_metrics.portfolio_irr < 0:
            # For moderately negative IRRs
            row["Portfolio IRR"] = f"Loss ({result.stressed_metrics.portfolio_irr*100:.2f}%)"
        elif abs(result.stressed_metrics.portfolio_irr) > 1.0:
            # Handle extremely large positive values (rare in credit)
            row["Portfolio IRR"] = f"Extreme Gain ({result.stressed_metrics.portfolio_irr*100:.1f}%)"
        else:
            # Normal case - positive IRR
            row["Portfolio IRR"] = f"{result.stressed_metrics.portfolio_irr * 100:.2f}%"
            
        # Format IRR change with description
        irr_change = result.impact_summary.get('IRR Change (%)', np.nan)
        if np.isnan(irr_change):
            # If we have both valid IRRs, we can calculate the change directly
            if not np.isnan(result.base_metrics.portfolio_irr) and not np.isnan(result.stressed_metrics.portfolio_irr):
                irr_change = 100 * (result.stressed_metrics.portfolio_irr - result.base_metrics.portfolio_irr) / abs(result.base_metrics.portfolio_irr)
                if abs(irr_change) > 500:
                    direction = "+" if irr_change > 0 else ""
                    row["IRR Change"] = f"Extreme ({direction}{min(abs(irr_change), 999.99):.0f}%)"
                else:
                    row["IRR Change"] = f"{irr_change:.2f}%"
            else:
                # If we can't calculate the change directly
                if result.stressed_metrics.portfolio_irr < 0 and result.base_metrics.portfolio_irr > 0:
                    row["IRR Change"] = "Turned negative"
                elif result.stressed_metrics.portfolio_irr < -0.2:
                    row["IRR Change"] = "Major loss"
                else:
                    row["IRR Change"] = "Not comparable"
        elif abs(irr_change) > 500:
            # Extreme changes
            if irr_change < -500:
                row["IRR Change"] = "Severe deterioration"
            else:
                row["IRR Change"] = f"Extreme (+{min(irr_change, 999.99):.0f}%)"
        elif irr_change < -100:
            # Significant negative change
            row["IRR Change"] = f"Major decline ({irr_change:.0f}%)"
        else:
            row["IRR Change"] = f"{irr_change:.2f}%"
        
        # Format portfolio WAL with better error handling
        if np.isnan(result.stressed_metrics.portfolio_wal):
            row["Portfolio WAL"] = "Not calculable"
        elif np.isinf(result.stressed_metrics.portfolio_wal):
            row["Portfolio WAL"] = "∞"
        else:
            row["Portfolio WAL"] = f"{result.stressed_metrics.portfolio_wal:.2f} months"
            
        # Format WAL Change
        wal_change = result.impact_summary.get('WAL Change (%)', np.nan)
        if np.isnan(wal_change):
            row["WAL Change"] = "Not calculable"
        elif abs(wal_change) > 50:
            row["WAL Change"] = f"Extreme ({wal_change:.1f}%)"
        else:
            row["WAL Change"] = f"{wal_change:.2f}%"
        
        # Format Default Impact with explanatory text for extreme values
        default_impact = result.impact_summary.get('Default Impact (%)', np.nan)
        if np.isnan(default_impact):
            row["Default Impact"] = "Not calculable"
        elif default_impact > 100:
            row["Default Impact"] = f"Severe (+{min(default_impact, 999.99):.0f}%)"
        elif default_impact < -50:
            row["Default Impact"] = f"Improved ({default_impact:.0f}%)"
        else:
            row["Default Impact"] = f"{default_impact:.2f}%"
        
        # Format Volatility with better error handling
        if np.isnan(result.stressed_metrics.volatility):
            row["Volatility"] = "Not calculable"
        elif np.isinf(result.stressed_metrics.volatility):
            row["Volatility"] = "∞"
        elif result.stressed_metrics.volatility > 10:
            row["Volatility"] = f"Very high ({min(result.stressed_metrics.volatility, 99.9):.1f})"
        else:
            row["Volatility"] = f"{result.stressed_metrics.volatility:.2f}"
            
        # Format Volatility Change
        vol_change = result.impact_summary.get('Cash Flow Volatility Change (%)', np.nan)
        if np.isnan(vol_change):
            row["Volatility Change"] = "Not calculable"
        elif abs(vol_change) > 500:
            direction = "increase" if vol_change > 0 else "decrease"
            row["Volatility Change"] = f"Extreme {direction}"
        else:
            row["Volatility Change"] = f"{vol_change:.2f}%"
        
        report_data.append(row)
        print(f"Added row to report: {row}")
    
    df = pd.DataFrame(report_data)
    print(f"Generated report with {len(df)} rows")
    return df

def plot_stress_test_impacts(results: Dict[str, StressTestResult], 
                           metrics: List[str] = None) -> plt.Figure:
    """
    Generate a plot showing the impact of different stress scenarios.
    
    Args:
        results: Dictionary of stress test results
        metrics: List of metrics to plot (default: all impact metrics)
        
    Returns:
        Matplotlib figure with the plot
    """
    if metrics is None:
        # Use a sample of the first result to get all metrics
        first_result = next(iter(results.values()))
        metrics = list(first_result.impact_summary.keys())
    
    # Extract data for plotting
    scenarios = list(results.keys())
    data = {metric: [results[s].impact_summary.get(metric, 0) for s in scenarios] 
            for metric in metrics}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(scenarios))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width - 0.4 + width/2, data[metric], width, label=metric)
    
    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Impact of Stress Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def run_sensitivity_analysis(base_params: InputParams, 
                           parameter_ranges: Dict[str, List[float]]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Run a sensitivity analysis by varying one parameter at a time.
    
    Args:
        base_params: Original input parameters
        parameter_ranges: Dictionary mapping parameter names to ranges of values to test
        
    Returns:
        Dictionary mapping parameter names to lists of (parameter_value, IRR) pairs
    """
    results = {}
    
    for param_name, param_values in parameter_ranges.items():
        param_results = []
        
        for value in param_values:
            # Create a copy of base parameters with one parameter modified
            test_params = InputParams(
                initial_capital=base_params.initial_capital,
                monthly_interest=base_params.monthly_interest,
                num_months=base_params.num_months,
                default_rates=base_params.default_rates.copy(),
                allocation=base_params.allocation.copy(),
                show_def_overlay=base_params.show_def_overlay
            )
            
            # Modify the specific parameter
            if param_name == "monthly_interest":
                test_params.monthly_interest = value
            elif param_name.startswith("default_rate_"):
                tenor = int(param_name.split("_")[-1])
                test_params.default_rates[tenor] = value
            elif param_name.startswith("allocation_"):
                tenor = int(param_name.split("_")[-1])
                # Adjust allocation for this tenor
                shift = value - test_params.allocation[tenor]
                test_params.allocation[tenor] = value
                
                # Redistribute the shift proportionally among other tenors
                other_tenors = [t for t in test_params.allocation if t != tenor]
                other_total = sum(test_params.allocation[t] for t in other_tenors)
                
                if other_total > 0:
                    for t in other_tenors:
                        test_params.allocation[t] -= shift * (test_params.allocation[t] / other_total)
                        test_params.allocation[t] = max(0, test_params.allocation[t])
                
                # Normalize to ensure allocations sum to 1
                total = sum(test_params.allocation.values())
                if total > 0:
                    for t in test_params.allocation:
                        test_params.allocation[t] /= total
            
            # Run simulation with modified parameter
            sim = run_simulation(test_params)
            metrics = compute_extended_metrics(test_params, sim)
            
            # Record the parameter value and IRR
            param_results.append((value, metrics.portfolio_irr))
        
        results[param_name] = param_results
    
    return results

def plot_sensitivity_analysis(sensitivity_results: Dict[str, List[Tuple[float, float]]]) -> plt.Figure:
    """
    Plot sensitivity analysis results.
    
    Args:
        sensitivity_results: Results from run_sensitivity_analysis
        
    Returns:
        Matplotlib figure with sensitivity plots
    """
    num_params = len(sensitivity_results)
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 4 * num_params))
    
    # If only one parameter, axes is not a list
    if num_params == 1:
        axes = [axes]
    
    for i, (param_name, results) in enumerate(sensitivity_results.items()):
        x_values, y_values = zip(*results)
        axes[i].plot(x_values, [y * 100 for y in y_values], marker='o')
        axes[i].set_xlabel(f'{param_name} Value')
        axes[i].set_ylabel('Portfolio IRR (%)')
        axes[i].set_title(f'Sensitivity of IRR to {param_name}')
        axes[i].grid(True)
    
    plt.tight_layout()
    return fig 