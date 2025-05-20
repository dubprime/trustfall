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
    # Run base case simulation
    base_sim = run_simulation(base_params)
    base_metrics = compute_extended_metrics(base_params, base_sim)
    
    # Apply stress scenario and run stressed simulation
    stressed_params = apply_stress_scenario(base_params, scenario)
    stressed_sim = run_simulation(stressed_params)
    stressed_metrics = compute_extended_metrics(stressed_params, stressed_sim)
    
    # Calculate impact on key metrics
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
    """Calculate percentage change between two values."""
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
    return 100 * (new_value - old_value) / abs(old_value)

def calculate_average_percentage_change(new_dict: Dict[int, float], 
                                      old_dict: Dict[int, float],
                                      invert: bool = False) -> float:
    """Calculate average percentage change across dictionary values."""
    changes = []
    for key in new_dict:
        if key in old_dict:
            if invert:
                # For metrics where decrease is negative impact (like recovery rates)
                changes.append(calculate_percentage_change(old_dict[key], new_dict[key]))
            else:
                changes.append(calculate_percentage_change(new_dict[key], old_dict[key]))
    
    return sum(changes) / len(changes) if changes else 0

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
    report_data = []
    
    for scenario_name, result in results.items():
        row = {
            "Scenario": scenario_name,
            "Description": result.scenario.description,
            "Default Multiplier": result.scenario.default_rate_multiplier,
            "Interest Rate Delta": f"{result.scenario.interest_rate_delta * 100:+.2f}%",
            "Portfolio IRR": f"{result.stressed_metrics.portfolio_irr * 100:.2f}%",
            "IRR Change": f"{result.impact_summary['IRR Change (%)']:.2f}%",
            "Portfolio WAL": f"{result.stressed_metrics.portfolio_wal:.2f} months",
            "WAL Change": f"{result.impact_summary['WAL Change (%)']:.2f}%",
            "Default Impact": f"{result.impact_summary['Default Impact (%)']:.2f}%",
            "Volatility": f"{result.stressed_metrics.volatility:.2f}",
            "Volatility Change": f"{result.impact_summary['Cash Flow Volatility Change (%)']:.2f}%"
        }
        report_data.append(row)
    
    return pd.DataFrame(report_data)

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