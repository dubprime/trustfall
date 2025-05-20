import numpy as np
import numpy_financial as npf
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pie_matrix import InputParams, SimulationResults

@dataclass
class ExtendedFinancialMetrics:
    """Extended financial metrics beyond the basic simulation output."""
    irr: Dict[int, float]  # Internal Rate of Return by tenor
    portfolio_irr: float   # Weighted IRR for the entire portfolio
    sharpe_ratios: Dict[int, float]  # Risk-adjusted returns by tenor
    recovery_rates: Dict[int, float]  # Effective recovery rates by tenor
    weighted_average_life: Dict[int, float]  # WAL by tenor
    portfolio_wal: float   # Portfolio WAL
    concentration_hhi: float  # Herfindahl-Hirschman Index for concentration
    duration: Dict[int, float]  # Modified duration by tenor
    convexity: Dict[int, float]  # Convexity by tenor
    volatility: float  # Cash flow volatility
    
def calculate_irr(cash_flows: np.ndarray) -> float:
    """
    Calculate the Internal Rate of Return for a series of cash flows.
    
    Args:
        cash_flows: NumPy array of cash flows (negative for outflows, positive for inflows)
        
    Returns:
        IRR as a decimal (e.g., 0.12 for 12%), or np.nan if IRR cannot be calculated
    """
    # Debug information
    print(f"Calculating IRR for cash flows: {cash_flows}")
    
    # Check if cash flow pattern is valid for IRR calculation
    if len(cash_flows) < 2:
        print("ERROR: Cash flow array too short for IRR calculation")
        return np.nan
        
    # Check if all values are negative or zero (no inflows)
    if np.all(cash_flows <= 0):
        print("ERROR: Cash flow pattern invalid for IRR - all values are negative or zero (no inflows)")
        return np.nan
        
    # Check if all values are positive or zero (no outflows)
    if np.all(cash_flows >= 0):
        print("ERROR: Cash flow pattern invalid for IRR - all values are positive or zero (no outflows)")
        return np.nan
    
    # Additional validation: ensure the cash flows sum isn't too close to zero
    # Can happen in stressed scenarios where losses almost exactly equal gains
    if abs(np.sum(cash_flows)) < 0.0001 * np.max(np.abs(cash_flows)):
        print("ERROR: Cash flow sum too close to zero relative to the magnitude of flows")
        return np.nan
    
    try:
        # Starting with a reasonable guess based on common rates in private credit
        irr = npf.irr(cash_flows)
        
        # Validate the IRR result
        if np.isnan(irr):
            print("ERROR: IRR calculation resulted in NaN")
            return np.nan
        if np.isinf(irr):
            print(f"ERROR: IRR calculation resulted in infinity: {irr}")
            return np.nan
        if abs(irr) > 1.0:  # IRR over 100% or under -100% is likely an error
            print(f"WARNING: Extreme IRR value calculated: {irr}")
            # Still return it, but flag it
        
        print(f"Successfully calculated IRR: {irr}")
        return irr
    except Exception as e:
        print(f"ERROR in primary IRR calculation: {str(e)}")
        # If IRR calculation fails, try with a different approach
        # This can happen with unconventional cash flow patterns
        for guess in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, -0.10, -0.20]:
            try:
                print(f"Attempting IRR calculation with guess: {guess}")
                irr = npf.irr(cash_flows, guess)
                
                # Validate the IRR result again
                if np.isnan(irr) or np.isinf(irr) or abs(irr) > 1.0:
                    print(f"Invalid IRR result with guess {guess}: {irr}")
                    continue
                    
                print(f"Successfully calculated IRR with fallback: {irr}")
                return irr
            except Exception as e:
                print(f"Fallback IRR calculation failed with guess {guess}: {str(e)}")
                continue
        # If all guesses fail, return NaN
        print("All IRR calculation attempts failed")
        return np.nan

def calculate_weighted_average_life(principal_flows: np.ndarray, periods: np.ndarray) -> float:
    """
    Calculate Weighted Average Life (WAL) of cash flows.
    
    Args:
        principal_flows: Principal repayments in each period
        periods: Period numbers (months)
        
    Returns:
        WAL in periods (months)
    """
    if sum(principal_flows) == 0:
        return 0
    
    weights = principal_flows / sum(principal_flows)
    return sum(weights * periods)

def calculate_recovery_rate(gross_outstanding: np.ndarray, 
                           defaults: np.ndarray, 
                           default_rate: float) -> float:
    """
    Calculate the effective recovery rate on defaulted loans.
    
    Args:
        gross_outstanding: Outstanding principal before defaults
        defaults: Default amounts in each period
        default_rate: Expected default rate at maturity
        
    Returns:
        Effective recovery rate (1 - LGD)
    """
    total_exposure = sum(gross_outstanding)
    total_defaults = sum(defaults)
    
    # Expected defaults based on default rate
    expected_defaults = total_exposure * default_rate
    
    # If no defaults are expected, return 1.0 (100% recovery)
    if expected_defaults == 0:
        return 1.0
        
    # Recovery rate = 1 - (actual_defaults / expected_defaults)
    return max(0, 1 - (total_defaults / expected_defaults))

def calculate_concentration_hhi(allocations: Dict[int, float]) -> float:
    """
    Calculate Herfindahl-Hirschman Index for measuring concentration risk.
    
    Args:
        allocations: Dictionary mapping tenor to allocation percentage
        
    Returns:
        HHI value (0-1 scale, higher means more concentrated)
    """
    # Calculate sum of squared percentages (as decimals)
    return sum(alloc**2 for alloc in allocations.values())

def calculate_cashflow_volatility(monthly_cash_flows: np.ndarray) -> float:
    """
    Calculate the coefficient of variation of cash flows.
    
    Args:
        monthly_cash_flows: Cash flows by month
        
    Returns:
        Coefficient of variation (standard deviation / mean)
    """
    if len(monthly_cash_flows) <= 1 or np.mean(monthly_cash_flows) == 0:
        return 0
        
    return np.std(monthly_cash_flows) / np.mean(monthly_cash_flows)

def calculate_modified_duration(cash_flows: np.ndarray, 
                              periods: np.ndarray, 
                              yield_rate: float) -> float:
    """
    Calculate modified duration (sensitivity to interest rate changes).
    
    Args:
        cash_flows: Cash flows in each period
        periods: Period numbers (months)
        yield_rate: Yield rate per period
        
    Returns:
        Modified duration in periods
    """
    if sum(cash_flows) == 0 or yield_rate == 0:
        return 0
        
    pv_factors = 1 / (1 + yield_rate) ** periods
    pvs = cash_flows * pv_factors
    total_pv = sum(pvs)
    
    if total_pv == 0:
        return 0
        
    weighted_periods = sum((pvs * periods) / total_pv)
    return weighted_periods / (1 + yield_rate)

def calculate_convexity(cash_flows: np.ndarray, 
                       periods: np.ndarray, 
                       yield_rate: float) -> float:
    """
    Calculate convexity (second derivative of price with respect to yield).
    
    Args:
        cash_flows: Cash flows in each period
        periods: Period numbers (months)
        yield_rate: Yield rate per period
        
    Returns:
        Convexity measure
    """
    if sum(cash_flows) == 0 or yield_rate == 0:
        return 0
        
    pv_factors = 1 / (1 + yield_rate) ** periods
    pvs = cash_flows * pv_factors
    total_pv = sum(pvs)
    
    if total_pv == 0:
        return 0
        
    weighted_periods_squared = sum((pvs * periods * (periods + 1)) / total_pv)
    return weighted_periods_squared / ((1 + yield_rate) ** 2)

def calculate_sharpe_ratio(returns: float, 
                         risk_free_rate: float, 
                         volatility: float,
                         default_risk_premium: float = 0.02) -> float:
    """
    Calculate Sharpe ratio (return per unit of risk).
    
    Args:
        returns: Annualized returns
        risk_free_rate: Risk-free rate
        volatility: Standard deviation of returns
        default_risk_premium: Additional risk premium for credit assets
        
    Returns:
        Sharpe ratio
    """
    if volatility == 0:
        return 0
        
    return (returns - risk_free_rate - default_risk_premium) / volatility

def compute_extended_metrics(params: InputParams, 
                           sim: SimulationResults, 
                           risk_free_rate: float = 0.04) -> ExtendedFinancialMetrics:
    """
    Compute extended financial metrics from simulation results.
    
    Args:
        params: Input parameters
        sim: Simulation results
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Extended financial metrics
    """
    # Extract necessary data from simulation results
    months = sim.months
    irr_by_tenor = {}
    recovery_rates = {}
    wal_by_tenor = {}
    duration_by_tenor = {}
    convexity_by_tenor = {}
    sharpe_ratios = {}
    
    # Calculate metrics for each tenor
    for n in [1, 2, 3]:
        # Prepare cash flows for IRR calculation (-outflow, +inflow)
        tenor_outflows = -sim.new_by_t[:, n-1]
        tenor_inflows = np.array([sim.payment_by_tenor[n][m] for m in months + 1])
        
        # Combine outflows and inflows into one array for IRR calculation
        cash_flows = np.zeros(len(months) + 1)
        cash_flows[:len(months)] += tenor_outflows
        cash_flows[1:] += tenor_inflows
        
        # Calculate IRR
        irr_by_tenor[n] = calculate_irr(cash_flows)
        
        # Calculate recovery rate
        recovery_rates[n] = calculate_recovery_rate(
            sim.out_gross[n], 
            sim.def_out[n], 
            params.default_rates[n]
        )
        
        # Calculate WAL
        principal_flows = np.array([sim.principal_by_tenor[n][m] for m in months + 1])
        wal_by_tenor[n] = calculate_weighted_average_life(principal_flows, months + 1)
        
        # Calculate duration and convexity
        yield_rate = params.monthly_interest
        all_flows = np.array([sim.payment_by_tenor[n][m] for m in months + 1])
        duration_by_tenor[n] = calculate_modified_duration(all_flows, months + 1, yield_rate)
        convexity_by_tenor[n] = calculate_convexity(all_flows, months + 1, yield_rate)
        
        # Calculate Sharpe ratio 
        # Convert monthly returns to annual
        annual_return = (1 + irr_by_tenor[n])**12 - 1 if not np.isnan(irr_by_tenor[n]) else 0
        # Use default rate as a proxy for volatility in credit assets
        volatility = params.default_rates[n]
        sharpe_ratios[n] = calculate_sharpe_ratio(annual_return, risk_free_rate, volatility)
    
    # Calculate portfolio-wide metrics using two different approaches
    # Approach 1: Direct aggregate cash flow approach
    portfolio_cash_flows = np.zeros(len(months) + 1)
    for m in range(len(months)):
        portfolio_cash_flows[m] -= sum(sim.new_by_t[m])
    
    for m in months + 1:
        portfolio_cash_flows[m-1 if m > 0 else 0] += sum(sim.payment_by_tenor[n][m] for n in [1, 2, 3])
    
    portfolio_irr = calculate_irr(portfolio_cash_flows)
    print(f"Direct portfolio IRR calculation: {portfolio_irr}")
    
    # Approach 2: Alternative portfolio IRR calculation using combined tenor cash flows
    # This can work even if some individual tenor IRRs are uncalculable
    if np.isnan(portfolio_irr):
        print("Direct portfolio IRR calculation failed, trying alternative approach")
        # Prepare combined cash flows across all tenors
        combined_cash_flows = np.zeros(len(months) + 1)
        total_investment = 0
        
        # First, sum all investments (outflows)
        for m in range(len(months)):
            month_investment = sum(sim.new_by_t[m])
            combined_cash_flows[m] -= month_investment
            total_investment += month_investment
        
        # Then add all inflows
        for m in months + 1:
            period_index = m-1 if m > 0 else 0
            period_inflow = sum(sim.payment_by_tenor[n][m] for n in [1, 2, 3])
            combined_cash_flows[period_index] += period_inflow
        
        # If we have both inflows and outflows, attempt IRR calculation
        if np.any(combined_cash_flows < 0) and np.any(combined_cash_flows > 0):
            print(f"Trying with combined cash flows: {combined_cash_flows}")
            portfolio_irr = calculate_irr(combined_cash_flows)
            print(f"Alternative portfolio IRR calculation: {portfolio_irr}")
        
        # If still NaN, try a modified version using totals
        if np.isnan(portfolio_irr) and total_investment > 0:
            print("Still unable to calculate IRR, using simplified approach")
            # Calculate a simple ROI-based estimate
            total_inflows = sum(combined_cash_flows[combined_cash_flows > 0])
            total_outflows = abs(sum(combined_cash_flows[combined_cash_flows < 0]))
            
            if total_outflows > 0:
                # Estimate as a simple MOIC-based IRR over the simulation period
                moic = total_inflows / total_outflows
                print(f"MOIC ratio: {moic}")
                
                if moic > 1.0:
                    # Positive return case
                    estimated_irr = (moic**(1/len(months)) - 1)
                    print(f"Estimated portfolio IRR from MOIC ({moic}): {estimated_irr}")
                    portfolio_irr = estimated_irr
                elif moic > 0.0001:
                    # Loss but not total loss
                    n_periods = len(months)
                    # Calculate terminal value as a percentage of initial investment
                    terminal_value_pct = moic
                    # Estimate IRR with assumption of linear loss over the period
                    estimated_irr = -((1 - terminal_value_pct**(1/n_periods)))
                    print(f"Estimated negative portfolio IRR from MOIC ({moic}): {estimated_irr}")
                    portfolio_irr = max(estimated_irr, -0.50)  # Floor at -50% monthly
                else:
                    # Severe or total loss case - cap at a significant negative return
                    # For severe losses (>95%), use a scaling factor based on default rate
                    avg_default_rate = sum(params.default_rates.values()) / len(params.default_rates)
                    severity_factor = min(5.0, avg_default_rate * 10)  # Scale based on default rates
                    estimated_irr = -0.25 * severity_factor  # From -25% to -125% monthly
                    print(f"Extreme loss scenario, assigning IRR of {estimated_irr}")
                    portfolio_irr = estimated_irr
            else:
                # Pathological case - no outflows at all
                print("No investment outflows detected, cannot calculate IRR")
                portfolio_irr = np.nan
    
    # Calculate concentration using allocation
    concentration_hhi = calculate_concentration_hhi(params.allocation)
    
    # Calculate portfolio WAL
    all_principal_flows = np.array([
        sum(sim.principal_by_tenor[n][m] for n in [1, 2, 3]) 
        for m in months + 1
    ])
    portfolio_wal = calculate_weighted_average_life(all_principal_flows, months + 1)
    
    # Calculate volatility of cash flows
    monthly_inflows = np.array([
        sum(sim.payment_by_tenor[n][m] for n in [1, 2, 3]) 
        for m in months[1:] + 1  # Skip month 0 as it's initial investment
    ])
    volatility = calculate_cashflow_volatility(monthly_inflows)
    
    return ExtendedFinancialMetrics(
        irr=irr_by_tenor,
        portfolio_irr=portfolio_irr,
        sharpe_ratios=sharpe_ratios,
        recovery_rates=recovery_rates,
        weighted_average_life=wal_by_tenor,
        portfolio_wal=portfolio_wal,
        concentration_hhi=concentration_hhi,
        duration=duration_by_tenor,
        convexity=convexity_by_tenor,
        volatility=volatility
    )

def format_metrics_for_display(metrics: ExtendedFinancialMetrics) -> Dict:
    """
    Format metrics for display in the UI.
    
    Args:
        metrics: ExtendedFinancialMetrics object
        
    Returns:
        Dictionary of formatted metrics for display
    """
    results = {}
    
    # Format IRRs as percentages with better error handling
    results["IRR (%)"] = {}
    for n in [1, 2, 3]:
        if np.isnan(metrics.irr[n]):
            results["IRR (%)"][f"{n}-Month"] = "N/A"
        elif np.isinf(metrics.irr[n]):
            results["IRR (%)"][f"{n}-Month"] = "∞" if metrics.irr[n] > 0 else "-∞"
        elif abs(metrics.irr[n]) > 1:
            # Handle extremely large values
            results["IRR (%)"][f"{n}-Month"] = "Extreme" 
        else:
            results["IRR (%)"][f"{n}-Month"] = f"{metrics.irr[n]*100:.2f}%"
    
    # Debug IRR calculation results
    print(f"IRR values before formatting: {metrics.irr}")
    print(f"Portfolio IRR before formatting: {metrics.portfolio_irr}")
    
    # More robust handling of portfolio IRR
    if np.isnan(metrics.portfolio_irr):
        results["Portfolio IRR"] = "N/A"
        print("Portfolio IRR is NaN - displaying as N/A")
    elif metrics.portfolio_irr == np.inf or metrics.portfolio_irr == -np.inf:
        results["Portfolio IRR"] = "∞" if metrics.portfolio_irr > 0 else "-∞"
        print(f"Portfolio IRR is {'positive' if metrics.portfolio_irr > 0 else 'negative'} infinity")
    elif abs(metrics.portfolio_irr) > 1:
        # Handle extremely large values
        results["Portfolio IRR"] = f"Extreme ({metrics.portfolio_irr*100:.1f}%)"
        print(f"Portfolio IRR is extremely large: {metrics.portfolio_irr}")
    else:
        results["Portfolio IRR"] = f"{metrics.portfolio_irr*100:.2f}%"
        print(f"Portfolio IRR formatted as: {results['Portfolio IRR']}")
    
    # Format Sharpe ratios with better error handling
    results["Sharpe Ratios"] = {}
    for n in [1, 2, 3]:
        if np.isnan(metrics.sharpe_ratios[n]):
            results["Sharpe Ratios"][f"{n}-Month"] = "N/A"
        elif np.isinf(metrics.sharpe_ratios[n]):
            results["Sharpe Ratios"][f"{n}-Month"] = "∞" if metrics.sharpe_ratios[n] > 0 else "-∞"
        elif abs(metrics.sharpe_ratios[n]) > 100:
            results["Sharpe Ratios"][f"{n}-Month"] = "Extreme"
        else:
            results["Sharpe Ratios"][f"{n}-Month"] = f"{metrics.sharpe_ratios[n]:.2f}"
    
    # Format recovery rates as percentages with better error handling
    results["Recovery Rates (%)"] = {}
    for n in [1, 2, 3]:
        if np.isnan(metrics.recovery_rates[n]):
            results["Recovery Rates (%)"][f"{n}-Month"] = "N/A"
        elif metrics.recovery_rates[n] == np.inf:
            results["Recovery Rates (%)"][f"{n}-Month"] = "∞"
        elif metrics.recovery_rates[n] < 0:
            results["Recovery Rates (%)"][f"{n}-Month"] = "0.00%"  # Floor at 0
        elif metrics.recovery_rates[n] > 1:
            results["Recovery Rates (%)"][f"{n}-Month"] = "100.00%"  # Cap at 100%
        else:
            results["Recovery Rates (%)"][f"{n}-Month"] = f"{metrics.recovery_rates[n]*100:.2f}%"
    
    # Format WAL in months with error handling
    results["Weighted Average Life (months)"] = {}
    for n in [1, 2, 3]:
        if np.isnan(metrics.weighted_average_life[n]):
            results["Weighted Average Life (months)"][f"{n}-Month"] = "N/A"
        elif np.isinf(metrics.weighted_average_life[n]):
            results["Weighted Average Life (months)"][f"{n}-Month"] = "∞"
        else:
            results["Weighted Average Life (months)"][f"{n}-Month"] = f"{metrics.weighted_average_life[n]:.2f}"
    
    # Format portfolio WAL with error handling
    if np.isnan(metrics.portfolio_wal):
        results["Portfolio WAL"] = "N/A"
    elif np.isinf(metrics.portfolio_wal):
        results["Portfolio WAL"] = "∞"
    else:
        results["Portfolio WAL"] = f"{metrics.portfolio_wal:.2f} months"
    
    # Format concentration risk (HHI) with error handling
    if np.isnan(metrics.concentration_hhi):
        results["Concentration Risk"] = "N/A"
    elif np.isinf(metrics.concentration_hhi):
        results["Concentration Risk"] = "∞"
    else:
        hhi_level = "Low" if metrics.concentration_hhi < 0.15 else "Medium" if metrics.concentration_hhi < 0.25 else "High"
        results["Concentration Risk"] = f"{metrics.concentration_hhi:.2f} ({hhi_level})"
    
    # Format duration and convexity with better error handling
    results["Duration (months)"] = {}
    for n in [1, 2, 3]:
        if np.isnan(metrics.duration[n]):
            results["Duration (months)"][f"{n}-Month"] = "N/A"
        elif np.isinf(metrics.duration[n]):
            results["Duration (months)"][f"{n}-Month"] = "∞"
        else:
            results["Duration (months)"][f"{n}-Month"] = f"{metrics.duration[n]:.2f}"
    
    results["Convexity"] = {}
    for n in [1, 2, 3]:
        if np.isnan(metrics.convexity[n]):
            results["Convexity"][f"{n}-Month"] = "N/A"
        elif np.isinf(metrics.convexity[n]):
            results["Convexity"][f"{n}-Month"] = "∞"
        elif abs(metrics.convexity[n]) > 1000:
            results["Convexity"][f"{n}-Month"] = "Extreme"
        else:
            results["Convexity"][f"{n}-Month"] = f"{metrics.convexity[n]:.2f}"
    
    # Format volatility with error handling
    if np.isnan(metrics.volatility):
        results["Cash Flow Volatility"] = "N/A"
    elif np.isinf(metrics.volatility):
        results["Cash Flow Volatility"] = "∞"
    elif metrics.volatility > 10:
        volatility_level = "Extreme"
        results["Cash Flow Volatility"] = f"{min(metrics.volatility, 999.99):.2f} ({volatility_level})"
    else:
        volatility_level = "Low" if metrics.volatility < 0.5 else "Medium" if metrics.volatility < 1.0 else "High"
        results["Cash Flow Volatility"] = f"{metrics.volatility:.2f} ({volatility_level})"
    
    print(f"Formatted metrics for display: {results}")
    return results 