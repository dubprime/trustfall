# DubPrime Loan Portfolio Simulator

A comprehensive simulation tool for managing and analyzing short-term loan portfolios with different tenors, default rates, and allocation strategies.

## Overview

The DubPrime Loan Portfolio Simulator allows credit analysts and portfolio managers to model the performance of loan portfolios with varying parameters:

- **Initial capital**: The starting investment amount
- **Interest rates**: Monthly interest rates for the portfolio
- **Tenor allocation**: Distribution of capital across different loan terms (1, 2, and 3 months)
- **Default rates**: Expected default rates for each tenor
- **Duration**: Simulation length in months

The simulator generates detailed visualizations of portfolio performance, cash flows, defaults, and returns.

## Key Features

- **Interactive UI**: Adjust parameters and see results in real-time
- **Cash Flow Modeling**: View detailed breakdowns of principal, interest, and defaults
- **Tenor-Based Analysis**: Compare performance of different loan durations
- **Default Visualization**: See the impact of defaults on portfolio performance
- **Advanced Financial Metrics**: Calculate IRR, WAL, duration, and risk-adjusted returns
- **Stress Testing**: Model portfolio performance under various economic scenarios
- **Robust IRR Calculations**: Accurately handle extreme stress scenarios with novel estimation techniques

## Financial Validation Suite

To ensure the financial soundness of the simulator, we've created a comprehensive validation suite:

### Validation Checklist

See the complete [Financial Validation Checklist](FINANCE_TODO.md) for a detailed breakdown of validation requirements.

### Testing Modules

- **test_finance_calculations.py**: Unit tests for core financial calculations
- **integration_test.py**: End-to-end tests for the entire system
- **financial_metrics.py**: Advanced financial metrics beyond basic simulation
- **stress_testing.py**: Tools for scenario analysis and stress testing

## Advanced Financial Metrics

The simulator now includes advanced financial metrics:

- **Internal Rate of Return (IRR)**: Time-weighted returns by tenor and portfolio-wide
- **Weighted Average Life (WAL)**: Average time to principal repayment
- **Recovery Rates**: Effective recovery on defaulted loans
- **Sharpe Ratios**: Risk-adjusted returns
- **Duration and Convexity**: Interest rate sensitivity measures
- **Concentration Risk**: Using Herfindahl-Hirschman Index (HHI)
- **Cash Flow Volatility**: Variability in periodic cash flows

## Stress Testing

Analyze portfolio performance under various scenarios:

- **Default Stress**: Increased default rates
- **Interest Rate Changes**: Rising or falling rate environments
- **Allocation Shifts**: Changes in tenor distribution
- **Combined Scenarios**: Multiple simultaneous stresses
- **Sensitivity Analysis**: Parameter-specific impact testing

### Robust Stress Scenario IRR Calculation

The simulator features a sophisticated multi-tiered approach to IRR calculation in extreme stress scenarios:

1. **Standard IRR Calculation**: First attempts the conventional IRR calculation using `numpy_financial.irr`
2. **Alternative Cash Flow Analysis**: If the standard approach fails, tries a restructured cash flow analysis
3. **MOIC-Based Estimation**: For severely distressed portfolios, estimates IRR using a modified MOIC approach
4. **Default-Rate Scaled Estimations**: In catastrophic scenarios, uses default rate scaling to provide meaningful negative IRR values

This advanced approach ensures that even in extreme stress cases where conventional IRR calculation fails (due to all-negative cash flows), the simulator still provides financially meaningful metrics, avoiding "N/A" or "Not calculable" results.

### Improved Financial Reporting

Stress test reports now use descriptive financial terminology for extreme scenarios:

- **Positive Returns**: Shown as standard percentages (e.g., "16.17%")
- **Minor Losses**: Displayed as "Loss (-X.XX%)" 
- **Significant Losses**: Shown as "Major Loss (-XX.X%)"
- **Severe Losses**: Displayed as "Severe Loss (-XX%)"
- **Catastrophic Losses**: Indicated as "Catastrophic (-XX%)"
- **Total Losses**: Clearly labeled as "Total Loss" rather than "-∞"

This improved terminology matches how credit analysts and portfolio managers would describe portfolio performance in real-world stressed environments.

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the simulator:
   ```
   streamlit run streamlit_stress_test_app.py
   ```

3. Adjust parameters in the sidebar and explore the visualizations

## Running Tests

```
python -m unittest test_finance_calculations.py
python -m unittest integration_test.py
```

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- NumPy-Financial
- Plotly
- Matplotlib

## Contributing

See the [Financial Validation Checklist](FINANCE_TODO.md) for current development priorities.

1. Implement items from the checklist
2. Add tests for new functionality
3. Ensure financial soundness of all calculations
4. Submit pull requests with detailed explanations


## Legal
This file is part of Mythral AI™ which is owned by DubPrime, Inc. Copyright (C) 2025 Vincent Lucero vincent@dubprime.com https://www.linkedin.com/in/vincent-lucero/

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see https://www.gnu.org/licenses/.

https://www.linkedin.com/posts/vincent-lucero_github-dubprimemythral-mythral-from-original-activity-7324903865089171456-2YZo?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAT4nS4BCc0uvhjl5cK8VfFtWKFPuI0Z3w0
