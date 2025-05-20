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

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the simulator:
   ```
   streamlit run streamlit_app.py
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
- Plotly
- Matplotlib

## Contributing

See the [Financial Validation Checklist](FINANCE_TODO.md) for current development priorities.

1. Implement items from the checklist
2. Add tests for new functionality
3. Ensure financial soundness of all calculations
4. Submit pull requests with detailed explanations

