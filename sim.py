# Re-import necessary packages after code state reset
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Define base assumptions again
loan_durations = [1, 2, 3]  # in months
interest_rates = [0.03, 0.06, 0.09]  # interest per loan
sweep = np.linspace(0.1, 0.9, 17)
results = []

for share_1m in sweep:
    share_2m = (1 - share_1m) / 2
    share_3m = 1 - share_1m - share_2m

    shares = [share_1m, share_2m, share_3m]
    cycles_per_year = [12 / d for d in loan_durations]
    
    # Weighted loan duration
    weighted_duration = sum([shares[i] * loan_durations[i] for i in range(3)])
    
    # Capital turnover (simple)
    capital_turnover = 12 / weighted_duration

    # Interest per loan
    interest_per_loan = sum([shares[i] * interest_rates[i] for i in range(3)])

    # Simple Annual Yield (Portfolio-Based)
    annual_yield = interest_per_loan * capital_turnover

    # Advanced Annual Yield (Cohort-Normalized)
    advanced_yield = sum([shares[i] * interest_rates[i] * cycles_per_year[i] for i in range(3)])

    results.append({
        "1M Share": share_1m,
        "2M Share": share_2m,
        "3M Share": share_3m,
        "Weighted Duration (Months)": weighted_duration,
        "Capital Turnover": capital_turnover,
        "Interest per Loan": interest_per_loan,
        "Simple Annual Yield": annual_yield,
        "Advanced Annual Yield": advanced_yield
    })

df_results = pd.DataFrame(results)

# Create interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_results["1M Share"], y=df_results["Simple Annual Yield"],
    mode='lines+markers',
    name='Simple Annual Yield',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=df_results["1M Share"], y=df_results["Advanced Annual Yield"],
    mode='lines+markers',
    name='Advanced Annual Yield',
    line=dict(color='green')
))

fig.update_layout(
    title="📈 Sensitivity of Annual Yield to 1-Month Loan Share",
    xaxis_title="Share of 1-Month Loans",
    yaxis_title="Annual Yield (Decimal)",
    hovermode="x unified",
    template="plotly_white"
)

fig.show()
