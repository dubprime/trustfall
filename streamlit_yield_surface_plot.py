
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Yield Surface Visualizer", layout="wide")

st.title("🧠 Yield Surface: Interest × Duration → Annualized Return")
st.markdown("Visualize how annualized yield varies with loan duration and interest per loan.")

# Grid of values
interest_vals = np.linspace(0.01, 0.25, 50)  # Interest per loan
duration_vals = np.linspace(1, 12, 50)       # Loan duration in months

I, D = np.meshgrid(interest_vals, duration_vals)
annual_yield = I * (12 / D)  # Annualized cohort yield

# 3D Surface plot
fig = go.Figure(data=[go.Surface(
    z=annual_yield,
    x=interest_vals,
    y=duration_vals,
    colorscale='Viridis'
)])

fig.update_layout(
    title="🎛️ Annualized Yield Surface",
    scene=dict(
        xaxis_title='Interest per Loan',
        yaxis_title='Loan Duration (Months)',
        zaxis_title='Annualized Yield'
    ),
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=50)
)

st.plotly_chart(fig, use_container_width=True)
