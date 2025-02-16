# frontend/pages/predictions.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from components.charts import create_consumption_chart

def show():
    st.markdown("""
        <div class="welcome-header">
            <h1>Energy Consumption Predictions</h1>
            <p class="subtitle">Forecast Future Energy Usage</p>
        </div>
    """, unsafe_allow_html=True)

    # Prediction Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        prediction_period = st.selectbox(
            "Prediction Period",
            ["24 Hours", "7 Days", "30 Days"]
        )
    with col2:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
    with col3:
        include_weather = st.checkbox("Include Weather Data", value=True)

    # Prediction Results
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    # Sample prediction data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    actual_values = np.random.normal(loc=50, scale=10, size=30)
    predicted_values = actual_values + np.random.normal(loc=0, scale=2, size=30)
    upper_bound = predicted_values + 5
    lower_bound = predicted_values - 5

    # Create prediction plot
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates[:15],
        y=actual_values[:15],
        name='Actual Consumption',
        line=dict(color='#FF4B4B', width=2)
    ))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=dates[14:],
        y=predicted_values[14:],
        name='Predicted Consumption',
        line=dict(color='#00ff00', width=2, dash='dash')
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=dates[14:],
        y=upper_bound[14:],
        fill=None,
        mode='lines',
        line_color='rgba(0,255,0,0)',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=dates[14:],
        y=lower_bound[14:],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,255,0,0)',
        name='Confidence Interval',
        fillcolor='rgba(0,255,0,0.1)'
    ))

    fig.update_layout(
        title='Energy Consumption Forecast',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA')
        ),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-emoji">📈</div>
                <div class="metric-label">Predicted Peak</div>
                <div class="metric-value">67.8 kWh</div>
                <div class="metric-delta positive">↑ 5.2%</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-emoji">📉</div>
                <div class="metric-label">Predicted Low</div>
                <div class="metric-value">23.4 kWh</div>
                <div class="metric-delta negative">↓ 3.1%</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-emoji">🎯</div>
                <div class="metric-label">Prediction Accuracy</div>
                <div class="metric-value">95.2%</div>
                <div class="metric-delta positive">↑ 1.3%</div>
            </div>
        """, unsafe_allow_html=True)