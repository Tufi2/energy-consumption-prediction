# frontend/pages/analysis.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def show():
    st.markdown("""
        <div class="welcome-header">
            <h1>Energy Consumption Analysis</h1>
            <p class="subtitle">Deep Insights into Energy Usage Patterns</p>
        </div>
    """, unsafe_allow_html=True)

    # Analysis Controls
    col1, col2 = st.columns(2)
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last Week", "Last Month", "Last Year", "Custom Range"]
        )
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Usage Patterns", "Cost Analysis", "Efficiency Metrics"]
        )

    # Create sample data for patterns
    hours = list(range(24))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data = np.random.normal(loc=50, scale=10, size=(7, 24))

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=hours,
        y=days,
        colorscale='Viridis',
        hoverongaps=False
    ))

    fig.update_layout(
        title='Weekly Energy Usage Pattern',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Additional Analysis Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-emoji">⚡</div>
                <div class="metric-label">Peak Usage Time</div>
                <div class="metric-value">2:00 PM</div>
                <div class="metric-delta">Tuesday</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-emoji">💡</div>
                <div class="metric-label">Average Daily Usage</div>
                <div class="metric-value">42.5 kWh</div>
                <div class="metric-delta positive">↑ 2.1%</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-emoji">📊</div>
                <div class="metric-label">Usage Variability</div>
                <div class="metric-value">±15%</div>
                <div class="metric-delta negative">↓ 3.4%</div>
            </div>
        """, unsafe_allow_html=True)