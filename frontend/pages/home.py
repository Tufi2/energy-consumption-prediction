# frontend/pages/home.py
import streamlit as st
import plotly.graph_objects as go
from components.charts import animated_metric, create_gauge_chart
import time

def show():
    # Animated welcome message
    st.markdown("""
        <div class="welcome-header">
            <h1>Welcome to Energy Consumption Predictor</h1>
            <p class="subtitle">Smart Analytics for Efficient Energy Management</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Animated metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        animated_metric(
            "Current Usage",
            "45.2 kWh",
            "2.3%",
            "positive",
            "⚡"
        )
    
    with col2:
        animated_metric(
            "Efficiency Score",
            "92/100",
            "4.5%",
            "positive",
            "🎯"
        )
    
    with col3:
        animated_metric(
            "Cost Savings",
            "$1,234",
            "12.3%",
            "positive",
            "💰"
        )
    
    # Main dashboard content
    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    
    # Create two columns for charts
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("""
            <div class="chart-container">
                <h3>Energy Consumption Trend</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Animated loading of chart
        with st.spinner("Loading chart..."):
            time.sleep(1)  # Simulate loading
            fig = create_consumption_chart()
            st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        st.markdown("""
            <div class="chart-container">
                <h3>Efficiency Gauge</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Animated gauge chart
        create_gauge_chart(92)
    
    # Bottom section with cards
    st.markdown("<div class='cards-section'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_info_card(
            "Predicted Usage",
            "Tomorrow's predicted consumption will be 5% lower",
            "📊"
        )
    
    with col2:
        create_info_card(
            "Weather Impact",
            "Sunny weather forecasted. Expected lower AC usage",
            "🌤️"
        )
    
    with col3:
        create_info_card(
            "Tips & Recommendations",
            "Consider reducing usage during peak hours (2-5 PM)",
            "💡"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_consumption_chart():
    # Create sample data for demonstration
    import numpy as np
    import pandas as pd
    
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    values = np.random.normal(loc=50, scale=10, size=30)
    values = np.cumsum(values) + 1000
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        fill='tonexty',
        line=dict(color='#FF4B4B', width=2),
        fillcolor='rgba(255, 75, 75, 0.2)'
    ))
    
    fig.update_layout(
        title=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title=None
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    
    return fig

def create_info_card(title, content, emoji):
    st.markdown(f"""
        <div class="info-card">
            <div class="card-emoji">{emoji}</div>
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)