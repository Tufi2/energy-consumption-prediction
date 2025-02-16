# frontend/components/charts.py
import streamlit as st
import plotly.graph_objects as go
import time

def animated_metric(label, value, delta, delta_type, emoji):
    """Animated metric display with emoji"""
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji}</div>
            <div style="font-size: 1.2rem; color: #FAFAFA;">{label}</div>
            <div style="font-size: 2rem; font-weight: 600; color: #FF4B4B;">{value}</div>
            <div style="color: {'#00ff00' if delta_type == 'positive' else '#ff0000'}">
                {delta}
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_gauge_chart(value):
    """Create an animated gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Efficiency Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#FF4B4B"},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(255, 75, 75, 0.4)'},
                {'range': [80, 100], 'color': 'rgba(255, 75, 75, 0.6)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FAFAFA"},
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)