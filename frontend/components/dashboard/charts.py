import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_consumption_chart():
    """Create main consumption trend chart"""
    # Sample data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    values = np.random.normal(loc=50, scale=10, size=30)
    values = np.cumsum(values) + 1000
    
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Consumption',
        line=dict(color='#2563EB', width=2),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)'
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=dates[-5:],
        y=values[-5:] * 1.1,
        mode='lines',
        name='Predicted',
        line=dict(color='#10B981', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=None,
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
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_distribution_chart(chart_type='time'):
    """Create distribution charts"""
    if chart_type == 'time':
        # Time distribution data
        hours = list(range(24))
        values = np.random.normal(loc=50, scale=15, size=24)
        
        fig = go.Figure(go.Bar(
            x=hours,
            y=values,
            marker_color='#2563EB'
        ))
        
    else:
        # Category distribution data
        categories = ['HVAC', 'Lighting', 'Equipment', 'Other']
        values = np.random.normal(loc=30, scale=10, size=4)
        
        fig = go.Figure(go.Pie(
            labels=categories,
            values=values,
            hole=0.4
        ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    
    return fig