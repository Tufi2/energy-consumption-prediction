import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def display_prediction_results(prediction_params):
    """Display prediction results including charts and metrics"""
    
    # Display loading state while generating predictions
    with st.spinner("Generating predictions..."):
        # Simulate prediction delay
        st.empty()
    
    # Create container for results
    st.markdown("""
        <div class="results-container">
            <h3 class="results-title">Prediction Results</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Display prediction metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Predicted Consumption</div>
                <div class="metric-value">567.8 kWh</div>
                <div class="metric-subtitle">Next 24 hours</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Confidence Interval</div>
                <div class="metric-value">±5.2%</div>
                <div class="metric-subtitle">Based on historical accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Potential Savings</div>
                <div class="metric-value">$123.45</div>
                <div class="metric-subtitle">Optimization potential</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Create prediction chart
    create_prediction_chart(prediction_params)
    
    # Display recommendations
    st.markdown("### Recommendations")
    display_recommendations()

def create_prediction_chart(params):
    """Create interactive prediction chart"""
    # Generate sample data
    dates = pd.date_range(
        start=params["start_date"],
        end=params["end_date"],
        freq='H'
    )
    
    # Actual values (historical)
    actual_values = np.random.normal(loc=50, scale=10, size=len(dates)//2)
    
    # Predicted values
    predicted_values = np.random.normal(
        loc=50,
        scale=10,
        size=len(dates)-len(actual_values)
    )
    
    # Confidence intervals
    confidence_upper = predicted_values + predicted_values * 0.1
    confidence_lower = predicted_values - predicted_values * 0.1
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates[:len(actual_values)],
        y=actual_values,
        name="Historical",
        line=dict(color="#2563EB", width=2)
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=dates[len(actual_values):],
        y=predicted_values,
        name="Predicted",
        line=dict(color="#10B981", width=2, dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=dates[len(actual_values):],
        y=confidence_upper,
        fill=None,
        mode='lines',
        line_color='rgba(16, 185, 129, 0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates[len(actual_values):],
        y=confidence_lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(16, 185, 129, 0)',
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title="Energy Consumption Forecast",
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
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations():
    """Display AI-generated recommendations"""
    recommendations = [
        {
            "type": "optimization",
            "title": "Peak Usage Optimization",
            "description": "Consider shifting heavy consumption tasks to off-peak hours (10 PM - 6 AM) to reduce costs.",
            "impact": "High",
            "savings": "15-20%"
        },
        {
            "type": "warning",
            "title": "Unusual Pattern Detected",
            "description": "Energy usage pattern suggests potential equipment inefficiency in Zone B.",
            "impact": "Medium",
            "savings": "5-10%"
        },
        {
            "type": "info",
            "title": "Weather Impact",
            "description": "Upcoming temperature rise may increase HVAC energy consumption.",
            "impact": "Medium",
            "savings": "8-12%"
        }
    ]
    
    for rec in recommendations:
        st.markdown(f"""
            <div class="recommendation-card {rec['type']}">
                <div class="rec-header">
                    <div class="rec-title">{rec['title']}</div>
                    <div class="rec-impact">{rec['impact']} Impact</div>
                </div>
                <div class="rec-description">{rec['description']}</div>
                <div class="rec-savings">Potential Savings: {rec['savings']}</div>
            </div>
        """, unsafe_allow_html=True)