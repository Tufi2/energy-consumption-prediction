import streamlit as st
from components.dashboard.metrics import display_metrics
from components.dashboard.charts import create_consumption_chart, create_distribution_chart
from components.dashboard.filters import create_filters

def show():
    st.markdown("""
        <div class="dashboard-container">
            <h1 class="page-title">Energy Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Filters Section
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_date = st.date_input("Select Date")
        with col2:
            selected_period = st.selectbox("Time Period", 
                ["Daily", "Weekly", "Monthly", "Yearly"])
        with col3:
            selected_type = st.selectbox("Consumption Type",
                ["All", "Residential", "Industrial"])

    # Metrics Row
    st.markdown("<div class='metrics-row'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Current Usage</div>
                <div class="metric-value">45.2 kWh</div>
                <div class="metric-change positive">+2.3% vs yesterday</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Predicted Peak</div>
                <div class="metric-value">78.5 kWh</div>
                <div class="metric-change negative">-5.1% vs last week</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Efficiency Score</div>
                <div class="metric-value">92/100</div>
                <div class="metric-change positive">+4 points</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Cost Savings</div>
                <div class="metric-value">$1,234</div>
                <div class="metric-change positive">+12.3% this month</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts Section
    st.markdown("<div class='charts-section'>", unsafe_allow_html=True)
    
    # Main consumption chart
    st.markdown("""
        <div class="chart-container">
            <h3>Energy Consumption Trend</h3>
        </div>
    """, unsafe_allow_html=True)
    create_consumption_chart()
    
    # Distribution charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="chart-container">
                <h3>Usage by Time of Day</h3>
            </div>
        """, unsafe_allow_html=True)
        create_distribution_chart('time')
    
    with col2:
        st.markdown("""
            <div class="chart-container">
                <h3>Usage by Category</h3>
            </div>
        """, unsafe_allow_html=True)
        create_distribution_chart('category')
    
    st.markdown("</div>", unsafe_allow_html=True)