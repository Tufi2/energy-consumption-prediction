import streamlit as st
from components.predictions.input_form import create_prediction_form
from components.predictions.results import display_prediction_results

def show():
    st.markdown("""
        <div class="predictions-container">
            <h1 class="page-title">Energy Consumption Predictions</h1>
            <p class="page-description">
                Get accurate predictions for future energy consumption based on historical data and external factors.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
            <div class="input-section">
                <h3>Prediction Parameters</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Date Range
        st.date_input("Start Date")
        st.date_input("End Date")
        
        # Additional Parameters
        st.selectbox("Prediction Type", 
            ["Short-term (24h)", "Medium-term (7d)", "Long-term (30d)"])
        
        st.selectbox("Location", 
            ["All Locations", "Location 1", "Location 2"])
        
        st.slider("Confidence Level", 80, 99, 95)
        
        # Weather Factors
        st.markdown("### Weather Factors")
        st.checkbox("Include Temperature")
        st.checkbox("Include Humidity")
        st.checkbox("Include Wind Speed")
        
        # Generate Button
        st.button("Generate Prediction", type="primary")

    with col2:
        st.markdown("""
            <div class="results-section">
                <h3>Prediction Results</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Prediction Metrics
        st.markdown("""
            <div class="prediction-metrics">
                <div class="metric-card">
                    <div class="metric-title">Predicted Consumption</div>
                    <div class="metric-value">567.8 kWh</div>
                    <div class="metric-range">Range: 550.2 - 585.4 kWh</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Prediction Chart
        st.markdown("""
            <div class="chart-container">
                <h4>Predicted vs Actual Consumption</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("""
            <div class="recommendations">
                <h4>Recommendations</h4>
                <ul class="recommendation-list">
                    <li>Consider shifting heavy consumption to off-peak hours</li>
                    <li>Expected high demand on Wednesday - optimize scheduling</li>
                    <li>Weather forecast suggests increased HVAC usage</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)