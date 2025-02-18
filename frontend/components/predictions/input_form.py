import streamlit as st
import datetime

def create_prediction_form():
    """Create the prediction input form with all necessary parameters"""
    
    st.markdown("""
        <div class="form-container">
            <h3 class="form-title">Prediction Parameters</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        # Time Range Selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                min_value=datetime.date.today(),
                value=datetime.date.today()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                min_value=start_date,
                value=start_date + datetime.timedelta(days=7)
            )
        
        # Prediction Type
        prediction_type = st.selectbox(
            "Prediction Type",
            options=[
                "Short-term (24 hours)",
                "Medium-term (7 days)",
                "Long-term (30 days)"
            ]
        )
        
        # Confidence Level
        confidence_level = st.slider(
            "Confidence Level",
            min_value=80,
            max_value=99,
            value=95,
            help="Higher confidence level means wider prediction intervals"
        )
        
        # Additional Factors
        st.markdown("### External Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            include_weather = st.checkbox("Include Weather Data", value=True)
            if include_weather:
                temperature = st.number_input(
                    "Expected Temperature (°C)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=25.0
                )
                humidity = st.number_input(
                    "Expected Humidity (%)",
                    min_value=0,
                    max_value=100,
                    value=60
                )
        
        with col2:
            include_events = st.checkbox("Include Special Events", value=False)
            if include_events:
                event_type = st.selectbox(
                    "Event Type",
                    options=[
                        "Holiday",
                        "Maintenance",
                        "Peak Production",
                        "Other"
                    ]
                )
        
        # Advanced Options
        with st.expander("Advanced Options"):
            model_type = st.selectbox(
                "Model Type",
                options=[
                    "Auto (Recommended)",
                    "LSTM",
                    "Prophet",
                    "XGBoost"
                ]
            )
            
            feature_importance = st.checkbox(
                "Show Feature Importance",
                value=True
            )
            
            export_results = st.checkbox(
                "Export Results",
                value=False
            )
        
        # Submit Button
        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )
        
        return {
            "submitted": submitted,
            "start_date": start_date,
            "end_date": end_date,
            "prediction_type": prediction_type,
            "confidence_level": confidence_level,
            "include_weather": include_weather,
            "temperature": temperature if include_weather else None,
            "humidity": humidity if include_weather else None,
            "include_events": include_events,
            "event_type": event_type if include_events else None,
            "model_type": model_type,
            "feature_importance": feature_importance,
            "export_results": export_results
        }