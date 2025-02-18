import streamlit as st

def create_metric_card(title, value, change, icon, change_type="positive"):
    """Create a metric card with animation"""
    return f"""
        <div class="card metric-card animate-slideUp">
            <div class="metric-icon">{icon}</div>
            <div class="metric-content">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-change {change_type}">
                    {change}
                </div>
            </div>
        </div>
    """

def display_metrics():
    """Display all metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card(
                "Current Usage",
                "45.2 kWh",
                "↑ 2.3%",
                "⚡"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card(
                "Efficiency Score",
                "92/100",
                "↑ 4.5%",
                "🎯"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_metric_card(
                "Cost Savings",
                "$1,234",
                "↑ 12.3%",
                "💰"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_metric_card(
                "Predicted Peak",
                "67.8 kWh",
                "↓ 3.1%",
                "📈",
                "negative"
            ),
            unsafe_allow_html=True
        )