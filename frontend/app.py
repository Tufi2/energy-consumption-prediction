import streamlit as st
import sys
import os
from pages import home, dashboard, predictions, reports, alerts, profile

def setup_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Energy Consumption Predictor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/energy-consumption-prediction',
            'Report a bug': "https://github.com/yourusername/energy-consumption-prediction/issues",
            'About': "# Energy Consumption Predictor\n This application predicts and analyzes energy consumption patterns."
        }
    )

def load_css():
    """Load all CSS files"""
    css_files = ['style.css', 'components.css', 'animations.css']
    for css in css_files:
        with open(os.path.join('frontend', 'static', 'css', css)) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <div class="logo-text">⚡ Energy Predictor</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        selected = st.radio(
            "Navigation",
            options=["Home", "Dashboard", "Predictions", "Reports", "Alerts", "Profile"],
            key="nav"
        )
        
        # Additional Sidebar Content
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        
        # Quick Filters
        st.markdown("### Quick Filters")
        date_range = st.select_slider(
            "Time Range",
            options=["1H", "24H", "7D", "30D", "1Y"],
            value="24H"
        )
        
        location = st.selectbox(
            "Location",
            options=["All Locations", "Location 1", "Location 2", "Location 3"]
        )
        
        # User Info
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="user-info">
                <img src="https://via.placeholder.com/40" class="user-avatar">
                <div class="user-details">
                    <div class="user-name">John Doe</div>
                    <div class="user-role">Energy Manager</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        return selected

def main():
    """Main application function"""
    # Setup page configuration
    setup_page_config()
    
    # Load CSS
    load_css()
    
    # Create sidebar and get selected page
    selected_page = create_sidebar()
    
    # Create top navigation bar
    st.markdown("""
        <nav class="navbar">
            <div class="navbar-content">
                <div class="search-box">
                    <input type="text" placeholder="Search...">
                </div>
                <div class="navbar-actions">
                    <button class="nav-button">
                        <i class="fas fa-bell"></i>
                        <span class="notification-badge">3</span>
                    </button>
                    <button class="nav-button">
                        <i class="fas fa-cog"></i>
                    </button>
                </div>
            </div>
        </nav>
    """, unsafe_allow_html=True)
    
    # Display selected page
    try:
        if selected_page == "Home":
            home.show()
        elif selected_page == "Dashboard":
            dashboard.show()
        elif selected_page == "Predictions":
            predictions.show()
        elif selected_page == "Reports":
            reports.show()
        elif selected_page == "Alerts":
            alerts.show()
        elif selected_page == "Profile":
            profile.show()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required components are properly set up.")
    
    # Footer
    st.markdown("""
        <footer class="footer">
            <div class="footer-content">
                <div class="footer-section">
                    <div class="footer-title">Energy Consumption Predictor</div>
                    <div class="footer-subtitle">Smart Analytics for Efficient Energy Management</div>
                </div>
                <div class="footer-section">
                    <div class="footer-links">
                        <a href="#" class="footer-link">Documentation</a>
                        <a href="#" class="footer-link">Support</a>
                        <a href="#" class="footer-link">Privacy Policy</a>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                © 2024 Energy Consumption Predictor. All rights reserved.
            </div>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()