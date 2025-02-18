import streamlit as st
from components.reports.table import create_data_table
from components.reports.export import create_export_section

def show():
    st.markdown("""
        <div class="page-header">
            <h1>Energy Consumption Reports</h1>
            <p>View, analyze, and export energy consumption data</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different report views
    tab1, tab2, tab3 = st.tabs([
        "Data Table",
        "Summary Reports",
        "Custom Reports"
    ])
    
    with tab1:
        # Display interactive data table
        data = create_data_table()
        
        # Add export functionality
        create_export_section(data)
    
    with tab2:
        st.markdown("### Summary Reports")
        # Add summary reports functionality
        
    with tab3:
        st.markdown("### Custom Reports")
        # Add custom report builder