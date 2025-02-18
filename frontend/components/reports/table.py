import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def create_data_table(data=None):
    """Create an interactive data table with filtering and sorting"""
    if data is None:
        # Sample data for demonstration
        data = generate_sample_data()
    
    st.markdown("""
        <div class="table-container">
            <div class="table-header">
                <h3>Energy Consumption Data</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=[pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()]
        )
    with col2:
        consumption_range = st.slider(
            "Consumption Range (kWh)",
            0, 1000, (0, 1000)
        )
    with col3:
        category = st.multiselect(
            "Category",
            options=["Residential", "Industrial", "Commercial"],
            default=["Residential", "Industrial", "Commercial"]
        )
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(data.columns),
            fill_color='#1e293b',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[data[col] for col in data.columns],
            fill_color='#0f172a',
            align='left',
            font=dict(color='white', size=11)
        )
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return data

def generate_sample_data():
    """Generate sample data for demonstration"""
    dates = pd.date_range(
        start=pd.Timestamp.now() - pd.Timedelta(days=30),
        end=pd.Timestamp.now(),
        freq='D'
    )
    
    data = pd.DataFrame({
        'Date': dates,
        'Consumption (kWh)': np.random.normal(500, 100, len(dates)),
        'Category': np.random.choice(
            ['Residential', 'Industrial', 'Commercial'],
            len(dates)
        ),
        'Cost ($)': np.random.normal(100, 20, len(dates)),
        'Efficiency Score': np.random.normal(85, 5, len(dates))
    })
    
    data['Consumption (kWh)'] = data['Consumption (kWh)'].round(2)
    data['Cost ($)'] = data['Cost ($)'].round(2)
    data['Efficiency Score'] = data['Efficiency Score'].round(1)
    
    return data