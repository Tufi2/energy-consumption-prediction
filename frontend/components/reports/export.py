import streamlit as st
import pandas as pd
import io
import base64
from datetime import datetime

def create_export_section(data):
    """Create export functionality for reports"""
    st.markdown("""
        <div class="export-container">
            <h3>Export Options</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "Excel", "PDF"]
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_analysis = st.checkbox("Include Analysis", value=True)
    
    with col2:
        st.markdown("### Report Customization")
        report_title = st.text_input(
            "Report Title",
            value=f"Energy Consumption Report - {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        report_description = st.text_area(
            "Report Description",
            value="Detailed analysis of energy consumption patterns and predictions."
        )
    
    if st.button("Generate Report", type="primary"):
        return generate_report(
            data,
            export_format,
            include_charts,
            include_analysis,
            report_title,
            report_description
        )

def generate_report(data, format, include_charts, include_analysis, title, description):
    """Generate and return the report in specified format"""
    try:
        if format == "CSV":
            return export_csv(data)
        elif format == "Excel":
            return export_excel(data, include_charts, include_analysis)
        elif format == "PDF":
            return export_pdf(data, include_charts, include_analysis, title, description)
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

def export_csv(data):
    """Export data as CSV"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="energy_report.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

def export_excel(data, include_charts, include_analysis):
    """Export data as Excel with optional charts"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write main data
        data.to_excel(writer, sheet_name='Data', index=False)
        
        if include_analysis:
            # Create analysis sheet
            analysis_df = create_analysis_summary(data)
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
        
        if include_charts:
            # Add charts sheet
            workbook = writer.book
            chart_sheet = workbook.add_worksheet('Charts')
            add_excel_charts(workbook, chart_sheet, data)
    
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="energy_report.xlsx">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)

def create_analysis_summary(data):
    """Create summary analysis of the data"""
    summary = pd.DataFrame({
        'Metric': [
            'Total Consumption (kWh)',
            'Average Daily Consumption (kWh)',
            'Peak Consumption (kWh)',
            'Total Cost ($)',
            'Average Efficiency Score'
        ],
        'Value': [
            data['Consumption (kWh)'].sum(),
            data['Consumption (kWh)'].mean(),
            data['Consumption (kWh)'].max(),
            data['Cost ($)'].sum(),
            data['Efficiency Score'].mean()
        ]
    })
    return summary

def add_excel_charts(workbook, worksheet, data):
    """Add charts to Excel worksheet"""
    # Add consumption trend chart
    chart1 = workbook.add_chart({'type': 'line'})
    chart1.add_series({
        'name': 'Consumption',
        'categories': '=Data!$A$2:$A$32',
        'values': '=Data!$B$2:$B$32'
    })
    worksheet.insert_chart('B2', chart1)

def export_pdf(data, include_charts, include_analysis, title, description):
    """Export data as PDF with optional charts and analysis"""
    # Note: This is a placeholder. In a real implementation,
    # you would use a PDF generation library like ReportLab
    st.warning("PDF export functionality coming soon!")