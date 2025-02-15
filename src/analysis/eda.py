# src/analysis/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go
import logging

class EnergyDataAnalyzer:
    """
    Think of this as our data detective! 🔍
    It helps us understand patterns and relationships in our energy data.
    """
    
    def __init__(self, data_path):
        """
        Getting ready to analyze our data!
        """
        self.data = pd.read_csv(data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory for plots
        import os
        self.output_dir = 'reports/figures'
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_energy_over_time(self):
        """
        Like drawing a picture of how energy use changes over time! 📈
        """
        fig = px.line(
            self.data, 
            x='timestamp', 
            y='energy_consumption',
            title='Energy Consumption Over Time'
        )
        
        # Save the plot
        fig.write_html(f"{self.output_dir}/energy_over_time.html")
        self.logger.info("Created energy over time plot")
        return fig

    def plot_daily_patterns(self):
        """
        Like looking at how energy use changes during the day! 🌞
        """
        daily_avg = self.data.groupby('hour')['energy_consumption'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_avg.index,
            y=daily_avg.values,
            mode='lines+markers',
            name='Average Energy Consumption'
        ))
        
        fig.update_layout(
            title='Average Daily Energy Consumption Pattern',
            xaxis_title='Hour of Day',
            yaxis_title='Average Energy Consumption'
        )
        
        fig.write_html(f"{self.output_dir}/daily_patterns.html")
        self.logger.info("Created daily patterns plot")
        return fig

    def create_heatmap(self):
        """
        Like making a color-coded calendar of energy usage! 🎨
        """
        # Create pivot table for hour vs day of week
        pivot_data = self.data.pivot_table(
            index='day_of_week',
            columns='hour',
            values='energy_consumption',
            aggfunc='mean'
        )
        
        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Energy Consumption Heatmap by Hour and Day',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week'
        )
        
        fig.write_html(f"{self.output_dir}/heatmap.html")
        self.logger.info("Created heatmap")
        return fig

    def decompose_time_series(self):
        """
        Like breaking down energy usage into different patterns! 🧩
        Shows us the trend, seasonal patterns, and random changes.
        """
        # Prepare data for decomposition
        ts_data = self.data.set_index('timestamp')['energy_consumption']
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            ts_data,
            period=24  # For hourly data, period=24 for daily seasonality
        )
        
        # Create subplot figure
        fig = go.Figure()
        
        # Add traces for each component
        components = {
            'Observed': decomposition.observed,
            'Trend': decomposition.trend,
            'Seasonal': decomposition.seasonal,
            'Residual': decomposition.resid
        }
        
        for title, data in components.items():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=title
                )
            )
        
        fig.update_layout(
            title='Time Series Decomposition',
            height=1000
        )
        
        fig.write_html(f"{self.output_dir}/decomposition.html")
        self.logger.info("Created time series decomposition plot")
        return fig

    def check_stationarity(self):
        """
        Like checking if our energy usage patterns are stable! 📊
        """
        # Perform Augmented Dickey-Fuller test
        result = adfuller(self.data['energy_consumption'].dropna())
        
        stationarity_report = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
        
        # Create a more detailed report
        with open(f"{self.output_dir}/stationarity_report.txt", 'w') as f:
            f.write("Augmented Dickey-Fuller Test Results:\n\n")
            f.write(f"Test Statistic: {result[0]}\n")
            f.write(f"p-value: {result[1]}\n\n")
            f.write("Critical Values:\n")
            for key, value in result[4].items():
                f.write(f"\t{key}: {value}\n")
            
            # Add interpretation
            f.write("\nInterpretation:\n")
            if result[1] < 0.05:
                f.write("The time series is stationary (rejects the null hypothesis)")
            else:
                f.write("The time series is non-stationary (fails to reject the null hypothesis)")
        
        self.logger.info("Created stationarity report")
        return stationarity_report

    def generate_summary_statistics(self):
        """
        Like creating a report card for our energy data! 📋
        """
        summary_stats = self.data['energy_consumption'].describe()
        
        # Additional statistics
        summary_stats['skewness'] = self.data['energy_consumption'].skew()
        summary_stats['kurtosis'] = self.data['energy_consumption'].kurtosis()
        
        # Save summary statistics
        summary_stats.to_csv(f"{self.output_dir}/summary_statistics.csv")
        self.logger.info("Generated summary statistics")
        return summary_stats

def main():
    """
    Let's run our analysis! 🚀
    """
    try:
        # Initialize analyzer with our processed data
        analyzer = EnergyDataAnalyzer('data/processed/featured_energy_data.csv')
        
        # Generate all plots and analyses
        analyzer.plot_energy_over_time()
        analyzer.plot_daily_patterns()
        analyzer.create_heatmap()
        analyzer.decompose_time_series()
        analyzer.check_stationarity()
        analyzer.generate_summary_statistics()
        
        print("EDA completed successfully! 🎉")
        print("Check the 'reports/figures' directory for all visualizations and reports!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()