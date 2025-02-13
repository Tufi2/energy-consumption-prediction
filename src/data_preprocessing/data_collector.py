# src/data_preprocessing/data_collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

class WeatherDataCollector:
    """
    This is like our weather reporter! 🌤️
    It goes out and gets weather information that might affect energy usage.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather_data(self, city):
        """
        Like asking: "Hey, what's the weather like in this city?"
        """
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'  # Get temperature in Celsius
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            weather_info = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed']
            }
            return weather_info
        except Exception as e:
            logging.error(f"Couldn't get weather data: {e}")
            return None

class EnergyDataCollector:
    """
    This is like our energy detective! 🔍
    It collects information about how much energy people are using.
    """
    def __init__(self, data_path):
        self.data_path = data_path
    
    def collect_smart_meter_data(self, start_date, end_date):
        """
        Like reading a diary of energy usage from smart meters!
        """
        try:
            # In real project, this would connect to smart meter API
            # For now, we'll simulate data
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
            energy_data = pd.DataFrame({
                'timestamp': dates,
                'energy_consumption': np.random.normal(loc=50, scale=10, size=len(dates))
            })
            return energy_data
        except Exception as e:
            logging.error(f"Error collecting smart meter data: {e}")
            return None

def main():
    """
    This is like our data collection recipe!
    It tells us step by step how to collect all the data we need.
    """
    # Setup logging to keep track of what's happening
    logging.basicConfig(level=logging.INFO)
    
    # Create our data collectors
    weather_collector = WeatherDataCollector(api_key='48ff73d2a21cb95acf08ca9a497a31ad')
    energy_collector = EnergyDataCollector(data_path='data/raw')
    
    # Collect data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Get energy data
    logging.info("Collecting energy consumption data...")
    energy_data = energy_collector.collect_smart_meter_data(start_date, end_date)
    
    # Get weather data for a specific city
    logging.info("Collecting weather data...")
    weather_data = weather_collector.get_weather_data('London')
    
    # Save the collected data
    if energy_data is not None:
        energy_data.to_csv('data/raw/energy_consumption.csv', index=False)
        logging.info("Successfully saved energy data! 🎉")
    
    if weather_data is not None:
        pd.DataFrame([weather_data]).to_csv('data/raw/weather_data.csv', index=False)
        logging.info("Successfully saved weather data! 🎉")

if __name__ == "__main__":
    main()