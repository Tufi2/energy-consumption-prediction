import requests
import pandas as pd
from datetime import datetime

# Step 1: Define API Key and City
API_KEY = "48ff73d2a21cb95acf08ca9a497a31ad"  # Replace with your OpenWeatherMap API key
CITY = "New York"        # Replace with your target city (e.g., "New York")
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"


import os

def save_weather_data(df, filepath):
    """
    Saves weather data to a CSV file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the DataFrame to CSV
    df.to_csv(filepath, index=False)
    print(f"Weather data saved to {filepath}")

# Step 2: Fetch Weather Data
def fetch_weather_data(api_key, city):
    """
    Fetches weather data from OpenWeatherMap API.
    """
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # Use "imperial" for Fahrenheit
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch data (Status Code: {response.status_code})")
        return None

# Step 3: Parse and Format Data
def parse_weather_data(weather_json):
    """
    Parses JSON weather data into a structured DataFrame.
    """
    weather_data = []
    
    for entry in weather_json["list"]:
        timestamp = datetime.fromtimestamp(entry["dt"])
        temperature = entry["main"]["temp"]
        humidity = entry["main"]["humidity"]
        wind_speed = entry["wind"]["speed"]
        weather_description = entry["weather"][0]["description"]
        
        weather_data.append({
            "timestamp": timestamp,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "weather_description": weather_description
        })
    
    return pd.DataFrame(weather_data)

# Step 4: Save Data to CSV
def save_weather_data(df, filepath):
    """
    Saves weather data to a CSV file.
    """
    df.to_csv(filepath, index=False)
    print(f"Weather data saved to {filepath}")

# Step 5: Main Function
def main():
    # Fetch weather data
    weather_json = fetch_weather_data(API_KEY, CITY)
    
    if weather_json:
        # Parse and format data
        weather_df = parse_weather_data(weather_json)
        
        # Save data to CSV
        save_weather_data(weather_df, "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/raw/weather.csv")

if __name__ == "__main__":
    main()