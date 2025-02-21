import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

class FeatureEngineer:
    """
    Enhanced utility for creating meaningful features from energy consumption data.
    """
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def add_time_features(self, df, time_column):
        """
        Add comprehensive time-based features from the timestamp column.
        """
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['month'] = df[time_column].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Additional time features
        df['is_holiday'] = self._is_holiday(df[time_column])
        df['day_period'] = pd.cut(df['hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['night', 'morning', 'afternoon', 'evening'])
        df['is_peak_hour'] = df['hour'].isin([9, 10, 11, 17, 18, 19]).astype(int)
        df['season'] = pd.cut(df['month'],
                            bins=[0, 3, 6, 9, 12],
                            labels=['winter', 'spring', 'summer', 'fall'])
        return df

    def _is_holiday(self, dates):
        """
        Determine if date is a holiday (simplified version - expand based on your country)
        """
        return dates.dt.dayofweek.isin([5, 6]).astype(int)  # Weekend as holiday example

    def add_weather_features(self, df):
        """
        Add weather-related interaction features.
        """
        if all(col in df.columns for col in ['temperature', 'humidity', 'wind_speed']):
            # Weather interactions
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['feels_like'] = self._calculate_feels_like(
                df['temperature'], df['humidity'], df['wind_speed']
            )
            
            # Extreme weather indicators
            df['is_extreme_temp'] = (
                (df['temperature'] > df['temperature'].quantile(0.95)) | 
                (df['temperature'] < df['temperature'].quantile(0.05))
            ).astype(int)
        return df

    def _calculate_feels_like(self, temp, humidity, wind_speed):
        """
        Calculate "feels like" temperature using simplified formula
        """
        return temp - 0.2 * (1 - humidity/100) + 0.1 * wind_speed

    def add_lag_features(self, df, target_column, lags=[24, 48, 168]):  # 168 = 1 week
        """
        Add enhanced lagged features for the target column.
        """
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
            # Add diff features
            if lag == 24:  # daily change
                df[f'{target_column}_diff_day'] = df[target_column] - df[f'{target_column}_lag_{lag}']
        return df

    def add_rolling_features(self, df, target_column, windows=[6, 12, 24, 168]):
        """
        Add comprehensive rolling statistics for the target column.
        """
        for window in windows:
            roll = df[target_column].rolling(window=window)
            df[f'{target_column}_rolling_mean_{window}'] = roll.mean()
            df[f'{target_column}_rolling_std_{window}'] = roll.std()
            df[f'{target_column}_rolling_min_{window}'] = roll.min()
            df[f'{target_column}_rolling_max_{window}'] = roll.max()
            
            # Add rate of change
            df[f'{target_column}_rate_of_change_{window}'] = (
                df[target_column] - df[f'{target_column}_rolling_mean_{window}']
            ) / window
        return df

    def add_consumption_patterns(self, df, consumption_column):
        """
        Add features related to consumption patterns.
        """
        # Daily patterns
        df['daily_peak'] = df.groupby(df['timestamp'].dt.date)[consumption_column].transform('max')
        df['daily_valley'] = df.groupby(df['timestamp'].dt.date)[consumption_column].transform('min')
        df['peak_valley_ratio'] = df['daily_peak'] / (df['daily_valley'] + 1e-10)
        
        # Consumption variability
        df['consumption_volatility'] = df.groupby(df['timestamp'].dt.date)[consumption_column].transform('std')
        return df

    def encode_categorical(self, df, categorical_columns):
        """
        One-hot encode categorical columns with improved handling.
        """
        # Handle missing values before encoding
        for col in categorical_columns:
            df[col] = df[col].fillna('unknown')
            
        encoded = self.encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=self.encoder.get_feature_names_out(categorical_columns),
            index=df.index
        )
        return pd.concat([df.drop(categorical_columns, axis=1), encoded_df], axis=1)

def main():
    """
    Enhanced main function with comprehensive feature engineering pipeline.
    """
    try:
        # Load cleaned data
        cleaned_data_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/processed/cleaned_data.csv"
        df = pd.read_csv(cleaned_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Add all features
        df = engineer.add_time_features(df, 'timestamp')
        df = engineer.add_weather_features(df)
        
        if 'energy_consumption' in df.columns:
            df = engineer.add_lag_features(df, 'energy_consumption')
            df = engineer.add_rolling_features(df, 'energy_consumption')
            df = engineer.add_consumption_patterns(df, 'energy_consumption')
            
        if 'weather_description' in df.columns:
            df = engineer.encode_categorical(df, ['weather_description'])
            
        # Remove rows with NaN values created by lag/rolling features
        df = df.dropna()
        
        # Save engineered data
        featured_data_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/processed/featured_data.csv"
        df.to_csv(featured_data_path, index=False)
        
        print("Enhanced feature engineering completed successfully! 🎉")
        print(f"Total features created: {len(df.columns)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()