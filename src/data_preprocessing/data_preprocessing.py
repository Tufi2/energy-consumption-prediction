# src/data_preprocessing/feature_engineer.py
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import logging

class FeatureEngineer:
    """
    Think of this as our feature factory! 🏭
    It creates new useful information from our existing data.
    """
    
    def __init__(self):
        self.uk_holidays = holidays.UK()  # You can change this to your country
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_time_features(self, df, timestamp_column):
        """
        Like extracting special information about time!
        Just like looking at a clock and calendar together! 📅
        """
        df = df.copy()
        
        # Convert timestamp to datetime if it's not already
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Extract basic time features
        df['hour'] = df[timestamp_column].dt.hour
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['month'] = df[timestamp_column].dt.month
        df['year'] = df[timestamp_column].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create season feature
        df['season'] = df['month'].map({
            1: 'Winter', 2: 'Winter', 3: 'Spring',
            4: 'Spring', 5: 'Spring', 6: 'Summer',
            7: 'Summer', 8: 'Summer', 9: 'Fall',
            10: 'Fall', 11: 'Fall', 12: 'Winter'
        })
        
        # Create time of day feature
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        self.logger.info("Created time-based features")
        return df

    def create_lag_features(self, df, target_column, lag_periods=[1, 24, 168]):
        """
        Like looking at past energy usage to predict future usage!
        Similar to knowing you use more electricity in the evening because you did yesterday too! 📊
        """
        df = df.copy()
        
        for lag in lag_periods:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
        self.logger.info(f"Created lag features for periods: {lag_periods}")
        return df

    def create_rolling_features(self, df, target_column, windows=[24, 168]):
        """
        Like calculating average energy usage over different time periods!
        Similar to knowing your average daily or weekly usage 📈
        """
        df = df.copy()
        
        for window in windows:
            # Calculate rolling mean
            df[f'{target_column}_rolling_mean_{window}'] = (
                df[target_column].rolling(window=window).mean()
            )
            
            # Calculate rolling standard deviation
            df[f'{target_column}_rolling_std_{window}'] = (
                df[target_column].rolling(window=window).std()
            )
            
        self.logger.info(f"Created rolling features for windows: {windows}")
        return df

    def create_holiday_features(self, df, timestamp_column):
        """
        Like marking special days on a calendar!
        Helps us know when people might use more or less energy 📅
        """
        df = df.copy()
        
        # Is holiday
        df['is_holiday'] = df[timestamp_column].apply(
            lambda x: 1 if x in self.uk_holidays else 0
        )
        
        # Is day before holiday
        df['is_day_before_holiday'] = df[timestamp_column].apply(
            lambda x: 1 if (x + pd.Timedelta(days=1)) in self.uk_holidays else 0
        )
        
        self.logger.info("Created holiday features")
        return df

def main():
    """
    Let's test our feature engineering!
    """
    try:
        # Load the cleaned data
        df = pd.read_csv('data/processed/cleaned_energy_data.csv')
        
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Add all our new features
        df = engineer.create_time_features(df, 'timestamp')
        df = engineer.create_lag_features(df, 'energy_consumption')
        df = engineer.create_rolling_features(df, 'energy_consumption')
        df = engineer.create_holiday_features(df, 'timestamp')
        
        # Save the feature-engineered data
        df.to_csv('data/processed/featured_energy_data.csv', index=False)
        print("Feature engineering completed successfully! 🎉")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()