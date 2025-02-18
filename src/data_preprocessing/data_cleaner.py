import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

class DataCleaner:
    """
    A data cleaning utility for energy consumption prediction.
    Think of this as our data washing machine! 🧺
    It takes dirty data and makes it squeaky clean for our analysis.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    # Configure logging to show messages in the terminal
        logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
        self.logger = logging.getLogger(__name__)

    def remove_duplicates(self, df):
        """
        Removes duplicate rows from the dataset.
        Like removing duplicate socks from your drawer!
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_rows = initial_rows - len(df)
        self.logger.info(f"Removed {dropped_rows} duplicate rows")
        return df

    def handle_missing_values(self, df):
        """
        Handles missing values in the dataset.
        Like filling in missing puzzle pieces! 🧩
        """
        # For numeric columns, use interpolation (like connecting the dots)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                # First, try linear interpolation
                df[col] = df[col].interpolate(method='linear')
                # If any values are still missing (at edges), use forward/backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
        # For categorical columns, use mode (most frequent value)
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
        self.logger.info("Handled missing values in all columns")
        return df

    def remove_outliers(self, df, columns, method='iqr'):
        """
        Removes outliers from the dataset.
        Like removing the weird data points that don't fit in! 
        """
        for column in columns:
            if method == 'iqr':
                # Calculate IQR (Interquartile Range)
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                
            elif method == 'zscore':
                # Remove values that are more than 3 standard deviations away
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                df = df[z_scores <= 3]
                
        self.logger.info(f"Removed outliers using {method} method")
        return df

    def normalize_data(self, df, columns):
        """
        Normalizes numeric columns using StandardScaler.
        Like making sure all our numbers are playing on the same playground! 
        """
        df_copy = df.copy()
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        self.logger.info(f"Normalized columns: {columns}")
        return df_copy

    def validate_data(self, df):
        """
        Validates the cleaned dataset.
        Like a quality check before sending the data to the next step!
        """
        validation_results = {
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'row_count': len(df),
            'column_count': len(df.columns),
            'negative_values': (df.select_dtypes(include=[np.number]) < 0).sum().to_dict()
        }
        
        self.logger.info("Data validation complete")
        return validation_results

    def resample_time_series(self, df, time_column, freq='H'):
        """
        Resamples time-series data to a consistent frequency.
        """
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
        df = df.resample(freq).mean().reset_index()
        self.logger.info(f"Resampled data to {freq} frequency")
        return df

def main():
    """
    Main function to test the data cleaner.
    """
    try:
        print("Starting data cleaning process...")
        
        # Load raw data
        raw_data_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/raw/weather.csv"
        print(f"Loading data from: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Create our data cleaner
        print("Creating DataCleaner instance...")
        cleaner = DataCleaner()
        
        # Clean the data step by step
        print("Removing duplicates...")
        df = cleaner.remove_duplicates(df)
        
        print("Handling missing values...")
        df = cleaner.handle_missing_values(df)
        
        # Remove outliers from numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        print(f"Removing outliers from columns: {numeric_columns}")
        df = cleaner.remove_outliers(df, numeric_columns)
        
        # Resample time-series data (if applicable)
        if 'timestamp' in df.columns:
            print("Resampling time-series data...")
            df = cleaner.resample_time_series(df, time_column='timestamp')
        
        # Normalize numeric columns
        print("Normalizing numeric columns...")
        df = cleaner.normalize_data(df, numeric_columns)
        
        # Validate the cleaned data
        print("Validating cleaned data...")
        validation_results = cleaner.validate_data(df)
        print("Validation results:", validation_results)
        
        # Save the cleaned data
        cleaned_data_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/processed/cleaned_data.csv"
        print(f"Saving cleaned data to: {cleaned_data_path}")
        df.to_csv(cleaned_data_path, index=False)
        
        print("Data cleaning completed successfully! 🎉")
        
    except Exception as e:
        print(f"An error occurred: {e}")