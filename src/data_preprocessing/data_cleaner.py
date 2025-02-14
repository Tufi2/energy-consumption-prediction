# src/data_preprocessing/data_cleaner.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

class DataCleaner:
    """
    Think of this as our data washing machine! 🧺
    It takes dirty data and makes it squeaky clean for our analysis.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def remove_duplicates(self, df):
        """
        Like removing duplicate socks from your drawer!
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_rows = initial_rows - len(df)
        self.logger.info(f"Removed {dropped_rows} duplicate rows")
        return df

    def handle_missing_values(self, df):
        """
        Like filling in missing puzzle pieces! 🧩
        We use different methods depending on what kind of data is missing.
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
        Like removing the weird data points that don't fit in! 
        Just like finding the one sock that's not like the others 🧦
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
        Like making sure all our numbers are playing on the same playground! 
        Makes big numbers and small numbers play nice together 🎮
        """
        df_copy = df.copy()
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        self.logger.info(f"Normalized columns: {columns}")
        return df_copy

    def validate_data(self, df):
        """
        Like a quality check before sending the data to the next step!
        Makes sure our data is good to go! ✅
        """
        validation_results = {
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        self.logger.info("Data validation complete")
        return validation_results

def main():
    """
    Let's test our data cleaner!
    """
    # Load sample data (you should replace this with your actual data)
    try:
        df = pd.read_csv('data/raw/energy_consumption.csv')
        
        # Create our data cleaner
        cleaner = DataCleaner()
        
        # Clean the data step by step
        df = cleaner.remove_duplicates(df)
        df = cleaner.handle_missing_values(df)
        
        # Remove outliers from numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = cleaner.remove_outliers(df, numeric_columns)
        
        # Normalize numeric columns
        df = cleaner.normalize_data(df, numeric_columns)
        
        # Validate the cleaned data
        validation_results = cleaner.validate_data(df)
        
        # Save the cleaned data
        df.to_csv('data/processed/cleaned_energy_data.csv', index=False)
        print("Data cleaning completed successfully! 🎉")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()