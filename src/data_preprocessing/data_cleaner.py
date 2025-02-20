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
        """Removes duplicate rows from the dataset."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        return df

    def handle_missing_values(self, df):
        """Robust missing value handling with multi-stage imputation"""
        # First pass: temporal imputation
        df = df.ffill().bfill()
        
        # Second pass: numeric mean imputation
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Third pass: categorical mode imputation
        categorical_cols = df.select_dtypes(include='object').columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Final check: drop any remaining NaNs
        df = df.dropna()
        self.logger.info("Completed missing value handling")
        return df

    def remove_outliers(self, df, columns, method='iqr'):
        """Outlier handling with clipping instead of removal"""
        df_clean = df.copy()
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR
                upper = Q3 + 1.5*IQR
                df_clean[col] = df_clean[col].clip(lower, upper)
        self.logger.info(f"Clipped outliers using {method} method")
        return df_clean

    def resample_time_series(self, df, time_column, freq='h'):
        """Time-series resampling with gap filling"""
        # Convert and set index
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
        
        # Create complete time index
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        df = df.reindex(full_index)
        
        # Separate numeric/categorical columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        
        # Resample with different methods
        df_num = df[numeric_cols].resample(freq).mean()
        df_cat = df[categorical_cols].resample(freq).first()
        
        # Forward-fill gaps
        df_num = df_num.ffill().bfill()
        df_cat = df_cat.ffill().bfill()
        
        # Combine results
        df = pd.concat([df_num, df_cat], axis=1).reset_index()
        self.logger.info(f"Resampled to {freq} frequency with gap filling")
        return df.rename(columns={'index': time_column})

    def normalize_data(self, df, columns):
        """Normalization after data validation"""
        df_copy = df.copy()
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        self.logger.info(f"Normalized columns: {columns}")
        return df_copy

    def validate_data(self, df):
        """Comprehensive data validation"""
        report = {
        'missing_values': df.isna().sum().to_dict(),
        'negative_values': (df.select_dtypes(include=np.number) < 0).sum().to_dict(),
        'inf_values': np.isinf(df.select_dtypes(include=np.number)).sum().to_dict(),  # Corrected line
        'dtype_issues': {
            col: str(df[col].map(type).unique()) 
            for col in df.columns 
            if df[col].map(type).nunique() > 1
        }
    }
        self.logger.info("Data validation completed")
        return report

def main():
    """Updated workflow with proper execution order"""
    try:
        print("Starting data cleaning process...")
        
        # Initialize cleaner and load data
        cleaner = DataCleaner()
        raw_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/raw/weather.csv"
        df = pd.read_csv(raw_path)
        print(f"Loaded raw data: {df.shape}")

        # Pipeline execution
        df = (
            df.pipe(cleaner.remove_duplicates)
              .pipe(cleaner.remove_outliers, columns=df.select_dtypes(include=np.number).columns)
              .pipe(cleaner.resample_time_series, time_column='timestamp', freq='h')
              .pipe(cleaner.handle_missing_values)
              .pipe(cleaner.normalize_data, columns=df.select_dtypes(include=np.number).columns)
        )

        # Final validation and save
        validation = cleaner.validate_data(df)
        print("Validation Report:", validation)
        
        clean_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/processed/cleaned_data.csv"
        df.to_csv(clean_path, index=False)
        print(f"Saved clean data: {clean_path}")

    except Exception as e:
        print(f"Critical error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()