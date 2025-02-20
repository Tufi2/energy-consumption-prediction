# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Set up visualization style
sns.set(style="whitegrid")
plt.style.use("seaborn")

# Load the cleaned data
cleaned_data_path = "D:/ProjectCQ/Energy-consumption-prediction/energy-consumption-prediction/data/processed/cleaned_data.csv"
df = pd.read_csv(cleaned_data_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Display basic information about the dataset
print("Dataset Info:")
df.info()

print("\nFirst 5 rows:")
display(df.head())

print("\nSummary Statistics:")
display(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Plot time-series trends
def plot_time_series(df, x_col, y_cols, title, ylabel):
    """
    Plot time-series data for multiple columns.
    """
    plt.figure(figsize=(14, 6))
    for col in y_cols:
        plt.plot(df[x_col], df[col], label=col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Plot temperature and humidity over time
plot_time_series(df, 'timestamp', ['temperature', 'humidity'], 
                 'Temperature and Humidity Over Time', 'Value')

# Plot wind speed over time
plot_time_series(df, 'timestamp', ['wind_speed'], 
                 'Wind Speed Over Time', 'Wind Speed')

# Distribution plots
def plot_distribution(df, columns):
    """
    Plot distribution of numeric columns.
    """
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# Plot distributions of numeric columns
numeric_cols = ['temperature', 'humidity', 'wind_speed']
plot_distribution(df, numeric_cols)

# Correlation heatmap
def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for numeric columns.
    """
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

plot_correlation_heatmap(df)

# Stationarity check
def check_stationarity(series, column_name):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series)
    print(f'ADF Statistic for {column_name}: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print(f"Result: {column_name} is stationary.")
    else:
        print(f"Result: {column_name} is non-stationary.")

# Check stationarity of numeric columns
for col in numeric_cols:
    check_stationarity(df[col], col)

# Boxplots for outlier detection
def plot_boxplots(df, columns):
    """
    Plot boxplots for numeric columns to detect outliers.
    """
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

plot_boxplots(df, numeric_cols)

# Pairplot for multivariate analysis
sns.pairplot(df[numeric_cols])
plt.suptitle('Pairplot of Numeric Columns', y=1.02)
plt.show()