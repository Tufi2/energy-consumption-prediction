import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_data():
    """Load the engineered dataset."""
    data_path = "../../data/processed/featured_data.csv"
    df = pd.read_csv(data_path)
    return df

def encode_categorical(df, categorical_columns):
    """One-hot encode categorical columns."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
    return pd.concat([df.drop(categorical_columns, axis=1), encoded_df], axis=1)

from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning using Grid Search."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main():
    """Main function to train and evaluate models."""
    try:
        # Load data
        df = load_data()
        
        # Configuration
        TARGET_COLUMN = 'temperature'
        CATEGORICAL_COLUMNS = ['day_period', 'season']
        DROP_COLUMNS = ['timestamp']
        MODEL_PATH = "../../models/trained_models/random_forest.pkl"
        
        # Create model directory if needed
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        
        # Feature engineering
        df = encode_categorical(df, CATEGORICAL_COLUMNS)
        X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)
        y = df[TARGET_COLUMN]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training baseline model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print(f"R²: {r2_score(y_test, y_pred):.4f}")
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved successfully to {MODEL_PATH}")
        
        # Replace the model training section with:
        print("Tuning hyperparameters...")
        model = tune_hyperparameters(X_train, y_train)
        print("Best hyperparameters:", model.get_params())
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()