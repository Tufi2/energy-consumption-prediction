import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data():
    """Load the engineered dataset."""
    data_path = "../../data/processed/featured_data.csv"
    return pd.read_csv(data_path)

def train_baseline_model(X_train, y_train):
    """Train a Random Forest Regressor as a baseline model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using MAE, RMSE, and R²."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

def main():
    """Main function to train and evaluate models."""
    try:
        # Load data
        df = load_data()
        
        # Prepare features and target
        X = df.drop(columns=['energy_consumption', 'timestamp'])  # Adjust based on your dataset
        y = df['energy_consumption']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training baseline model...")
        model = train_baseline_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = "../../models/trained_models/random_forest.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()