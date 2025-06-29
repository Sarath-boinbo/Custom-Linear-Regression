import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path


class MultipleLinearRegression:
    def __init__(self):
        """Initializes the model. The coefficients are set to None until the model is trained."""
        self.coefficients = None
    
    def fit(self, X, y):
        """
        Trains the model by finding the optimal coefficients using the Normal Equation.
        
        This function implements the mathematical formula: β = (X^T * X)^(-1) * X^T * y
        It calculates the best coefficients that minimize the sum of squared errors
        between the predicted and actual target values.
        """
        # Add a bias (intercept) term to the feature matrix.
        # This column of ones allows the model to learn a baseline value (the intercept).
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        try:
            # --- The Normal Equation ---
            # 1. Calculate (X^T * X)^-1
            xtx_inv = np.linalg.inv(X_b.T @ X_b)
            
            # 2. Calculate X^T * y
            xt_y = X_b.T @ y
            
            # 3. Multiply them to get the coefficients
            self.coefficients = xtx_inv @ xt_y

        except np.linalg.LinAlgError:
            # This block runs if the matrix is "singular" (cannot be inverted),
            # which is usually caused by perfect multicollinearity in the data.
            print("Error: Singular matrix. Cannot compute the inverse.")
            print("Model training failed. This can be caused by redundant features in the data.")
            self.coefficients = None

    def predict(self, X):
        """
        Predicts target values for new input data using the learned coefficients.
        """
        # Safety check: ensure the model has been trained before prediction.
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted yet or training failed.")
        
        # Add the bias term to the new data so its format matches the training data.
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Make predictions using the formula: y_pred = X_b * coefficients
        return X_b @ self.coefficients

def rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Lower RMSE values indicate a better model fit.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def plot_data(data, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(data, predictions, alpha=0.7, edgecolors='k')
    plt.plot([min(data), max(data)], [min(data), max(data)], 'r--', lw=2, label='Perfect Prediction Line')
    plt.title('Actual vs. Predicted Life Expectancy')
    plt.xlabel('Actual Life Expectancy (Years)')
    plt.ylabel('Predicted Life Expectancy (Years)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main function to run the regression analysis."""
    # --- 1. Load and Prepare Data ---
    script_dir = Path(__file__).parent
    data_path = script_dir / 'WHO_LE_Data.csv'
    data = pd.read_csv(data_path)
    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()

    # Scale GDP Column from dollars to thousands of dollars
    data['GDP'] = data['GDP'] / 1000

    # Drop columns that are identifiers or potentially non-predictive
    features_to_remove = [
        'infant deaths', 
        'percentage expenditure', 
        'thinness 5-9 years',
        'Country'
    ]
    data = data.drop(features_to_remove, axis=1)

    # One-Hot Encode the 'Status' column (e.g., 'Developed', 'Developing')
    # drop_first=True prevents multicollinearity between the new dummy columns
    data = pd.get_dummies(data, columns=['Status'], drop_first=True)

    # IMPORTANT: Remove rows with any missing values FIRST
    data = data.dropna()
    
    # THEN, separate the clean data into features (X) and target (y)
    X = data.drop('Life expectancy', axis=1)
    y = data['Life expectancy']

    X = X.astype(float)

    # Store feature names for later printing and convert to numpy arrays
    feature_names = X.columns
    X = X.values
    y = y.values

    # Split data into training and testing sets for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 2. Train The Custom Model ---
    regressor = MultipleLinearRegression()
    regressor.fit(X_train, y_train)

    # --- 3. Evaluate the Model ---
    # IMPORTANT FIX: Check if the model was successfully trained before trying to use it.
    if regressor.coefficients is not None:
        predictions = regressor.predict(X_test)
        error = rmse(y_test, predictions)
    else:
        print("\nCould not evaluate the model because training failed.")
    
    # Train Scikit on my model
    sklearn_regressor = LinearRegression()
    sklearn_regressor.fit(X_train, y_train)
    sklearn_predictions = sklearn_regressor.predict(X_test)

    sklearn_rmse = np.sqrt(mean_squared_error(y_test, sklearn_predictions))

    # Compare results
    print("\n--- Comparison ---")
    if regressor.coefficients is not None:
        print(f"My Model's Intercept:       {regressor.coefficients[0]}")
        print(f"Scikit-learn's Intercept:   {sklearn_regressor.intercept_}")

        print(f"\nMy Model's RMSE:            {error}")
        print(f"Scikit-learn's RMSE:        {sklearn_rmse}")
    else:
        print("My Model: Training failed - no coefficients available")
        print(f"Scikit-learn's Intercept:   {sklearn_regressor.intercept_}")
        print(f"Scikit-learn's RMSE:        {sklearn_rmse}")

    # Compare coefficients
    print("\n--- Coefficient Comparison ---")
    if regressor.coefficients is not None:
        for i, name in enumerate(feature_names):
            print(f"{name:>20}: My Model = {regressor.coefficients[i+1]:.4f}, Scikit-learn = {sklearn_regressor.coef_[i]:.4f}")
        
        # Plot the data
        plot_data(y_test, predictions)
    else:
        print("Cannot compare coefficients - model training failed")

    


if __name__ == "__main__":
    main()

"""
# --- Model Improvement & Analysis ---
#
# **Initial Model Issues (The "Why"):**
# The first version of the model produced a lower RMSE (around 3.61) but had several illogical
# and counter-intuitive coefficients. For example, 'infant deaths' had a positive coefficient,
# implying that more infant deaths led to a higher life expectancy. This did not mean the model's
# math was wrong; it meant the model's *interpretation* was broken. Another minor issue was that
# GDP was not scaled properly so it's positive effect was near negligible.
#
# The root cause was **multicollinearity**: many features in the dataset were highly correlated
# (e.g., 'infant deaths' and 'under-five deaths'). When features are redundant, the model gets
# confused about how to assign credit or blame, leading to unstable and nonsensical coefficients.
#
# **Changes Made (The "How"):**
# To fix this, we identified and removed the most redundant features:
#   - 'infant deaths' (information is captured by 'under-five deaths')
#   - 'percentage expenditure' (information is captured by 'GDP')
#   - 'thinness 5-9 years' (information is captured by 'thinness 1-19 years')
#   - Scaled GDP from dollars to thousands of dollars
#
# **New Model Results (The "What"):**
# The new model's RMSE increased slightly (to around 3.66). This is expected. We removed a tiny
# amount of unique information (or statistical noise) that the original model was using to
# minimize its prediction error, even at the cost of logic.
#
# **Why the New Model is Better:**
# 1.  **Interpretability:** The coefficients are now stable and make logical sense. We can now
#     confidently explain the impact of each feature. The model is no longer a "black box."
# 2.  **Reliability:** The model is more robust. Its coefficients are less likely to swing wildly
#     if the training data changes slightly.
#
# **Conclusion:** I made a deliberate and correct trade-off. I sacrificed a negligible amount
# of raw predictive accuracy for a massive gain in model interpretability and reliability.
# For the goal of understanding *how* a model works, the new version is far superior.
#
"""
