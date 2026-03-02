import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data = pd.read_csv("stock_data.csv")
print("First 5 Rows:")
print(data.head())
data = data[['Open', 'High', 'Low', 'Volume', 'Close']]
data.dropna(inplace=True)
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
# =====================================
# 8. Evaluation Function
# =====================================
def evaluate_model(name, y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name} Performance:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)

# Evaluate Models
evaluate_model("Linear Regression", y_test, lr_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# =====================================
# 9. Compare Actual vs Predicted
# =====================================
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "LR Predicted": lr_predictions,
    "RF Predicted": rf_predictions
})

print("\nActual vs Predicted:")
print(comparison.head())

# =====================================
# 10. Plot Results
# =====================================
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual Price")
plt.plot(rf_predictions, label="RF Predicted")
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.show()