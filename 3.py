import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#load dataset
data = pd.read_csv("C:\\Users\\Rd\\Downloads\\House Price India.csv.zip")
#clean column
#replace spaces with underscores and make to lowercase
data.columns = data.columns.str.replace(" ", "_").str.lower()
#check cleaned column names
print("Columns:", data.columns.tolist())
#feature selection
X = data[['number_of_bedrooms', 'number_of_bathrooms', 'living_area', 'lot_area']]
y = data['price']
# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#model building
model = LinearRegression()
model.fit(X_train, y_train)
#predictions
y_pred = model.predict(X_test)
#model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
#visualization
#actual vs predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
#residuals
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()