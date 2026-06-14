# Sales Prediction Using Python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("advertising.csv")

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Model coefficients
print("\nModel Coefficients:")
print("TV:", model.coef_[0])
print("Radio:", model.coef_[1])
print("Newspaper:", model.coef_[2])
print("Intercept:", model.intercept_)

# Predict sales for new advertising budgets
tv = float(input("\nEnter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))

new_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper]
})

predicted_sales = model.predict(new_data)

print(f"\nPredicted Sales: {predicted_sales[0]:.2f}")