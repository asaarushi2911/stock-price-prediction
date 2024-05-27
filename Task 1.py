# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('stock_data.csv')

# Check for missing values
print("Missing values:")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Prepare the data
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]  # Independent variables
y = data['Close']  # Dependent variable

# Split the dataset into training and testing sets
X_trn, X_tst, y_trn, y_tst  = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model  =  LinearRegression()
model.fit(X_trn, y_trn)


# Make predictions
y_pred = model.predict(X_tst)

# Evaluate the model
mse = mean_squared_error(y_tst, y_pred)
r2 = r2_score(y_tst, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot predicted vs actual prices
plt.scatter(y_tst, y_pred)
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.show()
