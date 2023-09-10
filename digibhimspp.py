import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the stock symbol and download historical data
stock_symbol = 'TSLA'
start_date = '2020-01-01'
end_date = '2021-12-31'

df = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate daily returns
df['Daily_Return'] = df['Adj Close'].pct_change()

# Drop missing values
df.dropna(inplace=True)

# Features (X) and target (y)
X = df[['Open', 'High', 'Low', 'Volume']].values
y = df['Adj Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()
