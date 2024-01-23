import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def prepare_data(data, target_column, lag=1):
    X = data.copy()
    X['lag'] = X[target_column].shift(lag)
    X = X.dropna()
    y = X[target_column]
    X = X.drop(columns=[target_column])
    return X, y

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

def main():
    # User input for stock symbol, start date, and end date
    stock_symbol = input("Enter stock symbol (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # Get stock data
    stock_data = get_stock_data(stock_symbol, start_date, end_date)

    # Prepare data for machine learning
    X, y = prepare_data(stock_data, target_column='Close', lag=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = make_predictions(model, X_test)

    # Evaluate the model
    mse = evaluate_model(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Example usage: Predict the stock price for the next day
    last_day_data = X.iloc[-1].values.reshape(1, -1)
    predicted_price = make_predictions(model, last_day_data)
    print(f'Predicted Stock Price for the Next Day: {predicted_price[0]}')

if __name__ == "__main__":
    main()
