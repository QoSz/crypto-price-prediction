import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import ta

# Function to fetch current price data
def get_current_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data[symbol]['usd']

# Function to fetch historical price data
def get_historical_data(symbol, days):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date')
    return df[['price']]

# Function to add technical indicators
def add_indicators(df):
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=14)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=14)
    df['RSI'] = ta.momentum.rsi(df['price'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['price'])
    df['BB_high'] = ta.volatility.bollinger_hband(df['price'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['price'])
    df = df.dropna()
    return df

# Function to prepare data for LSTM model
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['price']])
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Function to create and train LSTM model with attention
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    attention = Attention()([lstm_out, lstm_out])
    lstm_out2 = LSTM(50)(attention)
    concat = Concatenate()([lstm_out2, Dense(50)(inputs[:,-1,:])])
    outputs = Dense(1)(concat)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to make predictions
def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    return predictions.flatten()

# Function to analyze buy/sell opportunities
def analyze_opportunities(data, predictions, threshold=0.02):
    buy_opportunities = []
    sell_opportunities = []
    for i in range(1, len(predictions)):
        price_change = (predictions[i] - data.iloc[-len(predictions)+i]['price']) / data.iloc[-len(predictions)+i]['price']
        if price_change > threshold:
            buy_opportunities.append((data.index[-len(predictions)+i], data.iloc[-len(predictions)+i]['price'], predictions[i]))
        elif price_change < -threshold:
            sell_opportunities.append((data.index[-len(predictions)+i], data.iloc[-len(predictions)+i]['price'], predictions[i]))
    return buy_opportunities, sell_opportunities

# Main function
def main():
    symbols = ['bitcoin', 'ethereum']
    for symbol in symbols:
        print(f"\nAnalyzing {symbol.capitalize()}:")
        
        # Get current price
        current_price = get_current_price(symbol)
        print(f"Current {symbol.capitalize()} price: ${current_price:.2f}")
        
        # Get historical data
        data = get_historical_data(symbol, 365)
        
        # Add technical indicators
        data = add_indicators(data)
        
        # Prepare data for LSTM model
        X, y, scaler = prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train LSTM model
        model = create_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        # Make predictions with LSTM
        lstm_predictions = make_predictions(model, X_test, scaler)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        rf_predictions = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
        rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1)).flatten()
        
        # Ensemble predictions
        predictions = (lstm_predictions + rf_predictions) / 2
        
        # Analyze opportunities
        buy_opportunities, sell_opportunities = analyze_opportunities(data, predictions)
        
        print("\nBuy opportunities:")
        for date, price, predicted_price in buy_opportunities[:5]:
            print(f"Date: {date.date()}, Current Price: ${price:.2f}, Predicted Price: ${predicted_price:.2f}")
        
        print("\nSell opportunities:")
        for date, price, predicted_price in sell_opportunities[:5]:
            print(f"Date: {date.date()}, Current Price: ${price:.2f}, Predicted Price: ${predicted_price:.2f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[-len(predictions):], data.iloc[-len(predictions):]['price'], label='Actual Price')
        plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price')
        plt.title(f"{symbol.capitalize()} Price Prediction")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
