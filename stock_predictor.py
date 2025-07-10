import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# List of tickers to predict
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Download historical data
def download_data(tickers, period='1y'):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period)
        if not df.empty:
            data[ticker] = df
    return data

def prepare_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    X = df[['Return', 'MA5', 'MA10']]
    y = df['Close'].shift(-1).dropna()
    X = X.iloc[:-1]
    return X, y

def train_and_predict(data):
    predictions = {}
    last_actuals = {}
    for ticker, df in data.items():
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict([X.iloc[-1]])
        predictions[ticker] = y_pred[0]
        last_actuals[ticker] = df['Close'].iloc[-1]
    return predictions, last_actuals

def main():
    print("Downloading data...")
    data = download_data(TICKERS)
    print("Training models and predicting...")
    predictions, last_actuals = train_and_predict(data)
    print("\nPredicted next closing prices:")
    actions = {}
    for ticker, price in predictions.items():
        last_price = last_actuals[ticker]
        # Ensure both price and last_price are scalars (not Series)
        price_scalar = float(price) if hasattr(price, '__iter__') and not isinstance(price, str) else price
        last_price_scalar = float(last_price) if hasattr(last_price, '__iter__') and not isinstance(last_price, str) else last_price
        pct_change = (price_scalar - last_price_scalar) / last_price_scalar
        if pct_change > 0.02:
            action = "BUY"
        elif pct_change < -0.02:
            action = "SELL"
        else:
            action = "HOLD"
        actions[ticker] = action
        print(f"{ticker}: ${price_scalar:.2f} | Last: ${last_price_scalar:.2f} | Action: {action}")

    # Plot actual vs predicted for each ticker
    fig, axs = plt.subplots(len(TICKERS), 1, figsize=(8, 2*len(TICKERS)))
    if len(TICKERS) == 1:
        axs = [axs]
    for i, ticker in enumerate(TICKERS):
        df = data[ticker]
        axs[i].plot(df.index, df['Close'], label='Actual Close')
        axs[i].axhline(predictions[ticker], color='r', linestyle='--', label='Predicted Next Close')
        axs[i].set_title(f"{ticker} - Action: {actions[ticker]}")
        axs[i].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
