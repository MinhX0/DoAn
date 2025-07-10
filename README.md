# Stock Prediction with Random Forest

This project predicts stock prices for multiple tickers using Random Forest. It downloads historical data from yfinance, trains a model for each ticker, and displays predicted prices.

## Features
- Download historical stock data for multiple tickers
- Train a Random Forest model for each ticker
- Display a simple screen with predicted prices

## Requirements
- Python 3.8+
- yfinance
- scikit-learn
- pandas
- matplotlib (for optional plotting)

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python stock_predictor.py`

## Customization
- Edit the `TICKERS` list in `stock_predictor.py` to change the stocks to predict.
