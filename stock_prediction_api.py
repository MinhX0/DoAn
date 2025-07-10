from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback
import pandas as pd

# Import the StockPredictor class from the current file
from stock_prediction_rf import StockPredictor


# Global cache for ticker data
ticker_data_cache = {}
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

app = FastAPI()

class PredictionRequest(BaseModel):
    symbol: str
    lookback_days: Optional[int] = 10


# Endpoint 2: Get prediction data for a ticker (GET version)
@app.get("/prediction_data")
def get_prediction_data(symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"), lookback_days: int = Query(10, description="Lookback days for features")):
    try:
        predictor = StockPredictor(symbol, lookback_days=lookback_days)
        if not predictor.fetch_data():
            return {"error": "Could not fetch data for symbol."}
        if not predictor.train_model():
            return {"error": "Not enough data to train model."}
        prediction = predictor.predict_next_day()
        if prediction is None:
            return {"error": "Prediction failed."}
        advice = predictor.get_investment_advice(prediction)
        direction_text = predictor.get_direction_text(prediction['direction'])
        return {
            "symbol": symbol,
            "current_price": prediction['current_price'],
            "predicted_price": prediction['predicted_price'],
            "predicted_return": prediction['predicted_return'],
            "direction": direction_text,
            "confidence": prediction['confidence'],
            "probabilities": list(prediction['probabilities']),
            "advice": advice
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# Prefetch ticker data at startup
@app.on_event("startup")
def prefetch_ticker_data():
    print("Prefetching ticker data for:", DEFAULT_TICKERS)
    for symbol in DEFAULT_TICKERS:
        try:
            predictor = StockPredictor(symbol)
            if predictor.fetch_data():
                ticker_data_cache[symbol] = predictor.data.copy()
                print(f"Prefetched {symbol} ({len(predictor.data)} rows)")
            else:
                print(f"Failed to prefetch {symbol}")
        except Exception as e:
            print(f"Error prefetching {symbol}: {e}")

# Endpoint 1: Get ticker data for charting
@app.get("/ticker_data")
def get_ticker_data(symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"), days: int = Query(30, description="Number of days to fetch")):
    try:
        df = None
        if symbol in ticker_data_cache:
            df = ticker_data_cache[symbol].tail(days)
        else:
            predictor = StockPredictor(symbol)
            if not predictor.fetch_data():
                return {"error": "Could not fetch data for symbol."}
            df = predictor.data.tail(days)
            ticker_data_cache[symbol] = predictor.data.copy()
        # Fix: ensure index is serializable (convert DatetimeIndex to string)
        df = df.copy()
        if not df.empty:
            # Always convert the index to a string column, but do not rename if already present
            df = df.reset_index()
            # Ensure all column names are strings (avoid tuple column names from MultiIndex)
            df.columns = [str(col) if not isinstance(col, str) else col for col in df.columns]
            # Convert the first column (index) to string type for JSON serialization
            first_col = df.columns[0]
            df[first_col] = df[first_col].astype(str)
            # Convert all columns to native Python types to avoid numpy types in JSON
            data = df.applymap(lambda x: x.item() if hasattr(x, 'item') else x).to_dict(orient="records")
        else:
            data = []
        return {"symbol": symbol, "data": data}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# Endpoint 2: Get prediction data for a ticker
@app.get("/prediction_data")
def get_prediction_data(symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"), lookback_days: int = Query(10, description="Lookback days for features")):
    try:
        predictor = StockPredictor(symbol, lookback_days=lookback_days)
        if not predictor.fetch_data():
            return {"error": "Could not fetch data for symbol."}
        if not predictor.train_model():
            return {"error": "Not enough data to train model."}
        prediction = predictor.predict_next_day()
        if prediction is None:
            return {"error": "Prediction failed."}
        advice = predictor.get_investment_advice(prediction)
        direction_text = predictor.get_direction_text(prediction['direction'])
        return {
            "symbol": symbol,
            "current_price": prediction['current_price'],
            "predicted_price": prediction['predicted_price'],
            "predicted_return": prediction['predicted_return'],
            "direction": direction_text,
            "confidence": prediction['confidence'],
            "probabilities": list(prediction['probabilities']),
            "advice": advice
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# For local testing
if __name__ == "__main__":
    uvicorn.run("stock_prediction_api:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback

# Import the StockPredictor class from the current file
from stock_prediction_rf import StockPredictor


class PredictionRequest(BaseModel):
    symbol: str
    lookback_days: Optional[int] = 10

@app.post("/predict")
def predict_stock(req: PredictionRequest):
    try:
        predictor = StockPredictor(req.symbol, lookback_days=req.lookback_days)
        if not predictor.fetch_data():
            return {"error": "Could not fetch data for symbol."}
        if not predictor.train_model():
            return {"error": "Not enough data to train model."}
        prediction = predictor.predict_next_day()
        if prediction is None:
            return {"error": "Prediction failed."}
        advice = predictor.get_investment_advice(prediction)
        direction_text = predictor.get_direction_text(prediction['direction'])
        return {
            "symbol": req.symbol,
            "current_price": prediction['current_price'],
            "predicted_price": prediction['predicted_price'],
            "predicted_return": prediction['predicted_return'],
            "direction": direction_text,
            "confidence": prediction['confidence'],
            "probabilities": list(prediction['probabilities']),
            "advice": advice
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# For local testing
if __name__ == "__main__":
    uvicorn.run("stock_prediction_api:app", host="0.0.0.0", port=8000, reload=True)
