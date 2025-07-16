from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback
import pandas as pd
import openai
import os

# Import the StockPredictor class from the current file
from stock_prediction_rf import StockPredictor
from chatgpt_ticker_suggest import get_similar_tickers_from_gemini



# Global cache for ticker data
ticker_data_cache = {}
DEFAULT_TICKERS = ["VCB.VN", "VIC.VN", "VHM.VN", "HPG.VN", "FPT.VN"] 

# CSV file paths
TICKER_DATA_CSV = "prefetched_ticker_data.csv"
USER_PREF_CSV = "user_preferences.csv"

# User preferences cache (in-memory, loaded from CSV)
user_preferences = set()

app = FastAPI()

# Endpoint: Get all tickers with current price and up/down indicator
@app.get("/ticker_status") 
def get_ticker_status():
    result = []
    for symbol in DEFAULT_TICKERS:
        # Try to get latest data from cache
        df = ticker_data_cache.get(symbol)
        price = None
        prev_price = None
        if df is not None and not df.empty:
            # Assume last row is the latest
            last_row = df.iloc[-1]
            price = last_row['Close'] if 'Close' in last_row else None
            # Try to get previous close for up/down
            if len(df) > 1:
                prev_row = df.iloc[-2]
                prev_price = prev_row['Close'] if 'Close' in prev_row else None
        else:
            # Fallback: fetch data
            predictor = StockPredictor(symbol)
            if not predictor.fetch_data():
                result.append({"symbol": symbol, "price": None, "indicator": "unknown"})
                continue
            df = predictor.data
            if df is not None and not df.empty:
                last_row = df.iloc[-1]
                price = last_row['Close'] if 'Close' in last_row else None
                if len(df) > 1:
                    prev_row = df.iloc[-2]
                    prev_price = prev_row['Close'] if 'Close' in prev_row else None
                ticker_data_cache[symbol] = df.copy()
                save_prefetched_data_to_csv()
            else:
                result.append({"symbol": symbol, "price": None, "indicator": "unknown"})
                continue
        # Clean up price values: convert NaN/inf to None, and cast to float
        def safe_float(val):
            try:
                f = float(val)
                if pd.isna(f) or pd.isnull(f) or f != f or f == float('inf') or f == float('-inf'):
                    return None
                return f
            except Exception:
                return None
        price = safe_float(price)
        prev_price = safe_float(prev_price)
        # Determine up/down/unknown
        if price is not None and prev_price is not None:
            if price > prev_price:
                indicator = "up"
            elif price < prev_price:
                indicator = "down"
            else:
                indicator = "no_change"
        else:
            indicator = "unknown"
        result.append({"symbol": symbol, "price": price, "indicator": indicator})
    return {"tickers": result}


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



# Helper: Save all prefetched ticker data to CSV (one file, with symbol column)
def save_prefetched_data_to_csv():
    all_data = []
    for symbol, df in ticker_data_cache.items():
        df = df.copy()
        df['symbol'] = symbol
        all_data.append(df)
    if all_data:
        full_df = pd.concat(all_data)
        full_df.to_csv(TICKER_DATA_CSV, index=False)

# Helper: Load prefetched ticker data from CSV
def load_prefetched_data_from_csv():
    try:
        df = pd.read_csv(TICKER_DATA_CSV)
        for symbol in df['symbol'].unique():
            ticker_data_cache[symbol] = df[df['symbol'] == symbol].drop(columns=['symbol'])
        print(f"Loaded prefetched ticker data from {TICKER_DATA_CSV}")
    except Exception as e:
        print(f"No prefetched ticker data found: {e}")

# Helper: Save user preferences to CSV
def save_user_preferences_to_csv():
    if user_preferences:
        pref_df = pd.DataFrame({"symbol": list(user_preferences)})
        pref_df.to_csv(USER_PREF_CSV, index=False)

# Helper: Load user preferences from CSV
def load_user_preferences_from_csv():
    try:
        df = pd.read_csv(USER_PREF_CSV)
        user_preferences.clear()
        for _, row in df.iterrows():
            user_preferences.add(row['symbol'])
        print(f"Loaded user preferences from {USER_PREF_CSV}")
    except Exception as e:
        print(f"No user preferences found: {e}")

# Prefetch ticker data at startup
@app.on_event("startup")
def prefetch_ticker_data():
    print("Prefetching ticker data for:", DEFAULT_TICKERS)
    load_prefetched_data_from_csv()
    load_user_preferences_from_csv()
    for symbol in DEFAULT_TICKERS:
        if symbol in ticker_data_cache:
            continue
        try:
            predictor = StockPredictor(symbol)
            if predictor.fetch_data():
                ticker_data_cache[symbol] = predictor.data.copy()
                print(f"Prefetched {symbol} ({len(predictor.data)} rows)")
            else:
                print(f"Failed to prefetch {symbol}")
        except Exception as e:
            print(f"Error prefetching {symbol}: {e}")
    save_prefetched_data_to_csv()

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
            save_prefetched_data_to_csv()
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

# Endpoint: Save user preference (POST)
from fastapi import Body
from typing import Dict

class UserPreferenceRequest(BaseModel):
    symbol: str

@app.post("/save_preference")
def save_user_preference(req: UserPreferenceRequest):
    user_preferences.add(req.symbol)
    save_user_preferences_to_csv()
    return {"status": "success"}

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
        # Determine user risk appetite based on like status
        if req.symbol in user_preferences:
            risk_appetite = "high"
        else:
            risk_appetite = "low"
        # Generate advice based on both prediction and risk appetite
        advice = generate_precise_advice(prediction, risk_appetite)
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

# Helper: Generate advice based on prediction and user risk appetite

def generate_precise_advice(prediction, risk_appetite):
    pred_price = prediction.get('predicted_price')
    curr_price = prediction.get('current_price')
    pred_return = prediction.get('predicted_return')
    direction = prediction.get('direction')
    confidence = prediction.get('confidence')
    # Example logic:
    if pred_price is None or curr_price is None:
        return "Not enough data for advice."
    if risk_appetite == "high":
        if direction == 1 and pred_return > 0.01 and confidence > 0.6:
            return "Aggressive buy: Model predicts strong upward movement."
        elif direction == 1:
            return "Buy: Model predicts price increase."
        elif direction == -1 and pred_return < -0.01:
            return "Consider selling: Model predicts a drop."
        else:
            return "Hold: No strong signal."
    else:  # low risk appetite
        if direction == 1 and pred_return > 0.02 and confidence > 0.7:
            return "Buy (conservative): Model predicts a solid upward trend."
        elif direction == -1 and pred_return < -0.01:
            return "Sell: Model predicts a possible drop."
        else:
            return "Hold: Wait for a clearer signal."


@app.get("/suggest_similar_tickers", tags=["Discovery"], summary="Suggest similar tickers using ChatGPT")
def suggest_similar_tickers():
    """
    Suggests similar tickers to those liked by the user, using ChatGPT API.
    """
    liked_tickers = list(user_preferences)
    if not liked_tickers:
        return {"error": "No liked tickers found."}
    similar_tickers = get_similar_tickers_from_gemini(liked_tickers)
    return {"liked_tickers": liked_tickers, "suggested_tickers": similar_tickers}

# For local testing
if __name__ == "__main__":
    uvicorn.run("stock_prediction_api:app", host="0.0.0.0", port=8000, reload=True)