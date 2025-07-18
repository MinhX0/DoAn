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
DEFAULT_TICKERS = ["VCB.VN", "VIC.VN", "VHM.VN", "HPG.VN", "FPT.VN", "BID.VN", "GAS.VN", "VNM.VN", "TCB.VN", "CTG.VN", "VPB.VN", "MBB.VN", "ACB.VN", "MSN.VN", "MWG.VN", "GVR.VN", "STB.VN", "HDB.VN", "SSI.VN", "VRE.VN", "SAB.VN", "PLX.VN", "VJC.VN", "TPB.VN", "POW.VN", "DGC.VN", "PNJ.VN", "BVH.VN", "REE.VN", "KDH.VN", "EIB.VN", "OCB.VN", "MSB.VN", "LPB.VN", "SHB.VN", "VIB.VN", "PDR.VN", "DXG.VN", "HSG.VN", "PC1.VN"]
# CSV file paths
USER_PREF_CSV = "user_preferences.csv"

# User preferences cache (in-memory, loaded from CSV)
user_preferences = set()

app = FastAPI()

# Endpoint: Get all tickers with current price and up/down indicator
@app.get("/ticker_status") 
def get_ticker_status():
    result = []
    for symbol in DEFAULT_TICKERS:
        df = ticker_data_cache.get(symbol)
        price = None
        prev_price = None
        # Try cache first, then fetch if needed
        if df is None or df.empty or 'Close' not in df.columns:
            predictor = StockPredictor(symbol)
            if not predictor.fetch_data():
                result.append({"symbol": symbol, "price": None, "indicator": "unknown"})
                continue
            df = predictor.data
            if df is not None and not df.empty and 'Close' in df.columns:
                ticker_data_cache[symbol] = df.copy()
            else:
                result.append({"symbol": symbol, "price": None, "indicator": "unknown"})
                continue
        # Now df should be valid
        try:
            last_row = df.iloc[-1]
            price = last_row['Close'] if 'Close' in last_row else None
            if len(df) > 1:
                prev_row = df.iloc[-2]
                prev_price = prev_row['Close'] if 'Close' in prev_row else None
        except Exception:
            price = None
            prev_price = None
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



# Helper: Save all prefetched ticker data to CSV (one file, with symbol column)# Helper: Load prefetched ticker data from CSV
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


# Helper: Check if cache file is from today


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
    load_user_preferences_from_csv()

# Endpoint 1: Get ticker data for charting
@app.get("/ticker_data")
def get_ticker_data(symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"), days: int = Query(30, description="Number of days to fetch")):
    try:
        import yfinance as yf
        df = ticker_data_cache.get(symbol)
        if df is None or df.empty or 'Close' not in df.columns:
            predictor = StockPredictor(symbol)
            if not predictor.fetch_data():
                return {"error": "Could not fetch data for symbol."}
            df = predictor.data
            if df is not None and not df.empty and 'Close' in df.columns:
                ticker_data_cache[symbol] = df.copy()
            else:
                return {"error": "No valid data for symbol."}
        df = df.tail(days)
        # Get company name using yfinance (do not cache)
        try:
            ticker_obj = yf.Ticker(symbol)
            info = ticker_obj.info
            company_name = info.get('shortName') or info.get('longName') or info.get('name') or symbol
        except Exception:
            company_name = symbol
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
        return {"symbol": symbol, "company_name": company_name, "data": data}
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
    similar_tickers = get_similar_tickers_from_gemini(liked_tickers, DEFAULT_TICKERS)
    return {"liked_tickers": liked_tickers, "suggested_tickers": similar_tickers}

@app.get("/ticker_closing_prices", summary="Get closing prices for a ticker")
def get_ticker_closing_prices(symbol: str = Query(..., description="Ticker symbol, e.g. VCB.VN"), days: int = Query(30, description="Number of days to fetch")):
    """
    Returns a list of closing prices for the given ticker, for the specified number of days (most recent first).
    """
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
        if df is not None and not df.empty and 'Close' in df.columns:
            closing_prices = [float(x) for x in df['Close'].values.flatten()]
        else:
            closing_prices = []
        return {"symbol": symbol, "closing_prices": closing_prices}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# For local testing
if __name__ == "__main__":
    uvicorn.run("stock_prediction_api:app", host="0.0.0.0", port=8000, reload=True)