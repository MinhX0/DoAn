from google import genai
from google.genai import types
import os
import ast
import re
import traceback

def get_similar_tickers_from_gemini(liked_tickers, ticker_list):
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY", "AIzaSyC0yEBIWflQgoeTcrv6ASQBuOx58oUNkW4")
    prompt = (
        f"Given these user preference stock tickers: {', '.join(liked_tickers)} and this list of default tickers: {', '.join(ticker_list)}"
        "suggest 5 other VietNam stock tickers that are similar in sector, industry, or business model. "
        "Return only the ticker symbols as a Python list."
    )
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        text = response.candidates[0].content.parts[0].text
        print(f"Raw Gemini response text: {text}")

        # Use regex to extract the Python list string,
        # including re.DOTALL to match across newlines
        match = re.search(r"\[.*?\]", text, re.DOTALL) # <--- IMPORTANT CHANGE HERE
        if match:
            list_str = match.group(0)
            print(f"Extracted list string: {list_str}")
            similar_tickers = ast.literal_eval(list_str)
        else:
            print(f"Warning: Could not find a Python list in Gemini's response. Raw response was: '{text}'")
            similar_tickers = []
        
        return similar_tickers
    except Exception as e:
        print(f"Error calling Gemini API or parsing response: {e}")
        traceback.print_exc()
        raise Exception(e)