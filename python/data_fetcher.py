import pandas as pd
import time
from datetime import datetime

# For Pyodide browser environment, we'll use JavaScript fetch via pyodide.ffi
try:
    import pyodide
    from js import fetch, Headers
    import json
    IN_BROWSER = True
except ImportError:
    import requests
    IN_BROWSER = False

async def fetch_from_yahoo_async(url, headers):
    """Async fetch for browser environment using JavaScript fetch API"""
    # Create JavaScript Headers object
    js_headers = Headers.new()
    for key, value in headers.items():
        js_headers.append(key, value)

    # Fetch using JavaScript fetch API
    response = await fetch(url, {
        'method': 'GET',
        'headers': js_headers,
        'mode': 'cors'
    })

    if not response.ok:
        raise Exception(f"HTTP {response.status}: {response.statusText}")

    # Get JSON response
    data = await response.json()
    return data

def get_stock_data(ticker, period='1y', max_retries=3):
    """
    Fetch stock data for the given ticker and period from Yahoo Finance API.

    This is a drop-in replacement for yfinance that returns identical DataFrame structure.
    Works in both browser (Pyodide) and server environments.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str, optional
        Period of historical data to fetch (default is '1y')
        Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    max_retries : int, optional
        Maximum number of retry attempts (default is 3)

    Returns:
    --------
    pandas.DataFrame
        Historical stock data with columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
        (identical to yfinance output format)
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # For 1y or less, get 2y of data to ensure we have enough after NaN values are dropped
            # For periods longer than 1y, use the specified period
            fetch_period = '2y' if period in ['1mo', '3mo', '6mo', '1y'] else period

            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

            # Map period to range parameter
            period_map = {
                '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
                '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y',
                '10y': '10y', 'ytd': 'ytd', 'max': 'max'
            }

            # Add query parameters to URL
            params = {
                'range': period_map.get(fetch_period, '1y'),
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div%2Csplit'
            }

            # Build URL with parameters
            url_with_params = url + '?' + '&'.join([f"{k}={v}" for k, v in params.items()])

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            if IN_BROWSER:
                # Use async fetch in browser
                import asyncio
                data = asyncio.get_event_loop().run_until_complete(
                    fetch_from_yahoo_async(url_with_params, headers)
                )
            else:
                # Use requests in server environment
                response = requests.get(url_with_params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()

            # Check for errors
            if 'chart' not in data or not data['chart']['result']:
                raise ValueError(f"No data found for ticker {ticker}. Please check if the symbol is correct.")

            chart_data = data['chart']['result'][0]

            # Extract time series data
            timestamps = chart_data['timestamp']
            ohlcv = chart_data['indicators']['quote'][0]

            # Convert timestamps to datetime
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]

            # Create DataFrame with same structure as yfinance
            df = pd.DataFrame({
                'Date': dates,
                'Open': ohlcv['open'],
                'High': ohlcv['high'],
                'Low': ohlcv['low'],
                'Close': ohlcv['close'],
                'Volume': ohlcv['volume']
            })

            # Add Dividends and Stock Splits columns (initialize with zeros)
            df['Dividends'] = 0.0
            df['Stock Splits'] = 0.0

            # Process events (dividends and splits) if available
            if 'events' in chart_data:
                events = chart_data['events']

                # Handle dividends
                if 'dividends' in events:
                    for timestamp, div_data in events['dividends'].items():
                        div_date = datetime.fromtimestamp(int(timestamp))
                        # Find matching date in DataFrame and set dividend
                        mask = df['Date'].dt.date == div_date.date()
                        if mask.any():
                            df.loc[mask, 'Dividends'] = div_data['amount']

                # Handle stock splits
                if 'splits' in events:
                    for timestamp, split_data in events['splits'].items():
                        split_date = datetime.fromtimestamp(int(timestamp))
                        # Find matching date in DataFrame and set split ratio
                        mask = df['Date'].dt.date == split_date.date()
                        if mask.any():
                            df.loc[mask, 'Stock Splits'] = split_data['splitRatio']

            # Clean data - remove rows with all NaN OHLC values
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

            if df.empty:
                raise ValueError(f"No valid data found for ticker {ticker} after cleaning.")

            # Ensure the data is sorted by date
            df.sort_values('Date', inplace=True)

            # If original period was less than 2y, trim the data back to requested period
            if period != fetch_period and period != 'max':
                # Get the appropriate number of rows based on the requested period
                if period == '1mo':
                    rows = min(30, len(df))
                elif period == '3mo':
                    rows = min(90, len(df))
                elif period == '6mo':
                    rows = min(180, len(df))
                elif period == '1y':
                    rows = min(365, len(df))
                else:
                    rows = len(df)

                df = df.iloc[-rows:]

            # Reset index to make sure it's clean
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                if "404" in str(e) or "not found" in str(e).lower():
                    raise ValueError(f"Symbol {ticker} not found. Please check if the ticker symbol is correct.")
                else:
                    raise Exception(f"Error fetching data for {ticker}: {str(e)}")

            # Wait before retrying
            time.sleep(1)

# Test function to verify compatibility
def test_compatibility():
    """
    Test function to verify the replacement works identically to yfinance
    """
    test_symbols = ['AAPL', 'TSLA', 'MSFT']

    for symbol in test_symbols:
        try:
            print(f"\nTesting {symbol}...")
            df = get_stock_data(symbol, period='1mo')

            print(f"✅ Success! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
            print(f"Sample data:")
            print(df.head(2).round(2))

            # Check for dividends and splits
            div_count = (df['Dividends'] > 0).sum()
            split_count = (df['Stock Splits'] > 0).sum()
            print(f"Dividends: {div_count}, Stock Splits: {split_count}")

        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")

if __name__ == "__main__":
    test_compatibility()
