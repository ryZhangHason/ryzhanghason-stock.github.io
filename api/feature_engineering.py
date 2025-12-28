import pandas as pd
import numpy as np

# Import the 'ta' package - check which modules are available
from ta import momentum, trend, volatility, volume

def _handle_infinite_values(df):
    """
    Handle infinite values in DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with potential infinite values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with infinite values replaced
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace inf and -inf with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # For each column, replace NaN with the column median or 0 if median is unavailable
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                df_clean[col].fillna(0, inplace=True)
            else:
                df_clean[col].fillna(median_val, inplace=True)
    
    return df_clean

def safe_division(a, b, default=0):
    """
    Safely divide two numbers, avoiding division by zero.
    
    Parameters:
    -----------
    a : numeric or array-like
        Numerator
    b : numeric or array-like
        Denominator
    default : numeric, optional
        Default value when division is invalid (default is 0)
        
    Returns:
    --------
    numeric or array-like
        Result of division or default value where division is invalid
    """
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return pd.Series(np.where(b != 0, a / b, default), index=a.index)
    else:
        return a / b if b != 0 else default

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with stock data
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added technical indicators
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        df_features = df.copy()
        
        # Calculate basic indicators
        # Moving averages
        df_features['MA5'] = df_features['Close'].rolling(window=5).mean()
        df_features['MA10'] = df_features['Close'].rolling(window=10).mean()
        df_features['MA20'] = df_features['Close'].rolling(window=20).mean()
        df_features['MA50'] = df_features['Close'].rolling(window=50).mean()
        df_features['MA200'] = df_features['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df_features['EMA5'] = df_features['Close'].ewm(span=5, adjust=False).mean()
        df_features['EMA20'] = df_features['Close'].ewm(span=20, adjust=False).mean()
        df_features['EMA50'] = df_features['Close'].ewm(span=50, adjust=False).mean()
        
        # Price changes
        df_features['Price_Change'] = df_features['Close'].pct_change()
        df_features['Price_Change_1d'] = df_features['Close'].pct_change(periods=1)
        df_features['Price_Change_5d'] = df_features['Close'].pct_change(periods=5)
        df_features['Price_Change_10d'] = df_features['Close'].pct_change(periods=10)
        
        # Volatility
        df_features['Volatility_5d'] = df_features['Price_Change'].rolling(window=5).std()
        df_features['Volatility_10d'] = df_features['Price_Change'].rolling(window=10).std()
        df_features['Volatility_20d'] = df_features['Price_Change'].rolling(window=20).std()
        
        # Moving Average Convergence Divergence (MACD)
        exp12 = df_features['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df_features['Close'].ewm(span=26, adjust=False).mean()
        df_features['MACD'] = exp12 - exp26
        df_features['MACD_signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
        df_features['MACD_hist'] = df_features['MACD'] - df_features['MACD_signal']
        
        # Add advanced indicators using 'ta' package
        try:
            # Add RSI (Relative Strength Index)
            indicator_rsi = momentum.RSIIndicator(close=df_features['Close'], window=14)
            df_features['RSI'] = indicator_rsi.rsi()
            
            # Add Stochastic Oscillator
            indicator_stoch = momentum.StochasticOscillator(
                high=df_features['High'],
                low=df_features['Low'],
                close=df_features['Close'],
                window=14,
                smooth_window=3
            )
            df_features['Stoch_K'] = indicator_stoch.stoch()
            df_features['Stoch_D'] = indicator_stoch.stoch_signal()
            
            # Add Bollinger Bands
            indicator_bb = volatility.BollingerBands(close=df_features['Close'], window=20, window_dev=2)
            df_features['BB_middle'] = indicator_bb.bollinger_mavg()
            df_features['BB_upper'] = indicator_bb.bollinger_hband()
            df_features['BB_lower'] = indicator_bb.bollinger_lband()
            df_features['BB_width'] = indicator_bb.bollinger_wband()
            df_features['BB_pct'] = indicator_bb.bollinger_pband()  # Percent B
            
            # Add Average Directional Index (ADX)
            indicator_adx = trend.ADXIndicator(
                high=df_features['High'],
                low=df_features['Low'],
                close=df_features['Close'],
                window=14
            )
            df_features['ADX'] = indicator_adx.adx()
            df_features['ADX_pos'] = indicator_adx.adx_pos()
            df_features['ADX_neg'] = indicator_adx.adx_neg()
            
            # Use a standard approach for CCI instead of the class
            # Since the CCIIndicator might not be available in the installed version
            def calculate_cci(high, low, close, window=20):
                tp = (high + low + close) / 3
                tp_mean = tp.rolling(window=window).mean()
                tp_mean_dev = tp.subtract(tp_mean).abs().rolling(window=window).mean()
                # Use safe division to avoid infinity
                cci = safe_division(tp - tp_mean, 0.015 * tp_mean_dev, default=0)
                return cci
            
            df_features['CCI'] = calculate_cci(df_features['High'], df_features['Low'], df_features['Close'])
            
            # Add MACD with ta package (in case the custom calculation above fails)
            if 'MACD' not in df_features.columns:
                indicator_macd = trend.MACD(
                    close=df_features['Close'],
                    window_slow=26,
                    window_fast=12,
                    window_sign=9
                )
                df_features['MACD'] = indicator_macd.macd()
                df_features['MACD_signal'] = indicator_macd.macd_signal()
                df_features['MACD_hist'] = indicator_macd.macd_diff()
            
            # Add Money Flow Index (MFI)
            try:
                indicator_mfi = momentum.MFIIndicator(
                    high=df_features['High'],
                    low=df_features['Low'],
                    close=df_features['Close'],
                    volume=df_features['Volume'],
                    window=14
                )
                df_features['MFI'] = indicator_mfi.money_flow_index()
            except Exception as mfi_err:
                print(f"MFI calculation error: {mfi_err}")
                # Calculate MFI manually if the class is not available
                df_features['Typical_Price'] = (df_features['High'] + df_features['Low'] + df_features['Close']) / 3
                df_features['Money_Flow'] = df_features['Typical_Price'] * df_features['Volume']
                
                df_features['TP_Delta'] = df_features['Typical_Price'].diff()
                df_features['Positive_Flow'] = np.where(df_features['TP_Delta'] > 0, df_features['Money_Flow'], 0)
                df_features['Negative_Flow'] = np.where(df_features['TP_Delta'] < 0, df_features['Money_Flow'], 0)
                
                df_features['Positive_Flow_Sum'] = df_features['Positive_Flow'].rolling(window=14).sum()
                df_features['Negative_Flow_Sum'] = df_features['Negative_Flow'].rolling(window=14).sum()
                
                # Use safe division to avoid infinity
                money_ratio = safe_division(df_features['Positive_Flow_Sum'], df_features['Negative_Flow_Sum'], default=1)
                df_features['MFI'] = 100 - (100 / (1 + money_ratio))
            
            # Add On-Balance Volume (OBV)
            indicator_obv = volume.OnBalanceVolumeIndicator(close=df_features['Close'], volume=df_features['Volume'])
            df_features['OBV'] = indicator_obv.on_balance_volume()
        
        except Exception as e:
            print(f"Error with ta package indicators: {str(e)}")
            # Fallback to manual calculations for basic indicators
            
            # Calculate RSI manually if not already calculated
            if 'RSI' not in df_features.columns:
                delta = df_features['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # Use safe division to avoid infinity
                rs = safe_division(avg_gain, avg_loss, default=1)
                df_features['RSI'] = 100 - (100 / (1 + rs))
        
        # Add mean reversion indicators
        df_features['Distance_MA20'] = safe_division(df_features['Close'], df_features['MA20'], default=1) - 1
        df_features['Distance_MA50'] = safe_division(df_features['Close'], df_features['MA50'], default=1) - 1
        
        # Add directional indicators
        df_features['MA_Cross_Signal'] = np.where(df_features['MA5'] > df_features['MA20'], 1, -1)
        df_features['Price_MA_Cross'] = np.where(df_features['Close'] > df_features['MA20'], 1, -1)
        
        # Volume indicators
        df_features['Volume_Change'] = df_features['Volume'].pct_change()
        df_features['Volume_MA20'] = df_features['Volume'].rolling(window=20).mean()
        df_features['Relative_Volume'] = safe_division(df_features['Volume'], df_features['Volume_MA20'], default=1)
        
        # High-Low range
        df_features['HL_Range'] = safe_division(df_features['High'] - df_features['Low'], df_features['Close'], default=0)
        df_features['HL_Range_MA10'] = df_features['HL_Range'].rolling(window=10).mean()
        
        # Calculate ratios between different indicators
        if 'RSI' in df_features.columns:
            df_features['RSI_MA'] = df_features['RSI'].rolling(window=5).mean()
            df_features['RSI_Divergence'] = df_features['RSI'] - df_features['RSI_MA']
        
        # Custom Composite Signal (will be further developed in the model)
        signal_components = []
        
        if 'RSI' in df_features.columns:
            signal_components.append((70 - df_features['RSI']) / 40)  # RSI: 30->1.0, 70->0.0
            
        if 'MACD_hist' in df_features.columns:
            max_macd_hist = df_features['MACD_hist'].abs().rolling(window=50).max()
            # Avoid division by zero
            max_macd_hist = max_macd_hist.replace(0, 1)
            df_features['MACD_hist_norm'] = df_features['MACD_hist'] / max_macd_hist
            signal_components.append((df_features['MACD_hist_norm'] + 1) / 2)  # -1->0.0, 1->1.0
            
        if 'BB_pct' in df_features.columns:
            signal_components.append(df_features['BB_pct'])  # 0->0.0, 1->1.0
            
        if 'Stoch_K' in df_features.columns:
            signal_components.append((100 - df_features['Stoch_K']) / 100)  # 0->1.0, 100->0.0
            
        if signal_components:
            # Create a proper DataFrame for all components to ensure proper alignment
            signal_df = pd.concat(signal_components, axis=1)
            # Handle any NaN values
            signal_df.fillna(0.5, inplace=True)  # Fill with neutral value
            df_features['Composite_Signal'] = signal_df.mean(axis=1)
        
        # Add lagged features for time-series prediction
        add_lagged_features(df_features)
        
        # Target variable: 1 if price goes up next day, 0 otherwise
        df_features['Target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
        
        # Handle infinite values before returning
        df_features = _handle_infinite_values(df_features)
        
        # Drop NaN values
        df_features.dropna(inplace=True)
        
        return df_features
        
    except Exception as e:
        print(f"Error in add_technical_indicators: {str(e)}")
        # Return original data if there's an error
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df

def add_lagged_features(df):
    """
    Add lagged features to the dataframe for time-series prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with stock data and technical indicators
    """
    try:
        # Columns to create lags for - use only a subset to avoid too many features
        base_columns = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'MA5', 'MA20', 'MA50',
            'RSI', 'MACD',
            'BB_upper', 'BB_lower',
            'Price_Change', 'Volatility_5d'
        ]
        
        # Filter to only include columns that exist in the dataframe
        base_columns = [col for col in base_columns if col in df.columns]
        
        # List of lag periods - use only a subset to avoid too many features
        lag_periods = [1, 5, 10, 20]
        
        # Create a new dataframe to store all new features to avoid fragmentation
        new_columns = {}
        
        # Create lagged features
        for col in base_columns:
            for lag in lag_periods:
                # Create lagged feature
                lag_col_name = f"{col}_lag{lag}"
                new_columns[lag_col_name] = df[col].shift(lag)
                
                # For price and indicator columns, create change feature
                # Only for critical columns to avoid feature explosion
                if col in ['Close', 'RSI', 'MACD']:
                    change_col_name = f"{col}_change{lag}"
                    # Get the lagged values, replacing zeros with NaN to avoid division by zero
                    lag_values = new_columns[lag_col_name].replace(0, np.nan)
                    # Calculate the percent change, handling NaN values
                    changes = (df[col] / lag_values - 1) * 100
                    # Replace infinite values and NaN with 0
                    changes.replace([np.inf, -np.inf], np.nan, inplace=True)
                    changes.fillna(0, inplace=True)
                    new_columns[change_col_name] = changes
        
        # Add all new columns to the dataframe at once to avoid fragmentation
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data
            
    except Exception as e:
        print(f"Error adding lagged features: {str(e)}")
        # Continue without adding more features if there's an error

def prepare_data_for_model(df_features):
    """
    Prepare data for the machine learning model.
    
    Parameters:
    -----------
    df_features : pandas.DataFrame
        Dataframe with technical indicators
    
    Returns:
    --------
    tuple
        X_train, y_train, latest_features, recent_data
    """
    try:
        # Clean data to handle any remaining infinity values
        df_features = _handle_infinite_values(df_features)
        
        # Get all numeric columns except for 'Target' and 'Date'
        exclude_cols = ['Target', 'Date', 'Datetime', 'Adj Close', 'Composite_Index']
        feature_columns = [col for col in df_features.columns if col not in exclude_cols and 
                        pd.api.types.is_numeric_dtype(df_features[col])]
        
        # Remove highly correlated features to improve training speed
        if len(feature_columns) > 20:  # Only do correlation filtering if we have many features
            # Calculate correlation matrix
            corr_matrix = df_features[feature_columns].corr().abs()
            
            # Create a mask for the upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than 0.98 (increased threshold)
            to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
            
            # Remove highly correlated features
            for col in to_drop:
                if col in feature_columns:
                    feature_columns.remove(col)
            
            print(f"Removed {len(to_drop)} highly correlated features, keeping {len(feature_columns)} features")
        
        # Create a clean copy of the data with selected features only
        clean_data = df_features[feature_columns + ['Target']].copy()
        
        # Create a copy of the Date column if it exists
        if 'Date' in df_features.columns:
            clean_data['Date'] = df_features['Date']
        
        # Save the latest data point for prediction
        latest_features = clean_data[feature_columns].iloc[-1:].copy()
        
        # Calculate how many days to use for recent accuracy
        # Use exactly 120 days or all available data if less than 120 days
        available_days = len(clean_data) - 1  # excluding the latest day which we use for prediction
        recent_data_size = min(120, available_days)
        
        if recent_data_size < available_days:
            # Use the most recent days if we have more than 120 days
            recent_data = clean_data.iloc[-(recent_data_size+1):-1].copy()
        else:
            # Use all available data if we have less than 120 days
            recent_data = clean_data.iloc[:-1].copy()
        
        print(f"Using {len(recent_data)} days for recent accuracy calculation")
        
        # Prepare training data (excluding the last data point)
        train_data = clean_data.iloc[:-1].copy()
        
        X_train = train_data[feature_columns]
        y_train = train_data['Target']
        
        return X_train, y_train, latest_features, recent_data
    except Exception as e:
        print(f"Error in prepare_data_for_model: {str(e)}")
        # If there's an error, return minimal data
        # Get basic features
        basic_features = ['Close', 'Open', 'High', 'Low', 'Volume']
        basic_features = [f for f in basic_features if f in df_features.columns]
        
        X_train = df_features[basic_features].iloc[:-1]
        y_train = df_features['Target'].iloc[:-1]
        latest_features = df_features[basic_features].iloc[-1:].copy()
        recent_data = df_features[basic_features + ['Target']].iloc[-120:].copy()
        
        return X_train, y_train, latest_features, recent_data