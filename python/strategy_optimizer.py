import pandas as pd
import numpy as np
from itertools import product

class StrategyOptimizer:
    """
    Class to optimize trading strategy thresholds based on historical performance.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.results = None
        
    def calculate_returns(self, buy_threshold, sell_threshold, exit_buy_threshold=None, exit_sell_threshold=None):
        """
        Calculate returns for a given set of thresholds.
        
        Parameters:
        -----------
        buy_threshold : float
            Composite Index threshold to generate buy signal
        sell_threshold : float
            Composite Index threshold to generate sell signal
        exit_buy_threshold : float, optional
            Threshold to exit a buy position, defaults to sell_threshold
        exit_sell_threshold : float, optional
            Threshold to exit a sell position, defaults to buy_threshold
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if 'Composite_Index' not in self.df.columns or 'Close' not in self.df.columns:
            return {'total_return': 0, 'max_drawdown': -100, 'sharpe_ratio': 0}
        
        # Set default exit thresholds if not provided
        if exit_buy_threshold is None:
            exit_buy_threshold = sell_threshold + 5  # Default to 5 points above sell threshold
        if exit_sell_threshold is None:
            exit_sell_threshold = buy_threshold - 5  # Default to 5 points below buy threshold
            
        try:
            df = self.df.copy()
            
            # Initialize position column (1 = long, -1 = short, 0 = cash)
            df['Position'] = 0
            
            # Initialize signals
            current_position = 0
            
            # Go through each row to determine position based on thresholds and trend
            for i in range(1, len(df)):
                idx = df.index[i]
                prev_idx = df.index[i-1]
                composite_value = df.at[idx, 'Composite_Index']
                
                # Current position is the previous row's position initially
                current_position = df.at[prev_idx, 'Position']
                
                # Logic for entering/exiting positions
                if current_position == 0:  # Currently in cash
                    if composite_value >= buy_threshold:
                        current_position = 1  # Enter long position
                    elif composite_value <= sell_threshold:
                        current_position = -1  # Enter short position
                elif current_position == 1:  # Currently long
                    if composite_value <= exit_buy_threshold:
                        current_position = 0  # Exit long position
                elif current_position == -1:  # Currently short
                    if composite_value >= exit_sell_threshold:
                        current_position = 0  # Exit short position
                
                # Store the current position
                df.at[idx, 'Position'] = current_position
            
            # Calculate daily returns
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Calculate strategy returns (position * next day's return)
            df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)
            
            # Drop NaN values
            df = df.dropna(subset=['Strategy_Return'])
            
            if len(df) < 2:
                return {'total_return': 0, 'max_drawdown': -100, 'sharpe_ratio': 0}
                
            # Calculate cumulative returns
            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
            
            # Calculate Buy & Hold returns
            df['BuyHold_Return'] = (1 + df['Daily_Return']).cumprod()
            
            # Calculate drawdowns
            df['Peak'] = df['Cumulative_Return'].cummax()
            df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak']
            
            # Calculate metrics
            total_return = df['Cumulative_Return'].iloc[-1] - 1
            max_drawdown = df['Drawdown'].min()
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = np.sqrt(252) * df['Strategy_Return'].mean() / df['Strategy_Return'].std() if df['Strategy_Return'].std() > 0 else 0
            
            # Calculate win rate and average trade metrics
            df['Trade_Start'] = df['Position'] != df['Position'].shift(1)
            trade_starts = df[df['Trade_Start']].index
            
            trade_returns = []
            winning_trades = 0
            
            for i in range(len(trade_starts) - 1):
                start_idx = trade_starts[i]
                end_idx = trade_starts[i+1]
                
                # Skip cash positions
                if df.at[start_idx, 'Position'] == 0:
                    continue
                    
                # Calculate trade return
                trade_return = df.loc[start_idx:end_idx, 'Strategy_Return'].sum()
                trade_returns.append(trade_return)
                
                if trade_return > 0:
                    winning_trades += 1
            
            # Add last trade if it's not cash
            if len(trade_starts) > 0 and df.at[trade_starts[-1], 'Position'] != 0:
                last_trade_return = df.loc[trade_starts[-1]:, 'Strategy_Return'].sum()
                trade_returns.append(last_trade_return)
                if last_trade_return > 0:
                    winning_trades += 1
            
            # Calculate win rate and average trade metrics
            num_trades = len(trade_returns)
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            avg_win = np.mean([r for r in trade_returns if r > 0]) if len([r for r in trade_returns if r > 0]) > 0 else 0
            avg_loss = np.mean([r for r in trade_returns if r <= 0]) if len([r for r in trade_returns if r <= 0]) > 0 else 0
            profit_factor = -sum([r for r in trade_returns if r > 0]) / sum([r for r in trade_returns if r <= 0]) if sum([r for r in trade_returns if r <= 0]) < 0 else 0
            
            return {
                'total_return': total_return * 100,  # as percentage
                'max_drawdown': max_drawdown * 100,  # as percentage
                'sharpe_ratio': sharpe_ratio,
                'num_trades': num_trades,
                'win_rate': win_rate * 100,  # as percentage
                'avg_win': avg_win * 100,  # as percentage
                'avg_loss': avg_loss * 100,  # as percentage
                'profit_factor': profit_factor,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'exit_buy_threshold': exit_buy_threshold,
                'exit_sell_threshold': exit_sell_threshold
            }
            
        except Exception as e:
            print(f"Error in calculate_returns: {str(e)}")
            return {'total_return': 0, 'max_drawdown': -100, 'sharpe_ratio': 0}
    
    def optimize_thresholds(self, min_period=0):
        """
        Find optimal buy/sell thresholds by testing various combinations.
        
        Parameters:
        -----------
        min_period : int, optional
            Minimum number of recent days to consider (0 = all data)
        
        Returns:
        --------
        dict
            Dictionary containing optimal thresholds and performance metrics
        """
        if 'Composite_Index' not in self.df.columns:
            return None
            
        # Filter to recent data if specified
        if min_period > 0 and len(self.df) > min_period:
            df = self.df.iloc[-min_period:].copy()
        else:
            df = self.df.copy()
            
        # Store original dataframe and use filtered one
        original_df = self.df
        self.df = df
        
        try:
            # Define range of thresholds to test
            buy_thresholds = range(55, 75, 5)  # From 55 to 70 in steps of 5
            sell_thresholds = range(45, 25, -5)  # From 45 to 30 in steps of 5
            
            # Also test exit thresholds (tighter and looser)
            exit_buy_offsets = [0, 5, 10]  # e.g., exit a buy when index falls to sell_threshold + offset
            exit_sell_offsets = [0, 5, 10]  # e.g., exit a sell when index rises to buy_threshold - offset
            
            results = []
            
            # Try each combination
            for buy, sell, exit_buy_offset, exit_sell_offset in product(buy_thresholds, sell_thresholds, exit_buy_offsets, exit_sell_offsets):
                if buy <= sell:  # Skip invalid combinations
                    continue
                    
                # Calculate exit thresholds
                exit_buy = sell + exit_buy_offset
                exit_sell = buy - exit_sell_offset
                
                # Calculate returns for this combination
                result = self.calculate_returns(buy, sell, exit_buy, exit_sell)
                results.append(result)
                
            # Find best strategy by Sharpe ratio
            results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
            
            # Store results for later use
            self.results = results
            
            # Restore original dataframe
            self.df = original_df
            
            return results[0] if results else None
            
        except Exception as e:
            print(f"Error in optimize_thresholds: {str(e)}")
            # Restore original dataframe
            self.df = original_df
            return None
    
    def apply_optimal_strategy(self, thresholds=None, min_period=0):
        """
        Apply the optimal (or provided) strategy to the dataframe.
        
        Parameters:
        -----------
        thresholds : dict, optional
            Dictionary containing threshold values, if None, will find optimal
        min_period : int, optional
            Minimum number of recent days to consider for optimization
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with strategy applied
        """
        if thresholds is None:
            thresholds = self.optimize_thresholds(min_period)
            
        if thresholds is None:
            return self.df.copy()
            
        # Apply the strategy to the full dataset
        buy_threshold = thresholds['buy_threshold']
        sell_threshold = thresholds['sell_threshold']
        exit_buy_threshold = thresholds.get('exit_buy_threshold', sell_threshold + 5)
        exit_sell_threshold = thresholds.get('exit_sell_threshold', buy_threshold - 5)
        
        df = self.df.copy()
        
        # Initialize position column (1 = long, -1 = short, 0 = cash)
        df['Position'] = 0
        
        # Initialize signals
        current_position = 0
        
        # Go through each row to determine position based on thresholds
        for i in range(1, len(df)):
            idx = df.index[i]
            prev_idx = df.index[i-1]
            composite_value = df.at[idx, 'Composite_Index']
            
            # Current position is the previous row's position initially
            current_position = df.at[prev_idx, 'Position']
            
            # Logic for entering/exiting positions
            if current_position == 0:  # Currently in cash
                if composite_value >= buy_threshold:
                    current_position = 1  # Enter long position
                elif composite_value <= sell_threshold:
                    current_position = -1  # Enter short position
            elif current_position == 1:  # Currently long
                if composite_value <= exit_buy_threshold:
                    current_position = 0  # Exit long position
            elif current_position == -1:  # Currently short
                if composite_value >= exit_sell_threshold:
                    current_position = 0  # Exit short position
            
            # Store the current position
            df.at[idx, 'Position'] = current_position
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate strategy returns (position * next day's return)
        df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)
        
        # Calculate Buy & Hold returns (starting with $1000)
        df['BuyHold_Value'] = 1000 * (1 + df['Daily_Return']).cumprod().fillna(1)
        
        # Calculate Strategy Value (starting with $1000)
        df['Strategy_Value'] = 1000 * (1 + df['Strategy_Return']).cumprod().fillna(1)
        
        # Calculate drawdowns for both
        df['BuyHold_Peak'] = df['BuyHold_Value'].cummax()
        df['BuyHold_Drawdown'] = (df['BuyHold_Value'] - df['BuyHold_Peak']) / df['BuyHold_Peak']
        
        df['Strategy_Peak'] = df['Strategy_Value'].cummax()
        df['Strategy_Drawdown'] = (df['Strategy_Value'] - df['Strategy_Peak']) / df['Strategy_Peak']
        
        # Add signals for plotting
        df['Buy_Signal'] = (df['Position'].shift(1) == 0) & (df['Position'] == 1)
        df['Sell_Signal'] = (df['Position'].shift(1) == 0) & (df['Position'] == -1)
        df['Exit_Signal'] = ((df['Position'].shift(1) != 0) & (df['Position'] == 0))
        
        return df
