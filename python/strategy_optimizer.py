import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# Minimum gap between buy and sell thresholds
MIN_THRESHOLD_GAP = 15


def validate_thresholds(buy_threshold, sell_threshold, min_gap=MIN_THRESHOLD_GAP):
    """
    Validate and correct thresholds to ensure buy > sell with minimum gap.

    Parameters:
    -----------
    buy_threshold : float
        Buy threshold value
    sell_threshold : float
        Sell threshold value
    min_gap : float
        Minimum required gap between buy and sell

    Returns:
    --------
    tuple : (corrected_buy, corrected_sell)
    """
    # Ensure valid ranges first
    buy_threshold = np.clip(buy_threshold, 50, 80)
    sell_threshold = np.clip(sell_threshold, 20, 50)

    # If sell >= buy or gap too small, fix it
    if sell_threshold >= buy_threshold or (buy_threshold - sell_threshold) < min_gap:
        # Calculate midpoint and spread from there
        mid = (buy_threshold + sell_threshold) / 2
        mid = np.clip(mid, 35, 65)  # Keep midpoint reasonable

        buy_threshold = mid + min_gap / 2
        sell_threshold = mid - min_gap / 2

        # Re-clip to valid ranges
        buy_threshold = np.clip(buy_threshold, 50, 80)
        sell_threshold = np.clip(sell_threshold, 20, 50)

        # Final safety check - if still invalid, use safe defaults
        if sell_threshold >= buy_threshold:
            buy_threshold = 65
            sell_threshold = 35

    return round(buy_threshold), round(sell_threshold)


class MarketRegimeDetector:
    """
    Detects market regimes using multiple indicators:
    - Trending Up: Strong upward momentum
    - Trending Down: Strong downward momentum
    - Ranging: Sideways movement with low volatility
    - High Volatility: Choppy market conditions
    """

    REGIME_TRENDING_UP = 'trending_up'
    REGIME_TRENDING_DOWN = 'trending_down'
    REGIME_RANGING = 'ranging'
    REGIME_HIGH_VOLATILITY = 'high_volatility'

    def __init__(self, lookback_period=20):
        self.lookback_period = lookback_period

    def detect_regime(self, df):
        """
        Detect the current market regime based on price action and indicators.

        Returns:
        --------
        str: Current regime classification
        dict: Regime features for meta-learning
        """
        if len(df) < self.lookback_period:
            return self.REGIME_RANGING, self._default_features()

        recent = df.iloc[-self.lookback_period:]

        # Calculate regime indicators
        returns = recent['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Trend strength using linear regression slope
        prices = recent['Close'].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        normalized_slope = slope / prices.mean() * 100  # Percentage slope

        # ADX for trend strength (if available)
        adx = recent['ADX'].iloc[-1] if 'ADX' in recent.columns else 25

        # RSI extremes
        rsi = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50

        # Bollinger Band width for volatility
        bb_width = recent['BB_width'].iloc[-1] if 'BB_width' in recent.columns else volatility

        # Price position relative to moving averages
        ma20 = recent['MA20'].iloc[-1] if 'MA20' in recent.columns else prices.mean()
        ma50 = recent['MA50'].iloc[-1] if 'MA50' in recent.columns else prices.mean()
        price_vs_ma20 = (prices[-1] - ma20) / ma20 * 100 if ma20 > 0 else 0
        price_vs_ma50 = (prices[-1] - ma50) / ma50 * 100 if ma50 > 0 else 0

        # Composite Index trend
        composite_trend = 0
        if 'Composite_Index' in recent.columns:
            ci = recent['Composite_Index'].values
            composite_trend = (ci[-1] - ci[0]) / max(ci.std(), 0.01)

        # Determine regime
        regime = self._classify_regime(
            normalized_slope, volatility, adx, rsi, bb_width,
            price_vs_ma20, price_vs_ma50, composite_trend
        )

        features = {
            'slope': normalized_slope,
            'volatility': volatility,
            'adx': adx,
            'rsi': rsi,
            'bb_width': bb_width,
            'price_vs_ma20': price_vs_ma20,
            'price_vs_ma50': price_vs_ma50,
            'composite_trend': composite_trend,
            'regime': regime
        }

        return regime, features

    def _classify_regime(self, slope, volatility, adx, rsi, bb_width,
                         price_vs_ma20, price_vs_ma50, composite_trend):
        """Classify market regime based on indicators."""

        # High volatility check first
        if volatility > 0.4 or bb_width > 0.1:  # Very high volatility
            return self.REGIME_HIGH_VOLATILITY

        # Strong trend indicators
        trending_threshold = 25  # ADX threshold for trending

        if adx > trending_threshold:
            if slope > 0.5 and price_vs_ma20 > 0 and composite_trend > 0:
                return self.REGIME_TRENDING_UP
            elif slope < -0.5 and price_vs_ma20 < 0 and composite_trend < 0:
                return self.REGIME_TRENDING_DOWN

        # RSI extremes with trend confirmation
        if rsi > 60 and slope > 0.3:
            return self.REGIME_TRENDING_UP
        elif rsi < 40 and slope < -0.3:
            return self.REGIME_TRENDING_DOWN

        # Default to ranging
        return self.REGIME_RANGING

    def _default_features(self):
        """Return default features when insufficient data."""
        return {
            'slope': 0,
            'volatility': 0.2,
            'adx': 25,
            'rsi': 50,
            'bb_width': 0.05,
            'price_vs_ma20': 0,
            'price_vs_ma50': 0,
            'composite_trend': 0,
            'regime': self.REGIME_RANGING
        }


class MetaLearner:
    """
    Meta-learner that predicts optimal strategy parameters based on market conditions.
    Uses historical performance data to learn which parameters work best in different regimes.
    """

    def __init__(self):
        self.buy_threshold_model = None
        self.sell_threshold_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'slope', 'volatility', 'adx', 'rsi', 'bb_width',
            'price_vs_ma20', 'price_vs_ma50', 'composite_trend'
        ]

        # Default parameters for each regime (prior knowledge)
        self.regime_defaults = {
            MarketRegimeDetector.REGIME_TRENDING_UP: {
                'buy_threshold': 55, 'sell_threshold': 35,
                'exit_buy_offset': 5, 'exit_sell_offset': 10
            },
            MarketRegimeDetector.REGIME_TRENDING_DOWN: {
                'buy_threshold': 70, 'sell_threshold': 45,
                'exit_buy_offset': 10, 'exit_sell_offset': 5
            },
            MarketRegimeDetector.REGIME_RANGING: {
                'buy_threshold': 65, 'sell_threshold': 35,
                'exit_buy_offset': 5, 'exit_sell_offset': 5
            },
            MarketRegimeDetector.REGIME_HIGH_VOLATILITY: {
                'buy_threshold': 70, 'sell_threshold': 30,
                'exit_buy_offset': 10, 'exit_sell_offset': 10
            }
        }

    def train(self, training_data):
        """
        Train the meta-learner on historical regime-performance data.

        Parameters:
        -----------
        training_data : list of dicts
            Each dict contains: features (market conditions) and optimal_params (best parameters found)
        """
        if len(training_data) < 5:
            print("Insufficient training data for meta-learner")
            return False

        try:
            X = []
            y_buy = []
            y_sell = []

            for record in training_data:
                features = [record['features'].get(col, 0) for col in self.feature_columns]
                X.append(features)
                y_buy.append(record['optimal_params']['buy_threshold'])
                y_sell.append(record['optimal_params']['sell_threshold'])

            X = np.array(X)
            y_buy = np.array(y_buy)
            y_sell = np.array(y_sell)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train models
            self.buy_threshold_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            )
            self.sell_threshold_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            )

            self.buy_threshold_model.fit(X_scaled, y_buy)
            self.sell_threshold_model.fit(X_scaled, y_sell)

            self.is_trained = True
            return True

        except Exception as e:
            print(f"Error training meta-learner: {e}")
            return False

    def predict_parameters(self, features, regime):
        """
        Predict optimal parameters for given market conditions.

        Parameters:
        -----------
        features : dict
            Market condition features
        regime : str
            Current market regime

        Returns:
        --------
        dict : Predicted optimal parameters (guaranteed buy > sell)
        """
        # Start with regime defaults
        defaults = self.regime_defaults.get(regime, self.regime_defaults[MarketRegimeDetector.REGIME_RANGING])

        if not self.is_trained:
            # Validate even defaults
            buy, sell = validate_thresholds(defaults['buy_threshold'], defaults['sell_threshold'])
            return {
                'buy_threshold': buy,
                'sell_threshold': sell,
                'exit_buy_offset': defaults['exit_buy_offset'],
                'exit_sell_offset': defaults['exit_sell_offset']
            }

        try:
            X = np.array([[features.get(col, 0) for col in self.feature_columns]])
            X_scaled = self.scaler.transform(X)

            buy_pred = self.buy_threshold_model.predict(X_scaled)[0]
            sell_pred = self.sell_threshold_model.predict(X_scaled)[0]

            # Blend ML prediction with regime defaults (70% ML, 30% prior)
            buy_threshold = 0.7 * buy_pred + 0.3 * defaults['buy_threshold']
            sell_threshold = 0.7 * sell_pred + 0.3 * defaults['sell_threshold']

            # Validate thresholds (ensures buy > sell with minimum gap)
            buy_threshold, sell_threshold = validate_thresholds(buy_threshold, sell_threshold)

            return {
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'exit_buy_offset': defaults['exit_buy_offset'],
                'exit_sell_offset': defaults['exit_sell_offset']
            }

        except Exception as e:
            print(f"Error in meta-learner prediction: {e}")
            buy, sell = validate_thresholds(defaults['buy_threshold'], defaults['sell_threshold'])
            return {
                'buy_threshold': buy,
                'sell_threshold': sell,
                'exit_buy_offset': defaults['exit_buy_offset'],
                'exit_sell_offset': defaults['exit_sell_offset']
            }


class WalkForwardOptimizer:
    """
    Implements walk-forward optimization to avoid overfitting.
    Tests strategies on out-of-sample data using rolling windows.
    """

    def __init__(self, train_window=60, test_window=20, n_splits=5):
        self.train_window = train_window
        self.test_window = test_window
        self.n_splits = n_splits

    def optimize(self, df, calculate_returns_func):
        """
        Perform walk-forward optimization.

        Parameters:
        -----------
        df : pandas.DataFrame
            Historical data with Composite_Index
        calculate_returns_func : callable
            Function to calculate strategy returns for given thresholds

        Returns:
        --------
        dict : Best parameters based on out-of-sample performance
        list : Performance history across windows
        """
        if len(df) < self.train_window + self.test_window:
            return None, []

        results_history = []
        param_performance = {}  # Track cumulative OOS performance per parameter set

        # Define parameter grid
        buy_thresholds = range(50, 76, 5)  # 50-75 in steps of 5
        sell_thresholds = range(25, 51, 5)  # 25-50 in steps of 5

        total_len = len(df)
        step_size = max(1, (total_len - self.train_window - self.test_window) // self.n_splits)

        for split_idx in range(self.n_splits):
            start_idx = split_idx * step_size
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            if test_end > total_len:
                break

            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            # Find best parameters on training data
            best_params = None
            best_train_sharpe = -np.inf

            for buy, sell in product(buy_thresholds, sell_thresholds):
                if buy <= sell + 10:  # Minimum gap
                    continue

                result = calculate_returns_func(train_df, buy, sell)
                if result['sharpe_ratio'] > best_train_sharpe:
                    best_train_sharpe = result['sharpe_ratio']
                    best_params = {'buy_threshold': buy, 'sell_threshold': sell}

            if best_params:
                # Test on out-of-sample data
                oos_result = calculate_returns_func(test_df, best_params['buy_threshold'],
                                                    best_params['sell_threshold'])

                results_history.append({
                    'split': split_idx,
                    'train_sharpe': best_train_sharpe,
                    'test_sharpe': oos_result['sharpe_ratio'],
                    'test_return': oos_result['total_return'],
                    'params': best_params
                })

                # Track parameter performance
                param_key = (best_params['buy_threshold'], best_params['sell_threshold'])
                if param_key not in param_performance:
                    param_performance[param_key] = []
                param_performance[param_key].append(oos_result['sharpe_ratio'])

        # Find parameters with best average OOS performance
        best_avg_sharpe = -np.inf
        best_params = {'buy_threshold': 60, 'sell_threshold': 40}

        for param_key, sharpes in param_performance.items():
            avg_sharpe = np.mean(sharpes)
            if avg_sharpe > best_avg_sharpe:
                best_avg_sharpe = avg_sharpe
                best_params = {'buy_threshold': param_key[0], 'sell_threshold': param_key[1]}

        return best_params, results_history


class StrategyEnsemble:
    """
    Combines multiple strategy configurations with adaptive weighting.
    Uses recent performance to adjust weights dynamically.
    """

    def __init__(self, n_strategies=3, decay_factor=0.9):
        self.n_strategies = n_strategies
        self.decay_factor = decay_factor
        self.strategies = []
        self.weights = []

    def initialize_strategies(self, regime_params, meta_params, walk_forward_params):
        """
        Initialize ensemble with diverse strategies.

        Parameters:
        -----------
        regime_params : dict
            Parameters from regime detection
        meta_params : dict
            Parameters from meta-learner
        walk_forward_params : dict
            Parameters from walk-forward optimization
        """
        self.strategies = []

        # Strategy 1: Regime-based
        self.strategies.append({
            'name': 'regime',
            'params': regime_params,
            'performance_history': []
        })

        # Strategy 2: Meta-learner
        self.strategies.append({
            'name': 'meta',
            'params': meta_params,
            'performance_history': []
        })

        # Strategy 3: Walk-forward optimized
        if walk_forward_params:
            self.strategies.append({
                'name': 'walk_forward',
                'params': walk_forward_params,
                'performance_history': []
            })

        # Initialize equal weights
        n = len(self.strategies)
        self.weights = [1.0 / n] * n

    def get_ensemble_parameters(self):
        """
        Get weighted average of strategy parameters.

        Returns:
        --------
        dict : Ensemble-weighted parameters (guaranteed buy > sell)
        """
        if not self.strategies:
            return {'buy_threshold': 65, 'sell_threshold': 35, 'exit_buy_offset': 5, 'exit_sell_offset': 5}

        # Weighted average of thresholds
        buy_threshold = sum(
            s['params']['buy_threshold'] * w
            for s, w in zip(self.strategies, self.weights)
        )
        sell_threshold = sum(
            s['params']['sell_threshold'] * w
            for s, w in zip(self.strategies, self.weights)
        )

        # Validate thresholds (ensures buy > sell with minimum gap)
        buy_threshold, sell_threshold = validate_thresholds(buy_threshold, sell_threshold)

        # Get exit offsets from best performing strategy
        best_idx = np.argmax(self.weights)
        exit_buy_offset = self.strategies[best_idx]['params'].get('exit_buy_offset', 5)
        exit_sell_offset = self.strategies[best_idx]['params'].get('exit_sell_offset', 5)

        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'exit_buy_offset': exit_buy_offset,
            'exit_sell_offset': exit_sell_offset
        }

    def update_weights(self, performances):
        """
        Update strategy weights based on recent performance.

        Parameters:
        -----------
        performances : list
            Recent Sharpe ratios for each strategy
        """
        if len(performances) != len(self.strategies):
            return

        # Add to performance history
        for i, perf in enumerate(performances):
            self.strategies[i]['performance_history'].append(perf)

        # Calculate exponentially weighted average performance
        weighted_perfs = []
        for strategy in self.strategies:
            hist = strategy['performance_history']
            if not hist:
                weighted_perfs.append(0)
                continue

            # Apply exponential decay to older performance
            weights = [self.decay_factor ** i for i in range(len(hist) - 1, -1, -1)]
            weighted_avg = np.average(hist, weights=weights)
            weighted_perfs.append(weighted_avg)

        # Convert to weights (softmax-like transformation)
        perfs = np.array(weighted_perfs)
        perfs = perfs - perfs.min()  # Shift to positive

        if perfs.sum() > 0:
            self.weights = (perfs / perfs.sum()).tolist()
        else:
            n = len(self.strategies)
            self.weights = [1.0 / n] * n


class StrategyOptimizer:
    """
    Smart Strategy Optimizer with Meta-Learning capabilities.

    Combines multiple optimization approaches:
    1. Market Regime Detection - adapts to market conditions
    2. Meta-Learning - learns optimal parameters from historical performance
    3. Walk-Forward Optimization - prevents overfitting
    4. Ensemble Methods - combines multiple strategies
    """

    def __init__(self, df):
        self.df = df.copy()
        self.results = None
        self.regime_detector = MarketRegimeDetector()
        self.meta_learner = MetaLearner()
        self.walk_forward = WalkForwardOptimizer(train_window=60, test_window=20, n_splits=5)
        self.ensemble = StrategyEnsemble()
        self.optimization_history = []

    def calculate_returns(self, buy_threshold, sell_threshold, exit_buy_threshold=None,
                          exit_sell_threshold=None, df=None):
        """
        Calculate returns for a given set of thresholds.
        """
        if df is None:
            df = self.df

        if 'Composite_Index' not in df.columns or 'Close' not in df.columns:
            return {'total_return': 0, 'max_drawdown': -100, 'sharpe_ratio': 0}

        if exit_buy_threshold is None:
            exit_buy_threshold = sell_threshold + 5
        if exit_sell_threshold is None:
            exit_sell_threshold = buy_threshold - 5

        try:
            df = df.copy()
            df['Position'] = 0
            current_position = 0

            for i in range(1, len(df)):
                idx = df.index[i]
                prev_idx = df.index[i-1]
                composite_value = df.at[idx, 'Composite_Index']
                current_position = df.at[prev_idx, 'Position']

                if current_position == 0:
                    if composite_value >= buy_threshold:
                        current_position = 1
                    elif composite_value <= sell_threshold:
                        current_position = -1
                elif current_position == 1:
                    if composite_value <= exit_buy_threshold:
                        current_position = 0
                elif current_position == -1:
                    if composite_value >= exit_sell_threshold:
                        current_position = 0

                df.at[idx, 'Position'] = current_position

            df['Daily_Return'] = df['Close'].pct_change()
            df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)
            df = df.dropna(subset=['Strategy_Return'])

            if len(df) < 2:
                return {'total_return': 0, 'max_drawdown': -100, 'sharpe_ratio': 0}

            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
            df['Peak'] = df['Cumulative_Return'].cummax()
            df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak']

            total_return = df['Cumulative_Return'].iloc[-1] - 1
            max_drawdown = df['Drawdown'].min()
            sharpe_ratio = np.sqrt(252) * df['Strategy_Return'].mean() / df['Strategy_Return'].std() if df['Strategy_Return'].std() > 0 else 0

            # Calculate trade metrics
            df['Trade_Start'] = df['Position'] != df['Position'].shift(1)
            trade_starts = df[df['Trade_Start']].index

            trade_returns = []
            winning_trades = 0

            for i in range(len(trade_starts) - 1):
                start_idx = trade_starts[i]
                end_idx = trade_starts[i+1]

                if df.at[start_idx, 'Position'] == 0:
                    continue

                trade_return = df.loc[start_idx:end_idx, 'Strategy_Return'].sum()
                trade_returns.append(trade_return)

                if trade_return > 0:
                    winning_trades += 1

            if len(trade_starts) > 0 and df.at[trade_starts[-1], 'Position'] != 0:
                last_trade_return = df.loc[trade_starts[-1]:, 'Strategy_Return'].sum()
                trade_returns.append(last_trade_return)
                if last_trade_return > 0:
                    winning_trades += 1

            num_trades = len(trade_returns)
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
            profit_factor = -sum([r for r in trade_returns if r > 0]) / sum([r for r in trade_returns if r <= 0]) if sum([r for r in trade_returns if r <= 0]) < 0 else 0

            return {
                'total_return': total_return * 100,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': num_trades,
                'win_rate': win_rate * 100,
                'avg_win': avg_win * 100,
                'avg_loss': avg_loss * 100,
                'profit_factor': profit_factor,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'exit_buy_threshold': exit_buy_threshold,
                'exit_sell_threshold': exit_sell_threshold
            }

        except Exception as e:
            print(f"Error in calculate_returns: {str(e)}")
            return {'total_return': 0, 'max_drawdown': -100, 'sharpe_ratio': 0}

    def _calculate_returns_for_df(self, df, buy_threshold, sell_threshold):
        """Helper function for walk-forward optimization."""
        return self.calculate_returns(buy_threshold, sell_threshold, df=df)

    def optimize_thresholds(self, min_period=0):
        """
        Smart optimization using meta-learning approach.

        Parameters:
        -----------
        min_period : int, optional
            Minimum number of recent days to consider (0 = all data)

        Returns:
        --------
        dict : Optimal thresholds and performance metrics
        """
        if 'Composite_Index' not in self.df.columns:
            return None

        # Filter to recent data if specified
        if min_period > 0 and len(self.df) > min_period:
            df = self.df.iloc[-min_period:].copy()
        else:
            df = self.df.copy()

        original_df = self.df
        self.df = df

        try:
            print("Starting smart optimization...")

            # Step 1: Detect market regime
            regime, regime_features = self.regime_detector.detect_regime(df)
            print(f"Detected regime: {regime}")

            # Step 2: Get regime-based parameters
            regime_params = self.meta_learner.regime_defaults.get(
                regime,
                self.meta_learner.regime_defaults[MarketRegimeDetector.REGIME_RANGING]
            )

            # Step 3: Get meta-learner predictions
            meta_params = self.meta_learner.predict_parameters(regime_features, regime)

            # Step 4: Perform walk-forward optimization
            walk_forward_params, wf_history = self.walk_forward.optimize(
                df,
                lambda d, b, s: self.calculate_returns(b, s, df=d)
            )

            # Step 5: Initialize ensemble with all strategies
            self.ensemble.initialize_strategies(regime_params, meta_params, walk_forward_params)

            # Step 6: Evaluate each strategy and update weights
            performances = []
            for strategy in self.ensemble.strategies:
                result = self.calculate_returns(
                    strategy['params']['buy_threshold'],
                    strategy['params']['sell_threshold'],
                    strategy['params']['sell_threshold'] + strategy['params'].get('exit_buy_offset', 5),
                    strategy['params']['buy_threshold'] - strategy['params'].get('exit_sell_offset', 5)
                )
                performances.append(result['sharpe_ratio'])

            self.ensemble.update_weights(performances)

            # Step 7: Get ensemble parameters
            ensemble_params = self.ensemble.get_ensemble_parameters()

            # Step 8: Fine-tune around ensemble parameters using local search
            best_params = self._local_search_optimization(ensemble_params)

            # Step 9: Calculate final metrics
            exit_buy = best_params['sell_threshold'] + best_params.get('exit_buy_offset', 5)
            exit_sell = best_params['buy_threshold'] - best_params.get('exit_sell_offset', 5)

            final_result = self.calculate_returns(
                best_params['buy_threshold'],
                best_params['sell_threshold'],
                exit_buy,
                exit_sell
            )

            # Store optimization history for meta-learning
            self.optimization_history.append({
                'features': regime_features,
                'optimal_params': best_params,
                'performance': final_result
            })

            # Train meta-learner with accumulated data
            if len(self.optimization_history) >= 5:
                self.meta_learner.train(self.optimization_history)

            # Prepare result
            self.results = [final_result]

            # Restore original dataframe
            self.df = original_df

            result = {
                **final_result,
                'regime': regime,
                'ensemble_weights': {
                    s['name']: w for s, w in zip(self.ensemble.strategies, self.ensemble.weights)
                },
                'optimization_method': 'meta_learning_ensemble'
            }

            print(f"Optimization complete: buy={best_params['buy_threshold']}, sell={best_params['sell_threshold']}, sharpe={final_result['sharpe_ratio']:.2f}")

            return result

        except Exception as e:
            print(f"Error in optimize_thresholds: {str(e)}")
            self.df = original_df
            # Fallback to simple grid search
            return self._fallback_optimization()

    def _local_search_optimization(self, initial_params, search_range=5, step=2):
        """
        Fine-tune parameters using local search around initial values.

        Parameters:
        -----------
        initial_params : dict
            Initial parameter values
        search_range : int
            Range to search around initial values
        step : int
            Step size for search

        Returns:
        --------
        dict : Optimized parameters (guaranteed buy > sell)
        """
        # Validate initial params first
        buy_center, sell_center = validate_thresholds(
            initial_params['buy_threshold'],
            initial_params['sell_threshold']
        )

        best_params = {
            'buy_threshold': buy_center,
            'sell_threshold': sell_center,
            'exit_buy_offset': initial_params.get('exit_buy_offset', 5),
            'exit_sell_offset': initial_params.get('exit_sell_offset', 5)
        }
        best_sharpe = -np.inf

        for buy_offset in range(-search_range, search_range + 1, step):
            for sell_offset in range(-search_range, search_range + 1, step):
                buy = buy_center + buy_offset
                sell = sell_center + sell_offset

                # Skip invalid combinations (buy must be > sell + MIN_THRESHOLD_GAP)
                if buy <= sell + MIN_THRESHOLD_GAP or buy > 80 or buy < 50 or sell > 50 or sell < 20:
                    continue

                exit_buy = sell + initial_params.get('exit_buy_offset', 5)
                exit_sell = buy - initial_params.get('exit_sell_offset', 5)

                result = self.calculate_returns(buy, sell, exit_buy, exit_sell)

                if result['sharpe_ratio'] > best_sharpe:
                    best_sharpe = result['sharpe_ratio']
                    best_params = {
                        'buy_threshold': buy,
                        'sell_threshold': sell,
                        'exit_buy_offset': initial_params.get('exit_buy_offset', 5),
                        'exit_sell_offset': initial_params.get('exit_sell_offset', 5)
                    }

        # Final validation of best params
        best_params['buy_threshold'], best_params['sell_threshold'] = validate_thresholds(
            best_params['buy_threshold'],
            best_params['sell_threshold']
        )

        return best_params

    def _fallback_optimization(self):
        """Fallback to simple grid search if smart optimization fails."""
        print("Using fallback grid search optimization...")

        buy_thresholds = range(55, 75, 5)
        sell_thresholds = range(45, 25, -5)
        exit_buy_offsets = [0, 5, 10]
        exit_sell_offsets = [0, 5, 10]

        results = []

        for buy, sell, exit_buy_offset, exit_sell_offset in product(
            buy_thresholds, sell_thresholds, exit_buy_offsets, exit_sell_offsets
        ):
            # Skip invalid combinations (buy must be > sell + MIN_THRESHOLD_GAP)
            if buy <= sell + MIN_THRESHOLD_GAP:
                continue

            exit_buy = sell + exit_buy_offset
            exit_sell = buy - exit_sell_offset

            result = self.calculate_returns(buy, sell, exit_buy, exit_sell)
            results.append(result)

        results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        self.results = results

        if results:
            best = results[0]
            # Validate the best result
            best['buy_threshold'], best['sell_threshold'] = validate_thresholds(
                best['buy_threshold'],
                best['sell_threshold']
            )
            return best

        # Return safe defaults if no valid results
        return {
            'buy_threshold': 65,
            'sell_threshold': 35,
            'sharpe_ratio': 0,
            'total_return': 0
        }

    def optimize_all_strategies(self, min_period=0):
        """
        Optimize and compare all strategy approaches.

        Returns detailed comparison of different optimization methods.
        """
        if 'Composite_Index' not in self.df.columns:
            return None

        if min_period > 0 and len(self.df) > min_period:
            df = self.df.iloc[-min_period:].copy()
        else:
            df = self.df.copy()

        results = {}

        # 1. Regime-based strategy
        regime, features = self.regime_detector.detect_regime(df)
        regime_params = self.meta_learner.regime_defaults.get(regime)
        regime_result = self.calculate_returns(
            regime_params['buy_threshold'],
            regime_params['sell_threshold'],
            df=df
        )
        results['regime_based'] = {
            'params': regime_params,
            'performance': regime_result,
            'regime': regime
        }

        # 2. Meta-learner strategy
        meta_params = self.meta_learner.predict_parameters(features, regime)
        meta_result = self.calculate_returns(
            meta_params['buy_threshold'],
            meta_params['sell_threshold'],
            df=df
        )
        results['meta_learner'] = {
            'params': meta_params,
            'performance': meta_result
        }

        # 3. Walk-forward strategy
        wf_params, wf_history = self.walk_forward.optimize(
            df,
            lambda d, b, s: self.calculate_returns(b, s, df=d)
        )
        if wf_params:
            wf_result = self.calculate_returns(
                wf_params['buy_threshold'],
                wf_params['sell_threshold'],
                df=df
            )
            results['walk_forward'] = {
                'params': wf_params,
                'performance': wf_result,
                'history': wf_history
            }

        # 4. Ensemble strategy
        self.ensemble.initialize_strategies(
            regime_params, meta_params, wf_params
        )
        ensemble_params = self.ensemble.get_ensemble_parameters()
        ensemble_result = self.calculate_returns(
            ensemble_params['buy_threshold'],
            ensemble_params['sell_threshold'],
            df=df
        )
        results['ensemble'] = {
            'params': ensemble_params,
            'performance': ensemble_result,
            'weights': dict(zip(
                [s['name'] for s in self.ensemble.strategies],
                self.ensemble.weights
            ))
        }

        return results

    def apply_optimal_strategy(self, thresholds=None, min_period=0):
        """
        Apply the optimal (or provided) strategy to the dataframe.
        """
        if thresholds is None:
            thresholds = self.optimize_thresholds(min_period)

        if thresholds is None:
            return self.df.copy()

        buy_threshold = thresholds['buy_threshold']
        sell_threshold = thresholds['sell_threshold']
        exit_buy_threshold = thresholds.get('exit_buy_threshold', sell_threshold + 5)
        exit_sell_threshold = thresholds.get('exit_sell_threshold', buy_threshold - 5)

        df = self.df.copy()
        df['Position'] = 0
        current_position = 0

        for i in range(1, len(df)):
            idx = df.index[i]
            prev_idx = df.index[i-1]
            composite_value = df.at[idx, 'Composite_Index']
            current_position = df.at[prev_idx, 'Position']

            if current_position == 0:
                if composite_value >= buy_threshold:
                    current_position = 1
                elif composite_value <= sell_threshold:
                    current_position = -1
            elif current_position == 1:
                if composite_value <= exit_buy_threshold:
                    current_position = 0
            elif current_position == -1:
                if composite_value >= exit_sell_threshold:
                    current_position = 0

            df.at[idx, 'Position'] = current_position

        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)
        df['BuyHold_Value'] = 1000 * (1 + df['Daily_Return']).cumprod().fillna(1)
        df['Strategy_Value'] = 1000 * (1 + df['Strategy_Return']).cumprod().fillna(1)

        df['BuyHold_Peak'] = df['BuyHold_Value'].cummax()
        df['BuyHold_Drawdown'] = (df['BuyHold_Value'] - df['BuyHold_Peak']) / df['BuyHold_Peak']
        df['Strategy_Peak'] = df['Strategy_Value'].cummax()
        df['Strategy_Drawdown'] = (df['Strategy_Value'] - df['Strategy_Peak']) / df['Strategy_Peak']

        df['Buy_Signal'] = (df['Position'].shift(1) == 0) & (df['Position'] == 1)
        df['Sell_Signal'] = (df['Position'].shift(1) == 0) & (df['Position'] == -1)
        df['Exit_Signal'] = ((df['Position'].shift(1) != 0) & (df['Position'] == 0))

        return df
