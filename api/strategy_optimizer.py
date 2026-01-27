import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# Minimum gap between buy and sell thresholds
MIN_THRESHOLD_GAP = 15


class BayesianThresholdOptimizer:
    """
    Bayesian Optimization using Gaussian Process for efficient threshold search.

    Much more efficient than grid search - finds optimal thresholds in ~15-20 evaluations
    instead of 100+ grid search evaluations.

    Uses Expected Improvement (EI) acquisition function to balance exploration vs exploitation.
    """

    def __init__(self, buy_bounds=(50, 80), sell_bounds=(20, 50), n_initial=5, n_iterations=15):
        """
        Initialize Bayesian optimizer.

        Parameters:
        -----------
        buy_bounds : tuple
            (min, max) for buy threshold
        sell_bounds : tuple
            (min, max) for sell threshold
        n_initial : int
            Number of random initial points
        n_iterations : int
            Number of optimization iterations after initial sampling
        """
        self.buy_bounds = buy_bounds
        self.sell_bounds = sell_bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations

        # Gaussian Process with Matern kernel (good for noisy objectives)
        kernel = ConstantKernel(1.0) * Matern(length_scale=[10, 10], nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True
        )

        # History of evaluations
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf

    def _expected_improvement(self, X, xi=0.01):
        """
        Calculate Expected Improvement acquisition function.

        EI balances exploration (high uncertainty) vs exploitation (high predicted value).

        Parameters:
        -----------
        X : array
            Points to evaluate
        xi : float
            Exploration-exploitation trade-off parameter
        """
        X = np.atleast_2d(X)

        mu, sigma = self.gp.predict(X, return_std=True)

        # Handle zero sigma
        sigma = np.maximum(sigma, 1e-8)

        # Calculate improvement
        imp = mu - self.best_score - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    def _acquisition_function(self, x):
        """Negative EI for minimization."""
        # Ensure valid thresholds
        buy, sell = x[0], x[1]

        # Penalize invalid configurations
        if buy <= sell + MIN_THRESHOLD_GAP:
            return 1e6  # Large penalty

        return -self._expected_improvement(x.reshape(1, -1))[0]

    def _generate_initial_points(self):
        """Generate initial random points with Latin Hypercube-like sampling."""
        points = []

        # Add corner points
        points.append([self.buy_bounds[0], self.sell_bounds[0]])  # Low buy, low sell
        points.append([self.buy_bounds[1], self.sell_bounds[1]])  # High buy, high sell
        points.append([65, 35])  # Center point

        # Add random points
        while len(points) < self.n_initial:
            buy = np.random.uniform(*self.buy_bounds)
            sell = np.random.uniform(*self.sell_bounds)

            # Only keep valid configurations
            if buy > sell + MIN_THRESHOLD_GAP:
                points.append([buy, sell])

        return np.array(points)

    def optimize(self, objective_function, verbose=True):
        """
        Run Bayesian optimization to find optimal thresholds.

        Parameters:
        -----------
        objective_function : callable
            Function that takes (buy_threshold, sell_threshold) and returns Sharpe ratio
        verbose : bool
            Print progress

        Returns:
        --------
        dict : Optimal thresholds and performance metrics
        """
        if verbose:
            print("Starting Bayesian Optimization with Gaussian Process...")

        # Phase 1: Initial random sampling
        initial_points = self._generate_initial_points()

        for i, point in enumerate(initial_points):
            buy, sell = validate_thresholds(point[0], point[1])
            score = objective_function(buy, sell)

            self.X_observed.append([buy, sell])
            self.y_observed.append(score)

            if score > self.best_score:
                self.best_score = score
                self.best_params = {'buy_threshold': buy, 'sell_threshold': sell}

            if verbose:
                print(f"  Initial {i+1}/{self.n_initial}: Buy={buy}, Sell={sell}, Sharpe={score:.3f}")

        # Fit GP on initial observations
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        # Phase 2: Bayesian optimization iterations
        for i in range(self.n_iterations):
            # Find next point by maximizing acquisition function
            best_acquisition = -np.inf
            best_next_point = None

            # Multi-start optimization of acquisition function
            for _ in range(20):
                x0 = [
                    np.random.uniform(*self.buy_bounds),
                    np.random.uniform(*self.sell_bounds)
                ]

                # Skip invalid starting points
                if x0[0] <= x0[1] + MIN_THRESHOLD_GAP:
                    continue

                try:
                    result = minimize(
                        self._acquisition_function,
                        x0,
                        bounds=[self.buy_bounds, self.sell_bounds],
                        method='L-BFGS-B'
                    )

                    if -result.fun > best_acquisition:
                        # Validate the point
                        buy, sell = result.x[0], result.x[1]
                        if buy > sell + MIN_THRESHOLD_GAP:
                            best_acquisition = -result.fun
                            best_next_point = result.x
                except:
                    continue

            if best_next_point is None:
                # Fallback to random valid point
                while True:
                    buy = np.random.uniform(*self.buy_bounds)
                    sell = np.random.uniform(*self.sell_bounds)
                    if buy > sell + MIN_THRESHOLD_GAP:
                        best_next_point = [buy, sell]
                        break

            # Evaluate the objective at the new point
            buy, sell = validate_thresholds(best_next_point[0], best_next_point[1])
            score = objective_function(buy, sell)

            # Update observations
            self.X_observed.append([buy, sell])
            self.y_observed.append(score)

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = {'buy_threshold': buy, 'sell_threshold': sell}

            # Refit GP
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)

            if verbose:
                print(f"  Iteration {i+1}/{self.n_iterations}: Buy={buy}, Sell={sell}, Sharpe={score:.3f} (Best: {self.best_score:.3f})")

        if verbose:
            print(f"Bayesian Optimization complete. Best: Buy={self.best_params['buy_threshold']}, Sell={self.best_params['sell_threshold']}, Sharpe={self.best_score:.3f}")

        return {
            **self.best_params,
            'sharpe_ratio': self.best_score,
            'n_evaluations': len(self.y_observed),
            'optimization_method': 'bayesian_gp'
        }

    def get_uncertainty_map(self, resolution=20):
        """
        Get GP uncertainty map for visualization.

        Returns:
        --------
        dict : Grid of buy/sell values with predicted mean and std
        """
        buy_range = np.linspace(*self.buy_bounds, resolution)
        sell_range = np.linspace(*self.sell_bounds, resolution)

        predictions = []
        for buy in buy_range:
            for sell in sell_range:
                if buy > sell + MIN_THRESHOLD_GAP:
                    mu, sigma = self.gp.predict([[buy, sell]], return_std=True)
                    predictions.append({
                        'buy': buy, 'sell': sell,
                        'predicted_sharpe': mu[0],
                        'uncertainty': sigma[0]
                    })

        return predictions


class ContextualBanditOptimizer:
    """
    Contextual Bandit with Thompson Sampling for online threshold adaptation.

    Maps market context (features) to threshold adjustments.
    Uses Bayesian linear regression for fast online updates.
    """

    def __init__(self, n_features=10, n_arms=9, alpha=1.0):
        """
        Initialize contextual bandit.

        Parameters:
        -----------
        n_features : int
            Number of context features
        n_arms : int
            Number of discrete threshold configurations (arms)
        alpha : float
            Exploration parameter for Thompson Sampling
        """
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = alpha

        # Define arms as (buy_offset, sell_offset) from base thresholds
        # Creates a 3x3 grid of adjustments
        self.arms = [
            (-5, -5), (-5, 0), (-5, 5),
            (0, -5),  (0, 0),  (0, 5),
            (5, -5),  (5, 0),  (5, 5)
        ]

        # Bayesian linear regression parameters for each arm
        # Using simplified version: maintain mean and precision for each arm-feature combo
        self.B = [np.eye(n_features) for _ in range(n_arms)]  # Precision matrices
        self.mu = [np.zeros(n_features) for _ in range(n_arms)]  # Mean vectors
        self.f = [np.zeros(n_features) for _ in range(n_arms)]  # Feature sums

        # History for analysis
        self.history = []
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)

    def _extract_context(self, alpha_factors, regime_features):
        """
        Extract context vector from alpha factors and regime features.

        Returns:
        --------
        np.array : Normalized context vector
        """
        context = []

        # Regime features (normalized)
        context.append(regime_features.get('slope', 0) / 10)
        context.append(regime_features.get('volatility', 0.2) / 0.5)
        context.append(regime_features.get('adx', 25) / 50)
        context.append((regime_features.get('rsi', 50) - 50) / 50)

        # Key alpha factors (normalized)
        context.append(alpha_factors.get('momentum_20d', 0) / 20)
        context.append(alpha_factors.get('zscore_ma20', 0) / 3)
        context.append(alpha_factors.get('vol_ratio', 1) - 1)
        context.append(alpha_factors.get('volume_price_trend', 0) / 50)
        context.append((alpha_factors.get('trend_consistency_20d', 50) - 50) / 50)
        context.append(alpha_factors.get('ma_alignment', 1.5) / 3)

        # Pad or truncate to n_features
        context = context[:self.n_features]
        while len(context) < self.n_features:
            context.append(0)

        return np.array(context)

    def select_arm(self, context):
        """
        Select arm using Thompson Sampling.

        Parameters:
        -----------
        context : np.array
            Context vector

        Returns:
        --------
        int : Selected arm index
        """
        samples = []

        for arm in range(self.n_arms):
            # Sample from posterior
            try:
                # Compute posterior variance
                B_inv = np.linalg.inv(self.B[arm])
                theta_sample = np.random.multivariate_normal(self.mu[arm], self.alpha * B_inv)

                # Predicted reward
                reward_sample = np.dot(context, theta_sample)
            except:
                # Fallback to prior
                reward_sample = np.random.normal(0, self.alpha)

            samples.append(reward_sample)

        return np.argmax(samples)

    def update(self, arm, context, reward):
        """
        Update posterior after observing reward.

        Parameters:
        -----------
        arm : int
            Arm that was pulled
        context : np.array
            Context vector
        reward : float
            Observed reward (e.g., Sharpe ratio)
        """
        # Update precision matrix
        self.B[arm] += np.outer(context, context)

        # Update feature sum
        self.f[arm] += reward * context

        # Update mean (solve B * mu = f)
        try:
            self.mu[arm] = np.linalg.solve(self.B[arm], self.f[arm])
        except:
            pass

        # Update statistics
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward

        # Record history
        self.history.append({
            'arm': arm,
            'context': context.tolist(),
            'reward': reward
        })

    def get_threshold_adjustment(self, base_buy, base_sell, alpha_factors, regime_features):
        """
        Get threshold adjustment based on current context.

        Parameters:
        -----------
        base_buy : int
            Base buy threshold from Bayesian optimization
        base_sell : int
            Base sell threshold from Bayesian optimization
        alpha_factors : dict
            Current alpha factor values
        regime_features : dict
            Current regime features

        Returns:
        --------
        tuple : (adjusted_buy, adjusted_sell, arm_index)
        """
        context = self._extract_context(alpha_factors, regime_features)
        arm = self.select_arm(context)

        buy_offset, sell_offset = self.arms[arm]

        adjusted_buy = base_buy + buy_offset
        adjusted_sell = base_sell + sell_offset

        # Validate
        adjusted_buy, adjusted_sell = validate_thresholds(adjusted_buy, adjusted_sell)

        return adjusted_buy, adjusted_sell, arm, context

    def learn_from_trade(self, arm, context, sharpe_improvement):
        """
        Update bandit after a trading period.

        Parameters:
        -----------
        arm : int
            Arm that was used
        context : np.array
            Context when arm was selected
        sharpe_improvement : float
            Improvement in Sharpe ratio vs baseline
        """
        # Normalize reward to [0, 1] range roughly
        reward = np.clip(sharpe_improvement + 0.5, 0, 1)
        self.update(arm, context, reward)

    def get_arm_statistics(self):
        """Get statistics for each arm."""
        stats = []
        for i, (buy_off, sell_off) in enumerate(self.arms):
            avg_reward = self.arm_rewards[i] / max(self.arm_counts[i], 1)
            stats.append({
                'arm': i,
                'buy_offset': buy_off,
                'sell_offset': sell_off,
                'times_selected': int(self.arm_counts[i]),
                'avg_reward': avg_reward
            })
        return stats


class HybridMLOptimizer:
    """
    Hybrid ML Optimizer combining Bayesian Optimization and Contextual Bandits.

    Stage 1: Bayesian Optimization finds globally optimal thresholds
    Stage 2: Contextual Bandit makes real-time adjustments based on market context

    This approach is both efficient (few evaluations) and adaptive (online learning).
    """

    def __init__(self):
        self.bayesian_optimizer = None
        self.contextual_bandit = None
        self.base_thresholds = {'buy_threshold': 65, 'sell_threshold': 35}
        self.is_initialized = False
        self.optimization_history = []

    def initialize(self, df, objective_function, verbose=True):
        """
        Initialize the hybrid optimizer with Bayesian optimization.

        Parameters:
        -----------
        df : pd.DataFrame
            Historical data for optimization
        objective_function : callable
            Function(buy, sell) -> Sharpe ratio
        verbose : bool
            Print progress

        Returns:
        --------
        dict : Optimal base thresholds
        """
        if verbose:
            print("\n" + "="*60)
            print("HYBRID ML OPTIMIZER - Stage 1: Bayesian Optimization")
            print("="*60)

        # Stage 1: Run Bayesian Optimization
        self.bayesian_optimizer = BayesianThresholdOptimizer(
            buy_bounds=(50, 80),
            sell_bounds=(20, 50),
            n_initial=5,
            n_iterations=12
        )

        result = self.bayesian_optimizer.optimize(objective_function, verbose=verbose)

        self.base_thresholds = {
            'buy_threshold': result['buy_threshold'],
            'sell_threshold': result['sell_threshold']
        }

        # Stage 2: Initialize Contextual Bandit
        if verbose:
            print("\n" + "="*60)
            print("HYBRID ML OPTIMIZER - Stage 2: Contextual Bandit Initialized")
            print("="*60)

        self.contextual_bandit = ContextualBanditOptimizer(n_features=10, n_arms=9)

        self.is_initialized = True

        if verbose:
            print(f"Base thresholds: Buy={self.base_thresholds['buy_threshold']}, Sell={self.base_thresholds['sell_threshold']}")
            print("Contextual Bandit ready for online adaptation")

        return {
            **result,
            'optimization_method': 'hybrid_bayesian_bandit'
        }

    def get_adaptive_thresholds(self, alpha_factors, regime_features, verbose=False):
        """
        Get contextually-adapted thresholds using the bandit.

        Parameters:
        -----------
        alpha_factors : dict
            Current alpha factor values
        regime_features : dict
            Current regime features
        verbose : bool
            Print debug info

        Returns:
        --------
        dict : Adapted thresholds with metadata
        """
        if not self.is_initialized:
            return {
                'buy_threshold': 65,
                'sell_threshold': 35,
                'adapted': False
            }

        # Get bandit's adjustment
        buy, sell, arm, context = self.contextual_bandit.get_threshold_adjustment(
            self.base_thresholds['buy_threshold'],
            self.base_thresholds['sell_threshold'],
            alpha_factors,
            regime_features
        )

        buy_offset, sell_offset = self.contextual_bandit.arms[arm]

        if verbose:
            print(f"Bandit selected arm {arm}: Buy offset={buy_offset}, Sell offset={sell_offset}")

        return {
            'buy_threshold': buy,
            'sell_threshold': sell,
            'base_buy': self.base_thresholds['buy_threshold'],
            'base_sell': self.base_thresholds['sell_threshold'],
            'arm_selected': arm,
            'buy_offset': buy_offset,
            'sell_offset': sell_offset,
            'context': context,
            'adapted': True
        }

    def update_from_performance(self, arm, context, actual_sharpe, baseline_sharpe):
        """
        Update the bandit based on observed performance.

        Parameters:
        -----------
        arm : int
            Arm that was used
        context : np.array
            Context when arm was selected
        actual_sharpe : float
            Actual Sharpe ratio achieved
        baseline_sharpe : float
            Baseline Sharpe (e.g., from base thresholds)
        """
        if self.contextual_bandit is not None:
            improvement = actual_sharpe - baseline_sharpe
            self.contextual_bandit.learn_from_trade(arm, context, improvement)

            self.optimization_history.append({
                'arm': arm,
                'actual_sharpe': actual_sharpe,
                'baseline_sharpe': baseline_sharpe,
                'improvement': improvement
            })

    def get_optimization_summary(self):
        """Get summary of optimization performance."""
        summary = {
            'base_thresholds': self.base_thresholds,
            'bayesian_evaluations': len(self.bayesian_optimizer.y_observed) if self.bayesian_optimizer else 0,
            'bayesian_best_sharpe': self.bayesian_optimizer.best_score if self.bayesian_optimizer else 0,
            'bandit_updates': len(self.optimization_history),
            'arm_statistics': self.contextual_bandit.get_arm_statistics() if self.contextual_bandit else []
        }

        if self.optimization_history:
            improvements = [h['improvement'] for h in self.optimization_history]
            summary['avg_improvement'] = np.mean(improvements)
            summary['total_improvement'] = np.sum(improvements)

        return summary


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


class AlphaFactorCalculator:
    """
    Calculates various alpha factors for the meta-learner to learn from.
    These factors capture different market dynamics and predictive signals.
    """

    def __init__(self, lookback_short=5, lookback_medium=20, lookback_long=60):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long

    def calculate_all_alphas(self, df):
        """
        Calculate all alpha factors from the dataframe.

        Returns:
        --------
        dict : Dictionary of alpha factor values
        """
        if len(df) < self.lookback_long:
            return self._default_alphas()

        alphas = {}

        # Momentum Alphas
        alphas.update(self._momentum_alphas(df))

        # Mean Reversion Alphas
        alphas.update(self._mean_reversion_alphas(df))

        # Volatility Alphas
        alphas.update(self._volatility_alphas(df))

        # Volume Alphas
        alphas.update(self._volume_alphas(df))

        # Technical Indicator Alphas
        alphas.update(self._technical_alphas(df))

        # Cross-Sectional / Relative Alphas
        alphas.update(self._relative_alphas(df))

        return alphas

    def _momentum_alphas(self, df):
        """Calculate momentum-based alpha factors."""
        close = df['Close']
        alphas = {}

        # Price momentum at different horizons
        alphas['momentum_5d'] = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
        alphas['momentum_10d'] = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) > 10 else 0
        alphas['momentum_20d'] = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0

        # Momentum acceleration (change in momentum)
        if len(close) > 10:
            mom_recent = close.iloc[-1] / close.iloc[-6] - 1
            mom_prior = close.iloc[-6] / close.iloc[-11] - 1
            alphas['momentum_acceleration'] = (mom_recent - mom_prior) * 100
        else:
            alphas['momentum_acceleration'] = 0

        # Rate of change smoothed
        roc_5 = close.pct_change(5).iloc[-5:].mean() * 100 if len(close) > 5 else 0
        alphas['roc_smoothed'] = roc_5

        # Trend consistency (% of up days in lookback)
        returns = close.pct_change().dropna()
        alphas['trend_consistency_20d'] = (returns.iloc[-20:] > 0).mean() * 100 if len(returns) >= 20 else 50

        return alphas

    def _mean_reversion_alphas(self, df):
        """Calculate mean reversion alpha factors."""
        close = df['Close']
        alphas = {}

        # Distance from moving averages (z-score style)
        if 'MA20' in df.columns:
            ma20 = df['MA20'].iloc[-1]
            std20 = close.iloc[-20:].std() if len(close) >= 20 else close.std()
            alphas['zscore_ma20'] = (close.iloc[-1] - ma20) / std20 if std20 > 0 else 0
        else:
            alphas['zscore_ma20'] = 0

        if 'MA50' in df.columns:
            ma50 = df['MA50'].iloc[-1]
            std50 = close.iloc[-50:].std() if len(close) >= 50 else close.std()
            alphas['zscore_ma50'] = (close.iloc[-1] - ma50) / std50 if std50 > 0 else 0
        else:
            alphas['zscore_ma50'] = 0

        # Bollinger Band position
        if 'BB_pct' in df.columns:
            alphas['bb_position'] = df['BB_pct'].iloc[-1] * 100 if not pd.isna(df['BB_pct'].iloc[-1]) else 50
        else:
            alphas['bb_position'] = 50

        # RSI extremes (deviation from 50)
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            alphas['rsi_extreme'] = abs(rsi - 50) if not pd.isna(rsi) else 0
        else:
            alphas['rsi_extreme'] = 0

        # Price vs 52-week high/low (if enough data)
        if len(close) >= 252:
            high_52w = close.iloc[-252:].max()
            low_52w = close.iloc[-252:].min()
            range_52w = high_52w - low_52w
            alphas['position_52w'] = ((close.iloc[-1] - low_52w) / range_52w * 100) if range_52w > 0 else 50
        else:
            alphas['position_52w'] = 50

        return alphas

    def _volatility_alphas(self, df):
        """Calculate volatility-based alpha factors."""
        close = df['Close']
        returns = close.pct_change().dropna()
        alphas = {}

        # Historical volatility at different horizons
        alphas['vol_5d'] = returns.iloc[-5:].std() * np.sqrt(252) * 100 if len(returns) >= 5 else 20
        alphas['vol_20d'] = returns.iloc[-20:].std() * np.sqrt(252) * 100 if len(returns) >= 20 else 20

        # Volatility ratio (short-term vs long-term)
        if len(returns) >= 20:
            vol_short = returns.iloc[-5:].std()
            vol_long = returns.iloc[-20:].std()
            alphas['vol_ratio'] = (vol_short / vol_long) if vol_long > 0 else 1
        else:
            alphas['vol_ratio'] = 1

        # ATR-based volatility
        if 'High' in df.columns and 'Low' in df.columns:
            tr = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - close.shift(1)),
                abs(df['Low'] - close.shift(1))
            ], axis=1).max(axis=1)
            atr_14 = tr.iloc[-14:].mean() if len(tr) >= 14 else tr.mean()
            alphas['atr_pct'] = (atr_14 / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 2
        else:
            alphas['atr_pct'] = 2

        # Volatility trend (expanding or contracting)
        if 'BB_width' in df.columns and len(df) >= 10:
            bb_width_recent = df['BB_width'].iloc[-5:].mean()
            bb_width_prior = df['BB_width'].iloc[-10:-5].mean()
            alphas['vol_trend'] = ((bb_width_recent / bb_width_prior) - 1) * 100 if bb_width_prior > 0 else 0
        else:
            alphas['vol_trend'] = 0

        return alphas

    def _volume_alphas(self, df):
        """Calculate volume-based alpha factors."""
        alphas = {}

        if 'Volume' not in df.columns:
            return {'volume_ratio': 1, 'obv_trend': 0, 'volume_price_trend': 0, 'mfi_signal': 50}

        volume = df['Volume']
        close = df['Close']

        # Volume ratio (recent vs average)
        vol_avg_20 = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
        vol_recent = volume.iloc[-5:].mean() if len(volume) >= 5 else volume.mean()
        alphas['volume_ratio'] = (vol_recent / vol_avg_20) if vol_avg_20 > 0 else 1

        # OBV trend
        if 'OBV' in df.columns and len(df) >= 10:
            obv = df['OBV']
            obv_recent = obv.iloc[-5:].mean()
            obv_prior = obv.iloc[-10:-5].mean()
            alphas['obv_trend'] = ((obv_recent / obv_prior) - 1) * 100 if obv_prior != 0 else 0
        else:
            alphas['obv_trend'] = 0

        # Volume-price trend (positive volume on up days)
        if len(df) >= 10:
            returns = close.pct_change()
            up_vol = volume[returns > 0].iloc[-10:].sum() if len(volume[returns > 0]) > 0 else 0
            down_vol = volume[returns < 0].iloc[-10:].sum() if len(volume[returns < 0]) > 0 else 0
            total_vol = up_vol + down_vol
            alphas['volume_price_trend'] = ((up_vol - down_vol) / total_vol * 100) if total_vol > 0 else 0
        else:
            alphas['volume_price_trend'] = 0

        # MFI signal
        if 'MFI' in df.columns:
            alphas['mfi_signal'] = df['MFI'].iloc[-1] if not pd.isna(df['MFI'].iloc[-1]) else 50
        else:
            alphas['mfi_signal'] = 50

        return alphas

    def _technical_alphas(self, df):
        """Calculate technical indicator-based alpha factors."""
        alphas = {}

        # MACD histogram momentum
        if 'MACD_hist' in df.columns and len(df) >= 5:
            macd_hist = df['MACD_hist']
            alphas['macd_momentum'] = macd_hist.iloc[-1] - macd_hist.iloc[-5] if not pd.isna(macd_hist.iloc[-1]) else 0
            alphas['macd_signal'] = 1 if macd_hist.iloc[-1] > 0 else -1
        else:
            alphas['macd_momentum'] = 0
            alphas['macd_signal'] = 0

        # ADX trend strength
        if 'ADX' in df.columns:
            adx = df['ADX'].iloc[-1]
            alphas['adx_strength'] = adx if not pd.isna(adx) else 25
            alphas['strong_trend'] = 1 if adx > 25 else 0
        else:
            alphas['adx_strength'] = 25
            alphas['strong_trend'] = 0

        # Stochastic position
        if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
            stoch_k = df['Stoch_K'].iloc[-1]
            stoch_d = df['Stoch_D'].iloc[-1]
            alphas['stoch_position'] = stoch_k if not pd.isna(stoch_k) else 50
            alphas['stoch_crossover'] = 1 if stoch_k > stoch_d else -1
        else:
            alphas['stoch_position'] = 50
            alphas['stoch_crossover'] = 0

        # CCI signal
        if 'CCI' in df.columns:
            cci = df['CCI'].iloc[-1]
            alphas['cci_signal'] = np.clip(cci / 100, -2, 2) if not pd.isna(cci) else 0
        else:
            alphas['cci_signal'] = 0

        return alphas

    def _relative_alphas(self, df):
        """Calculate relative/cross-sectional alpha factors."""
        alphas = {}
        close = df['Close']

        # Price relative to recent range
        if len(close) >= 20:
            high_20 = close.iloc[-20:].max()
            low_20 = close.iloc[-20:].min()
            range_20 = high_20 - low_20
            alphas['range_position_20d'] = ((close.iloc[-1] - low_20) / range_20 * 100) if range_20 > 0 else 50
        else:
            alphas['range_position_20d'] = 50

        # MA alignment score (bullish when short > medium > long)
        if all(col in df.columns for col in ['MA5', 'MA20', 'MA50']):
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]

            if not any(pd.isna([ma5, ma20, ma50])):
                alignment_score = 0
                if ma5 > ma20:
                    alignment_score += 1
                if ma20 > ma50:
                    alignment_score += 1
                if close.iloc[-1] > ma5:
                    alignment_score += 1
                alphas['ma_alignment'] = alignment_score  # 0-3 scale
            else:
                alphas['ma_alignment'] = 1.5
        else:
            alphas['ma_alignment'] = 1.5

        # Composite index position
        if 'Composite_Index' in df.columns:
            ci = df['Composite_Index'].iloc[-1]
            alphas['composite_position'] = ci if not pd.isna(ci) else 50

            # Composite momentum
            if len(df) >= 5:
                ci_mom = df['Composite_Index'].iloc[-1] - df['Composite_Index'].iloc[-5]
                alphas['composite_momentum'] = ci_mom if not pd.isna(ci_mom) else 0
            else:
                alphas['composite_momentum'] = 0
        else:
            alphas['composite_position'] = 50
            alphas['composite_momentum'] = 0

        return alphas

    def _default_alphas(self):
        """Return default alpha values when insufficient data."""
        return {
            'momentum_5d': 0, 'momentum_10d': 0, 'momentum_20d': 0,
            'momentum_acceleration': 0, 'roc_smoothed': 0, 'trend_consistency_20d': 50,
            'zscore_ma20': 0, 'zscore_ma50': 0, 'bb_position': 50,
            'rsi_extreme': 0, 'position_52w': 50,
            'vol_5d': 20, 'vol_20d': 20, 'vol_ratio': 1, 'atr_pct': 2, 'vol_trend': 0,
            'volume_ratio': 1, 'obv_trend': 0, 'volume_price_trend': 0, 'mfi_signal': 50,
            'macd_momentum': 0, 'macd_signal': 0, 'adx_strength': 25, 'strong_trend': 0,
            'stoch_position': 50, 'stoch_crossover': 0, 'cci_signal': 0,
            'range_position_20d': 50, 'ma_alignment': 1.5,
            'composite_position': 50, 'composite_momentum': 0
        }

    def get_alpha_importance(self, alphas):
        """
        Analyze which alpha factors are showing strong signals.

        Returns:
        --------
        dict : Alpha factors with their signal strength and direction
        """
        signals = {}

        # Momentum signals
        if abs(alphas.get('momentum_20d', 0)) > 5:
            signals['momentum'] = {
                'strength': min(abs(alphas['momentum_20d']) / 10, 1),
                'direction': 'bullish' if alphas['momentum_20d'] > 0 else 'bearish',
                'value': alphas['momentum_20d']
            }

        # Mean reversion signals
        zscore = alphas.get('zscore_ma20', 0)
        if abs(zscore) > 1.5:
            signals['mean_reversion'] = {
                'strength': min(abs(zscore) / 3, 1),
                'direction': 'oversold' if zscore < 0 else 'overbought',
                'value': zscore
            }

        # Volatility signals
        vol_ratio = alphas.get('vol_ratio', 1)
        if vol_ratio > 1.5 or vol_ratio < 0.7:
            signals['volatility'] = {
                'strength': min(abs(vol_ratio - 1), 1),
                'direction': 'expanding' if vol_ratio > 1 else 'contracting',
                'value': vol_ratio
            }

        # Volume confirmation
        vol_price = alphas.get('volume_price_trend', 0)
        if abs(vol_price) > 30:
            signals['volume'] = {
                'strength': min(abs(vol_price) / 50, 1),
                'direction': 'accumulation' if vol_price > 0 else 'distribution',
                'value': vol_price
            }

        # Trend strength
        if alphas.get('strong_trend', 0) == 1:
            signals['trend'] = {
                'strength': alphas.get('adx_strength', 25) / 50,
                'direction': 'strong',
                'value': alphas.get('adx_strength', 25)
            }

        return signals


class TraderProfileGenerator:
    """
    Generates 3 different trader profiles with different risk appetites:
    - Aggressive: Lower thresholds, more frequent trading, higher risk tolerance
    - Medium: Balanced approach with moderate thresholds
    - Conservative: Higher thresholds, fewer trades, capital preservation focus
    """

    AGGRESSIVE = 'aggressive'
    MEDIUM = 'medium'
    CONSERVATIVE = 'conservative'

    def __init__(self, base_thresholds=None):
        """
        Initialize with base thresholds from optimization.

        Parameters:
        -----------
        base_thresholds : dict
            Base thresholds from optimization (buy_threshold, sell_threshold)
        """
        self.base_thresholds = base_thresholds or {'buy_threshold': 60, 'sell_threshold': 40}

    def generate_all_profiles(self, df, alpha_factors=None):
        """
        Generate all 3 trader profiles with their behavioral analysis.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price data and indicators
        alpha_factors : dict, optional
            Current alpha factor values

        Returns:
        --------
        dict : All 3 trader profiles with analysis
        """
        profiles = {}

        for profile_type in [self.AGGRESSIVE, self.MEDIUM, self.CONSERVATIVE]:
            thresholds = self._get_profile_thresholds(profile_type)
            df_with_positions = self._apply_strategy(df.copy(), thresholds)
            analysis = self._analyze_profile(df_with_positions, thresholds, profile_type, alpha_factors)
            profiles[profile_type] = {
                'thresholds': thresholds,
                'analysis': analysis,
                'metrics': self._calculate_profile_metrics(df_with_positions, thresholds)
            }

        return profiles

    def _get_profile_thresholds(self, profile_type):
        """Get thresholds for a specific profile type."""
        base_buy = self.base_thresholds.get('buy_threshold', 60)
        base_sell = self.base_thresholds.get('sell_threshold', 40)

        if profile_type == self.AGGRESSIVE:
            # Lower thresholds = more trades, earlier entry
            buy = max(50, base_buy - 10)
            sell = min(50, base_sell + 5)
            return {
                'buy_threshold': buy,
                'sell_threshold': sell,
                'exit_buy_offset': 3,
                'exit_sell_offset': 3,
                'description': 'Early entry, tight stops, high frequency'
            }
        elif profile_type == self.CONSERVATIVE:
            # Higher thresholds = fewer trades, stronger signals needed
            buy = min(80, base_buy + 10)
            sell = max(20, base_sell - 10)
            return {
                'buy_threshold': buy,
                'sell_threshold': sell,
                'exit_buy_offset': 10,
                'exit_sell_offset': 10,
                'description': 'Strong signals only, wide stops, capital preservation'
            }
        else:  # MEDIUM
            return {
                'buy_threshold': base_buy,
                'sell_threshold': base_sell,
                'exit_buy_offset': 5,
                'exit_sell_offset': 5,
                'description': 'Balanced approach, moderate frequency'
            }

    def _apply_strategy(self, df, thresholds):
        """Apply strategy with given thresholds to generate positions."""
        if 'Composite_Index' not in df.columns:
            df['Position'] = 0
            return df

        buy_th = thresholds['buy_threshold']
        sell_th = thresholds['sell_threshold']
        exit_buy = sell_th + thresholds.get('exit_buy_offset', 5)
        exit_sell = buy_th - thresholds.get('exit_sell_offset', 5)

        df['Position'] = 0
        current_position = 0

        for i in range(1, len(df)):
            idx = df.index[i]
            prev_idx = df.index[i-1]
            composite_value = df.at[idx, 'Composite_Index']
            current_position = df.at[prev_idx, 'Position']

            if current_position == 0:
                if composite_value >= buy_th:
                    current_position = 1
                elif composite_value <= sell_th:
                    current_position = -1
            elif current_position == 1:
                if composite_value <= exit_buy:
                    current_position = 0
            elif current_position == -1:
                if composite_value >= exit_sell:
                    current_position = 0

            df.at[idx, 'Position'] = current_position

        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)

        return df

    def _analyze_profile(self, df, thresholds, profile_type, alpha_factors):
        """Generate behavioral analysis for a profile."""
        positions = df['Position'] if 'Position' in df.columns else pd.Series([0])

        # Position breakdown
        long_pct = (positions == 1).mean() * 100
        short_pct = (positions == -1).mean() * 100
        cash_pct = (positions == 0).mean() * 100

        # Profile-specific descriptions
        profile_descriptions = {
            self.AGGRESSIVE: {
                'trading_style': 'High-frequency momentum trader',
                'entry_behavior': 'Enters positions early on emerging signals',
                'exit_behavior': 'Uses tight stops, quick to cut losses',
                'risk_approach': 'Accepts higher volatility for larger returns',
                'best_market': 'Strong trending markets with clear direction',
                'weakness': 'May get whipsawed in choppy/ranging markets',
                'indicator_focus': 'RSI momentum, MACD crossovers, volume spikes'
            },
            self.MEDIUM: {
                'trading_style': 'Balanced swing trader',
                'entry_behavior': 'Waits for confirmed signals before entry',
                'exit_behavior': 'Moderate stops, allows for normal pullbacks',
                'risk_approach': 'Balances risk and reward equally',
                'best_market': 'Most market conditions with some trend',
                'weakness': 'May miss early moves or hold through reversals',
                'indicator_focus': 'Composite index, trend confirmation, MA alignment'
            },
            self.CONSERVATIVE: {
                'trading_style': 'Position trader focused on capital preservation',
                'entry_behavior': 'Only enters on very strong, confirmed signals',
                'exit_behavior': 'Wide stops, holds through volatility',
                'risk_approach': 'Prioritizes capital preservation over gains',
                'best_market': 'Strong trending markets with clear momentum',
                'weakness': 'Misses many opportunities, slow to react',
                'indicator_focus': 'ADX trend strength, multi-MA alignment, volume confirmation'
            }
        }

        description = profile_descriptions.get(profile_type, profile_descriptions[self.MEDIUM])

        # Generate alpha usage description based on profile
        alpha_usage = self._describe_alpha_usage(profile_type, alpha_factors)

        return {
            'profile_type': profile_type,
            'profile_name': profile_type.replace('_', ' ').title(),
            'description': description,
            'position_breakdown': {
                'long_pct': round(long_pct, 1),
                'short_pct': round(short_pct, 1),
                'cash_pct': round(cash_pct, 1)
            },
            'threshold_interpretation': self._interpret_thresholds(thresholds, profile_type),
            'alpha_usage': alpha_usage,
            'risk_rating': {'aggressive': 'High', 'medium': 'Medium', 'conservative': 'Low'}.get(profile_type, 'Medium'),
            'trade_frequency': {'aggressive': 'High', 'medium': 'Medium', 'conservative': 'Low'}.get(profile_type, 'Medium')
        }

    def _describe_alpha_usage(self, profile_type, alpha_factors):
        """Describe how this profile uses alpha factors."""
        if not alpha_factors:
            return {'primary': [], 'secondary': [], 'description': 'No alpha factors available'}

        if profile_type == self.AGGRESSIVE:
            return {
                'primary': ['momentum_5d', 'momentum_acceleration', 'volume_ratio', 'macd_momentum'],
                'secondary': ['rsi_extreme', 'stoch_crossover'],
                'description': 'Focuses on short-term momentum and volume spikes for quick entries. Uses RSI and Stochastic for timing.'
            }
        elif profile_type == self.CONSERVATIVE:
            return {
                'primary': ['momentum_20d', 'adx_strength', 'ma_alignment', 'trend_consistency_20d'],
                'secondary': ['vol_ratio', 'zscore_ma50'],
                'description': 'Relies on longer-term trend indicators and multi-timeframe confirmation. Avoids trades in high volatility.'
            }
        else:  # MEDIUM
            return {
                'primary': ['momentum_20d', 'zscore_ma20', 'composite_position', 'volume_price_trend'],
                'secondary': ['vol_ratio', 'bb_position', 'macd_signal'],
                'description': 'Balanced use of momentum and mean-reversion signals. Confirms entries with volume analysis.'
            }

    def _interpret_thresholds(self, thresholds, profile_type):
        """Interpret what the thresholds mean for trading behavior."""
        buy = thresholds['buy_threshold']
        sell = thresholds['sell_threshold']
        spread = buy - sell

        interpretations = {
            self.AGGRESSIVE: f'Buy at {buy} (easier entry), Sell at {sell}. Narrow spread ({spread}pts) = more active trading.',
            self.MEDIUM: f'Buy at {buy}, Sell at {sell}. Moderate spread ({spread}pts) balances opportunity and risk.',
            self.CONSERVATIVE: f'Buy at {buy} (strong signal needed), Sell at {sell}. Wide spread ({spread}pts) = fewer, higher-quality trades.'
        }

        return interpretations.get(profile_type, f'Buy >= {buy}, Sell <= {sell}, Spread: {spread}pts')

    def _calculate_profile_metrics(self, df, thresholds):
        """Calculate performance metrics for this profile."""
        if 'Strategy_Return' not in df.columns or df['Strategy_Return'].isna().all():
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0
            }

        returns = df['Strategy_Return'].dropna()

        if len(returns) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0
            }

        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # Trade count
        position_changes = df['Position'].diff().fillna(0).abs()
        num_trades = int(position_changes.sum() / 2)

        # Win rate (simplified)
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])
        win_rate = winning_days / total_days * 100 if total_days > 0 else 0

        return {
            'total_return': round(total_return * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'num_trades': num_trades,
            'win_rate': round(win_rate, 1)
        }


class TradingBehaviorAnalyzer:
    """
    Analyzes the trading behavior of the optimized strategy.
    Provides insights into how the trader uses different indicators and data sources.
    """

    def __init__(self):
        self.analysis = {}

    def analyze_strategy(self, df, thresholds, alpha_factors=None):
        """
        Analyze the trading behavior of the optimized strategy.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with strategy positions and indicators
        thresholds : dict
            Optimized thresholds
        alpha_factors : dict, optional
            Current alpha factor values

        Returns:
        --------
        dict : Comprehensive behavioral analysis
        """
        if 'Position' not in df.columns:
            return {'error': 'No position data available'}

        analysis = {
            'strategy_profile': self._analyze_strategy_profile(df, thresholds),
            'entry_patterns': self._analyze_entry_patterns(df),
            'exit_patterns': self._analyze_exit_patterns(df),
            'indicator_usage': self._analyze_indicator_usage(df, thresholds),
            'risk_profile': self._analyze_risk_profile(df),
            'alpha_utilization': self._analyze_alpha_utilization(df, alpha_factors) if alpha_factors else {},
            'trading_summary': self._generate_trading_summary(df, thresholds)
        }

        self.analysis = analysis
        return analysis

    def _analyze_strategy_profile(self, df, thresholds):
        """Determine the overall strategy profile/style."""
        positions = df['Position']
        returns = df['Close'].pct_change()

        # Calculate position statistics
        long_pct = (positions == 1).mean() * 100
        short_pct = (positions == -1).mean() * 100
        cash_pct = (positions == 0).mean() * 100

        # Determine strategy type
        if long_pct > 60:
            style = 'Aggressive Long-Biased'
        elif short_pct > 40:
            style = 'Defensive/Hedged'
        elif cash_pct > 50:
            style = 'Conservative/Selective'
        else:
            style = 'Balanced/Adaptive'

        # Threshold analysis
        threshold_spread = thresholds.get('buy_threshold', 60) - thresholds.get('sell_threshold', 40)
        if threshold_spread > 30:
            selectivity = 'Highly Selective'
        elif threshold_spread > 20:
            selectivity = 'Moderately Selective'
        else:
            selectivity = 'Active Trading'

        return {
            'style': style,
            'selectivity': selectivity,
            'long_exposure': round(long_pct, 1),
            'short_exposure': round(short_pct, 1),
            'cash_exposure': round(cash_pct, 1),
            'threshold_spread': threshold_spread
        }

    def _analyze_entry_patterns(self, df):
        """Analyze patterns around entry points."""
        if 'Buy_Signal' not in df.columns or 'Sell_Signal' not in df.columns:
            return {}

        patterns = {'long_entries': [], 'short_entries': []}

        # Analyze long entries
        buy_signals = df[df['Buy_Signal'] == True]
        if len(buy_signals) > 0:
            for idx in buy_signals.index:
                loc = df.index.get_loc(idx)
                if loc >= 5:
                    pre_entry = df.iloc[loc-5:loc]
                    entry_pattern = {
                        'composite_at_entry': df.at[idx, 'Composite_Index'] if 'Composite_Index' in df.columns else None,
                        'rsi_at_entry': df.at[idx, 'RSI'] if 'RSI' in df.columns else None,
                        'momentum_before': ((df.at[idx, 'Close'] / pre_entry['Close'].iloc[0]) - 1) * 100,
                        'volume_spike': df.at[idx, 'Volume'] / pre_entry['Volume'].mean() if 'Volume' in df.columns else 1
                    }
                    patterns['long_entries'].append(entry_pattern)

        # Summarize long entry patterns
        if patterns['long_entries']:
            patterns['avg_long_entry'] = {
                'avg_composite': np.mean([p['composite_at_entry'] for p in patterns['long_entries'] if p['composite_at_entry']]),
                'avg_rsi': np.mean([p['rsi_at_entry'] for p in patterns['long_entries'] if p['rsi_at_entry']]),
                'avg_momentum': np.mean([p['momentum_before'] for p in patterns['long_entries']]),
                'avg_volume_spike': np.mean([p['volume_spike'] for p in patterns['long_entries']])
            }

        # Analyze short entries similarly
        sell_signals = df[df['Sell_Signal'] == True]
        if len(sell_signals) > 0:
            for idx in sell_signals.index:
                loc = df.index.get_loc(idx)
                if loc >= 5:
                    pre_entry = df.iloc[loc-5:loc]
                    entry_pattern = {
                        'composite_at_entry': df.at[idx, 'Composite_Index'] if 'Composite_Index' in df.columns else None,
                        'rsi_at_entry': df.at[idx, 'RSI'] if 'RSI' in df.columns else None,
                        'momentum_before': ((df.at[idx, 'Close'] / pre_entry['Close'].iloc[0]) - 1) * 100,
                    }
                    patterns['short_entries'].append(entry_pattern)

        if patterns['short_entries']:
            patterns['avg_short_entry'] = {
                'avg_composite': np.mean([p['composite_at_entry'] for p in patterns['short_entries'] if p['composite_at_entry']]),
                'avg_rsi': np.mean([p['rsi_at_entry'] for p in patterns['short_entries'] if p['rsi_at_entry']]),
                'avg_momentum': np.mean([p['momentum_before'] for p in patterns['short_entries']])
            }

        patterns['total_long_entries'] = len(patterns['long_entries'])
        patterns['total_short_entries'] = len(patterns['short_entries'])

        return patterns

    def _analyze_exit_patterns(self, df):
        """Analyze patterns around exit points."""
        if 'Exit_Signal' not in df.columns:
            return {}

        exit_signals = df[df['Exit_Signal'] == True]
        patterns = {'exits': [], 'profit_exits': 0, 'loss_exits': 0}

        if len(exit_signals) > 0 and 'Strategy_Return' in df.columns:
            for idx in exit_signals.index:
                loc = df.index.get_loc(idx)
                if loc >= 1:
                    # Calculate trade return (approximate)
                    prev_position = df['Position'].iloc[loc-1]
                    if prev_position != 0:
                        # Look back to find entry
                        entry_loc = loc - 1
                        while entry_loc > 0 and df['Position'].iloc[entry_loc-1] == prev_position:
                            entry_loc -= 1

                        trade_return = df['Strategy_Return'].iloc[entry_loc:loc].sum()
                        patterns['exits'].append({
                            'return': trade_return * 100,
                            'duration': loc - entry_loc,
                            'composite_at_exit': df.at[idx, 'Composite_Index'] if 'Composite_Index' in df.columns else None
                        })

                        if trade_return > 0:
                            patterns['profit_exits'] += 1
                        else:
                            patterns['loss_exits'] += 1

        if patterns['exits']:
            patterns['avg_trade_return'] = np.mean([e['return'] for e in patterns['exits']])
            patterns['avg_trade_duration'] = np.mean([e['duration'] for e in patterns['exits']])
            patterns['win_rate'] = patterns['profit_exits'] / len(patterns['exits']) * 100

        return patterns

    def _analyze_indicator_usage(self, df, thresholds):
        """Analyze how different indicators contribute to trading decisions."""
        usage = {
            'primary_signals': [],
            'confirmation_signals': [],
            'filter_signals': []
        }

        # Composite Index is primary
        usage['primary_signals'].append({
            'indicator': 'Composite Index',
            'role': 'Primary decision driver',
            'buy_condition': f'>= {thresholds.get("buy_threshold", 60)}',
            'sell_condition': f'<= {thresholds.get("sell_threshold", 40)}',
            'weight': 'High'
        })

        # Analyze correlation of indicators with positions
        if 'Position' in df.columns:
            position = df['Position']

            # RSI correlation
            if 'RSI' in df.columns:
                rsi_corr = position.corr(df['RSI'])
                usage['confirmation_signals'].append({
                    'indicator': 'RSI',
                    'correlation_with_position': round(rsi_corr, 3) if not pd.isna(rsi_corr) else 0,
                    'role': 'Momentum confirmation',
                    'weight': 'Medium'
                })

            # MACD correlation
            if 'MACD_hist' in df.columns:
                macd_corr = position.corr(df['MACD_hist'])
                usage['confirmation_signals'].append({
                    'indicator': 'MACD Histogram',
                    'correlation_with_position': round(macd_corr, 3) if not pd.isna(macd_corr) else 0,
                    'role': 'Trend confirmation',
                    'weight': 'Medium'
                })

            # ADX as filter
            if 'ADX' in df.columns:
                avg_adx_in_trade = df[position != 0]['ADX'].mean() if len(df[position != 0]) > 0 else 0
                usage['filter_signals'].append({
                    'indicator': 'ADX',
                    'avg_during_trades': round(avg_adx_in_trade, 1) if not pd.isna(avg_adx_in_trade) else 25,
                    'role': 'Trend strength filter',
                    'weight': 'Low-Medium'
                })

            # Volume as confirmation
            if 'Volume' in df.columns:
                vol_mean = df['Volume'].mean()
                vol_in_trades = df[position != 0]['Volume'].mean() if len(df[position != 0]) > 0 else vol_mean
                usage['filter_signals'].append({
                    'indicator': 'Volume',
                    'avg_ratio_during_trades': round(vol_in_trades / vol_mean, 2) if vol_mean > 0 else 1,
                    'role': 'Entry confirmation',
                    'weight': 'Low'
                })

        return usage

    def _analyze_risk_profile(self, df):
        """Analyze the risk management characteristics."""
        if 'Strategy_Return' not in df.columns:
            return {}

        returns = df['Strategy_Return'].dropna()

        profile = {
            'avg_daily_return': returns.mean() * 100,
            'return_volatility': returns.std() * 100,
            'max_daily_gain': returns.max() * 100,
            'max_daily_loss': returns.min() * 100,
            'positive_days_pct': (returns > 0).mean() * 100
        }

        # Drawdown analysis
        if 'Strategy_Drawdown' in df.columns:
            dd = df['Strategy_Drawdown']
            profile['max_drawdown'] = dd.min() * 100
            profile['avg_drawdown'] = dd[dd < 0].mean() * 100 if len(dd[dd < 0]) > 0 else 0

        # Risk-adjusted metrics
        if returns.std() > 0:
            profile['sharpe_estimate'] = (returns.mean() / returns.std()) * np.sqrt(252)

        # Tail risk
        profile['var_95'] = np.percentile(returns, 5) * 100  # 95% VaR
        profile['cvar_95'] = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) > 0 else 0

        return profile

    def _analyze_alpha_utilization(self, df, alpha_factors):
        """Analyze how alpha factors relate to strategy performance."""
        if not alpha_factors:
            return {}

        utilization = {
            'active_alphas': [],
            'alpha_summary': {}
        }

        # Identify which alphas are showing strong signals
        for alpha_name, value in alpha_factors.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Determine if alpha is active/significant
                if 'momentum' in alpha_name and abs(value) > 3:
                    utilization['active_alphas'].append({
                        'name': alpha_name,
                        'value': round(value, 2),
                        'signal': 'bullish' if value > 0 else 'bearish'
                    })
                elif 'zscore' in alpha_name and abs(value) > 1:
                    utilization['active_alphas'].append({
                        'name': alpha_name,
                        'value': round(value, 2),
                        'signal': 'oversold' if value < 0 else 'overbought'
                    })
                elif 'vol_ratio' in alpha_name and (value > 1.3 or value < 0.7):
                    utilization['active_alphas'].append({
                        'name': alpha_name,
                        'value': round(value, 2),
                        'signal': 'high volatility' if value > 1 else 'low volatility'
                    })

        # Categorize alphas
        momentum_alphas = {k: v for k, v in alpha_factors.items() if 'momentum' in k.lower() or 'trend' in k.lower()}
        reversion_alphas = {k: v for k, v in alpha_factors.items() if 'zscore' in k.lower() or 'bb_' in k.lower() or 'rsi' in k.lower()}
        vol_alphas = {k: v for k, v in alpha_factors.items() if 'vol' in k.lower() or 'atr' in k.lower()}

        utilization['alpha_summary'] = {
            'momentum_signal': 'bullish' if np.mean([v for v in momentum_alphas.values() if isinstance(v, (int, float))]) > 0 else 'bearish',
            'reversion_signal': 'oversold' if np.mean([v for v in reversion_alphas.values() if isinstance(v, (int, float)) and 'zscore' in str(v)]) < 0 else 'overbought',
            'volatility_regime': 'high' if alpha_factors.get('vol_ratio', 1) > 1.2 else 'normal'
        }

        return utilization

    def _generate_trading_summary(self, df, thresholds):
        """Generate a human-readable trading summary."""
        summary = []

        # Strategy style
        profile = self._analyze_strategy_profile(df, thresholds)
        summary.append(f"Strategy Style: {profile['style']}")
        summary.append(f"Selectivity: {profile['selectivity']}")

        # Position breakdown
        summary.append(f"Position Mix: {profile['long_exposure']:.0f}% Long, {profile['short_exposure']:.0f}% Short, {profile['cash_exposure']:.0f}% Cash")

        # Threshold interpretation
        buy_th = thresholds.get('buy_threshold', 60)
        sell_th = thresholds.get('sell_threshold', 40)

        if buy_th >= 70:
            summary.append("Entry Stance: Very conservative, waits for strong bullish signals")
        elif buy_th >= 60:
            summary.append("Entry Stance: Moderately selective, prefers confirmed uptrends")
        else:
            summary.append("Entry Stance: Aggressive, enters on early signals")

        if sell_th <= 30:
            summary.append("Exit Stance: Tolerant of pullbacks, holds through volatility")
        elif sell_th <= 40:
            summary.append("Exit Stance: Balanced risk management")
        else:
            summary.append("Exit Stance: Quick to exit, prioritizes capital preservation")

        return summary

    def get_behavior_report(self):
        """Generate a formatted behavior report."""
        if not self.analysis:
            return "No analysis available. Run analyze_strategy() first."

        report = []
        report.append("=" * 50)
        report.append("TRADING BEHAVIOR ANALYSIS REPORT")
        report.append("=" * 50)

        # Strategy Profile
        if 'strategy_profile' in self.analysis:
            sp = self.analysis['strategy_profile']
            report.append(f"\n[Strategy Profile]")
            report.append(f"  Style: {sp.get('style', 'N/A')}")
            report.append(f"  Selectivity: {sp.get('selectivity', 'N/A')}")
            report.append(f"  Exposure: {sp.get('long_exposure', 0):.1f}% Long / {sp.get('short_exposure', 0):.1f}% Short / {sp.get('cash_exposure', 0):.1f}% Cash")

        # Entry Patterns
        if 'entry_patterns' in self.analysis:
            ep = self.analysis['entry_patterns']
            report.append(f"\n[Entry Patterns]")
            report.append(f"  Total Entries: {ep.get('total_long_entries', 0)} Long, {ep.get('total_short_entries', 0)} Short")
            if 'avg_long_entry' in ep:
                ale = ep['avg_long_entry']
                report.append(f"  Avg Long Entry: Composite={ale.get('avg_composite', 0):.1f}, RSI={ale.get('avg_rsi', 0):.1f}")

        # Risk Profile
        if 'risk_profile' in self.analysis:
            rp = self.analysis['risk_profile']
            report.append(f"\n[Risk Profile]")
            report.append(f"  Avg Daily Return: {rp.get('avg_daily_return', 0):.3f}%")
            report.append(f"  Max Drawdown: {rp.get('max_drawdown', 0):.2f}%")
            report.append(f"  Win Rate: {rp.get('positive_days_pct', 0):.1f}%")
            if 'sharpe_estimate' in rp:
                report.append(f"  Sharpe Ratio (est): {rp['sharpe_estimate']:.2f}")

        # Indicator Usage
        if 'indicator_usage' in self.analysis:
            iu = self.analysis['indicator_usage']
            report.append(f"\n[Indicator Usage]")
            for sig in iu.get('primary_signals', []):
                report.append(f"  PRIMARY: {sig['indicator']} - {sig['role']}")
            for sig in iu.get('confirmation_signals', []):
                report.append(f"  CONFIRM: {sig['indicator']} (corr={sig.get('correlation_with_position', 0):.2f})")

        # Trading Summary
        if 'trading_summary' in self.analysis:
            report.append(f"\n[Summary]")
            for line in self.analysis['trading_summary']:
                report.append(f"  {line}")

        report.append("\n" + "=" * 50)

        return "\n".join(report)


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
    Uses historical performance data and alpha factors to learn which parameters work best.
    """

    def __init__(self):
        self.buy_threshold_model = None
        self.sell_threshold_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Core regime features
        self.regime_feature_columns = [
            'slope', 'volatility', 'adx', 'rsi', 'bb_width',
            'price_vs_ma20', 'price_vs_ma50', 'composite_trend'
        ]

        # Alpha factor features for enhanced learning
        self.alpha_feature_columns = [
            'momentum_20d', 'momentum_acceleration', 'trend_consistency_20d',
            'zscore_ma20', 'bb_position', 'rsi_extreme',
            'vol_ratio', 'atr_pct', 'vol_trend',
            'volume_ratio', 'volume_price_trend', 'mfi_signal',
            'macd_momentum', 'adx_strength', 'stoch_position',
            'range_position_20d', 'ma_alignment', 'composite_position'
        ]

        # Combined feature columns for ML
        self.feature_columns = self.regime_feature_columns + self.alpha_feature_columns

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


class AdaptiveStrategyOptimizer:
    """
    Adaptive Strategy Optimizer using HYBRID ML approach:

    1. Bayesian Optimization (Gaussian Process) - Finds optimal base thresholds efficiently
    2. Contextual Bandit (Thompson Sampling) - Makes real-time adjustments based on market context

    Re-optimizes every N days with ML-driven threshold selection.
    """

    def __init__(self, df, reoptimize_days=90):
        """
        Initialize the adaptive optimizer.

        Parameters:
        -----------
        df : pandas.DataFrame
            Full historical data with Composite_Index
        reoptimize_days : int
            Number of days between re-optimization (default: 90)
        """
        self.df = df.copy()
        self.reoptimize_days = reoptimize_days
        self.optimization_periods = []
        self.trade_signals = []
        self.period_thresholds = []
        self.hybrid_optimizer = None
        self.alpha_calculator = AlphaFactorCalculator()
        self.regime_detector = MarketRegimeDetector()

    def _create_objective_function(self, optimization_df):
        """
        Create objective function for Bayesian optimization.

        Returns:
        --------
        callable : Function(buy, sell) -> Sharpe ratio
        """
        def objective(buy_threshold, sell_threshold):
            if 'Composite_Index' not in optimization_df.columns:
                return 0

            df = optimization_df.copy()
            df['Position'] = 0
            current_position = 0

            exit_buy = sell_threshold + 5
            exit_sell = buy_threshold - 5

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
                    if composite_value <= exit_buy:
                        current_position = 0
                elif current_position == -1:
                    if composite_value >= exit_sell:
                        current_position = 0

                df.at[idx, 'Position'] = current_position

            df['Daily_Return'] = df['Close'].pct_change()
            df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)

            returns = df['Strategy_Return'].dropna()
            if len(returns) < 10 or returns.std() == 0:
                return 0

            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            return sharpe

        return objective

    def optimize_adaptive(self, lookback_days=60):
        """
        Perform adaptive optimization using HYBRID ML (Bayesian + Bandit).

        Parameters:
        -----------
        lookback_days : int
            Days of historical data to use for each optimization

        Returns:
        --------
        dict : Results including trade signals, period thresholds, and performance
        """
        if 'Composite_Index' not in self.df.columns:
            return None

        df = self.df.copy()
        n_rows = len(df)

        # Initialize columns
        df['Position'] = 0
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df['Exit_Signal'] = False
        df['Current_Buy_Threshold'] = 60
        df['Current_Sell_Threshold'] = 40

        self.trade_signals = []
        self.period_thresholds = []

        current_position = 0
        current_buy_threshold = 60
        current_sell_threshold = 40
        last_optimization_idx = 0

        # Track bandit context for learning
        current_arm = None
        current_context = None
        period_start_value = 1000

        print("\n" + "="*70)
        print("HYBRID ML ADAPTIVE OPTIMIZER")
        print("Bayesian Optimization + Contextual Bandit")
        print(f"Re-optimize every {self.reoptimize_days} days")
        print("="*70)

        for i in range(1, n_rows):
            idx = df.index[i]

            # Check if we need to re-optimize (every reoptimize_days)
            if i - last_optimization_idx >= self.reoptimize_days or i == 1:
                # Get lookback data for optimization
                start_idx = max(0, i - lookback_days)
                optimization_df = df.iloc[start_idx:i].copy()

                if len(optimization_df) >= 30:
                    print(f"\n--- Period {len(self.period_thresholds) + 1} Optimization ---")

                    # Update bandit from previous period's performance
                    if self.hybrid_optimizer is not None and current_arm is not None:
                        current_value = df.at[df.index[i-1], 'Strategy_Value'] if 'Strategy_Value' in df.columns else 1000
                        period_return = (current_value / period_start_value - 1) * 100
                        period_sharpe = period_return / 10  # Simplified sharpe proxy
                        self.hybrid_optimizer.update_from_performance(
                            current_arm, current_context, period_sharpe, 0
                        )
                        print(f"Bandit learning: Arm {current_arm} achieved {period_return:.2f}% return")

                    # Create objective function
                    objective = self._create_objective_function(optimization_df)

                    # Initialize or update hybrid optimizer
                    if self.hybrid_optimizer is None:
                        self.hybrid_optimizer = HybridMLOptimizer()
                        result = self.hybrid_optimizer.initialize(optimization_df, objective, verbose=True)
                    else:
                        # Run fresh Bayesian optimization for new period
                        bayesian_opt = BayesianThresholdOptimizer(
                            n_initial=3,
                            n_iterations=8
                        )
                        result = bayesian_opt.optimize(objective, verbose=True)
                        self.hybrid_optimizer.base_thresholds = {
                            'buy_threshold': result['buy_threshold'],
                            'sell_threshold': result['sell_threshold']
                        }

                    # Get regime and alpha factors for contextual adaptation
                    regime, regime_features = self.regime_detector.detect_regime(optimization_df)
                    alpha_factors = self.alpha_calculator.calculate_all_alphas(optimization_df)

                    # Get contextual bandit adjustment
                    if self.hybrid_optimizer.contextual_bandit is not None:
                        adapted = self.hybrid_optimizer.get_adaptive_thresholds(
                            alpha_factors, regime_features, verbose=True
                        )
                        current_buy_threshold = adapted['buy_threshold']
                        current_sell_threshold = adapted['sell_threshold']
                        current_arm = adapted['arm_selected']
                        current_context = adapted['context']

                        print(f"Bandit adjustment: Arm {current_arm} (Buy: {adapted['base_buy']}+{adapted['buy_offset']}, Sell: {adapted['base_sell']}+{adapted['sell_offset']})")
                    else:
                        current_buy_threshold = result['buy_threshold']
                        current_sell_threshold = result['sell_threshold']

                    # Record period info
                    period_info = {
                        'start_date': str(df.index[start_idx]) if hasattr(df.index[start_idx], 'strftime') else str(df.index[start_idx]),
                        'end_date': str(idx) if hasattr(idx, 'strftime') else str(idx),
                        'buy_threshold': current_buy_threshold,
                        'sell_threshold': current_sell_threshold,
                        'regime': regime,
                        'sharpe': result.get('sharpe_ratio', 0),
                        'optimization_method': result.get('optimization_method', 'bayesian_gp'),
                        'bayesian_evaluations': result.get('n_evaluations', 0)
                    }
                    self.period_thresholds.append(period_info)

                    print(f"\nPeriod {len(self.period_thresholds)}: Buy={current_buy_threshold}, Sell={current_sell_threshold}")
                    print(f"Regime: {regime}, Bayesian Sharpe: {result.get('sharpe_ratio', 0):.3f}")
                    print(f"Evaluations: {result.get('n_evaluations', 0)} (vs 144 for grid search)")

                    # Track period start for bandit learning
                    period_start_value = df.at[idx, 'Strategy_Value'] if 'Strategy_Value' in df.columns else 1000

                last_optimization_idx = i

            # Store current thresholds
            df.at[idx, 'Current_Buy_Threshold'] = current_buy_threshold
            df.at[idx, 'Current_Sell_Threshold'] = current_sell_threshold

            # Get composite value
            composite_value = df.at[idx, 'Composite_Index']

            # Trading logic
            exit_buy = current_sell_threshold + 5
            exit_sell = current_buy_threshold - 5

            if current_position == 0:
                if composite_value >= current_buy_threshold:
                    current_position = 1
                    df.at[idx, 'Buy_Signal'] = True
                    self.trade_signals.append({
                        'date': str(idx) if hasattr(idx, 'strftime') else str(idx),
                        'type': 'buy',
                        'price': df.at[idx, 'Close'],
                        'composite': composite_value,
                        'threshold': current_buy_threshold
                    })
                elif composite_value <= current_sell_threshold:
                    current_position = -1
                    df.at[idx, 'Sell_Signal'] = True
                    self.trade_signals.append({
                        'date': str(idx) if hasattr(idx, 'strftime') else str(idx),
                        'type': 'sell',
                        'price': df.at[idx, 'Close'],
                        'composite': composite_value,
                        'threshold': current_sell_threshold
                    })
            elif current_position == 1:
                if composite_value <= exit_buy:
                    current_position = 0
                    df.at[idx, 'Exit_Signal'] = True
                    self.trade_signals.append({
                        'date': str(idx) if hasattr(idx, 'strftime') else str(idx),
                        'type': 'exit_long',
                        'price': df.at[idx, 'Close'],
                        'composite': composite_value,
                        'threshold': exit_buy
                    })
            elif current_position == -1:
                if composite_value >= exit_sell:
                    current_position = 0
                    df.at[idx, 'Exit_Signal'] = True
                    self.trade_signals.append({
                        'date': str(idx) if hasattr(idx, 'strftime') else str(idx),
                        'type': 'exit_short',
                        'price': df.at[idx, 'Close'],
                        'composite': composite_value,
                        'threshold': exit_sell
                    })

            df.at[idx, 'Position'] = current_position

        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'] * df['Daily_Return'].shift(-1)
        df['Strategy_Value'] = 1000 * (1 + df['Strategy_Return'].fillna(0)).cumprod()
        df['BuyHold_Value'] = 1000 * (1 + df['Daily_Return'].fillna(0)).cumprod()

        # Drawdown calculations
        df['Strategy_Peak'] = df['Strategy_Value'].cummax()
        df['Strategy_Drawdown'] = (df['Strategy_Value'] - df['Strategy_Peak']) / df['Strategy_Peak']
        df['BuyHold_Peak'] = df['BuyHold_Value'].cummax()
        df['BuyHold_Drawdown'] = (df['BuyHold_Value'] - df['BuyHold_Peak']) / df['BuyHold_Peak']

        # Calculate performance metrics
        returns = df['Strategy_Return'].dropna()
        total_return = (df['Strategy_Value'].iloc[-1] / 1000 - 1) * 100 if len(df) > 0 else 0
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        max_dd = df['Strategy_Drawdown'].min() * 100

        # Count trades
        num_trades = len([s for s in self.trade_signals if s['type'] in ['buy', 'sell']])

        self.df = df

        # Print summary
        print("\n" + "="*70)
        print("HYBRID ML OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Periods: {len(self.period_thresholds)}")
        print(f"Trades: {num_trades}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Max Drawdown: {max_dd:.2f}%")

        if self.hybrid_optimizer:
            summary = self.hybrid_optimizer.get_optimization_summary()
            print(f"\nBayesian evaluations (total): ~{summary['bayesian_evaluations'] * len(self.period_thresholds)}")
            print(f"Grid search equivalent: {144 * len(self.period_thresholds)} evaluations")
            print(f"Efficiency gain: {144 / max(summary['bayesian_evaluations'], 1):.1f}x faster")

            if summary['arm_statistics']:
                print("\nContextual Bandit Arm Usage:")
                for arm_stat in summary['arm_statistics']:
                    if arm_stat['times_selected'] > 0:
                        print(f"  Arm {arm_stat['arm']}: Buy{arm_stat['buy_offset']:+d}, Sell{arm_stat['sell_offset']:+d} - Used {arm_stat['times_selected']}x, Avg reward: {arm_stat['avg_reward']:.3f}")

        return {
            'df': df,
            'trade_signals': self.trade_signals,
            'period_thresholds': self.period_thresholds,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'num_periods': len(self.period_thresholds),
            'optimization_method': 'hybrid_bayesian_bandit'
        }

    def get_trade_signals(self):
        """Get list of all trade signals for charting."""
        return self.trade_signals

    def get_period_thresholds(self):
        """Get thresholds for each optimization period."""
        return self.period_thresholds

    def get_optimization_summary(self):
        """Get summary of hybrid ML optimization."""
        if self.hybrid_optimizer:
            return self.hybrid_optimizer.get_optimization_summary()
        return {}


class StrategyOptimizer:
    """
    Smart Strategy Optimizer with Meta-Learning capabilities.

    Combines multiple optimization approaches:
    1. Market Regime Detection - adapts to market conditions
    2. Meta-Learning with Alpha Factors - learns optimal parameters from market features
    3. Walk-Forward Optimization - prevents overfitting
    4. Ensemble Methods - combines multiple strategies
    5. Behavioral Analysis - explains how the trader uses different indicators
    """

    def __init__(self, df):
        self.df = df.copy()
        self.results = None
        self.regime_detector = MarketRegimeDetector()
        self.meta_learner = MetaLearner()
        self.alpha_calculator = AlphaFactorCalculator()
        self.behavior_analyzer = TradingBehaviorAnalyzer()
        self.profile_generator = None  # Initialized after optimization
        self.walk_forward = WalkForwardOptimizer(train_window=60, test_window=20, n_splits=5)
        self.ensemble = StrategyEnsemble()
        self.optimization_history = []
        self.current_alphas = {}
        self.behavior_analysis = {}
        self.trader_profiles = {}

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
        Smart optimization using meta-learning approach with alpha factors.

        Parameters:
        -----------
        min_period : int, optional
            Minimum number of recent days to consider (0 = all data)

        Returns:
        --------
        dict : Optimal thresholds, performance metrics, alpha factors, and behavior analysis
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
            print("Starting smart optimization with alpha factors...")

            # Step 1: Detect market regime
            regime, regime_features = self.regime_detector.detect_regime(df)
            print(f"Detected regime: {regime}")

            # Step 2: Calculate alpha factors
            self.current_alphas = self.alpha_calculator.calculate_all_alphas(df)
            alpha_signals = self.alpha_calculator.get_alpha_importance(self.current_alphas)
            print(f"Active alpha signals: {len(alpha_signals)}")

            # Step 3: Merge regime features with alpha factors for meta-learner
            combined_features = {**regime_features, **self.current_alphas}

            # Step 4: Get regime-based parameters
            regime_params = self.meta_learner.regime_defaults.get(
                regime,
                self.meta_learner.regime_defaults[MarketRegimeDetector.REGIME_RANGING]
            )

            # Step 5: Get meta-learner predictions using combined features
            meta_params = self.meta_learner.predict_parameters(combined_features, regime)

            # Step 6: Perform walk-forward optimization
            walk_forward_params, wf_history = self.walk_forward.optimize(
                df,
                lambda d, b, s: self.calculate_returns(b, s, df=d)
            )

            # Step 7: Initialize ensemble with all strategies
            self.ensemble.initialize_strategies(regime_params, meta_params, walk_forward_params)

            # Step 8: Evaluate each strategy and update weights
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

            # Step 9: Get ensemble parameters
            ensemble_params = self.ensemble.get_ensemble_parameters()

            # Step 10: Fine-tune around ensemble parameters using local search
            best_params = self._local_search_optimization(ensemble_params)

            # Step 11: Calculate final metrics
            exit_buy = best_params['sell_threshold'] + best_params.get('exit_buy_offset', 5)
            exit_sell = best_params['buy_threshold'] - best_params.get('exit_sell_offset', 5)

            final_result = self.calculate_returns(
                best_params['buy_threshold'],
                best_params['sell_threshold'],
                exit_buy,
                exit_sell
            )

            # Store optimization history for meta-learning (with alpha factors)
            self.optimization_history.append({
                'features': combined_features,
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
                'optimization_method': 'meta_learning_ensemble',
                'alpha_signals': alpha_signals,
                'key_alphas': {
                    'momentum_20d': self.current_alphas.get('momentum_20d', 0),
                    'vol_ratio': self.current_alphas.get('vol_ratio', 1),
                    'zscore_ma20': self.current_alphas.get('zscore_ma20', 0),
                    'trend_consistency': self.current_alphas.get('trend_consistency_20d', 50),
                    'volume_price_trend': self.current_alphas.get('volume_price_trend', 0),
                    'ma_alignment': self.current_alphas.get('ma_alignment', 1.5)
                }
            }

            print(f"Optimization complete: buy={best_params['buy_threshold']}, sell={best_params['sell_threshold']}, sharpe={final_result['sharpe_ratio']:.2f}")

            # Generate 3 trader profiles
            self.profile_generator = TraderProfileGenerator(best_params)
            self.trader_profiles = self.profile_generator.generate_all_profiles(df, self.current_alphas)
            print(f"Generated 3 trader profiles: aggressive, medium, conservative")

            return result

        except Exception as e:
            print(f"Error in optimize_thresholds: {str(e)}")
            import traceback
            traceback.print_exc()
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

        # Analyze trading behavior
        self.behavior_analysis = self.behavior_analyzer.analyze_strategy(
            df, thresholds, self.current_alphas
        )

        return df

    def get_behavior_analysis(self):
        """
        Get the behavioral pattern analysis of the optimized trader.

        Returns:
        --------
        dict : Comprehensive analysis of trading behavior
        """
        return self.behavior_analysis

    def get_behavior_report(self):
        """
        Get a formatted human-readable behavior report.

        Returns:
        --------
        str : Formatted report string
        """
        return self.behavior_analyzer.get_behavior_report()

    def get_alpha_summary(self):
        """
        Get a summary of current alpha factor values and their signals.

        Returns:
        --------
        dict : Alpha factors summary with interpretations
        """
        if not self.current_alphas:
            return {'message': 'No alpha factors calculated yet. Run optimize_thresholds() first.'}

        summary = {
            'momentum': {
                'short_term': self.current_alphas.get('momentum_5d', 0),
                'medium_term': self.current_alphas.get('momentum_20d', 0),
                'acceleration': self.current_alphas.get('momentum_acceleration', 0),
                'interpretation': self._interpret_momentum()
            },
            'mean_reversion': {
                'zscore_ma20': self.current_alphas.get('zscore_ma20', 0),
                'bb_position': self.current_alphas.get('bb_position', 50),
                'rsi_extreme': self.current_alphas.get('rsi_extreme', 0),
                'interpretation': self._interpret_mean_reversion()
            },
            'volatility': {
                'vol_5d': self.current_alphas.get('vol_5d', 20),
                'vol_ratio': self.current_alphas.get('vol_ratio', 1),
                'atr_pct': self.current_alphas.get('atr_pct', 2),
                'interpretation': self._interpret_volatility()
            },
            'volume': {
                'volume_ratio': self.current_alphas.get('volume_ratio', 1),
                'volume_price_trend': self.current_alphas.get('volume_price_trend', 0),
                'mfi_signal': self.current_alphas.get('mfi_signal', 50),
                'interpretation': self._interpret_volume()
            },
            'trend': {
                'adx_strength': self.current_alphas.get('adx_strength', 25),
                'ma_alignment': self.current_alphas.get('ma_alignment', 1.5),
                'trend_consistency': self.current_alphas.get('trend_consistency_20d', 50),
                'interpretation': self._interpret_trend()
            }
        }

        return summary

    def _interpret_momentum(self):
        """Interpret momentum alpha factors."""
        mom_20d = self.current_alphas.get('momentum_20d', 0)
        accel = self.current_alphas.get('momentum_acceleration', 0)

        if mom_20d > 5 and accel > 0:
            return "Strong bullish momentum with acceleration"
        elif mom_20d > 5:
            return "Positive momentum, but slowing"
        elif mom_20d < -5 and accel < 0:
            return "Strong bearish momentum with acceleration"
        elif mom_20d < -5:
            return "Negative momentum, but stabilizing"
        else:
            return "Neutral momentum"

    def _interpret_mean_reversion(self):
        """Interpret mean reversion alpha factors."""
        zscore = self.current_alphas.get('zscore_ma20', 0)
        bb_pos = self.current_alphas.get('bb_position', 50)

        if zscore < -2 or bb_pos < 10:
            return "Extremely oversold - potential bounce"
        elif zscore < -1 or bb_pos < 25:
            return "Oversold conditions"
        elif zscore > 2 or bb_pos > 90:
            return "Extremely overbought - potential pullback"
        elif zscore > 1 or bb_pos > 75:
            return "Overbought conditions"
        else:
            return "Neutral - near fair value"

    def _interpret_volatility(self):
        """Interpret volatility alpha factors."""
        vol_ratio = self.current_alphas.get('vol_ratio', 1)
        vol_5d = self.current_alphas.get('vol_5d', 20)

        if vol_ratio > 1.5:
            return "Volatility expanding significantly - exercise caution"
        elif vol_ratio > 1.2:
            return "Volatility slightly elevated"
        elif vol_ratio < 0.7:
            return "Volatility contracting - potential breakout setup"
        elif vol_ratio < 0.8:
            return "Low volatility environment"
        else:
            return "Normal volatility conditions"

    def _interpret_volume(self):
        """Interpret volume alpha factors."""
        vol_ratio = self.current_alphas.get('volume_ratio', 1)
        vpt = self.current_alphas.get('volume_price_trend', 0)

        if vol_ratio > 1.5 and vpt > 30:
            return "High volume accumulation - bullish"
        elif vol_ratio > 1.5 and vpt < -30:
            return "High volume distribution - bearish"
        elif vol_ratio > 1.3:
            return "Above average volume activity"
        elif vol_ratio < 0.7:
            return "Low volume - potential consolidation"
        else:
            return "Normal volume activity"

    def _interpret_trend(self):
        """Interpret trend alpha factors."""
        adx = self.current_alphas.get('adx_strength', 25)
        ma_align = self.current_alphas.get('ma_alignment', 1.5)
        consistency = self.current_alphas.get('trend_consistency_20d', 50)

        if adx > 30 and ma_align >= 2.5:
            return "Strong uptrend with aligned MAs"
        elif adx > 30 and ma_align <= 0.5:
            return "Strong downtrend with aligned MAs"
        elif adx > 25:
            return "Moderate trend in progress"
        elif adx < 20:
            return "Weak trend / ranging market"
        else:
            return "Transitional trend state"

    def get_trader_profiles(self):
        """
        Get the 3 trader profiles (aggressive, medium, conservative) with their analysis.

        Returns:
        --------
        dict : Dictionary containing all 3 trader profiles with metrics and analysis
        """
        if not self.trader_profiles:
            # Generate profiles if not already done
            if self.results and len(self.results) > 0:
                base_thresholds = {
                    'buy_threshold': self.results[0].get('buy_threshold', 60),
                    'sell_threshold': self.results[0].get('sell_threshold', 40)
                }
            else:
                base_thresholds = {'buy_threshold': 60, 'sell_threshold': 40}

            self.profile_generator = TraderProfileGenerator(base_thresholds)
            self.trader_profiles = self.profile_generator.generate_all_profiles(
                self.df, self.current_alphas
            )

        return self.trader_profiles
