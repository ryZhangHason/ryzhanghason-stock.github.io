from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import get_stock_data
from feature_engineering import add_technical_indicators, prepare_data_for_model
from model import StockPredictor
from strategy_optimizer import StrategyOptimizer
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '2y')
        optimize = data.get('optimize', True)

        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400

        # Fetch stock data
        print(f"Fetching data for {symbol}...")
        df = get_stock_data(symbol, period)

        # Add technical indicators
        print(f"Adding technical indicators...")
        df_features = add_technical_indicators(df)

        # Prepare data for model
        print(f"Preparing data for model...")
        X_train, y_train, latest_features, recent_data = prepare_data_for_model(df_features)

        # Train/load model
        print(f"Training/loading model...")
        predictor = StockPredictor()
        model_path = f"{symbol}_model.pkl"

        if os.path.exists(model_path):
            predictor.load_model(model_path)
            predictor.calculate_recent_accuracy(recent_data)
        else:
            predictor.train_model(X_train, y_train, recent_data)
            predictor.save_model(model_path)

        # Make prediction
        prediction, probability = predictor.predict(latest_features)

        # Calculate composite index
        print(f"Calculating composite index...")
        df_features_copy = df_features.copy()
        composite_index = predictor.calculate_composite_index(df_features_copy)
        df_features_copy['Composite_Index'] = composite_index

        # Optimize strategy using smart meta-learning optimizer
        strategy_thresholds = {'buy_threshold': 60, 'sell_threshold': 40}
        strategy_metrics = None
        optimization_info = None

        if optimize:
            print(f"Running smart meta-learning optimization...")
            optimizer = StrategyOptimizer(df_features_copy)
            optimal_thresholds = optimizer.optimize_thresholds(min_period=120)

            if optimal_thresholds:
                df_features_copy = optimizer.apply_optimal_strategy(optimal_thresholds)
                strategy_thresholds = optimal_thresholds

                # Extract optimization info for frontend display
                optimization_info = {
                    'method': optimal_thresholds.get('optimization_method', 'grid_search'),
                    'regime': optimal_thresholds.get('regime', 'unknown'),
                    'ensemble_weights': optimal_thresholds.get('ensemble_weights', {}),
                    'win_rate': optimal_thresholds.get('win_rate', 0),
                    'num_trades': optimal_thresholds.get('num_trades', 0),
                    'profit_factor': optimal_thresholds.get('profit_factor', 0)
                }

                # Calculate strategy metrics
                last_120 = df_features_copy.iloc[-120:]
                strategy_metrics = {
                    'period': f"{len(last_120)} days",
                    'buyhold_return': ((last_120['BuyHold_Value'].iloc[-1] / last_120['BuyHold_Value'].iloc[0] - 1) * 100),
                    'strategy_return': ((last_120['Strategy_Value'].iloc[-1] / last_120['Strategy_Value'].iloc[0] - 1) * 100),
                    'strategy_max_dd': (last_120['Strategy_Drawdown'].min() * 100),
                    'strategy_sharpe': optimal_thresholds.get('sharpe_ratio', 0),
                    'alpha': optimal_thresholds.get('total_return', 0),
                    'beta': 1.0,
                    'trades': int(last_120['Position'].diff().fillna(0).abs().sum() / 2),
                    'optimization_info': optimization_info
                }

        df_features = df_features_copy

        # Prepare response
        response = {
            'symbol': symbol,
            'prediction': int(prediction),
            'probability': float(probability),
            'metrics': {
                'accuracy': predictor.recent_metrics['accuracy'],
                'f1_score': predictor.recent_metrics['f1_score'],
                'up_precision': predictor.recent_metrics['up_precision'],
                'down_precision': predictor.recent_metrics['down_precision'],
                'up_count': predictor.recent_metrics['up_count'],
                'down_count': predictor.recent_metrics['down_count'],
                'correct_predictions': predictor.recent_metrics['correct_predictions'],
                'total_predictions': predictor.recent_metrics['total_predictions'],
                'strategy_metrics': strategy_metrics
            },
            'price_data': {
                'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist()[-200:],
                'close': df['Close'].tolist()[-200:],
                'ma20': df['MA20'].tolist()[-200:] if 'MA20' in df.columns else [],
                'ma50': df['MA50'].tolist()[-200:] if 'MA50' in df.columns else []
            },
            'composite_data': {
                'dates': df_features['Date'].dt.strftime('%Y-%m-%d').tolist()[-200:],
                'values': df_features['Composite_Index'].tolist()[-200:]
            },
            'thresholds': {
                'buy_threshold': strategy_thresholds.get('buy_threshold', 60),
                'sell_threshold': strategy_thresholds.get('sell_threshold', 40)
            },
            'strategy_data': {
                'dates': [],
                'buyhold': [],
                'strategy': []
            }
        }

        # Add strategy performance data
        if 'BuyHold_Value' in df_features.columns and 'Strategy_Value' in df_features.columns:
            recent_df = df_features.iloc[-120:]
            initial_bh = recent_df['BuyHold_Value'].iloc[0]
            initial_st = recent_df['Strategy_Value'].iloc[0]

            response['strategy_data'] = {
                'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'buyhold': ((recent_df['BuyHold_Value'] / initial_bh - 1) * 100).tolist(),
                'strategy': ((recent_df['Strategy_Value'] / initial_st - 1) * 100).tolist()
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Stock Predictor API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
