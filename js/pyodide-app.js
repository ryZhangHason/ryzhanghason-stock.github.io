// Global variables
let pyodide;
let priceChart, compositeChart, strategyChart;
let isPythonReady = false;

const predictBtn = document.getElementById('predictBtn');
const stockSymbolInput = document.getElementById('stockSymbol');
const timePeriodSelect = document.getElementById('timePeriod');
const optimizeStrategyCheckbox = document.getElementById('optimizeStrategy');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const loadingMessage = document.getElementById('loadingMessage');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');

function updateProgress(percent, message) {
    if (progressBar) {
        progressBar.style.width = percent + '%';
        if (progressText) progressText.textContent = percent + '%';
    }
    if (loadingMessage && message) loadingMessage.textContent = message;
}

// Initialize Pyodide and load Python modules
async function initPython() {
    try {
        predictBtn.textContent = '🔄 Loading Python environment...';
        predictBtn.disabled = true;

        console.log('Loading Pyodide...');

        // Load Pyodide from CDN
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
        });

        console.log('Installing Python packages...');
        predictBtn.textContent = '🔄 Installing packages...';

        // Load required packages
        await pyodide.loadPackage(['numpy', 'pandas', 'micropip']);

        console.log('Installing additional packages...');
        const micropip = pyodide.pyimport('micropip');

        predictBtn.textContent = '🔄 Installing scikit-learn...';
        await micropip.install('scikit-learn');

        predictBtn.textContent = '🔄 Installing XGBoost...';
        await micropip.install('xgboost');

        console.log('Loading Python modules...');
        predictBtn.textContent = '🔄 Loading modules...';

        // Load the Python module files from the server
        const pythonModules = {
            'data_fetcher': await fetch('python/data_fetcher.py').then(r => r.text()),
            'feature_engineering': await fetch('python/feature_engineering.py').then(r => r.text()),
            'model': await fetch('python/model.py').then(r => r.text()),
            'strategy_optimizer': await fetch('python/strategy_optimizer.py').then(r => r.text())
        };

        // Execute the Python modules in Pyodide
        for (const [name, code] of Object.entries(pythonModules)) {
            console.log(`Loading ${name}...`);
            await pyodide.runPythonAsync(code);
        }

        console.log('Python environment ready!');
        isPythonReady = true;
        predictBtn.disabled = false;
        predictBtn.textContent = '🔮 Fetch & Predict';

    } catch (error) {
        console.error('Failed to initialize Python:', error);
        predictBtn.textContent = '❌ Failed to load Python';
        showError('Failed to initialize Python environment: ' + error.message);
    }
}

// Event listeners
predictBtn.addEventListener('click', handlePredict);
stockSymbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && isPythonReady) {
        handlePredict();
    }
});

// Main prediction function
async function handlePredict() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();

    if (!symbol) {
        showError('Please enter a stock symbol');
        return;
    }

    if (!isPythonReady) {
        showError('Python environment is still loading. Please wait...');
        return;
    }

    hideAllSections();
    loadingSection.style.display = 'block';
    updateProgress(0, `Initializing request for ${symbol}...`);
    predictBtn.disabled = true;

    try {
        const period = timePeriodSelect.value;
        const optimize = optimizeStrategyCheckbox.checked;

        updateProgress(10, 'Fetching stock data...');

        // Run the full Python pipeline
        const pythonCode = `
import json
from datetime import datetime
import sys

# Set up the ticker and period
ticker = "${symbol}"
period = "${period}"
optimize_strategy = ${optimize ? 'True' : 'False'}

print(f"Processing {ticker} for period {period}...")

# Step 1: Fetch stock data
print("Step 1: Fetching stock data...")
try:
    df = await get_stock_data_async(ticker, period=period)
    print(f"Fetched {len(df)} days of data")
except Exception as e:
    raise Exception(f"Failed to fetch data: {str(e)}")

# Step 2: Add technical indicators
print("Step 2: Calculating technical indicators...")
try:
    df_features = add_technical_indicators(df)
    print(f"Added technical indicators, features shape: {df_features.shape}")
except Exception as e:
    raise Exception(f"Failed to add technical indicators: {str(e)}")

# Step 3: Prepare data for model
print("Step 3: Preparing data for model...")
try:
    X_train, y_train, latest_features, recent_data = prepare_data_for_model(df_features)
    print(f"Training data shape: {X_train.shape}, Recent data: {len(recent_data)} days")
except Exception as e:
    raise Exception(f"Failed to prepare data: {str(e)}")

# Step 4: Train model
print("Step 4: Training XGBoost model...")
try:
    predictor = StockPredictor()
    predictor.train_model(X_train, y_train, recent_data)
    print("Model trained successfully")
except Exception as e:
    raise Exception(f"Failed to train model: {str(e)}")

# Step 5: Make prediction
print("Step 5: Making prediction...")
try:
    prediction, probability = predictor.predict(latest_features)
    print(f"Prediction: {prediction}, Probability: {probability:.2%}")
except Exception as e:
    raise Exception(f"Failed to make prediction: {str(e)}")

# Step 6: Calculate composite index
print("Step 6: Calculating composite index...")
try:
    composite_index = predictor.calculate_composite_index(df_features)
    df_features['Composite_Index'] = composite_index
    print(f"Composite index calculated")
except Exception as e:
    raise Exception(f"Failed to calculate composite index: {str(e)}")

# Step 7: Optimize strategy if requested using ADAPTIVE meta-learning optimizer (re-optimize every 90 days)
strategy_results = None
optimization_info = None
trade_signals_data = []
if optimize_strategy:
    print("Step 7: Running ADAPTIVE meta-learning optimization (re-optimize every 90 days)...")
    try:
        # Use AdaptiveStrategyOptimizer for 90-day rolling re-optimization
        adaptive_optimizer = AdaptiveStrategyOptimizer(df_features, reoptimize_days=90)
        adaptive_result = adaptive_optimizer.optimize_adaptive(lookback_days=60)

        if adaptive_result:
            df_with_strategy = adaptive_result['df']
            trade_signals_data = adaptive_result['trade_signals']
            period_thresholds = adaptive_result['period_thresholds']

            # Get latest threshold from the most recent period
            latest_period = period_thresholds[-1] if period_thresholds else {'buy_threshold': 60, 'sell_threshold': 40, 'regime': 'unknown'}

            # Also run standard optimizer for additional analysis
            optimizer = StrategyOptimizer(df_features)
            optimal_thresholds = optimizer.optimize_thresholds(min_period=120)

            # Extract optimization info
            optimization_info = {
                'method': 'adaptive_90day',
                'regime': latest_period.get('regime', 'unknown'),
                'buy_threshold': latest_period.get('buy_threshold', 60),
                'sell_threshold': latest_period.get('sell_threshold', 40),
                'sharpe_ratio': adaptive_result.get('sharpe_ratio', 0),
                'total_return': adaptive_result.get('total_return', 0),
                'win_rate': optimal_thresholds.get('win_rate', 0) if optimal_thresholds else 0,
                'num_trades': adaptive_result.get('num_trades', 0),
                'max_drawdown': adaptive_result.get('max_drawdown', 0),
                'num_periods': adaptive_result.get('num_periods', 1),
                'period_thresholds': period_thresholds
            }

            # Calculate strategy performance for chart (last 120 days)
            last_120 = df_with_strategy.tail(120)
            if 'Strategy_Value' in last_120.columns and 'BuyHold_Value' in last_120.columns:
                initial_st = last_120['Strategy_Value'].iloc[0]
                initial_bh = last_120['BuyHold_Value'].iloc[0]

                # Get trade signals within last 120 days
                last_120_dates = set([d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in last_120['Date']])
                filtered_signals = [s for s in trade_signals_data if s['date'] in last_120_dates]

                strategy_results = {
                    'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in last_120['Date']],
                    'strategy': [float((x / initial_st - 1) * 100) for x in last_120['Strategy_Value']],
                    'buyhold': [float((x / initial_bh - 1) * 100) for x in last_120['BuyHold_Value']],
                    'trade_signals': filtered_signals,
                    'optimization_info': optimization_info
                }

            # Get behavior analysis and alpha summary from standard optimizer
            if optimal_thresholds:
                df_features_copy = optimizer.apply_optimal_strategy(optimal_thresholds)
                behavior_analysis = optimizer.get_behavior_analysis()
                alpha_summary = optimizer.get_alpha_summary()
                trader_profiles = optimizer.get_trader_profiles()

                # Extract behavior summary
                behavior_summary = {}
                if behavior_analysis:
                    if 'strategy_profile' in behavior_analysis:
                        sp = behavior_analysis['strategy_profile']
                        behavior_summary['style'] = sp.get('style', 'Unknown')
                        behavior_summary['selectivity'] = sp.get('selectivity', 'Unknown')
                        behavior_summary['long_exposure'] = sp.get('long_exposure', 0)
                        behavior_summary['short_exposure'] = sp.get('short_exposure', 0)
                        behavior_summary['cash_exposure'] = sp.get('cash_exposure', 0)

                    if 'indicator_usage' in behavior_analysis:
                        iu = behavior_analysis['indicator_usage']
                        behavior_summary['primary_indicators'] = [s['indicator'] for s in iu.get('primary_signals', [])]
                        behavior_summary['confirmation_indicators'] = [s['indicator'] for s in iu.get('confirmation_signals', [])]

                    if 'trading_summary' in behavior_analysis:
                        behavior_summary['summary'] = behavior_analysis['trading_summary']

                # Extract alpha interpretations
                alpha_interpretations = {}
                if alpha_summary and 'message' not in alpha_summary:
                    for category, data in alpha_summary.items():
                        if isinstance(data, dict) and 'interpretation' in data:
                            alpha_interpretations[category] = data['interpretation']

                # Add to optimization info
                optimization_info['behavior_summary'] = behavior_summary
                optimization_info['alpha_interpretations'] = alpha_interpretations
                optimization_info['trader_profiles'] = trader_profiles

            print(f"Adaptive optimization complete - {optimization_info['num_periods']} periods, {optimization_info['num_trades']} trades")
            print(f"Latest thresholds - Buy: {optimization_info['buy_threshold']}, Sell: {optimization_info['sell_threshold']}")
    except Exception as e:
        print(f"Warning: Adaptive strategy optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        strategy_results = None
        optimization_info = None

# Prepare results
latest_data = df_features.iloc[-1]
recent_metrics = predictor.recent_metrics if hasattr(predictor, 'recent_metrics') and predictor.recent_metrics else {}

result = {
    'symbol': ticker,
    'prediction': int(prediction),
    'probability': float(probability),
    'latest_price': float(latest_data['Close']),
    'change_pct': float(((latest_data['Close'] - df_features.iloc[-2]['Close']) / df_features.iloc[-2]['Close']) * 100),
    'recent_accuracy': float(predictor.recent_accuracy) if predictor.recent_accuracy else 0.0,
    'recent_metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in recent_metrics.items()},
    'indicators': {
        'rsi': float(latest_data['RSI']) if 'RSI' in latest_data and not pd.isna(latest_data['RSI']) else 50.0,
        'macd': float(latest_data['MACD']) if 'MACD' in latest_data and not pd.isna(latest_data['MACD']) else 0.0,
        'ma20': float(latest_data['MA20']) if 'MA20' in latest_data and not pd.isna(latest_data['MA20']) else float(latest_data['Close']),
        'ma50': float(latest_data['MA50']) if 'MA50' in latest_data and not pd.isna(latest_data['MA50']) else float(latest_data['Close']),
        'bb_upper': float(latest_data['BB_upper']) if 'BB_upper' in latest_data and not pd.isna(latest_data['BB_upper']) else float(latest_data['Close']),
        'bb_lower': float(latest_data['BB_lower']) if 'BB_lower' in latest_data and not pd.isna(latest_data['BB_lower']) else float(latest_data['Close']),
        'volatility': float(latest_data['Volatility_20d']) if 'Volatility_20d' in latest_data and not pd.isna(latest_data['Volatility_20d']) else 0.0
    }
}

# Prepare chart data (last 60 days)
last_60 = df_features.tail(60)
chart_data = {
    'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in last_60['Date']],
    'close': [float(x) for x in last_60['Close']],
    'ma20': [float(x) if not pd.isna(x) else None for x in last_60['MA20']],
    'ma50': [float(x) if not pd.isna(x) else None for x in last_60['MA50']],
    'bb_upper': [float(x) if 'BB_upper' in last_60.columns and not pd.isna(x) else None for x in (last_60['BB_upper'] if 'BB_upper' in last_60.columns else [None] * len(last_60))],
    'bb_lower': [float(x) if 'BB_lower' in last_60.columns and not pd.isna(x) else None for x in (last_60['BB_lower'] if 'BB_lower' in last_60.columns else [None] * len(last_60))]
}

# Composite index chart data with period-varying thresholds
# Build threshold arrays that change based on optimization periods
composite_dates = chart_data['dates']
buy_threshold_array = []
sell_threshold_array = []

if optimization_info and 'period_thresholds' in optimization_info:
    period_thresholds = optimization_info['period_thresholds']

    for date_str in composite_dates:
        # Find which period this date belongs to
        matched_buy = 60
        matched_sell = 40

        for period in period_thresholds:
            period_end = period.get('end_date', '')
            if date_str <= period_end or period == period_thresholds[-1]:
                matched_buy = period.get('buy_threshold', 60)
                matched_sell = period.get('sell_threshold', 40)
                break

        buy_threshold_array.append(matched_buy)
        sell_threshold_array.append(matched_sell)
else:
    # Default static thresholds
    default_buy = optimization_info['buy_threshold'] if optimization_info else 60
    default_sell = optimization_info['sell_threshold'] if optimization_info else 40
    buy_threshold_array = [default_buy] * len(composite_dates)
    sell_threshold_array = [default_sell] * len(composite_dates)

composite_data = {
    'dates': composite_dates,
    'index': [float(x) if not pd.isna(x) else 50.0 for x in composite_index.tail(60)],
    'close': chart_data['close'],
    'buy_threshold': optimization_info['buy_threshold'] if optimization_info else 60,
    'sell_threshold': optimization_info['sell_threshold'] if optimization_info else 40,
    'buy_threshold_array': buy_threshold_array,
    'sell_threshold_array': sell_threshold_array,
    'period_thresholds': optimization_info.get('period_thresholds', []) if optimization_info else []
}

# Strategy chart data (now using smart optimizer output)
strategy_chart = None
if strategy_results:
    strategy_chart = {
        'dates': strategy_results['dates'],
        'strategy': strategy_results['strategy'],
        'buyhold': strategy_results['buyhold'],
        'optimization_info': strategy_results.get('optimization_info', {})
    }

output = {
    'result': result,
    'chart_data': chart_data,
    'composite_data': composite_data,
    'strategy_chart': strategy_chart
}

json.dumps(output)
`;

        updateProgress(20, 'Running Python analysis...');

        // Run the Python code
        const output = await pyodide.runPythonAsync(pythonCode);
        const data = JSON.parse(output);

        updateProgress(100, 'Complete!');

        // Display results
        displayResults(data.result, data.chart_data, data.composite_data, data.strategy_chart);

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to analyze stock. Please check the symbol and try again.');
    } finally {
        predictBtn.disabled = false;
    }
}

function displayResults(result, chartData, compositeData, strategyChart) {
    hideAllSections();
    resultSection.style.display = 'block';

    // Display prediction
    const predictionDiv = document.getElementById('predictionResult');

    if (result.prediction === 1) {
        predictionDiv.className = 'prediction-result prediction-up';
        predictionDiv.innerHTML = `
            📈 ${result.symbol}: Price likely to GO UP<br>
            <span style="font-size: 0.9em;">Confidence: ${(result.probability * 100).toFixed(1)}%</span><br>
            <span style="font-size: 0.8em;">Current: $${result.latest_price.toFixed(2)} (${result.change_pct > 0 ? '+' : ''}${result.change_pct.toFixed(2)}%)</span>
        `;
    } else {
        predictionDiv.className = 'prediction-result prediction-down';
        predictionDiv.innerHTML = `
            📉 ${result.symbol}: Price likely to GO DOWN<br>
            <span style="font-size: 0.9em;">Confidence: ${((1 - result.probability) * 100).toFixed(1)}%</span><br>
            <span style="font-size: 0.8em;">Current: $${result.latest_price.toFixed(2)} (${result.change_pct > 0 ? '+' : ''}${result.change_pct.toFixed(2)}%)</span>
        `;
    }

    // Display metrics with alpha factors, behavior analysis, and trader profiles
    displayMetrics(result, strategyChart);

    // Display price chart
    displayPriceChart(chartData, result.symbol);

    // Display composite index chart
    displayCompositeChart(compositeData);

    // Display strategy chart if available
    if (strategyChart) {
        displayStrategyChart(strategyChart);
    }
}

function displayMetrics(result, strategyChart) {
    const metricsDiv = document.getElementById('metricsResult');
    const metrics = result.recent_metrics;
    const oi = strategyChart ? strategyChart.optimization_info : null;

    let metricsHtml = `<pre style="text-align: left; line-height: 1.6; font-size: 0.85em;">`;

    // Model Performance
    metricsHtml += `<strong>═══════════════════════════════════════</strong>\n`;
    metricsHtml += `<strong>MODEL PERFORMANCE (Last 120 Days):</strong>\n`;
    metricsHtml += `<strong>═══════════════════════════════════════</strong>\n`;
    metricsHtml += `Accuracy: ${(result.recent_accuracy * 100).toFixed(1)}%\n`;
    if (metrics.precision) metricsHtml += `Precision: ${(metrics.precision * 100).toFixed(1)}%\n`;
    if (metrics.f1_score) metricsHtml += `F1 Score: ${(metrics.f1_score * 100).toFixed(1)}%\n`;
    if (metrics.correct_predictions) metricsHtml += `Correct: ${metrics.correct_predictions}/${metrics.total_predictions}\n`;

    // Technical Indicators
    metricsHtml += `\n<strong>TECHNICAL INDICATORS:</strong>\n`;
    metricsHtml += `Price: $${result.latest_price.toFixed(2)}\n`;
    metricsHtml += `MA20: $${result.indicators.ma20.toFixed(2)} | MA50: $${result.indicators.ma50.toFixed(2)}\n`;
    metricsHtml += `RSI: ${result.indicators.rsi.toFixed(1)} ${result.indicators.rsi < 30 ? '(Oversold)' : result.indicators.rsi > 70 ? '(Overbought)' : '(Neutral)'}\n`;
    metricsHtml += `MACD: ${result.indicators.macd.toFixed(2)}\n`;
    metricsHtml += `Volatility: ${(result.indicators.volatility * 100).toFixed(2)}%\n`;

    // Alpha Factor Analysis
    if (oi && oi.alpha_interpretations && Object.keys(oi.alpha_interpretations).length > 0) {
        metricsHtml += `\n<strong>═══════════════════════════════════════</strong>\n`;
        metricsHtml += `<strong>ALPHA FACTOR ANALYSIS:</strong>\n`;
        metricsHtml += `<strong>═══════════════════════════════════════</strong>\n`;
        for (const [category, interpretation] of Object.entries(oi.alpha_interpretations)) {
            const categoryName = category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ');
            metricsHtml += `${categoryName}: ${interpretation}\n`;
        }
    }

    // Behavior Analysis
    if (oi && oi.behavior_summary && Object.keys(oi.behavior_summary).length > 0) {
        const bs = oi.behavior_summary;
        metricsHtml += `\n<strong>═══════════════════════════════════════</strong>\n`;
        metricsHtml += `<strong>TRADER BEHAVIOR ANALYSIS:</strong>\n`;
        metricsHtml += `<strong>═══════════════════════════════════════</strong>\n`;

        if (bs.style) metricsHtml += `Strategy Style: ${bs.style}\n`;
        if (bs.selectivity) metricsHtml += `Selectivity: ${bs.selectivity}\n`;

        if (bs.long_exposure !== undefined) {
            metricsHtml += `\nPosition Exposure:\n`;
            metricsHtml += `  Long:  ${getPositionBar(bs.long_exposure)} ${bs.long_exposure.toFixed(0)}%\n`;
            metricsHtml += `  Short: ${getPositionBar(bs.short_exposure)} ${bs.short_exposure.toFixed(0)}%\n`;
            metricsHtml += `  Cash:  ${getPositionBar(bs.cash_exposure)} ${bs.cash_exposure.toFixed(0)}%\n`;
        }

        if (bs.primary_indicators && bs.primary_indicators.length > 0) {
            metricsHtml += `\nPrimary Signals: ${bs.primary_indicators.join(', ')}\n`;
        }
        if (bs.confirmation_indicators && bs.confirmation_indicators.length > 0) {
            metricsHtml += `Confirmation: ${bs.confirmation_indicators.join(', ')}\n`;
        }

        if (bs.summary && bs.summary.length > 0) {
            metricsHtml += `\nStrategy Summary:\n`;
            for (const line of bs.summary) {
                metricsHtml += `  • ${line}\n`;
            }
        }
    }

    // 3 Trader Profiles
    if (oi && oi.trader_profiles && Object.keys(oi.trader_profiles).length > 0) {
        metricsHtml += `\n<strong>═══════════════════════════════════════</strong>\n`;
        metricsHtml += `<strong>3 TRADER PROFILES COMPARISON:</strong>\n`;
        metricsHtml += `<strong>═══════════════════════════════════════</strong>\n`;

        const profileOrder = ['aggressive', 'medium', 'conservative'];
        const profileEmojis = { aggressive: '🔥', medium: '⚖️', conservative: '🛡️' };

        for (const profileType of profileOrder) {
            const profile = oi.trader_profiles[profileType];
            if (!profile) continue;

            const emoji = profileEmojis[profileType];
            const name = profileType.charAt(0).toUpperCase() + profileType.slice(1);
            const th = profile.thresholds;
            const metrics = profile.metrics;
            const analysis = profile.analysis;

            metricsHtml += `\n${emoji} <strong>${name} Trader</strong>\n`;
            metricsHtml += `${'─'.repeat(35)}\n`;
            metricsHtml += `Thresholds: Buy ≥ ${th.buy_threshold}, Sell ≤ ${th.sell_threshold}\n`;
            metricsHtml += `${th.description}\n\n`;

            // Performance metrics
            metricsHtml += `Performance:\n`;
            metricsHtml += `  Return: ${metrics.total_return >= 0 ? '+' : ''}${metrics.total_return}%\n`;
            metricsHtml += `  Sharpe: ${metrics.sharpe_ratio}\n`;
            metricsHtml += `  Max DD: ${metrics.max_drawdown}%\n`;
            metricsHtml += `  Trades: ${metrics.num_trades} | Win Rate: ${metrics.win_rate}%\n\n`;

            // Trading behavior
            if (analysis.description) {
                metricsHtml += `Trading Style: ${analysis.description.trading_style}\n`;
                metricsHtml += `Entry: ${analysis.description.entry_behavior}\n`;
                metricsHtml += `Exit: ${analysis.description.exit_behavior}\n`;
                metricsHtml += `Risk: ${analysis.description.risk_approach}\n`;
                metricsHtml += `Best Market: ${analysis.description.best_market}\n`;
                metricsHtml += `Focus: ${analysis.description.indicator_focus}\n`;
            }

            // Position breakdown
            if (analysis.position_breakdown) {
                const pb = analysis.position_breakdown;
                metricsHtml += `\nPositions: ${pb.long_pct}% Long | ${pb.short_pct}% Short | ${pb.cash_pct}% Cash\n`;
            }

            // Alpha usage
            if (analysis.alpha_usage) {
                metricsHtml += `\nAlpha Strategy:\n`;
                metricsHtml += `  ${analysis.alpha_usage.description}\n`;
            }
        }
    }

    metricsHtml += `\n<strong>═══════════════════════════════════════</strong>\n`;
    metricsHtml += `⚠️ Disclaimer: Educational purposes only.\n`;
    metricsHtml += `   Not financial advice.\n`;
    metricsHtml += `</pre>`;

    metricsDiv.innerHTML = metricsHtml;
}

function getPositionBar(value) {
    const filled = Math.round(value / 10);
    const empty = 10 - filled;
    return '[' + '█'.repeat(filled) + '░'.repeat(empty) + ']';
}

function displayPriceChart(chartData, symbol) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    const datasets = [{
        label: 'Close Price',
        data: chartData.close,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        tension: 0.1,
        borderWidth: 2,
        fill: false
    }, {
        label: '20-day MA',
        data: chartData.ma20,
        borderColor: 'rgb(255, 159, 64)',
        borderDash: [5, 5],
        tension: 0.1,
        fill: false,
        borderWidth: 1.5
    }, {
        label: '50-day MA',
        data: chartData.ma50,
        borderColor: 'rgb(153, 102, 255)',
        borderDash: [5, 5],
        tension: 0.1,
        fill: false,
        borderWidth: 1.5
    }];

    // Add Bollinger Bands if available
    if (chartData.bb_upper && chartData.bb_upper.some(x => x !== null)) {
        datasets.push({
            label: 'BB Upper',
            data: chartData.bb_upper,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            borderDash: [2, 2],
            tension: 0.1,
            fill: false,
            borderWidth: 1
        });
        datasets.push({
            label: 'BB Lower',
            data: chartData.bb_lower,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            borderDash: [2, 2],
            tension: 0.1,
            fill: false,
            borderWidth: 1
        });
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} - Last 60 Days`,
                    font: { size: 16 }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: value => '$' + value.toFixed(2)
                    }
                }
            }
        }
    });
}

function displayCompositeChart(compositeData) {
    const ctx = document.getElementById('compositeChart').getContext('2d');

    if (compositeChart) {
        compositeChart.destroy();
    }

    const numPoints = compositeData.dates.length;

    // Use period-varying threshold arrays if available, otherwise use static
    const buyThresholdLine = compositeData.buy_threshold_array || Array(numPoints).fill(compositeData.buy_threshold || 60);
    const sellThresholdLine = compositeData.sell_threshold_array || Array(numPoints).fill(compositeData.sell_threshold || 40);
    const neutralLine = Array(numPoints).fill(50);

    // Get unique threshold values for legend
    const uniqueBuyThresholds = [...new Set(buyThresholdLine)];
    const uniqueSellThresholds = [...new Set(sellThresholdLine)];
    const hasVaryingThresholds = uniqueBuyThresholds.length > 1 || uniqueSellThresholds.length > 1;

    // Color the composite index based on current threshold at each point
    const indexColors = compositeData.index.map((val, i) => {
        const buyTh = buyThresholdLine[i];
        const sellTh = sellThresholdLine[i];
        if (val >= buyTh) return 'rgba(34, 197, 94, 1)';      // Green - buy zone
        if (val <= sellTh) return 'rgba(239, 68, 68, 1)';     // Red - sell zone
        return 'rgba(99, 102, 241, 1)';                        // Blue - neutral
    });

    // Create subtitle based on whether thresholds vary
    let subtitleText;
    if (hasVaryingThresholds) {
        const periods = compositeData.period_thresholds || [];
        subtitleText = `Adaptive Thresholds (${periods.length} periods) | Green=BUY | Red=SELL | Blue=HOLD`;
    } else {
        const buyTh = buyThresholdLine[0];
        const sellTh = sellThresholdLine[0];
        subtitleText = `BUY >= ${buyTh} | SELL <= ${sellTh} | Blue = HOLD`;
    }

    // Build legend label for thresholds
    let buyLabel = 'Buy Threshold';
    let sellLabel = 'Sell Threshold';
    if (hasVaryingThresholds) {
        buyLabel = `Buy Threshold (${uniqueBuyThresholds.join('→')})`;
        sellLabel = `Sell Threshold (${uniqueSellThresholds.join('→')})`;
    } else {
        buyLabel = `BUY >= ${buyThresholdLine[0]}`;
        sellLabel = `SELL <= ${sellThresholdLine[0]}`;
    }

    compositeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: compositeData.dates,
            datasets: [
                // Composite Index line with colored points
                {
                    label: 'Composite Index',
                    data: compositeData.index,
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    tension: 0.3,
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 4,
                    pointBackgroundColor: indexColors,
                    pointBorderColor: indexColors,
                    yAxisID: 'y'
                },
                // Buy threshold line (varying)
                {
                    label: buyLabel,
                    data: buyThresholdLine,
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.15)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: {
                        target: { value: 100 },
                        above: 'rgba(34, 197, 94, 0.08)'
                    },
                    stepped: hasVaryingThresholds ? 'before' : false,
                    yAxisID: 'y'
                },
                // Sell threshold line (varying)
                {
                    label: sellLabel,
                    data: sellThresholdLine,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: {
                        target: { value: 0 },
                        below: 'rgba(239, 68, 68, 0.08)'
                    },
                    stepped: hasVaryingThresholds ? 'before' : false,
                    yAxisID: 'y'
                },
                // Neutral line (50)
                {
                    label: 'Neutral (50)',
                    data: neutralLine,
                    borderColor: 'rgba(156, 163, 175, 0.5)',
                    borderDash: [5, 5],
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y'
                },
                // Stock price on secondary axis
                {
                    label: 'Stock Price',
                    data: compositeData.close,
                    borderColor: 'rgba(75, 192, 192, 0.8)',
                    backgroundColor: 'transparent',
                    tension: 0.2,
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: hasVaryingThresholds
                        ? 'COMPOSITE INDEX - Adaptive Thresholds (Re-optimized Every 90 Days)'
                        : 'COMPOSITE INDEX - Trading Signal Zones',
                    font: { size: 16, weight: 'bold' }
                },
                subtitle: {
                    display: true,
                    text: subtitleText,
                    font: { size: 12 },
                    color: '#666'
                },
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 12
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const dataIndex = context.dataIndex;

                            if (context.dataset.label === 'Composite Index') {
                                const val = context.raw;
                                const buyTh = buyThresholdLine[dataIndex];
                                const sellTh = sellThresholdLine[dataIndex];
                                let signal = 'HOLD';
                                if (val >= buyTh) signal = 'BUY SIGNAL';
                                if (val <= sellTh) signal = 'SELL SIGNAL';
                                return `Index: ${val.toFixed(1)} - ${signal} (Buy>=${buyTh}, Sell<=${sellTh})`;
                            }
                            if (context.dataset.label === 'Stock Price') {
                                return `Price: $${context.raw.toFixed(2)}`;
                            }
                            if (context.dataset.label.includes('Buy')) {
                                return `Buy Threshold: ${context.raw}`;
                            }
                            if (context.dataset.label.includes('Sell')) {
                                return `Sell Threshold: ${context.raw}`;
                            }
                            return context.dataset.label + ': ' + context.raw;
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Composite Index (0-100)',
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        stepSize: 10
                    },
                    grid: {
                        color: function(context) {
                            if (context.tick.value === 50) return 'rgba(156, 163, 175, 0.4)';
                            return 'rgba(0, 0, 0, 0.05)';
                        }
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Stock Price ($)',
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        callback: value => '$' + value.toFixed(2)
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

function displayStrategyChart(strategyData) {
    const ctx = document.getElementById('strategyChart').getContext('2d');

    if (strategyChart) {
        strategyChart.destroy();
    }

    // Get optimization info
    const oi = strategyData.optimization_info || {};
    const regime = formatRegime(oi.regime || 'unknown');
    const sharpe = (oi.sharpe_ratio || 0).toFixed(2);
    const winRate = (oi.win_rate || 0).toFixed(1);
    const method = formatOptMethod(oi.method || 'grid_search');
    const numPeriods = oi.num_periods || 1;
    const numTrades = oi.num_trades || 0;
    const totalReturn = (oi.total_return || 0).toFixed(1);
    const maxDD = (oi.max_drawdown || 0).toFixed(1);

    // Process trade signals for markers
    const tradeSignals = strategyData.trade_signals || [];
    const dates = strategyData.dates;

    // Create arrays for buy/sell markers
    const buyPoints = new Array(dates.length).fill(null);
    const sellPoints = new Array(dates.length).fill(null);
    const exitLongPoints = new Array(dates.length).fill(null);
    const exitShortPoints = new Array(dates.length).fill(null);

    // Map trade signals to strategy value at that point
    tradeSignals.forEach(signal => {
        const dateIndex = dates.indexOf(signal.date);
        if (dateIndex >= 0) {
            const strategyValue = strategyData.strategy[dateIndex];
            if (signal.type === 'buy') {
                buyPoints[dateIndex] = strategyValue;
            } else if (signal.type === 'sell') {
                sellPoints[dateIndex] = strategyValue;
            } else if (signal.type === 'exit_long') {
                exitLongPoints[dateIndex] = strategyValue;
            } else if (signal.type === 'exit_short') {
                exitShortPoints[dateIndex] = strategyValue;
            }
        }
    });

    // Count signals for display
    const buyCount = tradeSignals.filter(s => s.type === 'buy').length;
    const sellCount = tradeSignals.filter(s => s.type === 'sell').length;

    strategyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                // Strategy performance line
                {
                    label: 'Adaptive Strategy',
                    data: strategyData.strategy,
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.15)',
                    tension: 0.2,
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0
                },
                // Buy & Hold line
                {
                    label: 'Buy & Hold',
                    data: strategyData.buyhold,
                    borderColor: 'rgb(107, 114, 128)',
                    backgroundColor: 'transparent',
                    tension: 0.2,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                },
                // BUY signals (green dots)
                {
                    label: `BUY Signals (${buyCount})`,
                    data: buyPoints,
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgb(34, 197, 94)',
                    pointRadius: 10,
                    pointHoverRadius: 14,
                    pointStyle: 'triangle',
                    rotation: 0,
                    showLine: false,
                    pointBorderWidth: 2,
                    pointBorderColor: 'white'
                },
                // SELL signals (red dots)
                {
                    label: `SELL Signals (${sellCount})`,
                    data: sellPoints,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgb(239, 68, 68)',
                    pointRadius: 10,
                    pointHoverRadius: 14,
                    pointStyle: 'triangle',
                    rotation: 180,
                    showLine: false,
                    pointBorderWidth: 2,
                    pointBorderColor: 'white'
                },
                // Exit long (yellow square)
                {
                    label: 'Exit Long',
                    data: exitLongPoints,
                    borderColor: 'rgb(245, 158, 11)',
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                    pointRadius: 7,
                    pointHoverRadius: 10,
                    pointStyle: 'rect',
                    showLine: false,
                    pointBorderWidth: 2,
                    pointBorderColor: 'white'
                },
                // Exit short (orange square)
                {
                    label: 'Exit Short',
                    data: exitShortPoints,
                    borderColor: 'rgb(249, 115, 22)',
                    backgroundColor: 'rgba(249, 115, 22, 0.8)',
                    pointRadius: 7,
                    pointHoverRadius: 10,
                    pointStyle: 'rect',
                    showLine: false,
                    pointBorderWidth: 2,
                    pointBorderColor: 'white'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'nearest',
                intersect: true
            },
            plugins: {
                title: {
                    display: true,
                    text: `ADAPTIVE STRATEGY - Re-optimized every 90 days (${numPeriods} periods)`,
                    font: { size: 16, weight: 'bold' }
                },
                subtitle: {
                    display: true,
                    text: `Return: ${totalReturn}% | Sharpe: ${sharpe} | Max DD: ${maxDD}% | Trades: ${numTrades} | Regime: ${regime}`,
                    font: { size: 12 },
                    color: '#666'
                },
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const datasetLabel = context.dataset.label;
                            if (datasetLabel.includes('BUY Signal')) {
                                const signal = tradeSignals.find(s => s.type === 'buy' && s.date === dates[context.dataIndex]);
                                if (signal) {
                                    return `BUY @ $${signal.price.toFixed(2)} (Index: ${signal.composite.toFixed(1)})`;
                                }
                            } else if (datasetLabel.includes('SELL Signal')) {
                                const signal = tradeSignals.find(s => s.type === 'sell' && s.date === dates[context.dataIndex]);
                                if (signal) {
                                    return `SELL @ $${signal.price.toFixed(2)} (Index: ${signal.composite.toFixed(1)})`;
                                }
                            } else if (datasetLabel === 'Exit Long') {
                                const signal = tradeSignals.find(s => s.type === 'exit_long' && s.date === dates[context.dataIndex]);
                                if (signal) {
                                    return `EXIT LONG @ $${signal.price.toFixed(2)}`;
                                }
                            } else if (datasetLabel === 'Exit Short') {
                                const signal = tradeSignals.find(s => s.type === 'exit_short' && s.date === dates[context.dataIndex]);
                                if (signal) {
                                    return `EXIT SHORT @ $${signal.price.toFixed(2)}`;
                                }
                            }
                            return `${datasetLabel}: ${context.raw !== null ? context.raw.toFixed(2) + '%' : 'N/A'}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Return (%)',
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        callback: value => value.toFixed(1) + '%'
                    },
                    grid: {
                        color: function(context) {
                            if (context.tick.value === 0) return 'rgba(0, 0, 0, 0.3)';
                            return 'rgba(0, 0, 0, 0.05)';
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

function formatOptMethod(method) {
    const methods = {
        'meta_learning_ensemble': 'Meta-Learning Ensemble',
        'grid_search': 'Grid Search',
        'walk_forward': 'Walk-Forward',
        'adaptive_90day': 'Adaptive 90-Day'
    };
    return methods[method] || method;
}

function formatRegime(regime) {
    const regimes = {
        'trending_up': 'Trending UP',
        'trending_down': 'Trending DOWN',
        'ranging': 'Ranging',
        'high_volatility': 'High Volatility'
    };
    return regimes[regime] || regime;
}

function showError(message) {
    hideAllSections();
    errorSection.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

function hideAllSections() {
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// Initialize Python when page loads
initPython();
