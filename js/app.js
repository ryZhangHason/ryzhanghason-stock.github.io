// API Configuration
// For local development, use: http://localhost:5000
// For production, replace with your deployed API URL
const API_URL = 'http://localhost:5000/api';

// Chart instances
let priceChart, compositeChart, strategyChart;

// Get DOM elements
const predictBtn = document.getElementById('predictBtn');
const stockSymbolInput = document.getElementById('stockSymbol');
const timePeriodSelect = document.getElementById('timePeriod');
const optimizeStrategyCheckbox = document.getElementById('optimizeStrategy');

const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

// Event listeners
predictBtn.addEventListener('click', handlePredict);

stockSymbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handlePredict();
    }
});

async function handlePredict() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();

    if (!symbol) {
        showError('Please enter a stock symbol');
        return;
    }

    // Show loading
    hideAllSections();
    loadingSection.style.display = 'block';
    predictBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                period: timePeriodSelect.value,
                optimize: optimizeStrategyCheckbox.checked
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to fetch prediction. Make sure the API server is running.');
    } finally {
        predictBtn.disabled = false;
    }
}

function displayResults(data) {
    hideAllSections();
    resultSection.style.display = 'block';

    // Display prediction
    const predictionDiv = document.getElementById('predictionResult');
    const prediction = data.prediction;
    const probability = data.probability;

    if (prediction === 1) {
        predictionDiv.className = 'prediction-result prediction-up';
        predictionDiv.innerHTML = `
            📈 ${data.symbol}: Price likely to GO UP<br>
            <span style="font-size: 0.8em;">Confidence: ${(probability * 100).toFixed(2)}%</span>
        `;
    } else {
        predictionDiv.className = 'prediction-result prediction-down';
        predictionDiv.innerHTML = `
            📉 ${data.symbol}: Price likely to GO DOWN<br>
            <span style="font-size: 0.8em;">Confidence: ${((1 - probability) * 100).toFixed(2)}%</span>
        `;
    }

    // Display metrics
    displayMetrics(data.metrics, data.symbol);

    // Display charts
    displayPriceChart(data.price_data, data.symbol);
    displayCompositeChart(data.composite_data, data.thresholds);
    displayStrategyChart(data.strategy_data);
}

function displayMetrics(metrics, symbol) {
    const metricsDiv = document.getElementById('metricsResult');

    let metricsText = `Model Performance for ${symbol}:\n\n`;
    metricsText += `Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%\n`;
    metricsText += `F1 Score: ${(metrics.f1_score * 100).toFixed(2)}%\n\n`;

    metricsText += `Precision:\n`;
    metricsText += `- UP predictions: ${(metrics.up_precision * 100).toFixed(2)}%\n`;
    metricsText += `- DOWN predictions: ${(metrics.down_precision * 100).toFixed(2)}%\n\n`;

    metricsText += `Recent ${metrics.total_predictions}-day Statistics:\n`;
    metricsText += `- Days price went UP: ${metrics.up_count}\n`;
    metricsText += `- Days price went DOWN: ${metrics.down_count}\n`;
    metricsText += `- Correct predictions: ${metrics.correct_predictions} / ${metrics.total_predictions}\n`;

    if (metrics.strategy_metrics) {
        const sm = metrics.strategy_metrics;
        metricsText += `\n${'='.repeat(40)}\n`;
        metricsText += `SMART TRADING STRATEGY (${sm.period}):\n`;
        metricsText += `${'='.repeat(40)}\n`;

        // Display optimization info if available
        if (sm.optimization_info) {
            const oi = sm.optimization_info;
            metricsText += `\nOptimization Method: ${formatOptMethod(oi.method)}\n`;
            metricsText += `Market Regime: ${formatRegime(oi.regime)}\n`;

            // Display ensemble weights
            if (oi.ensemble_weights && Object.keys(oi.ensemble_weights).length > 0) {
                metricsText += `\nEnsemble Strategy Weights:\n`;
                for (const [strategy, weight] of Object.entries(oi.ensemble_weights)) {
                    const pct = (weight * 100).toFixed(1);
                    const bar = getProgressBar(weight);
                    metricsText += `  ${formatStrategyName(strategy)}: ${bar} ${pct}%\n`;
                }
            }
            metricsText += `\n`;
        }

        metricsText += `ALPHA: ${sm.alpha.toFixed(2)}%\n`;
        metricsText += `BETA: ${sm.beta.toFixed(2)}\n\n`;
        metricsText += `Strategy Return: ${sm.strategy_return.toFixed(2)}%\n`;
        metricsText += `Buy & Hold Return: ${sm.buyhold_return.toFixed(2)}%\n`;
        metricsText += `Max Drawdown: ${sm.strategy_max_dd.toFixed(2)}%\n`;
        metricsText += `Sharpe Ratio: ${sm.strategy_sharpe.toFixed(2)}\n`;
        metricsText += `Number of Trades: ${sm.trades}\n`;

        // Show additional metrics if available
        if (sm.optimization_info) {
            const oi = sm.optimization_info;
            if (oi.win_rate) metricsText += `Win Rate: ${oi.win_rate.toFixed(1)}%\n`;
            if (oi.profit_factor) metricsText += `Profit Factor: ${oi.profit_factor.toFixed(2)}\n`;
        }
    }

    metricsDiv.textContent = metricsText;
}

function formatOptMethod(method) {
    const methods = {
        'meta_learning_ensemble': 'Meta-Learning Ensemble',
        'grid_search': 'Grid Search',
        'walk_forward': 'Walk-Forward'
    };
    return methods[method] || method;
}

function formatRegime(regime) {
    const regimes = {
        'trending_up': 'Trending UP',
        'trending_down': 'Trending DOWN',
        'ranging': 'Ranging/Sideways',
        'high_volatility': 'High Volatility'
    };
    return regimes[regime] || regime;
}

function formatStrategyName(name) {
    const names = {
        'regime': 'Regime-Based   ',
        'meta': 'Meta-Learner   ',
        'walk_forward': 'Walk-Forward   '
    };
    return names[name] || name.padEnd(15);
}

function getProgressBar(value) {
    const filled = Math.round(value * 10);
    const empty = 10 - filled;
    return '[' + '#'.repeat(filled) + '-'.repeat(empty) + ']';
}

function displayPriceChart(priceData, symbol) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: priceData.dates,
            datasets: [{
                label: 'Close Price',
                data: priceData.close,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1
            }, {
                label: '20-day MA',
                data: priceData.ma20,
                borderColor: 'rgb(255, 159, 64)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false
            }, {
                label: '50-day MA',
                data: priceData.ma50,
                borderColor: 'rgb(153, 102, 255)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} Stock Price`
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

function displayCompositeChart(compositeData, thresholds) {
    const ctx = document.getElementById('compositeChart').getContext('2d');

    if (compositeChart) {
        compositeChart.destroy();
    }

    const buyThreshold = thresholds.buy_threshold || 60;
    const sellThreshold = thresholds.sell_threshold || 40;
    const numPoints = compositeData.dates.length;

    // Create threshold line data (horizontal lines across all dates)
    const buyThresholdLine = Array(numPoints).fill(buyThreshold);
    const sellThresholdLine = Array(numPoints).fill(sellThreshold);

    compositeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: compositeData.dates,
            datasets: [{
                label: 'Composite Index',
                data: compositeData.values,
                borderColor: 'rgb(147, 51, 234)',
                backgroundColor: 'rgba(147, 51, 234, 0.1)',
                tension: 0.1,
                borderWidth: 2
            }, {
                label: `Buy Threshold (${buyThreshold})`,
                data: buyThresholdLine,
                borderColor: 'rgb(34, 197, 94)',
                borderDash: [8, 4],
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }, {
                label: `Sell Threshold (${sellThreshold})`,
                data: sellThresholdLine,
                borderColor: 'rgb(239, 68, 68)',
                borderDash: [8, 4],
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Composite Index with Buy/Sell Thresholds'
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100
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

    strategyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: strategyData.dates,
            datasets: [{
                label: 'Buy & Hold',
                data: strategyData.buyhold,
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                tension: 0.1
            }, {
                label: 'Strategy',
                data: strategyData.strategy,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Strategy Returns (Last 120 Days, %)'
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
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
