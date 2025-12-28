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
        metricsText += `\n${'='.repeat(30)}\n`;
        metricsText += `TRADING STRATEGY (${sm.period}):\n`;
        metricsText += `${'='.repeat(30)}\n`;
        metricsText += `ALPHA: ${sm.alpha.toFixed(2)}%\n`;
        metricsText += `BETA: ${sm.beta.toFixed(2)}\n\n`;
        metricsText += `Strategy Return: ${sm.strategy_return.toFixed(2)}%\n`;
        metricsText += `Buy & Hold Return: ${sm.buyhold_return.toFixed(2)}%\n`;
        metricsText += `Max Drawdown: ${sm.strategy_max_dd.toFixed(2)}%\n`;
        metricsText += `Sharpe Ratio: ${sm.strategy_sharpe.toFixed(2)}\n`;
        metricsText += `Number of Trades: ${sm.trades}\n`;
    }

    metricsDiv.textContent = metricsText;
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

    compositeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: compositeData.dates,
            datasets: [{
                label: 'Composite Index',
                data: compositeData.values,
                borderColor: 'rgb(147, 51, 234)',
                backgroundColor: 'rgba(147, 51, 234, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Composite Index (0-100)'
                },
                annotation: {
                    annotations: {
                        buyLine: {
                            type: 'line',
                            yMin: thresholds.buy_threshold,
                            yMax: thresholds.buy_threshold,
                            borderColor: 'green',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                display: true,
                                content: `Buy (${thresholds.buy_threshold})`,
                                position: 'end'
                            }
                        },
                        sellLine: {
                            type: 'line',
                            yMin: thresholds.sell_threshold,
                            yMax: thresholds.sell_threshold,
                            borderColor: 'red',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                display: true,
                                content: `Sell (${thresholds.sell_threshold})`,
                                position: 'end'
                            }
                        }
                    }
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
