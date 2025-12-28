// Global variables
let pyodide;
let priceChart;
let isPythonReady = false;

// Function to fetch stock data from Yahoo Finance API
async function fetchStockData(symbol, period) {
    // Convert period to range
    const periodMap = {
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730
    };
    const days = periodMap[period] || 180;

    try {
        // Using Yahoo Finance API v8
        const end = Math.floor(Date.now() / 1000);
        const start = end - (days * 24 * 60 * 60);

        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${start}&period2=${end}&interval=1d`;

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch data for ${symbol}`);
        }

        const data = await response.json();
        const result = data.chart.result[0];

        if (!result || !result.timestamp) {
            throw new Error('Invalid data received');
        }

        const timestamps = result.timestamp;
        const quotes = result.indicators.quote[0];

        return timestamps.map((timestamp, i) => ({
            date: new Date(timestamp * 1000).toISOString().split('T')[0],
            close: quotes.close[i] || 0,
            high: quotes.high[i] || 0,
            low: quotes.low[i] || 0,
            volume: quotes.volume[i] || 0
        })).filter(d => d.close > 0);  // Filter out invalid data

    } catch (error) {
        console.error('Error fetching stock data:', error);
        throw new Error(`Failed to fetch data for ${symbol}. Please check the symbol and try again.`);
    }
}

// DOM elements
const predictBtn = document.getElementById('predictBtn');
const stockSymbolInput = document.getElementById('stockSymbol');
const timePeriodSelect = document.getElementById('timePeriod');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const loadingMessage = document.getElementById('loadingMessage');
const loadingPython = document.getElementById('loadingPython');

// Initialize Pyodide and Python packages
async function initPython() {
    try {
        loadingPython.style.display = 'block';
        predictBtn.textContent = '🔄 Loading Python environment...';

        console.log('Loading Pyodide...');
        pyodide = await loadPyodide();

        console.log('Loading packages...');
        await pyodide.loadPackage(['numpy', 'pandas', 'micropip']);

        console.log('Installing additional packages...');
        const micropip = pyodide.pyimport('micropip');

        predictBtn.textContent = '🔄 Installing scikit-learn...';
        await micropip.install('scikit-learn');

        // Note: We'll fetch stock data directly via JavaScript/API instead of yfinance
        console.log('Using JavaScript for data fetching (yfinance not compatible with browser)');

        console.log('Python environment ready!');
        isPythonReady = true;
        loadingPython.style.display = 'none';
        predictBtn.disabled = false;
        predictBtn.textContent = '🔮 Fetch & Predict';

    } catch (error) {
        console.error('Failed to initialize Python:', error);
        loadingPython.style.display = 'none';
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
    loadingMessage.textContent = `Fetching data for ${symbol}...`;
    predictBtn.disabled = true;

    try {
        // Fetch stock data using JavaScript (Yahoo Finance API)
        const period = timePeriodSelect.value;
        loadingMessage.textContent = 'Downloading stock data...';

        const stockData = await fetchStockData(symbol, period);

        if (!stockData || stockData.length === 0) {
            throw new Error(`No data found for ${symbol}`);
        }

        loadingMessage.textContent = 'Analyzing data with Python...';

        // Prepare data for Python
        const dates = stockData.map(d => d.date);
        const closes = stockData.map(d => d.close);
        const highs = stockData.map(d => d.high);
        const lows = stockData.map(d => d.low);
        const volumes = stockData.map(d => d.volume);

        const pythonCode = `
import pandas as pd
import numpy as np
from datetime import datetime

# Create DataFrame from JavaScript data
dates = ${JSON.stringify(dates)}
closes = ${JSON.stringify(closes)}
highs = ${JSON.stringify(highs)}
lows = ${JSON.stringify(lows)}
volumes = ${JSON.stringify(volumes)}

df = pd.DataFrame({
    'Date': pd.to_datetime(dates),
    'Close': closes,
    'High': highs,
    'Low': lows,
    'Volume': volumes
})
df = df.set_index('Date')
symbol = "${symbol}"

print(f"Downloaded {len(df)} days of data")

# Calculate basic technical indicators
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Calculate RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Volatility
df['Volatility'] = df['Close'].rolling(window=20).std()

# Price momentum
df['Momentum'] = df['Close'].diff(10)

# Simple prediction based on technical indicators
latest = df.iloc[-1]
prev = df.iloc[-2]

# Scoring system
score = 0

# Moving averages
if latest['Close'] > latest['MA20']:
    score += 1
if latest['MA20'] > latest['MA50']:
    score += 1
if latest['MA5'] > latest['MA20']:
    score += 1

# RSI
if 30 < latest['RSI'] < 70:
    score += 1
elif latest['RSI'] < 30:
    score += 2  # Oversold, likely to go up

# MACD
if latest['MACD'] > latest['Signal']:
    score += 1

# Momentum
if latest['Momentum'] > 0:
    score += 1

# Price trend
if latest['Close'] > prev['Close']:
    score += 1

# Make prediction
total_indicators = 8
confidence = score / total_indicators
prediction = 1 if score >= 4 else 0

# Prepare results
result = {
    'symbol': symbol,
    'prediction': prediction,
    'confidence': confidence,
    'latest_price': float(latest['Close']),
    'change_pct': float((latest['Close'] - prev['Close']) / prev['Close'] * 100),
    'rsi': float(latest['RSI']) if not pd.isna(latest['RSI']) else 50,
    'macd': float(latest['MACD']) if not pd.isna(latest['MACD']) else 0,
    'ma20': float(latest['MA20']) if not pd.isna(latest['MA20']) else float(latest['Close']),
    'ma50': float(latest['MA50']) if not pd.isna(latest['MA50']) else float(latest['Close']),
    'score': score,
    'total_indicators': total_indicators
}

# Data for charts
chart_data = {
    'dates': df.index.strftime('%Y-%m-%d').tolist()[-60:],
    'close': df['Close'].tolist()[-60:],
    'ma20': df['MA20'].fillna(0).tolist()[-60:],
    'ma50': df['MA50'].fillna(0).tolist()[-60:]
}

import json
json.dumps({'result': result, 'chart_data': chart_data})
`;

        loadingMessage.textContent = 'Analyzing data with technical indicators...';

        const output = await pyodide.runPythonAsync(pythonCode);
        const data = JSON.parse(output);

        displayResults(data.result, data.chart_data);

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to analyze stock. Please check the symbol and try again.');
    } finally {
        predictBtn.disabled = false;
    }
}

function displayResults(result, chartData) {
    hideAllSections();
    resultSection.style.display = 'block';

    // Display prediction
    const predictionDiv = document.getElementById('predictionResult');

    if (result.prediction === 1) {
        predictionDiv.className = 'prediction-result prediction-up';
        predictionDiv.innerHTML = `
            📈 ${result.symbol}: Price likely to GO UP<br>
            <span style="font-size: 0.8em;">Confidence: ${(result.confidence * 100).toFixed(1)}%</span><br>
            <span style="font-size: 0.7em;">Current: $${result.latest_price.toFixed(2)} (${result.change_pct > 0 ? '+' : ''}${result.change_pct.toFixed(2)}%)</span>
        `;
    } else {
        predictionDiv.className = 'prediction-result prediction-down';
        predictionDiv.innerHTML = `
            📉 ${result.symbol}: Price likely to GO DOWN<br>
            <span style="font-size: 0.8em;">Confidence: ${(result.confidence * 100).toFixed(1)}%</span><br>
            <span style="font-size: 0.7em;">Current: $${result.latest_price.toFixed(2)} (${result.change_pct > 0 ? '+' : ''}${result.change_pct.toFixed(2)}%)</span>
        `;
    }

    // Display metrics
    const metricsDiv = document.getElementById('metricsResult');
    metricsDiv.textContent = `
Technical Indicators Summary:

Current Price: $${result.latest_price.toFixed(2)}
20-day MA: $${result.ma20.toFixed(2)}
50-day MA: $${result.ma50.toFixed(2)}

RSI (14): ${result.rsi.toFixed(2)} ${result.rsi < 30 ? '(Oversold)' : result.rsi > 70 ? '(Overbought)' : '(Neutral)'}
MACD: ${result.macd.toFixed(2)}

Bullish Indicators: ${result.score} / ${result.total_indicators}

Note: This is a simplified analysis based on technical indicators.
For investment decisions, please consult a financial advisor.
    `.trim();

    // Display chart
    displayPriceChart(chartData, result.symbol);
}

function displayPriceChart(chartData, symbol) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [{
                label: 'Close Price',
                data: chartData.close,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                borderWidth: 2
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
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} Stock Price (Last 60 Days)`,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
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

// Initialize Python when page loads
initPython();
