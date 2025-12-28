// Simplified version - No Python needed, pure JavaScript
let priceChart;

const predictBtn = document.getElementById('predictBtn');
const stockSymbolInput = document.getElementById('stockSymbol');
const timePeriodSelect = document.getElementById('timePeriod');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const loadingMessage = document.getElementById('loadingMessage');

// Ready immediately - no Python to load
predictBtn.disabled = false;
predictBtn.textContent = '🔮 Fetch & Predict';
document.getElementById('loadingPython').style.display = 'none';

predictBtn.addEventListener('click', handlePredict);
stockSymbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handlePredict();
});

async function handlePredict() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();
    if (!symbol) {
        showError('Please enter a stock symbol');
        return;
    }

    hideAllSections();
    loadingSection.style.display = 'block';
    loadingMessage.textContent = `Fetching data for ${symbol}...`;
    predictBtn.disabled = true;

    try {
        const period = timePeriodSelect.value;
        const stockData = await fetchStockData(symbol, period);

        if (!stockData || stockData.length === 0) {
            throw new Error(`No data found for ${symbol}`);
        }

        loadingMessage.textContent = 'Analyzing data...';
        const analysis = analyzeStock(stockData);
        displayResults(symbol, analysis, stockData);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to fetch data. Please try again.');
    } finally {
        predictBtn.disabled = false;
    }
}

async function fetchStockData(symbol, period) {
    const periodDays = { '3mo': 90, '6mo': 180, '1y': 365, '2y': 730 };
    const days = periodDays[period] || 180;

    // Use a reliable free API - Twelve Data (no API key needed for basic usage)
    try {
        const url = `https://api.twelvedata.com/time_series?symbol=${symbol}&interval=1day&outputsize=${Math.min(days, 5000)}&format=JSON`;

        const response = await fetch(url);
        const data = await response.json();

        if (data.status === 'error' || !data.values) {
            throw new Error('Failed to fetch stock data');
        }

        return data.values.reverse().map(item => ({
            date: item.datetime,
            open: parseFloat(item.open),
            high: parseFloat(item.high),
            low: parseFloat(item.low),
            close: parseFloat(item.close),
            volume: parseInt(item.volume)
        }));

    } catch (error) {
        console.error('Twelve Data failed, trying alternative...');

        // Fallback: Try Polygon.io aggregates (demo mode)
        try {
            const endDate = new Date();
            const startDate = new Date(endDate - days * 24 * 60 * 60 * 1000);
            const url = `https://api.polygon.io/v2/aggs/ticker/${symbol}/range/1/day/${startDate.toISOString().split('T')[0]}/${endDate.toISOString().split('T')[0]}?adjusted=true&sort=asc&apiKey=demo`;

            const response = await fetch(url);
            const data = await response.json();

            if (data.results && data.results.length > 0) {
                return data.results.map(item => ({
                    date: new Date(item.t).toISOString().split('T')[0],
                    open: item.o,
                    high: item.h,
                    low: item.l,
                    close: item.c,
                    volume: item.v
                }));
            }
        } catch (e) {
            console.error('Polygon also failed');
        }

        throw new Error(`Unable to fetch data for ${symbol}. Please check the symbol and try again.`);
    }
}

function analyzeStock(data) {
    const closes = data.map(d => d.close);
    const latest = closes[closes.length - 1];
    const previous = closes[closes.length - 2];

    // Calculate moving averages
    const ma5 = average(closes.slice(-5));
    const ma20 = average(closes.slice(-20));
    const ma50 = closes.length >= 50 ? average(closes.slice(-50)) : ma20;

    // Calculate RSI
    const rsi = calculateRSI(closes, 14);

    // Calculate MACD
    const macd = calculateMACD(closes);

    // Volatility (standard deviation of last 20 days)
    const volatility = stdDev(closes.slice(-20));

    // Momentum (10-day price change)
    const momentum = closes.length >= 10 ? latest - closes[closes.length - 11] : 0;

    // Scoring system
    let score = 0;
    const indicators = [];

    if (latest > ma20) { score++; indicators.push('Price > MA20'); }
    if (ma20 > ma50) { score++; indicators.push('MA20 > MA50'); }
    if (ma5 > ma20) { score++; indicators.push('MA5 > MA20'); }
    if (rsi > 30 && rsi < 70) { score++; indicators.push('RSI Neutral'); }
    if (rsi < 30) { score += 2; indicators.push('RSI Oversold (Bullish)'); }
    if (macd.histogram > 0) { score++; indicators.push('MACD Bullish'); }
    if (momentum > 0) { score++; indicators.push('Positive Momentum'); }
    if (latest > previous) { score++; indicators.push('Price Rising'); }

    const maxScore = 8;
    const confidence = score / maxScore;
    const prediction = score >= 4 ? 1 : 0;

    return {
        prediction,
        confidence,
        score,
        maxScore,
        latest,
        previous,
        change: ((latest - previous) / previous) * 100,
        ma5, ma20, ma50,
        rsi,
        macd: macd.macd,
        signal: macd.signal,
        volatility,
        momentum,
        indicators
    };
}

function calculateRSI(prices, period = 14) {
    if (prices.length < period + 1) return 50;

    const changes = [];
    for (let i = 1; i < prices.length; i++) {
        changes.push(prices[i] - prices[i - 1]);
    }

    const gains = changes.slice(-period).map(c => c > 0 ? c : 0);
    const losses = changes.slice(-period).map(c => c < 0 ? -c : 0);

    const avgGain = average(gains);
    const avgLoss = average(losses);

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}

function calculateMACD(prices) {
    const ema12 = EMA(prices, 12);
    const ema26 = EMA(prices, 26);
    const macdLine = ema12[ema12.length - 1] - ema26[ema26.length - 1];

    // Signal line is 9-period EMA of MACD
    const macdHistory = [];
    for (let i = 0; i < Math.min(prices.length, 26); i++) {
        if (ema12[i] !== undefined && ema26[i] !== undefined) {
            macdHistory.push(ema12[i] - ema26[i]);
        }
    }
    const signalLine = EMA(macdHistory, 9);
    const signal = signalLine[signalLine.length - 1];

    return {
        macd: macdLine,
        signal: signal,
        histogram: macdLine - signal
    };
}

function EMA(data, period) {
    const k = 2 / (period + 1);
    const ema = [data[0]];

    for (let i = 1; i < data.length; i++) {
        ema.push(data[i] * k + ema[i - 1] * (1 - k));
    }

    return ema;
}

function average(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdDev(arr) {
    const avg = average(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(average(squareDiffs));
}

function displayResults(symbol, analysis, stockData) {
    hideAllSections();
    resultSection.style.display = 'block';

    const predictionDiv = document.getElementById('predictionResult');
    if (analysis.prediction === 1) {
        predictionDiv.className = 'prediction-result prediction-up';
        predictionDiv.innerHTML = `
            📈 ${symbol}: Price likely to GO UP<br>
            <span style="font-size: 0.8em;">Confidence: ${(analysis.confidence * 100).toFixed(1)}%</span><br>
            <span style="font-size: 0.7em;">Current: $${analysis.latest.toFixed(2)} (${analysis.change > 0 ? '+' : ''}${analysis.change.toFixed(2)}%)</span>
        `;
    } else {
        predictionDiv.className = 'prediction-result prediction-down';
        predictionDiv.innerHTML = `
            📉 ${symbol}: Price likely to GO DOWN<br>
            <span style="font-size: 0.8em;">Confidence: ${(analysis.confidence * 100).toFixed(1)}%</span><br>
            <span style="font-size: 0.7em;">Current: $${analysis.latest.toFixed(2)} (${analysis.change > 0 ? '+' : ''}${analysis.change.toFixed(2)}%)</span>
        `;
    }

    // Display metrics
    const metricsDiv = document.getElementById('metricsResult');
    metricsDiv.textContent = `
Technical Indicators:

Current Price: $${analysis.latest.toFixed(2)}
5-day MA: $${analysis.ma5.toFixed(2)}
20-day MA: $${analysis.ma20.toFixed(2)}
50-day MA: $${analysis.ma50.toFixed(2)}

RSI (14): ${analysis.rsi.toFixed(2)} ${analysis.rsi < 30 ? '(Oversold 📈)' : analysis.rsi > 70 ? '(Overbought 📉)' : '(Neutral)'}
MACD: ${analysis.macd.toFixed(2)}
Signal: ${analysis.signal.toFixed(2)}
Histogram: ${(analysis.macd - analysis.signal).toFixed(2)} ${(analysis.macd - analysis.signal) > 0 ? '(Bullish ✓)' : '(Bearish ✗)'}

Volatility: ${analysis.volatility.toFixed(2)}
Momentum (10d): ${analysis.momentum.toFixed(2)}

Bullish Indicators: ${analysis.score} / ${analysis.maxScore}
Active Signals:
${analysis.indicators.map(i => '  • ' + i).join('\n')}

⚠️ Disclaimer: This is for educational purposes only.
Not financial advice. Always do your own research.
    `.trim();

    // Display chart
    displayChart(symbol, stockData, analysis);
}

function displayChart(symbol, data, analysis) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    const last60 = data.slice(-60);
    const dates = last60.map(d => d.date);
    const closes = last60.map(d => d.close);

    // Calculate MAs for chart
    const ma20Chart = [];
    const ma50Chart = [];

    for (let i = 0; i < last60.length; i++) {
        const closeSlice = data.slice(Math.max(0, data.indexOf(last60[i]) - 19), data.indexOf(last60[i]) + 1).map(d => d.close);
        ma20Chart.push(closeSlice.length >= 20 ? average(closeSlice) : null);

        const close50Slice = data.slice(Math.max(0, data.indexOf(last60[i]) - 49), data.indexOf(last60[i]) + 1).map(d => d.close);
        ma50Chart.push(close50Slice.length >= 50 ? average(close50Slice) : null);
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Close Price',
                data: closes,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                borderWidth: 2
            }, {
                label: '20-day MA',
                data: ma20Chart,
                borderColor: 'rgb(255, 159, 64)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false,
                borderWidth: 1.5
            }, {
                label: '50-day MA',
                data: ma50Chart,
                borderColor: 'rgb(153, 102, 255)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false,
                borderWidth: 1.5
            }]
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
