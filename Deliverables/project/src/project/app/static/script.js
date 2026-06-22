document.addEventListener('DOMContentLoaded', () => {
    // --- Global State & Config ---
    let priceChart;
    let isPlaying = false;
    let simulationSpeed = 1;
    let lastPrice = null;
    let totalTestDataPoints = 0;
    let currentIndex = 0;

    // --- DOM Element Cache ---
    const elements = {
        loadingOverlay: document.getElementById('loading-overlay'),
        statusIndicator: document.getElementById('status-indicator'),
        statusText: document.getElementById('status-text'),
        currentTime: document.getElementById('current-time'),
        playPauseBtn: document.getElementById('playPauseBtn'),
        resetBtn: document.getElementById('resetBtn'),
        speedSelect: document.getElementById('speedSelect'),
        currentPrice: document.getElementById('currentPrice'),
        priceChange: document.getElementById('priceChange'),
        currentVolume: document.getElementById('currentVolume'),
        anomalyRisk: document.getElementById('anomalyRisk'),
        anomalyLog: document.getElementById('anomalyLog'),
        priceChartCanvas: document.getElementById('priceChart'),
    };

    // --- Main Initialization ---
    async function initDashboard() {
        try {
            console.log("Initializing dashboard...");
            setupEventListeners();
            
            const response = await fetch('/api/init_data');
            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            
            const data = await response.json();
            console.log("Initial data received.");

            totalTestDataPoints = data.test_data_count;
            initChart(data.train_data);
            updateMarketData(data.initial_test_point, null); // Pass null for prediction initially
            updateClock(data.initial_test_point.Date); // Initialize clock

            elements.loadingOverlay.style.display = 'none';
            console.log("Dashboard initialized successfully.");

        } catch (error) {
            console.error("Fatal Error during initialization:", error);
            elements.loadingOverlay.innerHTML = `<div class="text-center"><i class="fas fa-exclamation-triangle text-4xl text-red-500"></i><p class="mt-4 text-lg">Failed to load dashboard.</p><p class="text-sm text-gray-400">Please check console (F12) for details.</p></div>`;
        }
    }

    // --- Charting ---
    function downsampleData(data, maxPoints = 2000) {
        if (data.length <= maxPoints) return data;
        const downsampled = [];
        const step = Math.ceil(data.length / maxPoints);
        for (let i = 0; i < data.length; i += step) {
            downsampled.push(data[i]);
        }
        return downsampled;
    }

    function initChart(trainData) {
        const displayableTrainData = downsampleData(trainData);
        const historicalAnomaliesData = trainData
            .filter(d => d.event && d.event !== 'normal')
            .map(d => ({ x: new Date(d.Date).valueOf(), y: d.Close }));

        console.log(`Found and plotted ${historicalAnomaliesData.length} historical anomalies.`);

        const datasets = {
            train: {
                label: 'Historical Price',
                data: displayableTrainData.map(d => ({ x: new Date(d.Date).valueOf(), y: d.Close })),
                borderColor: 'rgba(100, 100, 100, 0.7)',
                borderWidth: 1,
                pointRadius: 0,
            },
            live: {
                label: 'Live Price',
                data: [],
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            },
            historicalAnomaly: {
                label: 'Historical Anomalies',
                data: historicalAnomaliesData,
                pointStyle: 'star',
                pointRadius: 8,
                pointBorderWidth: 2,
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.7)',
                showLine: false,
            },
            liveAnomaly: {
                label: 'Live Anomaly',
                data: [],
                pointStyle: 'crossRot',
                pointRadius: 10,
                pointBorderWidth: 3,
                borderColor: 'rgba(239, 68, 68, 1)',
                backgroundColor: 'rgba(239, 68, 68, 0.7)',
                showLine: false,
            }
        };

        const chartConfig = {
            type: 'line',
            data: { datasets: Object.values(datasets) },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { intersect: false, mode: 'index' },
                scales: {
                    x: { type: 'time', ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
                },
                plugins: {
                    legend: { labels: { color: '#d1d5db' } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}` } },
                    zoom: {
                        pan: { enabled: true, mode: 'x' },
                        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }
                    }
                }
            }
        };
        priceChart = new Chart(elements.priceChartCanvas.getContext('2d'), chartConfig);
    }

    function updateChart(dataPoint, prediction) {
        if (!priceChart) return;
        const newTimestamp = new Date(dataPoint.Date).valueOf();
        const newPoint = { x: newTimestamp, y: dataPoint.Close };
        
        priceChart.data.datasets[1].data.push(newPoint); // Live data
        
        if (prediction?.is_anomaly) {
            priceChart.data.datasets[3].data.push(newPoint); // Live anomaly
        }
        
        const panRange = 100 * 60 * 1000; // Keep a 100-minute viewing window
        priceChart.options.scales.x.min = newTimestamp - panRange;
        priceChart.options.scales.x.max = newTimestamp;
        
        const liveData = priceChart.data.datasets[1].data;
        if(liveData.length > 2000) {
            liveData.shift();
            const liveAnomalyData = priceChart.data.datasets[3].data;
            if(liveAnomalyData.length > 0 && liveAnomalyData[0].x < liveData[0].x) {
                liveAnomalyData.shift();
            }
        }
        
        priceChart.update('none');
    }

    // --- Simulation Control ---
    function setupEventListeners() {
        elements.playPauseBtn.addEventListener('click', togglePlayPause);
        elements.resetBtn.addEventListener('click', resetSimulation);
        elements.speedSelect.addEventListener('change', updateSpeed);
    }

    async function togglePlayPause() {
        isPlaying = !isPlaying;
        updatePlayPauseButton();

        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: isPlaying ? 'play' : 'pause' })
        });
        
        if (isPlaying) {
            updateStatusDisplay('LIVE', 'green');
            advanceSimulation();
        } else {
            updateStatusDisplay('PAUSED', 'yellow');
        }
    }
    
    async function resetSimulation() {
        if (isPlaying) await togglePlayPause();
        
        console.log("Resetting simulation...");
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'reset' })
        });
        
        currentIndex = 0;
        priceChart.data.datasets[1].data = []; // Clear live data
        priceChart.data.datasets[3].data = []; // Clear live anomalies
        priceChart.options.scales.x.min = undefined; // Reset pan
        priceChart.options.scales.x.max = undefined;
        priceChart.resetZoom();
        priceChart.update('none');
        
        elements.anomalyLog.querySelectorAll('.log-entry:not(.welcome)').forEach(e => e.remove());
        
        const response = await fetch('/api/init_data');
        const data = await response.json();
        updateMarketData(data.initial_test_point, null);
        updateClock(data.initial_test_point.Date);
        lastPrice = null; // Reset price change calculation
        updateStatusDisplay('STANDBY', 'gray');
        console.log("Simulation reset.");
    }
    
    function updateSpeed() {
        simulationSpeed = Number(elements.speedSelect.value);
    }

    async function advanceSimulation() {
        if (!isPlaying) return;

        if (currentIndex >= totalTestDataPoints) {
            isPlaying = false;
            updatePlayPauseButton();
            addLogEntry({ title: "Simulation Complete", message: "Reached the end of the test dataset.", type: "info" });
            updateStatusDisplay('FINISHED', 'indigo');
            return;
        }

        try {
            const response = await fetch('/api/next_point');
            const result = await response.json();
            
            if (result.status === 'running') {
                currentIndex = result.index;
                updateUI(result.data, result.prediction);
            } else if (result.status === 'complete' || result.status === 'paused'){
                isPlaying = false;
                updatePlayPauseButton();
                if (result.status === 'complete') {
                    addLogEntry({ title: "Simulation Complete", message: "Server indicated end of data.", type: "info" });
                    updateStatusDisplay('FINISHED', 'indigo');
                }
            } else {
                throw new Error(result.message || 'Unknown server error');
            }
        } catch (error) {
            console.error("Error during simulation step:", error);
            addLogEntry({ title: "Simulation Error", message: error.message, type: "anomaly" });
            isPlaying = false;
            updatePlayPauseButton();
        }

        if (isPlaying) {
            setTimeout(advanceSimulation, 1000 / simulationSpeed);
        }
    }

    // --- UI Updates ---
    function updateUI(dataPoint, prediction) {
        updateChart(dataPoint, prediction);
        updateMarketData(dataPoint, prediction);
        updateClock(dataPoint.Date);
        
        // ** COMPLIANCE: Check for anomaly and log with full explanation **
        if (prediction?.is_anomaly) {
            addLogEntry({
                type: 'anomaly',
                title: prediction.explicit_label || 'AI-Generated Prediction', // Use explicit label from backend
                reason: prediction.reason || 'No specific factor provided.', // Use understandable explanation
                score: prediction.score,
                price: dataPoint.Close,
                timestamp: dataPoint.Date
            });
        }
    }
    
    function updateMarketData(dataPoint, prediction) {
        if(!dataPoint) return;
        const price = dataPoint.Close;
        
        elements.currentPrice.textContent = `$${price.toFixed(2)}`;
        elements.currentVolume.textContent = Number(dataPoint.Volume).toLocaleString();

        if (lastPrice !== null) {
            const change = price - lastPrice;
            const percent = (change / lastPrice * 100);
            elements.priceChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${percent.toFixed(2)}%)`;
            elements.priceChange.className = `stat-value ${change >= 0 ? 'text-green-400' : 'text-red-400'}`;
        }
        lastPrice = price;

        // Update Anomaly Risk card based on probability from backend
        if (prediction && typeof prediction.anomaly_probability === 'number') {
            const risk = (prediction.anomaly_probability * 100);
            elements.anomalyRisk.textContent = `${risk.toFixed(1)}%`;
            if (risk > 70) elements.anomalyRisk.className = 'stat-value text-red-400 animate-pulse';
            else if (risk > 40) elements.anomalyRisk.className = 'stat-value text-yellow-400';
            else elements.anomalyRisk.className = 'stat-value text-green-400';
        } else {
            elements.anomalyRisk.textContent = `0.0%`;
            elements.anomalyRisk.className = 'stat-value text-green-400';
        }
    }

    // ** COMPLIANCE: Centralized function for logging AI outputs **
    function addLogEntry(logData) {
        const entry = document.createElement('div');
        const logTime = new Date(logData.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const iconClass = logData.type === 'anomaly' ? 'fa-exclamation-triangle' : 'fa-info-circle';
        
        let messageHtml;
        if (logData.type === 'anomaly') {
            // Fulfills "Explicit Labeling" and "Understandable Explanations"
            messageHtml = `
                <p class="font-bold">${logData.title}: High Risk Detected
                    <span class="text-xs text-gray-400 font-mono ml-2">${logTime}</span>
                </p>
                <p class="text-sm text-gray-300 mt-1">
                    <span class="font-semibold">Reason:</span> ${logData.reason}
                </p>
                <p class="text-xs text-gray-400 mt-1">
                    Details: Score: ${logData.score?.toFixed(4)} | Price: $${logData.price?.toFixed(2)}
                </p>
            `;
        } else {
            // For general info logs
            messageHtml = `
                <p class="font-bold">${logData.title}
                    <span class="text-xs text-gray-400 font-mono ml-2">${logTime || ''}</span>
                </p>
                <p class="text-sm text-gray-300">${logData.message}</p>
            `;
        }
        
        entry.className = `log-entry ${logData.type}`;
        entry.innerHTML = `
            <div class="log-icon ${logData.type}"><i class="fas ${iconClass}"></i></div>
            <div>${messageHtml}</div>
        `;
        // Insert new logs after the "welcome" message
        elements.anomalyLog.insertBefore(entry, elements.anomalyLog.firstChild.nextSibling);
    }
    
    function updatePlayPauseButton() {
        const icon = elements.playPauseBtn.querySelector('i');
        const text = elements.playPauseBtn.querySelector('span');
        if (isPlaying) {
            icon.className = 'fas fa-pause';
            text.textContent = ' Pause';
            elements.playPauseBtn.classList.remove('bg-blue-500', 'hover:bg-blue-600');
            elements.playPauseBtn.classList.add('bg-yellow-500', 'hover:bg-yellow-600');
        } else {
            icon.className = 'fas fa-play';
            text.textContent = ' Play';
            elements.playPauseBtn.classList.remove('bg-yellow-500', 'hover:bg-yellow-600');
            elements.playPauseBtn.classList.add('bg-blue-500', 'hover:bg-blue-600');
        }
    }

    function updateStatusDisplay(text, color) {
        elements.statusText.textContent = text;
        const colorMap = {
            green: 'bg-green-500', yellow: 'bg-yellow-500',
            gray: 'bg-gray-500', indigo: 'bg-indigo-500',
            red: 'bg-red-500' // Added for errors
        };
        elements.statusIndicator.className = `w-3 h-3 rounded-full ${colorMap[color] || 'bg-gray-500'} transition-colors`;
        
        if(text === 'LIVE') elements.statusIndicator.classList.add('animate-ping');
        else elements.statusIndicator.classList.remove('animate-ping');
    }

    function updateClock(timestampStr) {
        const date = timestampStr ? new Date(timestampStr) : new Date();
        elements.currentTime.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    // --- Start the application ---
    initDashboard();
});