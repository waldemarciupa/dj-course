document.addEventListener('DOMContentLoaded', () => {
    displayStats();
});

function displayStats() {
    document.title = 'Time Statistics - Developer Distractor Destroyer';

    const timeStatsList = document.getElementById('statsList');
    const timeChartCanvas = document.getElementById('timeChart').getContext('2d');
    const clearTimeStatsBtn = document.getElementById('clearTimeStats');
    let timeChart = null;

    const gotchaStatsList = document.getElementById('gotchaList');
    const gotchaChartCanvas = document.getElementById('gotchaChart').getContext('2d');
    const clearGotchaStatsBtn = document.getElementById('clearGotchaStats');
    let gotchaChart = null;

    let intervalId = null;
    let gotchaFilter = 'allTime'; // 'today', 'thisWeek', 'thisMonth', 'allTime'

    function formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function updateStats() {
        chrome.storage.local.get(['timeData', 'blockEvents'], (result) => {
            // Time Stats (existing logic)
            timeStatsList.innerHTML = '';
            const timeData = result.timeData || {};
            const sortedTimeSites = Object.entries(timeData).sort((a, b) => b[1] - a[1]);
    
            if (sortedTimeSites.length === 0) {
                timeStatsList.innerHTML = '<div class="stat-item">No time tracking data yet.</div>';
                document.getElementById('timeChart').style.display = 'none';
            } else {
                document.getElementById('timeChart').style.display = 'block';
                sortedTimeSites.forEach(([site, time]) => {
                    const statItem = createStatItem(site, formatTime(time), 'timeData', timeChart, timeStatsList);
                    timeStatsList.appendChild(statItem);
                });
                renderPieChart(sortedTimeSites);
            }

            // "Gotcha" Stats (refactored logic)
            gotchaStatsList.innerHTML = '';
            const blockEvents = result.blockEvents || [];
            const filteredEvents = filterBlockEvents(blockEvents, gotchaFilter);
            const gotchaData = aggregateBlockEvents(filteredEvents);
            const sortedGotchaSites = Object.entries(gotchaData).sort((a, b) => b[1] - a[1]);

            if (sortedGotchaSites.length === 0) {
                gotchaStatsList.innerHTML = '<div class="stat-item">No "gotcha" data for this period.</div>';
                document.getElementById('gotchaChart').style.display = 'none';
            } else {
                document.getElementById('gotchaChart').style.display = 'block';
                sortedGotchaSites.forEach(([site, count]) => {
                    const statItem = createStatItem(site, `${count} times`, 'blockEvents', gotchaChart, gotchaStatsList);
                    gotchaStatsList.appendChild(statItem);
                });
                renderGotchaChart(sortedGotchaSites);
            }
        });
    }

    function filterBlockEvents(events, filter) {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        
        switch (filter) {
            case 'today':
                return events.filter(event => new Date(event.timestamp) >= today);
            case 'thisWeek':
                const firstDayOfWeek = new Date(today);
                firstDayOfWeek.setDate(today.getDate() - today.getDay());
                return events.filter(event => new Date(event.timestamp) >= firstDayOfWeek);
            case 'thisMonth':
                const firstDayOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
                return events.filter(event => new Date(event.timestamp) >= firstDayOfMonth);
            case 'allTime':
            default:
                return events;
        }
    }

    function aggregateBlockEvents(events) {
        return events.reduce((acc, event) => {
            acc[event.domain] = (acc[event.domain] || 0) + 1;
            return acc;
        }, {});
    }

    function createStatItem(site, value, statType, chart, listElement) {
        const statItem = document.createElement('div');
        statItem.className = 'stat-item';
        statItem.dataset.site = site;
    
        if (chart) {
            const index = chart.data.labels.indexOf(site);
            if (index !== -1 && !chart.getDataVisibility(index)) {
                statItem.classList.add('disabled');
            }
        }
    
        const siteText = document.createElement('span');
        siteText.textContent = site;
    
        const valueContainer = document.createElement('div');
        valueContainer.className = 'value-container';
    
        const valueText = document.createElement('span');
        valueText.textContent = value;
    
        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-stat-btn';
        deleteBtn.textContent = '❌';
    
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (confirm(`Are you sure you want to delete stats for "${site}"?`)) {
                removeStatEntry(statType, site);
            }
        });
    
        valueContainer.appendChild(valueText);
        valueContainer.appendChild(deleteBtn);
    
        statItem.appendChild(siteText);
        statItem.appendChild(valueContainer);
    
        statItem.addEventListener('click', () => {
            if (!chart) return;
            const index = chart.data.labels.indexOf(site);
            if (index !== -1) {
                chart.toggleDataVisibility(index);
                chart.update();
                statItem.classList.toggle('disabled', !chart.getDataVisibility(index));
            }
        });
    
        return statItem;
    }
    

    function removeStatEntry(statType, siteToRemove) {
        chrome.storage.local.get([statType], (result) => {
            let stats = result[statType];
            if (statType === 'blockEvents') {
                stats = stats.filter(event => event.domain !== siteToRemove);
            } else { // timeData
                delete stats[siteToRemove];
            }
            
            let dataToSet = {};
            dataToSet[statType] = stats;
            chrome.storage.local.set(dataToSet, updateStats);
        });
    }

    function renderPieChart(data) {
        const labels = data.map(item => item[0]);
        const values = data.map(item => item[1]);

        if (timeChart) {
            timeChart.data.labels = labels;
            timeChart.data.datasets[0].data = values;
            timeChart.update();
            return;
        }

        timeChart = new Chart(timeChartCanvas, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Time Spent (seconds)',
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: 'white'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += formatTime(context.parsed);
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderGotchaChart(data) {
        const labels = data.map(item => item[0]);
        const values = data.map(item => item[1]);

        if (gotchaChart) {
            gotchaChart.data.labels = labels;
            gotchaChart.data.datasets[0].data = values;
            gotchaChart.update();
            return;
        }

        gotchaChart = new Chart(gotchaChartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '"Gotcha" Count',
                    data: values,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: 'white'
                        }
                    },
                    y: {
                        ticks: {
                            color: 'white'
                        }
                    }
                }
            }
        });
    }

    clearTimeStatsBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all time statistics? This cannot be undone.')) {
            chrome.storage.local.set({ timeData: {}, currentSessionTime: 0 }, () => {
                if (timeChart) {
                    timeChart.destroy();
                    timeChart = null;
                }
                updateStats();
            });
        }
    });

    clearGotchaStatsBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all "gotcha" statistics? This cannot be undone.')) {
            chrome.storage.local.set({ blockEvents: [] }, () => {
                if (gotchaChart) {
                    gotchaChart.destroy();
                    gotchaChart = null;
                }
                updateStats();
            });
        }
    });

    // Filter button event listeners
    document.getElementById('today').addEventListener('click', () => setGotchaFilter('today'));
    document.getElementById('thisWeek').addEventListener('click', () => setGotchaFilter('thisWeek'));
    document.getElementById('thisMonth').addEventListener('click', () => setGotchaFilter('thisMonth'));
    document.getElementById('allTime').addEventListener('click', () => setGotchaFilter('allTime'));

    function setGotchaFilter(filter) {
        gotchaFilter = filter;
        document.querySelectorAll('.button-group button').forEach(btn => btn.classList.remove('active'));
        document.getElementById(filter).classList.add('active');
        updateStats();
    }

    // Initial update
    updateStats();

    // Set up auto-refresh
    intervalId = setInterval(updateStats, 5000);

    // Clean up the interval when the page is hidden
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearInterval(intervalId);
        } else {
            // Initial update on visibility change
            updateStats();
            intervalId = setInterval(updateStats, 5000);
        }
    });
}