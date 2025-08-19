// Global variables
let selectedTeams = {
    team1: null,
    team2: null
};

let searchTimeouts = {};

// API base URL - adjust this to match your Flask server
const API_BASE_URL = '';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadPopularMatchups();
    checkModelStatus();
});

function initializeApp() {
    // Set current season
    const currentYear = new Date().getFullYear();
    document.getElementById('season').value = currentYear;
    
    // Set current week (estimate based on date)
    const currentDate = new Date();
    const seasonStart = new Date(currentYear, 7, 25); // Approximate season start
    const weeksDiff = Math.floor((currentDate - seasonStart) / (7 * 24 * 60 * 60 * 1000));
    const currentWeek = Math.max(1, Math.min(15, weeksDiff + 1));
    document.getElementById('week').value = currentWeek;
}

function setupEventListeners() {
    // Team search inputs
    document.getElementById('team1-search').addEventListener('input', (e) => {
        handleTeamSearch(e.target.value, 1);
    });
    
    document.getElementById('team2-search').addEventListener('input', (e) => {
        handleTeamSearch(e.target.value, 2);
    });
    
    // Focus events to show search results
    document.getElementById('team1-search').addEventListener('focus', (e) => {
        if (e.target.value.length >= 2) {
            document.getElementById('team1-results').classList.add('show');
        }
    });
    
    document.getElementById('team2-search').addEventListener('focus', (e) => {
        if (e.target.value.length >= 2) {
            document.getElementById('team2-results').classList.add('show');
        }
    });
    
    // Hide search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            hideAllSearchResults();
        }
    });
    
    // Game settings change
    ['week', 'season', 'location', 'season-type'].forEach(id => {
        document.getElementById(id).addEventListener('change', updatePredictButton);
    });
}

function handleTeamSearch(query, teamNumber) {
    const resultsContainer = document.getElementById(`team${teamNumber}-results`);
    
    // Clear previous timeout
    if (searchTimeouts[teamNumber]) {
        clearTimeout(searchTimeouts[teamNumber]);
    }
    
    if (query.length < 2) {
        resultsContainer.classList.remove('show');
        return;
    }
    
    // Debounce search
    searchTimeouts[teamNumber] = setTimeout(() => {
        searchTeams(query, teamNumber);
    }, 300);
}

async function searchTeams(query, teamNumber) {
    const resultsContainer = document.getElementById(`team${teamNumber}-results`);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/teams/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (response.ok) {
            displaySearchResults(data.teams, teamNumber);
        } else {
            console.error('Search error:', data.error);
            showToast('Search failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Network error:', error);
        showToast('Network error. Please check your connection.', 'error');
    }
}

function displaySearchResults(teams, teamNumber) {
    const resultsContainer = document.getElementById(`team${teamNumber}-results`);
    
    if (teams.length === 0) {
        resultsContainer.innerHTML = '<div class="search-result-item">No teams found</div>';
        resultsContainer.classList.add('show');
        return;
    }
    
    resultsContainer.innerHTML = teams.map(team => `
        <div class="search-result-item" onclick="selectTeam(${team.team_id}, ${teamNumber})">
            <div class="result-name">${team.name}</div>
            <div class="result-details">${team.location || ''} ${team.mascot ? '• ' + team.mascot : ''}</div>
        </div>
    `).join('');
    
    resultsContainer.classList.add('show');
}

async function selectTeam(teamId, teamNumber) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}`);
        const data = await response.json();
        
        if (response.ok) {
            selectedTeams[`team${teamNumber}`] = {
                ...data.team,
                stats: data.stats
            };
            displaySelectedTeam(data.team, data.stats, teamNumber);
            hideSearchResults(teamNumber);
            updatePredictButton();
        } else {
            console.error('Team fetch error:', data.error);
            showToast('Failed to load team data', 'error');
        }
    } catch (error) {
        console.error('Network error:', error);
        showToast('Network error. Please check your connection.', 'error');
    }
}

function displaySelectedTeam(team, stats, teamNumber) {
    const container = document.getElementById(`team${teamNumber}-selected`);
    const searchInput = document.getElementById(`team${teamNumber}-search`);
    
    // Update team info
    container.querySelector('.team-name').textContent = team.name;
    container.querySelector('.team-details').textContent = `${team.location || ''} ${team.mascot ? '• ' + team.mascot : ''}`;
    
    // Update team stats
    const statsContainer = container.querySelector('.team-stats');
    statsContainer.innerHTML = `
        <div class="team-stat">
            <span class="stat-label">Record</span>
            <span class="stat-value">${stats.wins}-${stats.losses}</span>
        </div>
        <div class="team-stat">
            <span class="stat-label">Avg Score</span>
            <span class="stat-value">${stats.avg_score.toFixed(1)}</span>
        </div>
        <div class="team-stat">
            <span class="stat-label">Home</span>
            <span class="stat-value">${stats.home_record}</span>
        </div>
        <div class="team-stat">
            <span class="stat-label">Away</span>
            <span class="stat-value">${stats.away_record}</span>
        </div>
    `;
    
    // Show selected team, hide search
    container.style.display = 'block';
    searchInput.style.display = 'none';
}

function clearTeam(teamNumber) {
    selectedTeams[`team${teamNumber}`] = null;
    
    const container = document.getElementById(`team${teamNumber}-selected`);
    const searchInput = document.getElementById(`team${teamNumber}-search`);
    
    container.style.display = 'none';
    searchInput.style.display = 'block';
    searchInput.value = '';
    
    updatePredictButton();
    hideSearchResults(teamNumber);
}

function hideSearchResults(teamNumber) {
    document.getElementById(`team${teamNumber}-results`).classList.remove('show');
}

function hideAllSearchResults() {
    hideSearchResults(1);
    hideSearchResults(2);
}

function updatePredictButton() {
    const predictBtn = document.getElementById('predict-btn');
    const canPredict = selectedTeams.team1 && selectedTeams.team2 && 
                      selectedTeams.team1.team_id && selectedTeams.team2.team_id;
    
    predictBtn.disabled = !canPredict;
    
    if (canPredict) {
        predictBtn.innerHTML = '<i class="fas fa-magic"></i> Make Prediction';
        predictBtn.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
    } else {
        predictBtn.innerHTML = '<i class="fas fa-magic"></i> Select Two Teams';
        predictBtn.style.background = 'rgba(71, 85, 105, 0.5)';
    }
}

async function makePrediction() {
    if (!selectedTeams.team1 || !selectedTeams.team2 || 
        !selectedTeams.team1.team_id || !selectedTeams.team2.team_id) {
        showToast('Please select both teams first', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        const gameInfo = {
            team1_id: selectedTeams.team1.team_id,
            team2_id: selectedTeams.team2.team_id,
            week: parseInt(document.getElementById('week').value),
            season: parseInt(document.getElementById('season').value),
            season_type: parseInt(document.getElementById('season-type').value)
        };
        
        // Set home/away based on location setting
        const location = document.getElementById('location').value;
        if (location === 'home') {
            gameInfo.team1_home_away = 'home';
            gameInfo.team2_home_away = 'away';
        } else if (location === 'away') {
            gameInfo.team1_home_away = 'away';
            gameInfo.team2_home_away = 'home';
        } else {
            gameInfo.team1_home_away = 'neutral';
            gameInfo.team2_home_away = 'neutral';
            gameInfo.neutral_site = 1;
        }
        
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(gameInfo)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayPredictionResults(data.prediction);
            showToast('Prediction complete!', 'success');
        } else {
            console.error('Prediction error:', data.error);
            showToast('Prediction failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Network error:', error);
        showToast('Network error. Please check your connection.', 'error');
    } finally {
        showLoading(false);
    }
}

function displayPredictionResults(prediction) {
    const resultsSection = document.getElementById('prediction-results');
    
    // Update team cards
    updateTeamResultCard('result-team1', prediction.team1, prediction.prediction_details.winner === prediction.team1.name);
    updateTeamResultCard('result-team2', prediction.team2, prediction.prediction_details.winner === prediction.team2.name);
    
    // Update prediction summary
    document.getElementById('winner-text').textContent = `${prediction.prediction_details.winner} Wins`;
    document.getElementById('margin').textContent = `${prediction.prediction_details.margin} points`;
    document.getElementById('total').textContent = `${prediction.prediction_details.total_points} points`;
    
    const confidenceElement = document.getElementById('confidence');
    confidenceElement.textContent = prediction.prediction_details.confidence;
    confidenceElement.className = `value confidence ${prediction.prediction_details.confidence}`;
    
    // Update detailed stats
    updateStatsComparison(prediction.team1.stats, prediction.team2.stats);
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function updateTeamResultCard(cardId, team, isWinner) {
    const card = document.getElementById(cardId);
    
    card.querySelector('.team-name').textContent = team.name;
    card.querySelector('.team-record').textContent = `${team.stats.wins}-${team.stats.losses}`;
    card.querySelector('.predicted-score').textContent = team.predicted_score;
    
    // Add winner styling
    if (isWinner) {
        card.classList.add('winner');
    } else {
        card.classList.remove('winner');
    }
}

function updateStatsComparison(team1Stats, team2Stats) {
    document.getElementById('team1-wins').textContent = team1Stats.wins;
    document.getElementById('team2-wins').textContent = team2Stats.wins;
    
    document.getElementById('team1-losses').textContent = team1Stats.losses;
    document.getElementById('team2-losses').textContent = team2Stats.losses;
    
    document.getElementById('team1-avg-score').textContent = team1Stats.avg_score.toFixed(1);
    document.getElementById('team2-avg-score').textContent = team2Stats.avg_score.toFixed(1);
    
    document.getElementById('team1-avg-allowed').textContent = team1Stats.avg_allowed.toFixed(1);
    document.getElementById('team2-avg-allowed').textContent = team2Stats.avg_allowed.toFixed(1);
    
    document.getElementById('team1-home').textContent = team1Stats.home_record;
    document.getElementById('team2-home').textContent = team2Stats.home_record;
    
    document.getElementById('team1-away').textContent = team1Stats.away_record;
    document.getElementById('team2-away').textContent = team2Stats.away_record;
}

async function loadPopularMatchups() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/popular-matchups`);
        const data = await response.json();
        
        if (response.ok) {
            displayPopularMatchups(data.suggested_matchups);
        }
    } catch (error) {
        console.error('Failed to load popular matchups:', error);
    }
}

function displayPopularMatchups(matchups) {
    const container = document.getElementById('popular-matchups-grid');
    
    container.innerHTML = matchups.map(matchup => `
        <div class="matchup-card" onclick="loadPopularMatchup(${matchup.team1.id}, ${matchup.team2.id})">
            <div class="matchup-teams">
                <span class="matchup-team">${matchup.team1.name}</span>
                <span class="matchup-vs">VS</span>
                <span class="matchup-team">${matchup.team2.name}</span>
            </div>
            <div class="matchup-rivalry">${matchup.rivalry}</div>
        </div>
    `).join('');
}

async function loadPopularMatchup(team1Id, team2Id) {
    try {
        // Clear existing selections
        clearTeam(1);
        clearTeam(2);
        
        // Load both teams
        const [team1Response, team2Response] = await Promise.all([
            fetch(`${API_BASE_URL}/api/teams/${team1Id}`),
            fetch(`${API_BASE_URL}/api/teams/${team2Id}`)
        ]);
        
        const team1Data = await team1Response.json();
        const team2Data = await team2Response.json();
        
        if (team1Response.ok && team2Response.ok) {
            selectedTeams.team1 = {
                ...team1Data.team,
                stats: team1Data.stats
            };
            selectedTeams.team2 = {
                ...team2Data.team,
                stats: team2Data.stats
            };
            
            displaySelectedTeam(team1Data.team, team1Data.stats, 1);
            displaySelectedTeam(team2Data.team, team2Data.stats, 2);
            
            updatePredictButton();
            
            // Scroll to team selection
            document.querySelector('.team-selection').scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        console.error('Failed to load popular matchup:', error);
        showToast('Failed to load matchup', 'error');
    }
}

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model/status`);
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('model-status').textContent = data.model_loaded ? 'Ready' : 'Loading';
            
            if (data.model_trained_at) {
                const trainedDate = new Date(data.model_trained_at);
                document.getElementById('last-updated').textContent = formatDate(trainedDate);
            }
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
        document.getElementById('model-status').textContent = 'Unknown';
    }
}

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'flex' : 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 4000);
    
    // Remove on click
    toast.addEventListener('click', () => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    });
}

function formatDate(date) {
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffTime / (1000 * 60));
    
    if (diffDays > 0) {
        return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
        return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffMinutes > 0) {
        return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
    } else {
        return 'Just now';
    }
}

// Utility functions for quick team selection (for testing)
function quickSelectTeams(team1Name, team2Name) {
    searchAndSelect(team1Name, 1);
    setTimeout(() => searchAndSelect(team2Name, 2), 500);
}

async function searchAndSelect(teamName, teamNumber) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/teams/search?q=${encodeURIComponent(teamName)}`);
        const data = await response.json();
        
        if (response.ok && data.teams.length > 0) {
            await selectTeam(data.teams[0].team_id, teamNumber);
        }
    } catch (error) {
        console.error('Quick select failed:', error);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Escape to clear search results
    if (e.key === 'Escape') {
        hideAllSearchResults();
    }
    
    // Enter to make prediction if teams are selected
    if (e.key === 'Enter' && !e.target.closest('.team-search') && 
        selectedTeams.team1 && selectedTeams.team2 && 
        selectedTeams.team1.team_id && selectedTeams.team2.team_id) {
        makePrediction();
    }
    
    // Quick team selection for testing (Ctrl+1, Ctrl+2, etc.)
    if (e.ctrlKey && e.key >= '1' && e.key <= '9') {
        e.preventDefault();
        const presets = [
            ['Alabama', 'Georgia'],
            ['Michigan', 'Ohio State'],
            ['Texas', 'Oklahoma'],
            ['USC', 'Notre Dame'],
            ['Florida State', 'Miami'],
            ['Oregon', 'Washington'],
            ['Penn State', 'Michigan State'],
            ['LSU', 'Auburn'],
            ['Clemson', 'South Carolina']
        ];
        
        const presetIndex = parseInt(e.key) - 1;
        if (presets[presetIndex]) {
            quickSelectTeams(presets[presetIndex][0], presets[presetIndex][1]);
        }
    }
});

// Auto-refresh model status every 5 minutes
setInterval(checkModelStatus, 5 * 60 * 1000);

// Performance monitoring
window.addEventListener('load', function() {
    const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
    console.log(`Page load time: ${loadTime}ms`);
});

// Error handling for uncaught errors
window.addEventListener('error', function(e) {
    console.error('Uncaught error:', e.error);
    showToast('An unexpected error occurred', 'error');
});

// Service worker registration for potential offline functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment if you want to add a service worker later
        // navigator.serviceWorker.register('/sw.js');
    });
}

// Utility function to export prediction data
function exportPrediction() {
    if (!selectedTeams.team1 || !selectedTeams.team2) {
        showToast('No prediction data to export', 'error');
        return;
    }
    
    const predictionData = {
        team1: selectedTeams.team1.name,
        team2: selectedTeams.team2.name,
        prediction: {
            team1_score: document.getElementById('result-team1').querySelector('.predicted-score').textContent,
            team2_score: document.getElementById('result-team2').querySelector('.predicted-score').textContent,
            winner: document.getElementById('winner-text').textContent,
            margin: document.getElementById('margin').textContent,
            total: document.getElementById('total').textContent,
            confidence: document.getElementById('confidence').textContent
        },
        timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(predictionData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `prediction_${selectedTeams.team1.name}_vs_${selectedTeams.team2.name}_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    showToast('Prediction data exported', 'success');
}

// Add share functionality
function sharePrediction() {
    if (!selectedTeams.team1 || !selectedTeams.team2) {
        showToast('No prediction to share', 'error');
        return;
    }
    
    const team1Score = document.getElementById('result-team1').querySelector('.predicted-score').textContent;
    const team2Score = document.getElementById('result-team2').querySelector('.predicted-score').textContent;
    const winner = document.getElementById('winner-text').textContent;
    
    const shareText = `🏈 CFB Prediction: ${selectedTeams.team1.name} ${team1Score} - ${team2Score} ${selectedTeams.team2.name}. ${winner}! #CollegeFootball #CFBPredict`;
    
    if (navigator.share) {
        navigator.share({
            title: 'College Football Prediction',
            text: shareText,
            url: window.location.href
        });
    } else {
        // Fallback to clipboard
        navigator.clipboard.writeText(shareText).then(() => {
            showToast('Prediction copied to clipboard!', 'success');
        }).catch(() => {
            showToast('Failed to copy to clipboard', 'error');
        });
    }
}

// Analytics tracking (placeholder for future implementation)
function trackEvent(eventName, eventData = {}) {
    // Placeholder for analytics tracking
    console.log('Event:', eventName, eventData);
}

// Track user interactions
document.addEventListener('click', function(e) {
    if (e.target.closest('.predict-btn')) {
        trackEvent('prediction_made', {
            team1: selectedTeams.team1?.name,
            team2: selectedTeams.team2?.name
        });
    } else if (e.target.closest('.matchup-card')) {
        trackEvent('popular_matchup_selected');
    }
});
