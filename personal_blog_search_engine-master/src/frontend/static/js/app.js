// Constants
const MAX_HISTORY_ITEMS = 5;
const DEBOUNCE_DELAY = 300;

// DOM Elements
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const searchFiltersBtn = document.getElementById('search-filters');
const advancedFilters = document.getElementById('advanced-filters');
const searchDropdown = document.querySelector('.search-dropdown');
const searchHistoryList = document.getElementById('search-history-list');
const resultsContainer = document.getElementById('search-results');
const loadingIndicator = document.getElementById('loading');
const noResultsMessage = document.getElementById('no-results');
const resultsCount = document.getElementById('results-count');
const viewToggle = document.getElementById('view-toggle');
const sortToggle = document.getElementById('sort-toggle');
const sortMenu = document.querySelector('.sort-menu');
const themeToggle = document.getElementById('theme-toggle');
const suggestionTags = document.querySelectorAll('.suggestion-tag');

// State
let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];
let currentView = localStorage.getItem('viewMode') || 'grid';
let currentSort = localStorage.getItem('sortMode') || 'relevance';
let isLoading = false;

// Check for saved theme preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    document.body.setAttribute('data-theme', savedTheme);
    themeToggle.checked = savedTheme === 'dark';
}

// Theme toggle
themeToggle.addEventListener('change', () => {
    const theme = themeToggle.checked ? 'dark' : 'light';
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
});

// Check search engine status
checkStatus();

// Event listeners
searchForm.addEventListener('submit', handleSearch);
searchInput.addEventListener('input', debounce(handleInput, 300));

// Handle suggestion tags
suggestionTags.forEach(tag => {
    tag.addEventListener('click', (e) => {
        e.preventDefault();
        searchInput.value = tag.textContent;
        handleSearch(new Event('submit'));
    });
});

/**
 * Check the status of the search engine
 */
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (!data.index_built) {
            console.warn('Search index not built yet. Some features may be limited.');
        }
        
        console.log(`Search engine ready. ${data.documents_indexed} documents indexed.`);
    } catch (error) {
        console.error('Error checking search engine status:', error);
    }
}

/**
 * Handle the search form submission
 * @param {Event} e - The submit event
 */
async function handleSearch(e) {
    e.preventDefault();
    
    const query = searchInput.value.trim();
    if (!query) return;
    
    // Show loading state
    resultsContainer.innerHTML = '';
    resultsCount.innerHTML = '';
    loadingIndicator.classList.remove('hidden');
    noResultsMessage.classList.add('hidden');
    
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                filters: {
                    dateRange: 'any',
                    sortBy: currentSort,
                    contentType: 'all'
                }
            })
        });
        
        const data = await response.json();
        
        // Hide loading state
        loadingIndicator.classList.add('hidden');
        
        if (data.error) {
            console.error('Search error:', data.error);
            showNoResults();
            return;
        }
        
        if (data.results && data.results.length > 0) {
            displayResults(data.results);
            addToSearchHistory(query);
        } else {
            showNoResults();
        }
    } catch (error) {
        console.error('Error performing search:', error);
        loadingIndicator.classList.add('hidden');
        showNoResults();
    }
}

/**
 * Handle input changes for potential auto-suggestions
 * @param {Event} e - The input event
 */
function handleInput(e) {
    const query = e.target.value.trim();
    if (query.length < 2) return;
    
    // Here you could add auto-suggestion functionality
    // For now, we'll just update the UI to show the input is being processed
    searchInput.style.backgroundImage = 'var(--gradient)';
}

/**
 * Display search results
 * @param {Array} results - The search results
 */
function displayResults(results) {
    if (!results || results.length === 0) {
        showNoResults();
        return;
    }
    
    // Update results count with animation
    resultsCount.style.opacity = '0';
    setTimeout(() => {
        resultsCount.textContent = `Found ${results.length} result${results.length !== 1 ? 's' : ''}`;
        resultsCount.style.opacity = '1';
    }, 150);
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    // Create result items with staggered animation
    results.forEach((result, index) => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.style.animationDelay = `${index * 0.1}s`;
        
        resultCard.innerHTML = `
            <h3><a href="${result.url}" target="_blank">${result.title}</a></h3>
            <p class="result-url">${formatUrl(result.url)}</p>
            <p class="result-snippet">${result.snippet}</p>
            <div class="result-meta">
                <span><i class="far fa-calendar"></i> ${formatDate(result.date)}</span>
                ${result.readTime ? `<span><i class="far fa-clock"></i> ${result.readTime} min read</span>` : ''}
            </div>
        `;
        
        resultsContainer.appendChild(resultCard);
    });
    
    resultsContainer.classList.remove('hidden');
    noResultsMessage.classList.add('hidden');
}

/**
 * Show the "no results" message with animation
 */
function showNoResults() {
    resultsContainer.innerHTML = '';
    resultsCount.textContent = '';
    noResultsMessage.style.opacity = '0';
    noResultsMessage.classList.remove('hidden');
    requestAnimationFrame(() => {
        noResultsMessage.style.opacity = '1';
    });
}

/**
 * Format a URL for display
 * @param {string} url - The URL to format
 * @returns {string} The formatted URL
 */
function formatUrl(url) {
    try {
        const urlObj = new URL(url);
        let formattedUrl = urlObj.hostname;
        
        if (urlObj.pathname !== '/') {
            // Truncate pathname if it's too long
            const pathname = urlObj.pathname.length > 30 
                ? urlObj.pathname.substring(0, 30) + '...' 
                : urlObj.pathname;
            formattedUrl += pathname;
        }
        
        return formattedUrl;
    } catch (e) {
        return url;
    }
}

/**
 * Format a date for display
 * @param {string} dateString - The date string to format
 * @returns {string} The formatted date
 */
function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch {
        return dateString;
    }
}

/**
 * Debounce function to limit the rate at which a function is called
 * @param {Function} func - The function to debounce
 * @param {number} wait - The debounce delay in milliseconds
 * @returns {Function} The debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Search History Management
function addToSearchHistory(query) {
    if (!query.trim()) return;
    
    // Remove existing instance of the query if it exists
    searchHistory = searchHistory.filter(item => item.toLowerCase() !== query.toLowerCase());
    
    // Add new query to the beginning
    searchHistory.unshift(query);
    
    // Keep only the most recent items
    if (searchHistory.length > MAX_HISTORY_ITEMS) {
        searchHistory = searchHistory.slice(0, MAX_HISTORY_ITEMS);
    }
    
    // Save to localStorage
    localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
    
    // Update UI
    updateSearchHistoryUI();
}

function updateSearchHistoryUI() {
    searchHistoryList.innerHTML = '';
    searchHistory.forEach(query => {
        const li = document.createElement('li');
        li.className = 'search-history-item';
        li.innerHTML = `
            <i class="fas fa-history"></i>
            <span>${query}</span>
        `;
        li.addEventListener('click', () => {
            searchInput.value = query;
            searchDropdown.classList.add('hidden');
            handleSearch(new Event('submit'));
        });
        searchHistoryList.appendChild(li);
    });
}

// Search Functionality
async function performSearch(query, filters = {}) {
    if (!query.trim()) return;
    
    isLoading = true;
    updateUIState();
    
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                filters: {
                    dateRange: document.getElementById('date-filter')?.value || 'any',
                    sortBy: currentSort,
                    contentType: document.getElementById('type-filter')?.value || 'all',
                    numResults: 10
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Search error:', data.error);
            showError(data.error);
            return;
        }
        
        if (data.results && data.results.length > 0) {
            displayResults(data.results);
            addToSearchHistory(query);
        } else {
            showNoResults();
        }
    } catch (error) {
        console.error('Search error:', error);
        showError(error.message);
    } finally {
        isLoading = false;
        updateUIState();
    }
}

// UI State Management
function updateUIState() {
    loadingIndicator.classList.toggle('hidden', !isLoading);
    resultsContainer.classList.toggle('hidden', isLoading);
    
    if (isLoading) {
        noResultsMessage.classList.add('hidden');
    }
}

function showError(message = 'An error occurred while searching. Please try again.') {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <p>${message}</p>
    `;
    resultsContainer.innerHTML = '';
    resultsContainer.appendChild(errorDiv);
    resultsCount.textContent = '';
}

// Event Listeners
searchInput.addEventListener('focus', () => {
    searchDropdown.classList.remove('hidden');
});

document.addEventListener('click', (e) => {
    if (!searchDropdown.contains(e.target) && !searchInput.contains(e.target)) {
        searchDropdown.classList.add('hidden');
    }
});

searchFiltersBtn.addEventListener('click', () => {
    advancedFilters.classList.toggle('hidden');
});

viewToggle.addEventListener('click', () => {
    currentView = currentView === 'grid' ? 'list' : 'grid';
    localStorage.setItem('viewMode', currentView);
    resultsContainer.classList.toggle('grid-view', currentView === 'grid');
    resultsContainer.classList.toggle('list-view', currentView === 'list');
    viewToggle.innerHTML = `<i class="fas fa-${currentView === 'grid' ? 'th-large' : 'list'}"></i>`;
});

sortToggle.addEventListener('click', () => {
    sortMenu.classList.toggle('hidden');
});

document.querySelectorAll('.sort-menu a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        currentSort = e.target.dataset.sort;
        localStorage.setItem('sortMode', currentSort);
        sortMenu.classList.add('hidden');
        if (searchInput.value.trim()) {
            performSearch(searchInput.value.trim());
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    updateSearchHistoryUI();
    resultsContainer.classList.add(`${currentView}-view`);
});

// Debounced search input handler for live search
searchInput.addEventListener('input', debounce((e) => {
    const query = e.target.value.trim();
    if (query.length >= 2) {
        performSearch(query);
    }
}, DEBOUNCE_DELAY));

// Theme Management
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
        themeToggle.checked = savedTheme === 'dark';
    } else {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.body.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        themeToggle.checked = prefersDark;
    }
} 