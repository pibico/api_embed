#!/usr/bin/env python3
"""
Embedding & Search web interface following pibiCo guidelines.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def web_interface():
    """
    Serve pibiCo-branded Embedding & Search web interface at root path.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#4682B4">
    <meta name="description" content="Multi-tenant embedding and semantic search service by pibiCo">
    <title>Embedding & Search - pibiCo AI Services</title>

    <!-- Favicon -->
    <link rel="icon" type="image/svg+xml" href="/static/pibico_icon.svg">

    <!-- Styles -->
    <link rel="stylesheet" href="/static/css/bootstrap-icons.min.css">
    <link rel="stylesheet" href="/static/css/pibico.css">

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', Arial, sans-serif;
            background: linear-gradient(135deg, #2c5171 0%, #4682b4 50%, #6a9bc3 100%);
            min-height: 100vh;
            color: #1A1A2E;
        }

        /* Navbar Card */
        .navbar {
            position: fixed;
            top: 8px;
            left: 12px;
            right: 12px;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 1.5rem;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            background: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(70, 130, 180, 0.15),
                        0 0 4px rgba(70, 130, 180, 0.08);
            line-height: 1.1;
        }

        .navbar-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', Arial, sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4682B4 0%, #B33A2B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .navbar-nav {
            display: flex;
            flex-direction: row;
            gap: 0.5rem;
            align-items: center;
            list-style: none;
            margin: 0;
            padding: 0;
        }

        .btn-icon {
            width: 34px;
            height: 34px;
            border: none;
            border-radius: 8px;
            background: #E8E8EA;
            color: #4682B4;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .btn-icon:hover {
            background: #D6E8F5;
            color: #365F8A;
        }

        /* Main Content */
        .app-main {
            margin-top: 60px;
            margin-bottom: 44px;
            padding: 12px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Tab Navigation */
        .tab-nav {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .tab-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.85);
            color: #4682B4;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .tab-btn.active {
            background: #4682B4;
            color: #FFFFFF;
        }

        .tab-btn:hover:not(.active) {
            background: #D6E8F5;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Cards */
        .pibico-card {
            background: #FFFFFF;
            border: none;
            border-radius: 12px;
            padding: 24px;
            line-height: 1.2;
            box-shadow: 0 0 12px rgba(70, 130, 180, 0.15),
                        0 0 4px rgba(70, 130, 180, 0.08);
            transition: box-shadow 0.2s ease;
            margin-bottom: 20px;
        }

        .pibico-card:hover {
            box-shadow: 0 0 20px rgba(70, 130, 180, 0.25),
                        0 0 8px rgba(70, 130, 180, 0.12);
        }

        .card-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', Arial, sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: #1A1A2E;
            margin-bottom: 12px;
        }

        .pibico-input, .pibico-textarea {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', Arial, sans-serif;
            font-size: 0.9rem;
            padding: 10px 12px;
            border: none;
            border-radius: 8px;
            background: #E8E8EA;
            color: #1A1A2E;
            width: 100%;
            outline: none;
            transition: box-shadow 0.2s ease;
        }

        .pibico-textarea {
            min-height: 120px;
            resize: vertical;
            font-family: monospace;
        }

        .pibico-input:focus, .pibico-textarea:focus {
            box-shadow: 0 0 0 2px #4682B4;
            background: #FFFFFF;
        }

        .form-group {
            margin-bottom: 16px;
        }

        .form-label {
            display: block;
            font-weight: 500;
            margin-bottom: 6px;
            color: #1A1A2E;
        }

        .pibico-btn {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', Arial, sans-serif;
            font-weight: 500;
            font-size: 0.85rem;
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            line-height: 1.1;
        }

        .pibico-btn-primary {
            background: #4682B4;
            color: #FFFFFF;
        }

        .pibico-btn-primary:hover {
            background: #365F8A;
        }

        .pibico-btn-danger {
            background: #B33A2B;
            color: #FFFFFF;
        }

        .pibico-btn-danger:hover {
            background: #8A2A1F;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(70, 130, 180, 0.3);
            border-radius: 50%;
            border-top-color: #4682B4;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Tables */
        .pibico-table {
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', Arial, sans-serif;
            font-size: 0.85rem;
            line-height: 1.1;
        }

        .pibico-table th {
            background: linear-gradient(135deg, #4682B4 0%, #365F8A 100%);
            color: #FFFFFF;
            font-weight: 600;
            padding: 8px 12px;
            text-align: left;
        }

        .pibico-table td {
            padding: 6px 12px;
            border-bottom: 1px solid #E8E8EA;
        }

        .pibico-table tr:hover td {
            background: #D6E8F5;
        }

        .result-item {
            padding: 12px;
            margin: 8px 0;
            background: #F8F9FA;
            border-radius: 8px;
            border-left: 4px solid #4682B4;
        }

        .result-score {
            display: inline-block;
            padding: 2px 8px;
            background: #4682B4;
            color: #FFFFFF;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 70px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 400px;
        }

        .toast {
            background: #FFFFFF;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideIn 0.3s ease;
            border-left: 4px solid #4682B4;
        }

        .toast.success { border-left-color: #28a745; }
        .toast.error { border-left-color: #B33A2B; }
        .toast.warning { border-left-color: #FFC107; }

        .toast-icon { font-size: 1.5rem; flex-shrink: 0; }
        .toast.success .toast-icon { color: #28a745; }
        .toast.error .toast-icon { color: #B33A2B; }
        .toast.warning .toast-icon { color: #FFC107; }

        .toast-content {
            flex: 1;
            font-size: 0.9rem;
            color: #1A1A2E;
        }

        .toast-close {
            background: none;
            border: none;
            color: #6E6E76;
            cursor: pointer;
            font-size: 1.2rem;
            padding: 0;
            line-height: 1;
        }

        @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Footer Card */
        .app-footer {
            position: fixed;
            bottom: 8px;
            left: 12px;
            right: 12px;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.4rem 1.5rem;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            background: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(70, 130, 180, 0.15),
                        0 0 4px rgba(70, 130, 180, 0.08);
            font-size: 0.75rem;
            color: #6E6E76;
        }

        .footer-separator { color: #C8C8CC; }

        /* Slide Panel */
        .pibico-panel-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(26, 26, 46, 0.4);
            z-index: 999;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }

        .pibico-panel-backdrop.active {
            opacity: 1;
            visibility: visible;
        }

        .pibico-panel {
            position: fixed;
            top: 0;
            right: 0;
            width: 45%;
            height: 100vh;
            background: #FFFFFF;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            overflow-y: auto;
            box-shadow: -4px 0 24px rgba(26, 26, 46, 0.15);
        }

        .pibico-panel.active {
            transform: translateX(0);
        }

        .pibico-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: linear-gradient(135deg, #4682B4 0%, #365F8A 100%);
            color: #FFFFFF;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 1;
        }

        .pibico-panel-close {
            background: none;
            border: none;
            color: #FFFFFF;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 6px;
            line-height: 1;
        }

        .pibico-panel-close:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .pibico-panel-body {
            padding: 20px;
        }

        @media (max-width: 768px) {
            .pibico-panel { width: 100%; }
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span class="navbar-title">Embedding & Search</span>
        </div>
        <ul class="navbar-nav">
            <li><button class="btn-icon" onclick="openSettings()" title="Settings"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></button></li>
            <li><a href="/embed/api/v1/docs" style="color: #4682B4; text-decoration: none; font-weight: 500; padding: 0.35rem 0.75rem; border-radius: 8px; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s ease;" target="_blank" title="API Docs" onmouseover="this.style.background='#D6E8F5'" onmouseout="this.style.background=''"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg> API</a></li>
        </ul>
    </nav>

    <!-- Main Content -->
    <main class="app-main">
        <!-- Tab Navigation -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('index')">
                <i class="bi bi-database-add"></i> Index Documents
            </button>
            <button class="tab-btn" onclick="switchTab('search')">
                <i class="bi bi-search"></i> Semantic Search
            </button>
            <button class="tab-btn" onclick="switchTab('libraries')">
                <i class="bi bi-collection"></i> Libraries
            </button>
        </div>

        <!-- Index Tab -->
        <div id="indexTab" class="tab-content active">
            <div class="pibico-card">
                <h2 class="card-title">
                    <i class="bi bi-database-add"></i> Index Document
                </h2>
                <p style="margin-bottom: 20px; color: #6E6E76;">
                    Add documents to multi-tenant vector index
                </p>

                <div class="form-group">
                    <label class="form-label">Site</label>
                    <input type="text" id="indexSite" class="pibico-input" placeholder="e.g., dev.pibico.es">
                </div>
                <div class="form-group">
                    <label class="form-label">Library ID</label>
                    <input type="text" id="indexLibraryId" class="pibico-input" placeholder="e.g., APPS_LIBRARY">
                </div>
                <div class="form-group">
                    <label class="form-label">Document Name</label>
                    <input type="text" id="indexDocName" class="pibico-input" placeholder="e.g., user_manual.md">
                </div>
                <div class="form-group">
                    <label class="form-label">Content</label>
                    <textarea id="indexContent" class="pibico-textarea" placeholder="Enter document content..."></textarea>
                </div>

                <button class="pibico-btn pibico-btn-primary" onclick="indexDocument()">
                    <i class="bi bi-check-circle"></i> Index Document
                </button>
            </div>
        </div>

        <!-- Search Tab -->
        <div id="searchTab" class="tab-content">
            <div class="pibico-card">
                <h2 class="card-title">
                    <i class="bi bi-search"></i> Semantic Search
                </h2>
                <p style="margin-bottom: 20px; color: #6E6E76;">
                    Search indexed documents using natural language
                </p>

                <div class="form-group">
                    <label class="form-label">Site</label>
                    <input type="text" id="searchSite" class="pibico-input" placeholder="e.g., dev.pibico.es">
                </div>
                <div class="form-group">
                    <label class="form-label">Library ID</label>
                    <input type="text" id="searchLibraryId" class="pibico-input" placeholder="e.g., APPS_LIBRARY">
                </div>
                <div class="form-group">
                    <label class="form-label">Query</label>
                    <input type="text" id="searchQuery" class="pibico-input" placeholder="Enter search query...">
                </div>
                <div class="form-group">
                    <label class="form-label">Results (k)</label>
                    <input type="number" id="searchK" class="pibico-input" value="10" min="1" max="100">
                </div>

                <button class="pibico-btn pibico-btn-primary" onclick="performSearch()">
                    <i class="bi bi-search"></i> Search
                </button>

                <div id="searchResults" style="margin-top: 24px; display: none;">
                    <h3 style="font-weight: 600; margin-bottom: 12px;">Search Results</h3>
                    <div id="searchResultsContent"></div>
                </div>
            </div>
        </div>

        <!-- Libraries Tab -->
        <div id="librariesTab" class="tab-content">
            <div class="pibico-card">
                <h2 class="card-title">
                    <i class="bi bi-collection"></i> Indexed Libraries
                </h2>
                <p style="margin-bottom: 20px; color: #6E6E76;">
                    View all indexed libraries and their statistics
                </p>

                <button class="pibico-btn pibico-btn-primary" onclick="loadLibraries()">
                    <i class="bi bi-arrow-clockwise"></i> Refresh
                </button>

                <div id="librariesContent" style="margin-top: 24px;"></div>
            </div>
        </div>
    </main>

    <!-- Toast Container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Footer -->
    <footer class="app-footer">
        <span><svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" fill="none" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9.354a4 4 0 1 0 0 5.292" stroke-linecap="round"/></svg> pibiCo 2026</span>
        <span class="footer-separator">|</span>
        <a href="/embed/api/v1/redoc" target="_blank" style="color: #4682B4; text-decoration: none; display: inline-flex; align-items: center; gap: 4px;"><svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg> ReDoc</a>
        <span class="footer-separator">|</span>
        <span><svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg> v2.1.0</span>
    </footer>

    <!-- Settings Panel -->
    <div class="pibico-panel-backdrop" id="settingsBackdrop" onclick="closeSettings()"></div>
    <div class="pibico-panel" id="settingsPanel">
        <div class="pibico-panel-header">
            <span><i class="bi bi-gear"></i> Settings</span>
            <button class="pibico-panel-close" onclick="closeSettings()">&times;</button>
        </div>
        <div class="pibico-panel-body">
            <h3 style="margin-bottom: 16px; font-weight: 600;">Authentication</h3>
            <div class="form-group">
                <label class="form-label">API Key</label>
                <input type="password" id="settingsApiKey" class="pibico-input" placeholder="Enter your API Key">
            </div>

            <button class="pibico-btn pibico-btn-primary" onclick="saveSettings()">
                <i class="bi bi-check-circle"></i> Save Settings
            </button>
        </div>
    </div>

    <script>
        let apiKey = sessionStorage.getItem('embed_api_key') || '';

        // Toast notification system
        function showToast(message, type = 'info', duration = 4000) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast ' + type;

            const icons = {
                success: '<i class="bi bi-check-circle-fill"></i>',
                error: '<i class="bi bi-x-circle-fill"></i>',
                warning: '<i class="bi bi-exclamation-triangle-fill"></i>'
            };

            const iconDiv = document.createElement('div');
            iconDiv.className = 'toast-icon';
            iconDiv.innerHTML = icons[type] || icons.info;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'toast-content';
            contentDiv.textContent = message;

            const closeBtn = document.createElement('button');
            closeBtn.className = 'toast-close';
            closeBtn.innerHTML = '&times;';
            closeBtn.onclick = function() { this.parentElement.remove(); };

            toast.appendChild(iconDiv);
            toast.appendChild(contentDiv);
            toast.appendChild(closeBtn);
            container.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, duration);
        }

        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

            // Show selected tab
            document.getElementById(tabName + 'Tab').classList.add('active');
            event.target.classList.add('active');

            // Auto-load libraries when switching to libraries tab
            if (tabName === 'libraries') {
                loadLibraries();
            }
        }

        async function indexDocument() {
            if (!apiKey) {
                showToast('Please configure your API key in Settings first', 'warning');
                openSettings();
                return;
            }

            const site = document.getElementById('indexSite').value.trim();
            const library_id = document.getElementById('indexLibraryId').value.trim();
            const doc_name = document.getElementById('indexDocName').value.trim();
            const content = document.getElementById('indexContent').value.trim();

            if (!site || !library_id || !doc_name || !content) {
                showToast('All fields are required', 'warning');
                return;
            }

            try {
                const response = await fetch('/embed/api/v1/index', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': apiKey
                    },
                    body: JSON.stringify({
                        site: site,
                        library_id: library_id,
                        doc_name: doc_name,
                        content: content,
                        metadata: {}
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    showToast(`Document indexed successfully! Total: ${result.total_documents} docs`, 'success');

                    // Clear form
                    document.getElementById('indexDocName').value = '';
                    document.getElementById('indexContent').value = '';
                } else {
                    const error = await response.json();
                    showToast('Error: ' + (error.detail || 'Indexing failed'), 'error');
                }
            } catch (error) {
                showToast('Error: ' + error.message, 'error');
            }
        }

        async function performSearch() {
            if (!apiKey) {
                showToast('Please configure your API key in Settings first', 'warning');
                openSettings();
                return;
            }

            const site = document.getElementById('searchSite').value.trim();
            const library_id = document.getElementById('searchLibraryId').value.trim();
            const query = document.getElementById('searchQuery').value.trim();
            const k = parseInt(document.getElementById('searchK').value);

            if (!site || !library_id || !query) {
                showToast('Site, Library ID, and Query are required', 'warning');
                return;
            }

            try {
                const response = await fetch('/embed/api/v1/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': apiKey
                    },
                    body: JSON.stringify({
                        site: site,
                        library_id: library_id,
                        query: query,
                        k: k
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    displaySearchResults(result);
                    showToast(`Found ${result.total} results`, 'success');
                } else {
                    const error = await response.json();
                    showToast('Error: ' + (error.detail || 'Search failed'), 'error');
                }
            } catch (error) {
                showToast('Error: ' + error.message, 'error');
            }
        }

        function displaySearchResults(result) {
            const container = document.getElementById('searchResultsContent');
            const resultsDiv = document.getElementById('searchResults');

            if (!result.results || result.results.length === 0) {
                container.innerHTML = '<p style="color: #6E6E76;">No results found</p>';
                resultsDiv.style.display = 'block';
                return;
            }

            container.innerHTML = '';
            result.results.forEach((item, index) => {
                const div = document.createElement('div');
                div.className = 'result-item';
                div.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong>${item.doc_name}</strong>
                        <span class="result-score">${(item.similarity * 100).toFixed(1)}%</span>
                    </div>
                    <p style="margin: 0; color: #6E6E76; font-size: 0.85rem;">${item.text_snippet || 'No snippet available'}</p>
                `;
                container.appendChild(div);
            });

            resultsDiv.style.display = 'block';
        }

        async function loadLibraries() {
            if (!apiKey) {
                showToast('Please configure your API key in Settings first', 'warning');
                openSettings();
                return;
            }

            const container = document.getElementById('librariesContent');
            container.innerHTML = '<div style="text-align: center; padding: 20px;"><span class="spinner"></span></div>';

            try {
                const response = await fetch('/embed/api/v1/libraries', {
                    headers: {
                        'X-API-Key': apiKey
                    }
                });

                if (response.ok) {
                    const result = await response.json();
                    displayLibraries(result);
                } else {
                    const error = await response.json();
                    container.innerHTML = '<p style="color: #B33A2B;">Error loading libraries: ' + (error.detail || 'Unknown error') + '</p>';
                }
            } catch (error) {
                container.innerHTML = '<p style="color: #B33A2B;">Error: ' + error.message + '</p>';
            }
        }

        function displayLibraries(result) {
            const container = document.getElementById('librariesContent');

            if (!result.libraries || result.libraries.length === 0) {
                container.innerHTML = '<p style="color: #6E6E76;">No libraries indexed yet</p>';
                return;
            }

            const table = document.createElement('table');
            table.className = 'pibico-table';
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Site</th>
                        <th>Library ID</th>
                        <th>Documents</th>
                        <th>DB Size</th>
                        <th>Index Size</th>
                        <th>Last Modified</th>
                    </tr>
                </thead>
                <tbody id="librariesTableBody"></tbody>
            `;

            const tbody = table.querySelector('tbody');
            result.libraries.forEach(lib => {
                const row = document.createElement('tr');
                const lastMod = new Date(lib.last_modified * 1000).toLocaleString();
                const dbSize = (lib.db_size / 1024).toFixed(1) + ' KB';
                const indexSize = (lib.index_size / 1024).toFixed(1) + ' KB';

                row.innerHTML = `
                    <td>${lib.site}</td>
                    <td><strong>${lib.library_id}</strong></td>
                    <td>${lib.document_count}</td>
                    <td>${dbSize}</td>
                    <td>${indexSize}</td>
                    <td>${lastMod}</td>
                `;
                tbody.appendChild(row);
            });

            container.innerHTML = '';
            container.appendChild(table);
        }

        function openSettings() {
            document.getElementById('settingsPanel').classList.add('active');
            document.getElementById('settingsBackdrop').classList.add('active');
            document.getElementById('settingsApiKey').value = apiKey;
        }

        function closeSettings() {
            document.getElementById('settingsPanel').classList.remove('active');
            document.getElementById('settingsBackdrop').classList.remove('active');
        }

        function saveSettings() {
            apiKey = document.getElementById('settingsApiKey').value.trim();
            if (apiKey) {
                sessionStorage.setItem('embed_api_key', apiKey);
                closeSettings();
                showToast('Settings saved successfully!', 'success');
            } else {
                showToast('Please enter an API key', 'warning');
            }
        }

        // Close panels on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeSettings();
            }
        });
    </script>
</body>
</html>
    """

    return HTMLResponse(content=html_content)
