const chatMessages = document.getElementById('chatMessages');
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const statusBadge = document.getElementById('statusBadge');
const docList = document.getElementById('docList');
const chunkCount = document.getElementById('chunkCount');
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');


const apiKeyInput = document.getElementById('apiKeyInput');
const apiKeySaveBtn = document.getElementById('apiKeySaveBtn');
const apiKeyStatus = document.getElementById('apiKeyStatus');

const btnShowChat = document.getElementById('btnShowChat');
const btnShowPipeline = document.getElementById('btnShowPipeline');
const chatView = document.getElementById('chatView');
const pipelineView = document.getElementById('pipelineView');

let isProcessing = false;
let serverHasApiKey = false;

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadApiKey();
    
    queryForm.addEventListener('submit', handleSubmit);
    apiKeySaveBtn.addEventListener('click', saveApiKey);
    apiKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { 
            e.preventDefault(); 
            saveApiKey(); 
        }
    });

    document.querySelectorAll('.suggestion').forEach(el => {
        el.addEventListener('click', () => {
            queryInput.value = el.dataset.query;
            // Force a tab switch to Chat view if suggesting
            switchView('chat');
            handleSubmit(new Event('submit', { cancelable: true }));
        });
    });

    menuToggle.addEventListener('click', toggleSidebar);
    
    // Tab switching
    btnShowChat.addEventListener('click', () => switchView('chat'));
    btnShowPipeline.addEventListener('click', () => switchView('pipeline'));
});

// View switching logic
function switchView(view) {
    if (view === 'chat') {
        btnShowChat.classList.add('active');
        btnShowPipeline.classList.remove('active');
        chatView.classList.add('active');
        pipelineView.classList.remove('active');
    } else {
        btnShowChat.classList.remove('active');
        btnShowPipeline.classList.add('active');
        chatView.classList.remove('active');
        pipelineView.classList.add('active');
    }
}


// API Key overrides
function loadApiKey() {
    const key = localStorage.getItem('groq_api_key') || '';
    if (key) {
        apiKeyInput.value = key;
        updateApiKeyUI(true);
    } else {
        updateApiKeyUI(false);
    }
}

function saveApiKey() {
    const key = apiKeyInput.value.trim();
    if (!key) {
        localStorage.removeItem('groq_api_key');
        updateApiKeyUI(false);
        return;
    }
    localStorage.setItem('groq_api_key', key);
    updateApiKeyUI(true);
}

function updateApiKeyUI(hasCustomKey) {
    apiKeyStatus.className = 'api-key-status';
    if (hasCustomKey) {
        const key = localStorage.getItem('groq_api_key') || '';
        apiKeyStatus.textContent = `Custom Key Active (${key.slice(0, 6)}...)`;
        apiKeyStatus.classList.add('override');
    } else {
        if (serverHasApiKey) {
            apiKeyStatus.textContent = 'Server Key Active (from .env)';
            apiKeyStatus.classList.add('active');
        } else {
            apiKeyStatus.textContent = 'No API key set (Server or Custom)';
        }
    }
}

function getApiKey() {
    return localStorage.getItem('groq_api_key') || '';
}

async function checkHealth() {
    try {
        const data = await (await fetch('/api/health')).json();
        if (data.status === 'healthy') {
            serverHasApiKey = !!data.has_api_key;
            setStatus('online', `Grounded Engine Online — ${data.chunks} chunks`);
            chunkCount.textContent = `${data.chunks} chunks loaded across ${data.documents} knowledge base documents.`;
            
            // Re-render API key status after we know the server config
            updateApiKeyUI(!!getApiKey());
            
            docList.innerHTML = '';
            (data.document_names || []).forEach(name => {
                const li = document.createElement('li');
                li.className = 'doc-item';
                li.textContent = name.replace('.md', '').replace('.txt', '').replace(/_/g, ' ');
                docList.appendChild(li);
            });
        }
    } catch { 
        setStatus('error', 'Connecting to backend...'); 
    }
}

function setStatus(status, text) {
    const dot = statusBadge.querySelector('.status-dot');
    dot.className = 'status-dot';
    if (status === 'online') dot.classList.add('online');
    Array.from(statusBadge.childNodes).filter(n => n.nodeType === 3).forEach(n => n.remove());
    statusBadge.appendChild(document.createTextNode(` ${text}`));
}

async function handleSubmit(e) {
    e.preventDefault();
    if (isProcessing) return;
    const question = queryInput.value.trim();
    if (!question) return;

    const customApiKey = getApiKey();
    // Validate: if neither custom key nor server key is available, warn the user.
    if (!customApiKey && !serverHasApiKey) {
        const welcome = chatMessages.querySelector('.welcome-message');
        if (welcome) welcome.remove();
        appendMsg('assistant', '⚠️ Please enter a custom Groq API key in the sidebar. No server-side API key was found in the environment (.env file).', true);
        return;
    }

    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    appendMsg('user', question);
    queryInput.value = '';
    setProcessing(true);
    const typingEl = appendTyping();

    try {
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question, 
                top_k: 5, 
                api_key: customApiKey 
            }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        typingEl.remove();
        appendAnswer(data);
    } catch (err) {
        typingEl.remove();
        appendMsg('assistant', `⚠️ Error processing query: ${err.message}`, true);
    } finally { 
        setProcessing(false); 
    }
}

function setProcessing(state) {
    isProcessing = state;
    submitBtn.disabled = state;
    queryInput.disabled = state;
    if (!state) queryInput.focus();
}

function appendMsg(role, text, isError = false) {
    const msg = el('div', `message ${role}`);
    const avatar = el('div', 'message-avatar');
    avatar.textContent = role === 'user' ? 'U' : 'AI';
    const content = el('div', 'message-content');
    const bubble = el('div', 'message-bubble');
    
    bubble.innerHTML = isError ? `<div class="message-error">${esc(text)}</div>` :
        role === 'user' ? esc(text) : fmt(text);
        
    content.appendChild(bubble);
    msg.append(avatar, content);
    chatMessages.appendChild(msg);
    scroll();
}

function appendAnswer(data) {
    const msg = el('div', 'message assistant');
    const avatar = el('div', 'message-avatar');
    avatar.textContent = 'AI';
    const content = el('div', 'message-content');
    const bubble = el('div', 'message-bubble');
    bubble.innerHTML = fmt(data.answer);
    content.appendChild(bubble);

    if (data.retrieved_chunks?.length) {
        const section = el('div', 'context-section');
        const toggle = el('button', 'context-toggle');
        toggle.innerHTML = `📄 Retrieved Context (${data.retrieved_chunks.length} chunks) <span class="arrow">▼</span>`;
        const chunksDiv = el('div', 'context-chunks');

        data.retrieved_chunks.forEach(c => {
            const card = el('div', 'chunk-card');
            
            // Format metrics
            const denseMetric = `Dense Rank: #${c.dense_rank} (score: ${c.dense_score})`;
            const sparseMetric = c.sparse_rank 
                ? `Sparse Rank: #${c.sparse_rank} (score: ${c.sparse_score})`
                : 'Sparse: No keyword match (score: 0)';
            const rrfMetric = `RRF Score: ${c.rrf_score}`;

            card.innerHTML = `
                <div class="chunk-header">
                    <span class="chunk-source">${esc(c.source.replace(/_/g, ' '))}</span>
                </div>
                <div class="chunk-text">${esc(c.text)}</div>
                <div class="chunk-metrics">
                    <span class="badge dense">${denseMetric}</span>
                    <span class="badge sparse">${sparseMetric}</span>
                    <span class="badge rrf">${rrfMetric}</span>
                </div>
            `;
            chunksDiv.appendChild(card);
        });

        toggle.addEventListener('click', () => { 
            toggle.classList.toggle('open'); 
            chunksDiv.classList.toggle('open'); 
        });
        section.append(toggle, chunksDiv);
        content.appendChild(section);
    }

    msg.append(avatar, content);
    chatMessages.appendChild(msg);
    scroll();
}

function appendTyping() {
    const msg = el('div', 'message assistant');
    const avatar = el('div', 'message-avatar');
    avatar.textContent = 'AI';
    const content = el('div', 'message-content');
    const bubble = el('div', 'message-bubble');
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    content.appendChild(bubble);
    msg.append(avatar, content);
    chatMessages.appendChild(msg);
    scroll();
    return msg;
}

function toggleSidebar() {
    sidebar.classList.toggle('open');
    let backdrop = document.querySelector('.sidebar-backdrop');
    if (!backdrop) {
        backdrop = el('div', 'sidebar-backdrop');
        backdrop.addEventListener('click', toggleSidebar);
        document.body.appendChild(backdrop);
    }
    backdrop.classList.toggle('active');
}

function el(tag, cls) { 
    const e = document.createElement(tag); 
    e.className = cls; 
    return e; 
}

function scroll() { 
    requestAnimationFrame(() => { 
        chatMessages.scrollTop = chatMessages.scrollHeight; 
    }); 
}

function esc(t) { 
    const d = document.createElement('div'); 
    d.textContent = t; 
    return d.innerHTML; 
}

function fmt(t) {
    if (!t) return '';
    let h = esc(t);
    h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
    h = h.replace(/`(.*?)`/g, '<code>$1</code>');
    return h.split(/\n\n+/).map(p => p.trim()).filter(Boolean)
        .map(p => {
            const lines = p.split('\n');
            if (lines.every(l => /^\s*[-•●]\s/.test(l) || /^\s*\d+[.)]\s/.test(l)))
                return '<ul>' + lines.map(l => `<li>${l.replace(/^\s*[-•●\d.)\s]+/, '')}</li>`).join('') + '</ul>';
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');
}
