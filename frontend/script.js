const chatMessages = document.getElementById('chatMessages');
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const statusBadge = document.getElementById('statusBadge');
const docList = document.getElementById('docList');
const chunkCount = document.getElementById('chunkCount');
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.querySelector('.sidebar');
const apiKeyInput = document.getElementById('apiKeyInput');
const apiKeySaveBtn = document.getElementById('apiKeySaveBtn');
const apiKeyStatus = document.getElementById('apiKeyStatus');
let isProcessing = false;

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadApiKey();
    queryForm.addEventListener('submit', handleSubmit);
    apiKeySaveBtn.addEventListener('click', saveApiKey);
    apiKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); saveApiKey(); }
    });
    document.querySelectorAll('.suggestion').forEach(el => {
        el.addEventListener('click', () => {
            queryInput.value = el.dataset.query;
            handleSubmit(new Event('submit', { cancelable: true }));
        });
    });
    menuToggle.addEventListener('click', toggleSidebar);
});

function loadApiKey() {
    const key = localStorage.getItem('groq_api_key') || '';
    if (key) {
        apiKeyInput.value = key;
        setApiKeyStatus('saved', `Key saved (${key.slice(0, 8)}...)`);
    } else {
        setApiKeyStatus('none', 'No API key set');
    }
}

function saveApiKey() {
    const key = apiKeyInput.value.trim();
    if (!key) {
        localStorage.removeItem('groq_api_key');
        setApiKeyStatus('none', 'No API key set');
        return;
    }
    localStorage.setItem('groq_api_key', key);
    setApiKeyStatus('saved', `Key saved (${key.slice(0, 8)}...)`);
}

function setApiKeyStatus(state, text) {
    apiKeyStatus.textContent = text;
    apiKeyStatus.className = 'api-key-status';
    if (state === 'saved') apiKeyStatus.classList.add('saved');
}

function getApiKey() {
    return localStorage.getItem('groq_api_key') || '';
}

async function checkHealth() {
    try {
        const data = await (await fetch('/api/health')).json();
        if (data.status === 'healthy') {
            setStatus('online', `Ready — ${data.chunks} chunks indexed`);
            chunkCount.textContent = `${data.chunks} chunks across ${data.documents} documents.`;
            docList.innerHTML = '';
            (data.document_names || []).forEach(name => {
                const li = document.createElement('li');
                li.className = 'doc-item';
                li.textContent = name.replace('.md', '').replace('.txt', '').replace(/_/g, ' ');
                docList.appendChild(li);
            });
        }
    } catch { setStatus('error', 'Cannot connect to server'); }
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

    const apiKey = getApiKey();
    if (!apiKey) {
        appendMsg('assistant', '⚠️ Please set your Groq API key in the sidebar before asking questions.', true);
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
            body: JSON.stringify({ question, top_k: 5, api_key: apiKey }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        typingEl.remove();
        appendAnswer(data);
    } catch (err) {
        typingEl.remove();
        appendMsg('assistant', `⚠️ Error: ${err.message}`, true);
    } finally { setProcessing(false); }
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
    avatar.textContent = role === 'user' ? 'U' : '🏗️';
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
    avatar.textContent = '🏗️';
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
            card.innerHTML = `<div class="chunk-header"><span class="chunk-source">${esc(c.source.replace(/_/g, ' '))}</span><span class="chunk-rank">#${c.rank}</span></div><div class="chunk-text">${esc(c.text)}</div><div class="chunk-score">L2: ${c.score}</div>`;
            chunksDiv.appendChild(card);
        });

        toggle.addEventListener('click', () => { toggle.classList.toggle('open'); chunksDiv.classList.toggle('open'); });
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
    avatar.textContent = '🏗️';
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
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay) {
        overlay = el('div', 'sidebar-overlay');
        overlay.addEventListener('click', toggleSidebar);
        document.body.appendChild(overlay);
    }
    overlay.classList.toggle('active');
}

function el(tag, cls) { const e = document.createElement(tag); e.className = cls; return e; }
function scroll() { requestAnimationFrame(() => { chatMessages.scrollTop = chatMessages.scrollHeight; }); }
function esc(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
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
