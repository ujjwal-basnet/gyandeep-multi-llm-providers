const pdfUpload = document.getElementById('pdf-upload');
const pdfCanvas = document.getElementById('pdf-canvas');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const pageControls = document.getElementById('page-controls');
const prevBtn = document.getElementById('prev-page');
const nextBtn = document.getElementById('next-page');
const pageNumInput = document.getElementById('page-num-input');
const pageCountSpan = document.getElementById('page-count');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

const analyzeGlobalBtn = document.getElementById('analyze-global-btn');
const analyzeEnvBtn = document.getElementById('analyze-env-btn');
const chatModeSelect = document.getElementById('chat-mode');

let pdfDoc = null;
let pageNum = 1;
let pageRendering = false;
let pageNumPending = null;
let currentPdfFilename = null;

function renderPage(num) {
    pageRendering = true;
    pdfDoc.getPage(num).then(function(page) {
        let viewport = page.getViewport({scale: 1.5});
        
        const container = document.getElementById('pdf-viewer-container');
        const scale = (container.clientWidth - 48) / viewport.width; 
        if (scale < 1.5) {
            viewport = page.getViewport({scale: scale});
        }
        
        pdfCanvas.height = viewport.height;
        pdfCanvas.width = viewport.width;

        const renderContext = {
            canvasContext: pdfCanvas.getContext('2d'),
            viewport: viewport
        };
        const renderTask = page.render(renderContext);

        renderTask.promise.then(function() {
            pageRendering = false;
            if (pageNumPending !== null) {
                renderPage(pageNumPending);
                pageNumPending = null;
            }
        });
    });
    pageNumInput.value = num;
}

function queueRenderPage(num) {
    if (pageRendering) {
        pageNumPending = num;
    } else {
        renderPage(num);
    }
}

function loadPdfUrl(url) {
    uploadPlaceholder.style.display = 'none';
    pdfCanvas.style.display = 'block';
    pageControls.style.display = 'flex';
    
    pdfjsLib.getDocument(url).promise.then(function(pdfDoc_) {
        pdfDoc = pdfDoc_;
        pageCountSpan.textContent = pdfDoc.numPages;
        pageNum = 1;
        renderPage(pageNum);
        
        chatInput.disabled = false;
        sendBtn.disabled = false;
        analyzeGlobalBtn.disabled = false;
        analyzeEnvBtn.disabled = false;
        chatInput.focus();
    }).catch(function(err) {
        console.error('Error loading PDF: ', err);
        addMessage('system', 'Error rendering PDF on screen.');
    });
}

prevBtn.addEventListener('click', () => {
    if (pageNum <= 1) return;
    pageNum--;
    queueRenderPage(pageNum);
});

nextBtn.addEventListener('click', () => {
    if (pageNum >= pdfDoc.numPages) return;
    pageNum++;
    queueRenderPage(pageNum);
});

pageNumInput.addEventListener('change', (e) => {
    let num = parseInt(e.target.value, 10);
    if (isNaN(num) || num < 1) num = 1;
    if (num > pdfDoc.numPages) num = pdfDoc.numPages;
    pageNumInput.value = num; 
    if (num !== pageNum) {
        pageNum = num;
        queueRenderPage(pageNum);
    }
});

pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    addMessage('system', `Uploading ${file.name}...`);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentPdfFilename = data.filename;
            addMessage('system', `PDF successfully processed and initialized. Ready to answer questions (OCR will run automatically when you ask something).`);
            
            loadPdfUrl(`/uploads/${currentPdfFilename}`);
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (err) {
        console.error(err);
        addMessage('system', 'Error uploading PDF: ' + err.message);
    }
});

analyzeEnvBtn.addEventListener('click', async () => {
    analyzeEnvBtn.disabled = true;
    addMessage('system', 'Analyzing Context Environment (+/- 5 pages). Please wait...');

    try {
        const response = await fetch('/api/analyze_env', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ current_page: pageNum })
        });
        if (!response.ok) throw new Error('Analysis failed');
        
        const data = await response.json();
        addMessage('system', `Environment Analysis Complete!\n\n${data.analysis}`);
    } catch (err) {
        addMessage('system', 'Error analyzing: ' + err.message);
    } finally {
        analyzeEnvBtn.disabled = false;
    }
});

analyzeGlobalBtn.addEventListener('click', async () => {
    analyzeGlobalBtn.disabled = true;
    addMessage('system', 'Scanning and Analyzing the ENTIRE Book. This may take several minutes! Please wait...');

    try {
        const response = await fetch('/api/analyze_global', { method: 'POST' });
        if (!response.ok) throw new Error('Analysis failed');
        
        const data = await response.json();
        addMessage('system', `Global Analysis Complete!\n\n${data.analysis}`);
    } catch (err) {
        addMessage('system', 'Error analyzing: ' + err.message);
    } finally {
        analyzeGlobalBtn.disabled = false;
    }
});

function addMessage(role, text, isMeta = false) {
    const msgDiv = document.createElement('div');
    if (isMeta) {
        msgDiv.className = 'message meta-msg';
        msgDiv.textContent = text;
    } else {
        msgDiv.className = `message ${role}-msg`;
        msgDiv.innerHTML = `
            <div class="avatar">${role === 'user' ? '👤' : '🤖'}</div>
            <div class="content">${escapeHtml(text)}</div>
        `;
    }
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    addMessage('user', query);
    chatInput.value = '';
    
    chatInput.disabled = true;
    sendBtn.disabled = true;

    const mode = chatModeSelect.value;
    const botMsgDiv = addMessage('bot', `Consulting the ${mode === 'analyze' ? 'context.txt' : 'surrounding_context.txt'} file...`);
    const botContent = botMsgDiv.querySelector('.content');

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                mode: mode
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        botContent.innerHTML = '';
        let fullText = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, {stream: true});
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ') && line !== 'data: {}') {
                    try {
                        const dataMsg = JSON.parse(line.substring(6));
                        
                        if (dataMsg.error) {
                            botContent.innerHTML += `<br/><span style="color:red">Error: ${dataMsg.error}</span>`;
                        } else if (dataMsg.is_meta) {
                            addMessage('system', dataMsg.content, true);
                        } else if (dataMsg.content) {
                            fullText += dataMsg.content;
                            botContent.innerHTML = escapeHtml(fullText).replace(/\n/g, '<br/>');
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                    } catch (e) {}
                }
            }
        }
    } catch (err) {
        botContent.innerHTML = `Error: ${err.message}`;
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
});
