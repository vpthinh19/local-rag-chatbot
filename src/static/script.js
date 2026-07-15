/**
 * AI Chatbot - Frontend Script
 * Handles chat UI, file uploads, and streaming responses
 */

// DOM Elements
const container = document.querySelector(".container");
const chatsContainer = document.querySelector(".chats-container");
const promptForm = document.querySelector("#prompt-form");
const promptInput = document.querySelector("#prompt-input");
const fileInput = document.querySelector("#file-input");
const fileUploadWrapper = document.querySelector(".file-upload-wrapper");
const themeToggleBtn = document.querySelector("#theme-toggle-btn");
const stopResponseBtn = document.querySelector("#stop-response-btn");
const deleteChatsBtn = document.querySelector("#delete-chats-btn");
const addFileBtn = document.querySelector("#add-file-btn");
const cancelFileBtn = document.querySelector("#cancel-file-btn");
const sidebar = document.querySelector("#sidebar");
const documentsList = document.querySelector("#documents-list");
const toggleSidebarBtn = document.querySelector("#toggle-sidebar-btn");

// State
let uploadedFile = null;
let currentBotMessage = null;
let abortController = null;
let isRequestActive = false;

// Initialize theme
const isLightTheme = localStorage.getItem("themeColor") === "light_mode";
document.body.classList.toggle("light-theme", isLightTheme);
themeToggleBtn.textContent = isLightTheme ? "dark_mode" : "light_mode";

// Initialize sidebar state
const sidebarCollapsed = localStorage.getItem("sidebarCollapsed") === "true";
if (sidebarCollapsed) {
    sidebar.classList.add("collapsed");
}

/**
 * Create a message element
 */
const createMessageElement = (content, ...classes) => {
    const div = document.createElement("div");
    div.classList.add("message", ...classes);
    div.innerHTML = content;
    return div;
};

/**
 * Scroll to bottom of container
 */
const scrollToBottom = () => {
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
};

/**
 * Check if user is near bottom
 */
const isNearBottom = () => {
    return container.scrollHeight - container.scrollTop - container.clientHeight < 200;
};

/**
 * Smart scroll - only auto-scroll if near bottom
 */
const smartScroll = () => {
    if (isNearBottom()) {
        scrollToBottom();
    }
};

/**
 * Cleanup after request
 */
const cleanupRequest = () => {
    isRequestActive = false;
    abortController = null;
    
    if (currentBotMessage) {
        currentBotMessage.classList.remove("loading");
    }
    
    document.body.classList.remove("bot-responding");
};

/**
 * Load chat history from server
 */
const loadChatHistory = async () => {
    try {
        const response = await fetch('/api/chat-history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            for (const msg of data.history) {
                if (msg.role === "user") {
                    const userMsgHTML = `<p class="message-text"></p>`;
                    const userMsgDiv = createMessageElement(userMsgHTML, "user-message");
                    userMsgDiv.querySelector(".message-text").textContent = msg.content;
                    chatsContainer.appendChild(userMsgDiv);
                } else if (msg.role === "assistant") {
                    const botMsgHTML = `<p class="message-text"></p>`;
                    const botMsgDiv = createMessageElement(botMsgHTML, "bot-message");
                    botMsgDiv.querySelector(".message-text").textContent = msg.content;
                    chatsContainer.appendChild(botMsgDiv);
                }
            }
            scrollToBottom();
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
};

/**
 * Load documents list from server
 */
const loadDocuments = async () => {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        
        documentsList.innerHTML = '';
        
        if (data.documents && data.documents.length > 0) {
            for (const doc of data.documents) {
                const docItem = document.createElement('div');
                docItem.className = 'document-item';
                docItem.innerHTML = `
                    <div class="doc-header">
                        <span class="material-symbols-rounded doc-icon">description</span>
                        <span class="doc-name"></span>
                        <div class="doc-actions">
                            <button class="download-doc-btn material-symbols-rounded" 
                                    title="Tải xuống">download</button>
                            <button class="delete-doc-btn material-symbols-rounded" 
                                    title="Xóa">close</button>
                        </div>
                    </div>
                    <div class="doc-meta">
                        <span class="doc-chunks"></span>
                    </div>
                `;
                const docName = docItem.querySelector('.doc-name');
                const downloadBtn = docItem.querySelector('.download-doc-btn');
                const deleteBtn = docItem.querySelector('.delete-doc-btn');
                docName.textContent = doc.file_name;
                docName.title = doc.file_name;
                downloadBtn.dataset.fileId = doc.file_id;
                downloadBtn.dataset.fileName = doc.file_name;
                deleteBtn.dataset.fileId = doc.file_id;
                docItem.querySelector('.doc-chunks').textContent = `${doc.chunk_count} chunks`;
                documentsList.appendChild(docItem);
            }
            
            // Add click handlers for download buttons
            documentsList.querySelectorAll('.download-doc-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.fileId;
                    const fileName = btn.dataset.fileName;
                    await downloadDocument(fileId, fileName);
                });
            });
            
            // Add click handlers for delete buttons
            documentsList.querySelectorAll('.delete-doc-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.fileId;
                    if (confirm('Xóa tài liệu này?')) {
                        await deleteDocument(fileId);
                    }
                });
            });
        } else {
            documentsList.innerHTML = '<p class="no-docs">Chưa có tài liệu nào</p>';
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
};

/**
 * Download a document
 */
const downloadDocument = async (fileId, fileName) => {
    try {
        const response = await fetch(`/api/documents/${fileId}/download`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } else {
            alert('Không thể tải file');
        }
    } catch (error) {
        console.error('Error downloading document:', error);
        alert('Lỗi khi tải file');
    }
};

/**
 * Delete a document
 */
const deleteDocument = async (fileId) => {
    try {
        const response = await fetch(`/api/documents/${fileId}`, { method: 'DELETE' });
        if (response.ok) {
            await loadDocuments();
        }
    } catch (error) {
        console.error('Error deleting document:', error);
    }
};

/**
 * Stream response from server
 */
const streamResponse = async (formData) => {
    let reader = null;
    
    try {
        abortController = new AbortController();
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData,
            signal: abortController.signal
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        if (!response.body) {
            throw new Error('Response body is null');
        }
        
        reader = response.body.getReader();
        const decoder = new TextDecoder();
        const textElement = currentBotMessage.querySelector(".message-text");
        
        if (!textElement) {
            throw new Error('Text element not found');
        }
        
        textElement.textContent = "";
        let fullResponse = "";
        let buffer = '';
        let contentStarted = false;
        
        while (true) {
            if (!isRequestActive) {
                reader.cancel();
                break;
            }
            
            const { done, value } = await reader.read();
            
            if (done) {
                break;
            }
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
                if (line.trim().startsWith('data: ')) {
                    try {
                        const jsonStr = line.slice(6).trim();
                        if (!jsonStr) continue;
                        
                        const data = JSON.parse(jsonStr);
                        
                        // Handle error
                        if (data.error) {
                            textElement.textContent = data.error;
                            textElement.style.color = "#d62939";
                            break;
                        }
                        
                        // Handle status updates
                        if (data.status && !contentStarted) {
                            textElement.textContent = data.status;
                            textElement.classList.add("status-text");
                            smartScroll();
                        }
                        
                        // Handle content streaming
                        if (data.content) {
                            if (!contentStarted) {
                                textElement.textContent = "";
                                textElement.classList.remove("status-text");
                                contentStarted = true;
                            }
                            fullResponse += data.content;
                            textElement.textContent = fullResponse;
                            smartScroll();
                        }
                        
                        // Handle completion
                        if (data.done) {
                            cleanupRequest();
                            // Reload documents in case new file was uploaded
                            loadDocuments();
                        }
                        
                        // Handle cancellation from server
                        if (data.cancelled) {
                            cleanupRequest();
                            loadDocuments();
                        }
                    } catch (parseError) {
                        console.error('Parse error:', parseError);
                    }
                }
            }
        }
        
        cleanupRequest();
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Request aborted');
        } else {
            console.error('Streaming error:', error);
            if (currentBotMessage) {
                const textElement = currentBotMessage.querySelector(".message-text");
                if (textElement) {
                    textElement.textContent = "Lỗi: " + error.message;
                    textElement.style.color = "#d62939";
                }
            }
        }
        
        cleanupRequest();
        
    } finally {
        if (reader) {
            try {
                await reader.cancel();
            } catch (e) {
                // Reader already closed
            }
        }
    }
};

/**
 * Handle form submission
 */
const handleFormSubmit = async (e) => {
    e.preventDefault();
    
    const userMessage = promptInput.value.trim();
    if (!userMessage || isRequestActive) return;
    
    // Prepare form data
    const formData = new FormData();
    formData.append('message', userMessage);
    
    if (uploadedFile) {
        formData.append('file', uploadedFile);
    }
    
    // Clear input and set active state
    promptInput.value = "";
    isRequestActive = true;
    document.body.classList.add("bot-responding");
    fileUploadWrapper.classList.remove("file-attached", "img-attached", "active");
    
    // Create user message element
    const userMsgHTML = `<p class="message-text"></p>`;
    const userMsgDiv = createMessageElement(userMsgHTML, "user-message");
    userMsgDiv.querySelector(".message-text").textContent = userMessage;
    
    // Add file attachment indicator
    if (uploadedFile) {
        const fileAttachment = document.createElement('p');
        fileAttachment.className = 'file-attachment';
        const fileIcon = document.createElement('span');
        fileIcon.className = 'material-symbols-rounded';
        fileIcon.textContent = 'description';
        fileAttachment.append(fileIcon, document.createTextNode(uploadedFile.name));
        userMsgDiv.appendChild(fileAttachment);
    }
    
    chatsContainer.appendChild(userMsgDiv);
    scrollToBottom();
    
    // Clear uploaded file
    uploadedFile = null;
    
    // Create bot message after short delay (no avatar)
    setTimeout(async () => {
        const botMsgHTML = `<p class="message-text"></p>`;
        currentBotMessage = createMessageElement(botMsgHTML, "bot-message", "loading");
        chatsContainer.appendChild(currentBotMessage);
        scrollToBottom();
        
        await streamResponse(formData);
    }, 300);
};

// Event Listeners

// Form submission
promptForm.addEventListener("submit", handleFormSubmit);

// Stop response
stopResponseBtn.addEventListener("click", async () => {
    isRequestActive = false;
    
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    
    // Ask the server to cancel the active request pipeline.
    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (e) {
        console.log('Stop request failed:', e);
    }
    
    if (currentBotMessage) {
        const textElement = currentBotMessage.querySelector(".message-text");
        if (textElement) {
            if (!textElement.textContent || textElement.classList.contains("status-text")) {
                textElement.textContent = "Đã dừng.";
            } else {
                textElement.textContent += "\n[Đã dừng]";
            }
            textElement.classList.remove("status-text");
        }
    }
    
    cleanupRequest();
    loadDocuments();
});

// File input
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    
    uploadedFile = file;
    fileUploadWrapper.classList.add("active", "file-attached");
    fileInput.value = "";
});

// Cancel file
cancelFileBtn.addEventListener("click", () => {
    uploadedFile = null;
    fileUploadWrapper.classList.remove("file-attached", "img-attached", "active");
});

// Add file button
addFileBtn.addEventListener("click", () => fileInput.click());

// Theme toggle
themeToggleBtn.addEventListener("click", () => {
    const isLightTheme = document.body.classList.toggle("light-theme");
    localStorage.setItem("themeColor", isLightTheme ? "light_mode" : "dark_mode");
    themeToggleBtn.textContent = isLightTheme ? "dark_mode" : "light_mode";
});

// Toggle sidebar
toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle('collapsed');
    const isCollapsed = sidebar.classList.contains('collapsed');
    localStorage.setItem("sidebarCollapsed", isCollapsed);
});

// Add floating toggle button to main content
const mainContent = document.querySelector(".main-content");
const floatingToggle = document.createElement("button");
floatingToggle.className = "sidebar-toggle-floating material-symbols-rounded";
floatingToggle.textContent = "menu";
floatingToggle.addEventListener("click", () => {
    sidebar.classList.remove('collapsed');
    localStorage.setItem("sidebarCollapsed", "false");
});
mainContent.appendChild(floatingToggle);

// Delete chats
deleteChatsBtn.addEventListener("click", async () => {
    if (!confirm("Xóa toàn bộ lịch sử chat và tài liệu?")) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear-chat', { method: 'POST' });
        
        if (response.ok) {
            chatsContainer.innerHTML = "";
            cleanupRequest();
            // Reload documents list
            loadDocuments();
        } else {
            alert("Không thể xóa lịch sử");
        }
    } catch (error) {
        console.error('Clear chat error:', error);
        alert("Lỗi khi xóa lịch sử");
    }
});

// Mobile controls
document.addEventListener("click", ({ target }) => {
    const wrapper = document.querySelector(".prompt-wrapper");
    const shouldHide = target.classList.contains("prompt-input") || 
                      (wrapper.classList.contains("hide-controls") && 
                       (target.id === "add-file-btn" || target.id === "stop-response-btn"));
    wrapper.classList.toggle("hide-controls", shouldHide);
});

// Load history and documents on page load
loadChatHistory();
loadDocuments();
