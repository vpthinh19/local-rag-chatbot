import { documentActions, reduceStreamEvent, shouldPollDocuments } from "./state.mjs";

const $ = (selector) => document.querySelector(selector);
const chatsContainer = $(".chats-container");
const promptForm = $("#prompt-form");
const promptInput = $("#prompt-input");
const documentUploadForm = $("#document-upload-form");
const documentFileInput = $("#document-file-input");
const documentsList = $("#documents-list");
const sessionsList = $("#sessions-list");
const stopResponseBtn = $("#stop-response-btn");
const sidebar = $("#sidebar");

let selectedSessionId = null;
const streamControllers = new Map();
let streamBuffers = {};
let documentPollTimer = null;

const api = async (url, options) => {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Lỗi ${response.status}`);
  return response.status === 204 ? null : response.json();
};
const scrollToBottom = () => $(".container").scrollTo({ top: $(".container").scrollHeight, behavior: "smooth" });
const message = (role, content, extra = "") => {
  const row = document.createElement("div");
  row.className = `message ${role}-message ${extra}`;
  const text = document.createElement("p");
  text.className = "message-text";
  text.textContent = content;
  row.append(text);
  return row;
};

function renderMessages(messages = []) {
  chatsContainer.replaceChildren(...messages.map(({ role, content }) => message(role, content)));
  const buffer = streamBuffers[selectedSessionId];
  if (buffer) {
    if (buffer.user) chatsContainer.append(message("user", buffer.user));
    const pending = message("bot", buffer.text || buffer.status, "loading");
    pending.querySelector(".message-text").classList.toggle("status-text", !buffer.text);
    chatsContainer.append(pending);
  }
  scrollToBottom();
}

async function loadMessages(sessionId = selectedSessionId) {
  if (!sessionId) return;
  const data = await api(`/api/sessions/${sessionId}/messages`);
  if (sessionId === selectedSessionId) renderMessages(data.messages);
}

async function loadSessions() {
  const data = await api("/api/sessions");
  sessionsList.replaceChildren(...data.sessions.map((session) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button"; button.textContent = session.title;
    button.classList.toggle("selected", session.id === selectedSessionId);
    button.addEventListener("click", () => selectSession(session.id));
    item.append(button); return item;
  }));
  return data.sessions;
}

async function selectSession(sessionId) {
  selectedSessionId = sessionId;
  await Promise.all([loadSessions(), loadMessages(sessionId)]);
}

async function newSession() {
  const session = await api("/api/sessions", { method: "POST" });
  await selectSession(session.id);
  promptInput.focus();
}

async function loadDocuments() {
  const data = await api("/api/documents").catch(() => ({ documents: [] }));
  documentsList.replaceChildren(...data.documents.map(renderDocument));
  if (shouldPollDocuments(data.documents)) startDocumentPolling(); else stopDocumentPolling();
}

function renderDocument(doc) {
  const item = document.createElement("article"); item.className = "document-item";
  const name = document.createElement("strong"); name.className = "doc-name"; name.textContent = doc.file_name; name.title = doc.file_name;
  const meta = document.createElement("p"); meta.className = "doc-meta";
  meta.textContent = `${doc.status} · ${doc.chunk_count} đoạn${doc.error ? ` · ${doc.error}` : ""}`;
  const actions = document.createElement("div"); actions.className = "doc-actions";
  documentActions(doc).forEach((action) => {
    const button = document.createElement("button"); button.type = "button"; button.className = `${action}-doc-btn material-symbols-rounded`;
    button.textContent = { download: "download", retry: "refresh", delete: "close" }[action];
    button.title = { download: "Tải xuống", retry: "Thử lại", delete: "Xóa" }[action]; button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => documentAction(doc, action)); actions.append(button);
  });
  item.append(name, meta, actions); return item;
}

async function documentAction(doc, action) {
  if (action === "download") { window.location.assign(`/api/documents/${doc.id}/download`); return; }
  if (action === "delete" && !confirm(`Xóa ${doc.file_name}?`)) return;
  await api(`/api/documents/${doc.id}${action === "retry" ? "/retry" : ""}`, { method: action === "retry" ? "POST" : "DELETE" });
  await loadDocuments();
}

function startDocumentPolling() {
  if (!documentPollTimer) documentPollTimer = setInterval(loadDocuments, 1500);
}
function stopDocumentPolling() {
  if (documentPollTimer) clearInterval(documentPollTimer);
  documentPollTimer = null;
}

function renderStream(sessionId) { if (sessionId === selectedSessionId) loadMessages(sessionId); }

async function streamChat(sessionId, userMessage) {
  const controller = new AbortController(); streamControllers.set(sessionId, controller);
  streamBuffers = { ...streamBuffers, [sessionId]: { text: "", status: "Đang xử lý...", user: userMessage } }; renderStream(sessionId);
  try {
    const response = await fetch(`/api/sessions/${sessionId}/chat`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ message: userMessage }), signal: controller.signal });
    if (!response.ok || !response.body) throw new Error(`Lỗi ${response.status}`);
    const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = "";
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      pending += decoder.decode(value, { stream: true }); const lines = pending.split("\n"); pending = lines.pop();
      for (const line of lines) if (line.startsWith("data: ")) {
        const event = JSON.parse(line.slice(6));
        streamBuffers = reduceStreamEvent({ buffers: streamBuffers, sessionId, event });
        if (event.type === "error") streamBuffers[sessionId].text = "Lỗi khi trả lời.";
        renderStream(sessionId);
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") { streamBuffers = reduceStreamEvent({ buffers: streamBuffers, sessionId, event: { type: "error" } }); renderStream(sessionId); }
  } finally {
    streamControllers.delete(sessionId); delete streamBuffers[sessionId]; await loadMessages(sessionId).catch(() => {});
  }
}

promptForm.addEventListener("submit", async (event) => {
  event.preventDefault(); const userMessage = promptInput.value.trim();
  if (!userMessage || !selectedSessionId || streamControllers.has(selectedSessionId)) return;
  const sessionId = selectedSessionId; promptInput.value = "";
  chatsContainer.append(message("user", userMessage)); scrollToBottom();
  await streamChat(sessionId, userMessage);
});
documentUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault(); if (!documentFileInput.files[0]) return;
  await api("/api/documents", { method: "POST", body: new FormData(documentUploadForm) });
  documentUploadForm.reset(); await loadDocuments();
});
stopResponseBtn.addEventListener("click", async () => {
  if (!selectedSessionId) return;
  streamControllers.get(selectedSessionId)?.abort();
  await api(`/api/sessions/${selectedSessionId}/stop`, { method: "POST" }).catch(() => {});
});
$("#new-session-btn").addEventListener("click", newSession);
$("#rename-session-btn").addEventListener("click", async () => {
  const title = prompt("Tên cuộc trò chuyện mới:");
  if (title?.trim() && selectedSessionId) await api(`/api/sessions/${selectedSessionId}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ title: title.trim() }) }).then(loadSessions);
});
$("#delete-session-btn").addEventListener("click", async () => {
  if (!selectedSessionId || !confirm("Xóa cuộc trò chuyện này?")) return;
  await api(`/api/sessions/${selectedSessionId}`, { method: "DELETE" }); selectedSessionId = null;
  const sessions = await loadSessions(); if (sessions[0]) await selectSession(sessions[0].id); else await newSession();
});
$("#theme-toggle-btn").addEventListener("click", () => {
  const light = document.body.classList.toggle("light-theme"); localStorage.setItem("themeColor", light ? "light_mode" : "dark_mode");
});
$("#toggle-sidebar-btn").addEventListener("click", () => sidebar.classList.toggle("collapsed"));

(async () => {
  const light = localStorage.getItem("themeColor") === "light_mode"; document.body.classList.toggle("light-theme", light);
  let sessions = await loadSessions(); if (!sessions.length) { await newSession(); sessions = await loadSessions(); }
  if (!selectedSessionId) selectedSessionId = sessions[0].id;
  await Promise.all([loadSessions(), loadMessages(), loadDocuments()]);
})();
