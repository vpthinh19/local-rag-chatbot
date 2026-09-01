import { documentActions, reduceStreamEvent, shouldPollDocuments } from "./state.mjs";

const $ = (selector) => document.querySelector(selector);
const chatsContainer = $(".chats-container");
const promptForm = $("#prompt-form");
const promptInput = $("#prompt-input");
const sessionsList = $("#sessions-list");
const documentsList = $("#documents-list");
const sessionStatus = $("#session-status");
const documentStatus = $("#document-status");
const documentUploadForm = $("#document-upload-form");
const documentFileInput = $("#document-file-input");
const uploadDocumentBtn = $("#upload-document-btn");
const stopResponseBtn = $("#stop-response-btn");
const sessionToggleBtn = $("#toggle-session-sidebar-btn");
const documentToggleBtn = $("#toggle-document-sidebar-btn");
const themeToggleBtn = $("#theme-toggle-btn");
const mobileLayout = window.matchMedia("(max-width: 768px)");

let selectedSessionId = null;
let streamBuffers = {};
let documentPollTimer = null;
let openItemMenu = null;
let sessionSidebarCollapsed = localStorage.getItem("sessionSidebarCollapsed") === "true";
let documentSidebarCollapsed = localStorage.getItem("documentSidebarCollapsed") === "true";
const streamControllers = new Map();
const persistedMessages = new Map();

const api = async (url, options) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Lỗi ${response.status}`);
  }
  return response.status === 204 ? null : response.json();
};

const icon = (name) => {
  const value = document.createElement("span");
  value.className = "material-symbols-rounded";
  value.textContent = name;
  return value;
};

const setStatus = (element, text = "", error = false) => {
  element.textContent = text;
  element.classList.toggle("error", error);
};

const scrollToBottom = () => {
  const container = $(".container");
  container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
};

const message = (role, content, extra = "") => {
  const row = document.createElement("div");
  row.className = `message ${role}-message ${extra}`;
  const text = document.createElement("p");
  text.className = "message-text";
  text.textContent = content;
  row.append(text);
  return row;
};

function renderMessages(messages = persistedMessages.get(selectedSessionId) || []) {
  chatsContainer.replaceChildren(
    ...messages.map(({ role, content }) =>
      message(role === "assistant" ? "bot" : role, content)
    )
  );
  const buffer = streamBuffers[selectedSessionId];
  if (buffer) {
    if (buffer.user) chatsContainer.append(message("user", buffer.user));
    const pending = message("bot", buffer.text || buffer.status, "loading");
    pending.querySelector(".message-text").classList.toggle("status-text", !buffer.text);
    chatsContainer.append(pending);
  }
  scrollToBottom();
}

function syncResponseState() {
  document.body.classList.toggle("bot-responding", streamControllers.has(selectedSessionId));
}

function renderStream(sessionId) { if (sessionId === selectedSessionId) renderMessages(); }

async function loadMessages(sessionId = selectedSessionId) {
  if (!sessionId) return;
  const data = await api(`/api/sessions/${sessionId}/messages`);
  persistedMessages.set(sessionId, data.messages);
  if (sessionId === selectedSessionId) renderMessages();
}

function closeMenu() {
  if (!openItemMenu) return;
  openItemMenu.menu.hidden = true;
  openItemMenu.trigger.setAttribute("aria-expanded", "false");
  openItemMenu = null;
}

function connectMenu(trigger, menu) {
  trigger.addEventListener("click", (event) => {
    event.stopPropagation();
    const opening = menu.hidden;
    closeMenu();
    if (opening) {
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      openItemMenu = { menu, trigger };
    }
  });
  menu.addEventListener("click", (event) => event.stopPropagation());
}

function menuAction(label, iconName, action, danger = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `menu-action${danger ? " danger" : ""}`;
  button.append(icon(iconName), document.createTextNode(label));
  button.addEventListener("click", () => action(button));
  return button;
}

function menuTrigger(label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "icon-button item-menu-trigger";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", "false");
  button.append(icon("more_horiz"));
  return button;
}

function renameSession(menu, session) {
  const form = document.createElement("form");
  form.className = "rename-form";
  const input = document.createElement("input");
  input.className = "rename-input";
  input.value = session.title;
  input.maxLength = 80;
  input.setAttribute("aria-label", "Tên cuộc trò chuyện");
  const actions = document.createElement("div");
  actions.className = "rename-actions";
  const save = document.createElement("button");
  save.type = "submit";
  save.className = "pill-button primary-button";
  save.textContent = "Lưu";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "pill-button";
  cancel.textContent = "Hủy";
  actions.append(save, cancel);
  form.append(input, actions);
  menu.replaceChildren(form);
  input.focus();
  input.select();
  cancel.addEventListener("click", closeMenu);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const title = input.value.trim();
    if (!title) return;
    try {
      await api(`/api/sessions/${session.id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title }),
      });
      closeMenu();
      await loadSessions();
    } catch (error) {
      setStatus(sessionStatus, error.message, true);
    }
  });
}

async function deleteSession(button, session) {
  if (!button.classList.contains("confirming")) {
    button.classList.add("confirming");
    button.lastChild.textContent = " Xác nhận xóa";
    return;
  }
  try {
    await api(`/api/sessions/${session.id}`, { method: "DELETE" });
    persistedMessages.delete(session.id);
    delete streamBuffers[session.id];
    if (selectedSessionId === session.id) selectedSessionId = null;
    closeMenu();
    const sessions = await loadSessions();
    if (sessions[0]) await selectSession(sessions[0].id);
    else await newSession();
  } catch (error) {
    setStatus(sessionStatus, error.message, true);
  }
}

function renderSession(session) {
  const item = document.createElement("li");
  item.className = "session-item";
  item.classList.toggle("selected", session.id === selectedSessionId);
  const select = document.createElement("button");
  select.type = "button";
  select.className = "session-select";
  const title = document.createElement("span");
  title.className = "session-title";
  title.textContent = session.title;
  select.append(icon("chat_bubble"), title);
  select.querySelector(".material-symbols-rounded").classList.add("item-leading-icon");
  select.addEventListener("click", () => selectSession(session.id));

  const trigger = menuTrigger(`Tùy chọn cho ${session.title}`);
  const menu = document.createElement("div");
  menu.className = "item-menu session-item-menu";
  menu.hidden = true;
  menu.append(
    menuAction("Đổi tên", "edit", () => renameSession(menu, session)),
    menuAction("Xóa", "delete", (button) => deleteSession(button, session), true)
  );
  connectMenu(trigger, menu);
  item.append(select, trigger, menu);
  return item;
}

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    sessionsList.replaceChildren(...data.sessions.map(renderSession));
    setStatus(sessionStatus);
    return data.sessions;
  } catch (error) {
    setStatus(sessionStatus, error.message, true);
    return [];
  }
}

async function selectSession(sessionId) {
  selectedSessionId = sessionId;
  closeMenu();
  syncResponseState();
  await Promise.all([loadSessions(), loadMessages(sessionId)]);
  if (mobileLayout.matches) {
    sessionSidebarCollapsed = true;
    syncPanels();
  }
}

async function newSession() {
  try {
    const session = await api("/api/sessions", { method: "POST" });
    await selectSession(session.id);
    promptInput.focus();
  } catch (error) {
    setStatus(sessionStatus, error.message, true);
  }
}

function renderDocument(doc) {
  const item = document.createElement("article");
  item.className = "document-item";
  const summary = document.createElement("div");
  summary.className = "document-summary";
  const leading = icon("description");
  leading.classList.add("item-leading-icon");
  const copy = document.createElement("div");
  copy.className = "document-copy";
  const name = document.createElement("strong");
  name.className = "doc-name";
  name.textContent = doc.file_name;
  name.title = doc.file_name;
  const status = document.createElement("span");
  status.className = `status-pill ${doc.status}`;
  status.textContent = doc.status;
  const meta = document.createElement("span");
  meta.className = "doc-meta";
  meta.textContent = `${doc.chunk_count} đoạn${doc.error ? ` · ${doc.error}` : ""}`;
  copy.append(name, status, meta);
  summary.append(leading, copy);

  const trigger = menuTrigger(`Tùy chọn cho ${doc.file_name}`);
  const menu = document.createElement("div");
  menu.className = "item-menu document-item-menu";
  menu.hidden = true;
  for (const action of documentActions(doc)) {
    const labels = {
      download: ["Tải xuống", "download"],
      retry: ["Thử lại", "refresh"],
      delete: ["Xóa", "delete"],
    };
    menu.append(menuAction(labels[action][0], labels[action][1], (button) => documentAction(doc, action, button), action === "delete"));
  }
  connectMenu(trigger, menu);
  item.append(summary, trigger, menu);
  return item;
}

async function loadDocuments() {
  try {
    const data = await api("/api/documents");
    documentsList.replaceChildren(
      ...(data.documents.length ? data.documents.map(renderDocument) : [emptyList("Chưa có tài liệu")])
    );
    if (shouldPollDocuments(data.documents)) startDocumentPolling();
    else stopDocumentPolling();
    return data.documents;
  } catch (error) {
    setStatus(documentStatus, error.message, true);
    return [];
  }
}

function emptyList(text) {
  const value = document.createElement("p");
  value.className = "empty-list";
  value.textContent = text;
  return value;
}

async function documentAction(doc, action, button) {
  if (action === "download") {
    window.location.assign(`/api/documents/${doc.id}/download`);
    closeMenu();
    return;
  }
  if (action === "delete" && !button.classList.contains("confirming")) {
    button.classList.add("confirming");
    button.lastChild.textContent = " Xác nhận xóa";
    return;
  }
  try {
    await api(`/api/documents/${doc.id}${action === "retry" ? "/retry" : ""}`, {
      method: action === "retry" ? "POST" : "DELETE",
    });
    closeMenu();
    await loadDocuments();
  } catch (error) {
    setStatus(documentStatus, error.message, true);
  }
}

function startDocumentPolling() {
  if (!documentPollTimer) documentPollTimer = setInterval(loadDocuments, 1500);
}

function stopDocumentPolling() {
  if (documentPollTimer) clearInterval(documentPollTimer);
  documentPollTimer = null;
}

async function uploadDocument() {
  if (!documentFileInput.files[0]) return;
  uploadDocumentBtn.disabled = true;
  setStatus(documentStatus, `Đang tải ${documentFileInput.files[0].name}...`);
  try {
    await api("/api/documents", { method: "POST", body: new FormData(documentUploadForm) });
    setStatus(documentStatus, "Đã đưa tài liệu vào hàng đợi.");
    documentUploadForm.reset();
    await loadDocuments();
  } catch (error) {
    setStatus(documentStatus, error.message, true);
  } finally {
    uploadDocumentBtn.disabled = false;
  }
}

async function streamChat(sessionId, userMessage) {
  const controller = new AbortController();
  streamControllers.set(sessionId, controller);
  streamBuffers = { ...streamBuffers, [sessionId]: { text: "", status: "Đang xử lý...", user: userMessage } };
  syncResponseState();
  renderStream(sessionId);
  try {
    const response = await fetch(`/api/sessions/${sessionId}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: userMessage }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error(`Lỗi ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));
        streamBuffers = reduceStreamEvent({ buffers: streamBuffers, sessionId, event });
        if (event.type === "error") streamBuffers[sessionId].text = "Lỗi khi trả lời.";
        renderStream(sessionId);
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      streamBuffers = reduceStreamEvent({ buffers: streamBuffers, sessionId, event: { type: "error" } });
      streamBuffers[sessionId].text = "Lỗi khi trả lời.";
      renderStream(sessionId);
    }
  } finally {
    streamControllers.delete(sessionId);
    syncResponseState();
    if (streamBuffers[sessionId]?.terminal === "done") {
      delete streamBuffers[sessionId];
      await Promise.all([loadMessages(sessionId).catch(() => {}), loadSessions()]);
    }
  }
}

function syncPanels() {
  document.body.classList.toggle("session-sidebar-collapsed", sessionSidebarCollapsed);
  document.body.classList.toggle("document-sidebar-collapsed", documentSidebarCollapsed);
  sessionToggleBtn.setAttribute("aria-expanded", String(!sessionSidebarCollapsed));
  documentToggleBtn.setAttribute("aria-expanded", String(!documentSidebarCollapsed));
  sessionToggleBtn.querySelector("span").textContent = sessionSidebarCollapsed ? "left_panel_open" : "left_panel_close";
  documentToggleBtn.querySelector("span").textContent = documentSidebarCollapsed ? "right_panel_open" : "right_panel_close";
  localStorage.setItem("sessionSidebarCollapsed", String(sessionSidebarCollapsed));
  localStorage.setItem("documentSidebarCollapsed", String(documentSidebarCollapsed));
}

sessionToggleBtn.addEventListener("click", () => {
  sessionSidebarCollapsed = !sessionSidebarCollapsed;
  if (mobileLayout.matches && !sessionSidebarCollapsed) documentSidebarCollapsed = true;
  syncPanels();
});

documentToggleBtn.addEventListener("click", () => {
  documentSidebarCollapsed = !documentSidebarCollapsed;
  if (mobileLayout.matches && !documentSidebarCollapsed) sessionSidebarCollapsed = true;
  syncPanels();
});

mobileLayout.addEventListener("change", ({ matches }) => {
  if (matches) {
    sessionSidebarCollapsed = true;
    documentSidebarCollapsed = true;
    syncPanels();
  }
});

promptForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const userMessage = promptInput.value.trim();
  if (!userMessage || !selectedSessionId || streamControllers.has(selectedSessionId)) return;
  const sessionId = selectedSessionId;
  promptInput.value = "";
  await streamChat(sessionId, userMessage);
});

stopResponseBtn.addEventListener("click", async () => {
  if (!selectedSessionId) return;
  if (streamBuffers[selectedSessionId]) {
    streamBuffers = reduceStreamEvent({ buffers: streamBuffers, sessionId: selectedSessionId, event: { type: "cancelled" } });
    renderStream(selectedSessionId);
  }
  streamControllers.get(selectedSessionId)?.abort();
  await api(`/api/sessions/${selectedSessionId}/stop`, { method: "POST" }).catch(() => {});
});

$("#new-session-btn").addEventListener("click", newSession);
uploadDocumentBtn.addEventListener("click", () => documentFileInput.click());
documentFileInput.addEventListener("change", uploadDocument);
documentUploadForm.addEventListener("submit", (event) => event.preventDefault());

themeToggleBtn.addEventListener("click", () => {
  const light = document.body.classList.toggle("light-theme");
  localStorage.setItem("themeColor", light ? "light_mode" : "dark_mode");
  themeToggleBtn.querySelector("span").textContent = light ? "dark_mode" : "light_mode";
});

document.addEventListener("click", closeMenu);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(); });

(async () => {
  const light = localStorage.getItem("themeColor") === "light_mode";
  document.body.classList.toggle("light-theme", light);
  themeToggleBtn.querySelector("span").textContent = light ? "dark_mode" : "light_mode";
  if (mobileLayout.matches) {
    sessionSidebarCollapsed = true;
    documentSidebarCollapsed = true;
  }
  syncPanels();
  let sessions = await loadSessions();
  if (!sessions.length) {
    await newSession();
    sessions = await loadSessions();
  }
  if (!selectedSessionId && sessions[0]) await selectSession(sessions[0].id);
  await loadDocuments();
})();
