const state = {
  bootstrap: null,
  activeTab: "knowledge",
  selectedKbId: null,
  selectedDocumentId: null,
  selectedDocumentTitle: null,
  chatSessionId: null,
};

const els = {
  feedback: document.getElementById("feedback"),
  currentKbLabel: document.getElementById("current-kb-label"),
  chatKbSummary: document.getElementById("chat-kb-summary"),
  refreshBootstrap: document.getElementById("refresh-bootstrap"),
  kbForm: document.getElementById("kb-form"),
  kbName: document.getElementById("kb-name"),
  kbId: document.getElementById("kb-id"),
  kbEmbeddingModel: document.getElementById("kb-embedding-model"),
  kbList: document.getElementById("kb-list"),
  knowledgeDocumentList: document.getElementById("knowledge-document-list"),
  fileDocumentList: document.getElementById("file-document-list"),
  uploadDocForm: document.getElementById("upload-doc-form"),
  uploadDocFiles: document.getElementById("upload-doc-files"),
  uploadDocTrigger: document.getElementById("upload-doc-trigger"),
  uploadDocSelection: document.getElementById("upload-doc-selection"),
  chatConfigForm: document.getElementById("chat-config-form"),
  chatModel: document.getElementById("chat-model"),
  newChatSession: document.getElementById("new-chat-session"),
  chatTranscript: document.getElementById("chat-transcript"),
  chatMessageForm: document.getElementById("chat-message-form"),
  chunkModal: document.getElementById("chunk-modal"),
  chunkModalBackdrop: document.getElementById("chunk-modal-backdrop"),
  chunkModalClose: document.getElementById("chunk-modal-close"),
  previewModalKicker: document.getElementById("preview-modal-kicker"),
  chunkModalTitle: document.getElementById("chunk-modal-title"),
  chunkModalBody: document.getElementById("chunk-modal-body"),
};

function showFeedback(message, variant = "success") {
  els.feedback.textContent = message;
  els.feedback.className = `feedback ${variant}`;
}

function clearFeedback() {
  els.feedback.textContent = "";
  els.feedback.className = "feedback hidden";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatRichText(value) {
  return escapeHtml(value).replaceAll("\n", "<br>");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_error) {
      // keep status text
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

function currentKnowledgeBase() {
  return state.bootstrap?.knowledge_bases?.find((item) => item.identifier === state.selectedKbId) || null;
}

function renderHeaderContext() {
  const kb = currentKnowledgeBase();
  const label = kb ? kb.vector_db_name || kb.identifier : "尚未选择";
  els.currentKbLabel.textContent = label;
  els.chatKbSummary.textContent = label;
}

function fillModelSelects() {
  const embeddingModels = state.bootstrap?.embedding_models || [];
  const chatModels = state.bootstrap?.chat_models || [];

  els.kbEmbeddingModel.innerHTML = embeddingModels
    .map(
      (model) =>
        `<option value="${model.identifier}" ${
          model.identifier === state.bootstrap.default_embedding_model_id ? "selected" : ""
        }>${model.identifier}</option>`,
    )
    .join("");

  els.chatModel.innerHTML = chatModels
    .map(
      (modelId) =>
        `<option value="${modelId}" ${modelId === state.bootstrap.default_chat_model_id ? "selected" : ""}>${modelId}</option>`,
    )
    .join("");
}

function setActiveTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  });
}

function renderKnowledgeBases() {
  const items = state.bootstrap?.knowledge_bases || [];
  if (!items.length) {
    els.kbList.innerHTML = `<div class="empty-state">还没有知识库。先创建一个，再导入文件。</div>`;
    return;
  }

  els.kbList.innerHTML = items
    .map((kb) => {
      const active = kb.identifier === state.selectedKbId ? "active" : "";
      return `
        <div class="list-item ${active}">
          <div class="list-item-header">
            <div>
              <strong>${kb.vector_db_name || kb.identifier}</strong>
              <div class="document-meta">${kb.identifier}</div>
              <div class="document-meta">${kb.embedding_model} · ${kb.provider_id}</div>
            </div>
            <div class="list-actions">
              <button class="mini-button" data-action="select-kb" data-id="${kb.identifier}">选中</button>
              <button class="mini-button" data-action="delete-kb" data-id="${kb.identifier}">删除</button>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderDocuments(target, documents) {
  if (!state.selectedKbId) {
    target.innerHTML = `<div class="empty-state">先选择一个知识库。</div>`;
    return;
  }

  if (!documents.length) {
    target.innerHTML = `<div class="empty-state">当前知识库还没有文档。</div>`;
    return;
  }

  target.innerHTML = documents
    .map(
      (doc) => `
        <div class="list-item">
          <div class="list-item-header">
            <div>
              <strong>${doc.title}</strong>
              <div class="document-meta">${doc.document_id}</div>
              <div class="document-meta">${doc.source_type} · ${doc.mime_type || "未知类型"}</div>
            </div>
            <div class="list-actions">
              <button class="mini-button" data-action="view-chunks" data-document-id="${doc.document_id}" data-document-title="${doc.title}">查看分块</button>
              <button class="mini-button danger-button" data-action="delete-document" data-document-id="${doc.document_id}" data-document-title="${doc.title}">删除文件</button>
            </div>
          </div>
        </div>
      `,
    )
    .join("");
}

function openChunkModal() {
  els.chunkModal.classList.remove("hidden");
  els.chunkModal.setAttribute("aria-hidden", "false");
}

function closeChunkModal() {
  els.chunkModal.classList.add("hidden");
  els.chunkModal.setAttribute("aria-hidden", "true");
}

function openPreviewModal({ kicker, title, body }) {
  els.previewModalKicker.textContent = kicker;
  els.chunkModalTitle.textContent = title;
  els.chunkModalBody.innerHTML = body;
  openChunkModal();
}

function renderDocumentChunks(chunks) {
  const empty = `<div class="empty-state">没有可展示的分块。若这是较早导入的旧文档，需要重新导入一次，系统才会记录 chunk 明细。</div>`;
  if (!chunks.length) {
    openPreviewModal({
      kicker: "分块预览",
      title: state.selectedDocumentTitle ? `文档：${state.selectedDocumentTitle}` : "查看切块结果",
      body: empty,
    });
    return;
  }

  const html = chunks
    .map(
      (chunk) => `
        <div class="chunk-card">
          <div class="chunk-card-header">
            <strong>${escapeHtml(chunk.chunk_id)}</strong>
            <span>${escapeHtml(chunk.metadata.chunking_strategy || "unknown")}</span>
          </div>
          <p>${escapeHtml(chunk.content)}</p>
          <div class="chunk-meta">chunk_index: ${escapeHtml(chunk.metadata.chunk_index ?? "n/a")}</div>
        </div>
      `,
    )
    .join("");

  openPreviewModal({
    kicker: "分块预览",
    title: state.selectedDocumentTitle ? `文档：${state.selectedDocumentTitle}` : "查看切块结果",
    body: html,
  });
}

async function getDocumentChunks(documentId, documentTitle) {
  if (!state.selectedKbId) {
    return [];
  }
  state.selectedDocumentId = documentId;
  state.selectedDocumentTitle = documentTitle;
  const chunks = await api(`/api/knowledge-bases/${encodeURIComponent(state.selectedKbId)}/documents/${encodeURIComponent(documentId)}/chunks`);
  return chunks;
}

async function loadDocuments() {
  if (!state.selectedKbId) {
    renderDocuments(els.knowledgeDocumentList, []);
    renderDocuments(els.fileDocumentList, []);
    state.selectedDocumentId = null;
    state.selectedDocumentTitle = null;
    return;
  }

  const documents = await api(`/api/knowledge-bases/${state.selectedKbId}/documents`);
  renderDocuments(els.knowledgeDocumentList, documents);
  renderDocuments(els.fileDocumentList, documents);
  if (state.selectedDocumentId && !documents.some((doc) => doc.document_id === state.selectedDocumentId)) {
    state.selectedDocumentId = null;
    state.selectedDocumentTitle = null;
  }
}

function renderChatTranscript(entries) {
  els.chatTranscript.dataset.entries = JSON.stringify(entries);
  if (!entries.length) {
    els.chatTranscript.innerHTML = `<div class="empty-state">先创建会话，再开始提问。</div>`;
    return;
  }

  els.chatTranscript.innerHTML = entries
    .map(
      (entry) => `
        <div class="chat-bubble ${entry.role}">
          <p class="chat-role">${entry.role === "user" ? "用户" : "助手"}</p>
          <div>${formatRichText(entry.content)}</div>
          ${
            entry.citations?.length
              ? `<div class="citations">
                  ${entry.citations
                    .map(
                      (citation) => `
                        <div class="citation-card">
                          <div class="citation-header">
                            <strong>[${citation.citation_index ?? "?"}] ${escapeHtml(citation.document_id || "未知文档")}</strong>
                            <span class="citation-meta">${
                              citation.score !== null && citation.score !== undefined ? citation.score.toFixed(4) : "无分数"
                            }</span>
                          </div>
                          <div>${formatRichText(citation.snippet || "")}</div>
                        </div>
                      `,
                    )
                    .join("")}
                </div>`
              : ""
          }
        </div>
      `,
    )
    .join("");
}

async function refreshBootstrap() {
  clearFeedback();
  const bootstrap = await api("/api/bootstrap");
  state.bootstrap = bootstrap;

  if (state.selectedKbId && !bootstrap.knowledge_bases.some((kb) => kb.identifier === state.selectedKbId)) {
    state.selectedKbId = null;
    state.chatSessionId = null;
  }
  if (!state.selectedKbId && bootstrap.knowledge_bases.length) {
    state.selectedKbId = bootstrap.knowledge_bases[0].identifier;
  }

  fillModelSelects();
  renderKnowledgeBases();
  renderHeaderContext();
  await loadDocuments();
}

function requireKbSelection() {
  if (!state.selectedKbId) {
    throw new Error("请先选择或创建一个知识库");
  }
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

els.refreshBootstrap.addEventListener("click", async () => {
  try {
    await refreshBootstrap();
    showFeedback("状态已刷新。");
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

els.kbForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const payload = {
      name: els.kbName.value,
      vector_db_id: els.kbId.value || null,
      embedding_model: els.kbEmbeddingModel.value || null,
    };
    const kb = await api("/api/knowledge-bases", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refreshBootstrap();
    state.selectedKbId = kb.identifier;
    renderKnowledgeBases();
    renderHeaderContext();
    await loadDocuments();
    els.kbForm.reset();
    fillModelSelects();
    showFeedback(`知识库 ${kb.identifier} 已创建。`);
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

els.kbList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  const action = target.dataset.action;
  const id = target.dataset.id;
  if (!action || !id) return;

  try {
    if (action === "select-kb") {
      state.selectedKbId = id;
      state.chatSessionId = null;
      renderKnowledgeBases();
      renderHeaderContext();
      await loadDocuments();
      showFeedback(`已切换到 ${id}。`);
      return;
    }

    if (action === "delete-kb") {
      await api(`/api/knowledge-bases/${id}`, { method: "DELETE" });
      if (state.selectedKbId === id) {
        state.selectedKbId = null;
        state.chatSessionId = null;
      }
      await refreshBootstrap();
      showFeedback(`已删除 ${id}。`);
    }
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

function wireDocumentListClick(container) {
  container.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.dataset.action;
    const documentId = target.dataset.documentId;
    const documentTitle = target.dataset.documentTitle || documentId;
    if (!action || !documentId) return;

    try {
      if (action === "view-chunks") {
        const chunks = await getDocumentChunks(documentId, documentTitle);
        renderDocumentChunks(chunks);
        showFeedback(`已加载文档 ${documentTitle} 的分块结果。`);
        return;
      }

      if (action === "delete-document") {
        const confirmed = window.confirm(`删除文件后将同时移除对应分块和检索内容：${documentTitle}`);
        if (!confirmed) {
          return;
        }
        await api(
          `/api/knowledge-bases/${encodeURIComponent(state.selectedKbId)}/documents/${encodeURIComponent(documentId)}`,
          { method: "DELETE" },
        );
        if (state.selectedDocumentId === documentId) {
          state.selectedDocumentId = null;
          state.selectedDocumentTitle = null;
          closeChunkModal();
        }
        await loadDocuments();
        showFeedback(`已删除文件 ${documentTitle}。`);
      }
    } catch (error) {
      showFeedback(error.message, "error");
    }
  });
}

wireDocumentListClick(els.knowledgeDocumentList);
wireDocumentListClick(els.fileDocumentList);

els.chunkModalClose.addEventListener("click", closeChunkModal);
els.chunkModalBackdrop.addEventListener("click", closeChunkModal);

els.uploadDocForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    requireKbSelection();
    const files = els.uploadDocFiles.files;
    if (!files.length) throw new Error("请至少选择一个文件");

    showFeedback("文件已选中，正在上传并交给 MinerU 解析……");
    const formData = new FormData();
    formData.append("chunk_size_in_tokens", document.getElementById("upload-doc-chunk-size").value);
    for (const file of files) {
      formData.append("files", file);
    }
    await api(`/api/knowledge-bases/${state.selectedKbId}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    await loadDocuments();
    els.uploadDocForm.reset();
    els.uploadDocSelection.textContent = "尚未选择文件";
    showFeedback("文件已导入。");
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

els.uploadDocTrigger.addEventListener("click", () => {
  els.uploadDocFiles.click();
});

els.uploadDocFiles.addEventListener("change", () => {
  const files = Array.from(els.uploadDocFiles.files || []);
  if (!files.length) {
    els.uploadDocSelection.textContent = "尚未选择文件";
    return;
  }
  if (files.length === 1) {
    els.uploadDocSelection.textContent = files[0].name;
    showFeedback(`已选择文件：${files[0].name}`);
    return;
  }
  els.uploadDocSelection.textContent = `已选择 ${files.length} 个文件`;
  showFeedback(`已选择 ${files.length} 个文件`);
});

els.chatConfigForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    requireKbSelection();
    const session = await api("/api/chat/sessions", {
      method: "POST",
      body: JSON.stringify({
        vector_db_id: state.selectedKbId,
        model_id: document.getElementById("chat-model").value,
        instructions: document.getElementById("chat-instructions").value,
        mode: document.getElementById("chat-mode").value,
        max_chunks: Number(document.getElementById("chat-max-chunks").value),
        ranker_type: document.getElementById("chat-ranker").value,
        alpha: Number(document.getElementById("chat-alpha").value),
        impact_factor: Number(document.getElementById("chat-impact").value),
      }),
    });
    state.chatSessionId = session.session_id;
    renderChatTranscript([]);
    setActiveTab("chat");
    showFeedback(`会话 ${session.session_id} 已创建。`);
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

els.newChatSession.addEventListener("click", () => {
  state.chatSessionId = null;
  renderChatTranscript([]);
  showFeedback("当前会话已清空。");
});

els.chatMessageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (!state.chatSessionId) {
      throw new Error("请先创建对话会话");
    }
    const message = document.getElementById("chat-message").value;
    const entries = els.chatTranscript.dataset.entries ? JSON.parse(els.chatTranscript.dataset.entries) : [];
    entries.push({ role: "user", content: message });
    renderChatTranscript(entries);

    const response = await api(`/api/chat/sessions/${state.chatSessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    entries.push({ role: "assistant", content: response.answer, citations: response.citations });
    renderChatTranscript(entries);
    els.chatMessageForm.reset();
    clearFeedback();
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

window.addEventListener("load", async () => {
  try {
    setActiveTab("knowledge");
    await refreshBootstrap();
    renderChatTranscript([]);
  } catch (error) {
    showFeedback(error.message, "error");
  }
});
