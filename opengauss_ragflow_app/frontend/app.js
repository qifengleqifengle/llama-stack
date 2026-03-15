const state = {
  bootstrap: null,
  activeTab: "knowledge",
  selectedKbId: null,
  selectedDocumentId: null,
  selectedDocumentTitle: null,
  chatSessionId: null,
  chatSessionKbId: null,
  statusTimer: null,
};

const els = {
  currentKbLabel: document.getElementById("current-kb-label"),
  openKbModal: document.getElementById("open-kb-modal"),
  openUploadModal: document.getElementById("open-upload-modal"),
  kbForm: document.getElementById("kb-form"),
  kbName: document.getElementById("kb-name"),
  kbId: document.getElementById("kb-id"),
  kbEmbeddingModel: document.getElementById("kb-embedding-model"),
  kbList: document.getElementById("kb-list"),
  knowledgeDocumentList: document.getElementById("knowledge-document-list"),
  uploadDocForm: document.getElementById("upload-doc-form"),
  uploadDocFiles: document.getElementById("upload-doc-files"),
  uploadDocTrigger: document.getElementById("upload-doc-trigger"),
  uploadDocSelection: document.getElementById("upload-doc-selection"),
  uploadDocStrategy: document.getElementById("upload-doc-strategy"),
  uploadDocChunkSize: document.getElementById("upload-doc-chunk-size"),
  uploadDocCharSize: document.getElementById("upload-doc-char-size"),
  uploadDocTokenSizeGroup: document.getElementById("upload-doc-token-size-group"),
  uploadDocCharSizeGroup: document.getElementById("upload-doc-char-size-group"),
  chatKbSelect: document.getElementById("chat-kb-select"),
  chatModel: document.getElementById("chat-model"),
  chatQueryRewrite: document.getElementById("chat-query-rewrite"),
  newChatSession: document.getElementById("new-chat-session"),
  chatTranscript: document.getElementById("chat-transcript"),
  chatMessageForm: document.getElementById("chat-message-form"),
  chunkModal: document.getElementById("chunk-modal"),
  chunkModalBackdrop: document.getElementById("chunk-modal-backdrop"),
  chunkModalClose: document.getElementById("chunk-modal-close"),
  chunkModalTitle: document.getElementById("chunk-modal-title"),
  chunkModalBody: document.getElementById("chunk-modal-body"),
  kbModal: document.getElementById("kb-modal"),
  uploadModal: document.getElementById("upload-modal"),
  statusModal: document.getElementById("status-modal"),
  statusModalMessage: document.getElementById("status-modal-message"),
  confirmModal: document.getElementById("confirm-modal"),
  confirmModalMessage: document.getElementById("confirm-modal-message"),
  confirmModalOk: document.getElementById("confirm-modal-ok"),
  confirmModalCancel: document.getElementById("confirm-modal-cancel"),
};

function showFeedback(message, variant = "success") {
  if (state.statusTimer) {
    window.clearTimeout(state.statusTimer);
  }
  els.statusModalMessage.textContent = message;
  els.statusModal.className = `status-modal ${variant}`;
  els.statusModal.classList.remove("hidden");
  els.statusModal.setAttribute("aria-hidden", "false");
  state.statusTimer = window.setTimeout(() => {
    clearFeedback();
  }, 2200);
}

function clearFeedback() {
  if (state.statusTimer) {
    window.clearTimeout(state.statusTimer);
    state.statusTimer = null;
  }
  els.statusModal.className = "status-modal hidden";
  els.statusModal.setAttribute("aria-hidden", "true");
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
  const label = kb ? kb.vector_db_name || kb.identifier : "未选择";
  els.currentKbLabel.textContent = label;
}

function fillModelSelects() {
  const embeddingModels = state.bootstrap?.embedding_models || [];
  const chatModels = state.bootstrap?.chat_models || [];
  const knowledgeBases = state.bootstrap?.knowledge_bases || [];

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

  els.chatQueryRewrite.checked = Boolean(state.bootstrap?.default_query_rewrite);

  els.chatKbSelect.innerHTML = knowledgeBases.length
    ? knowledgeBases
        .map((kb) => {
          const selected = kb.identifier === state.selectedKbId ? "selected" : "";
          return `<option value="${kb.identifier}" ${selected}>${escapeHtml(kb.vector_db_name || kb.identifier)}</option>`;
        })
        .join("")
    : '<option value="">暂无知识库</option>';
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
      const name = kb.vector_db_name || kb.identifier;
      return `
        <div class="list-item kb-item ${active}" data-kb-id="${kb.identifier}">
          <button class="kb-row" type="button" data-action="select-kb" data-id="${kb.identifier}">
            <strong class="kb-name">${escapeHtml(name)}</strong>
            ${active ? '<span class="kb-active-dot" aria-hidden="true"></span>' : ""}
          </button>
          <div class="list-actions">
            <button class="icon-button danger-button" type="button" data-action="delete-kb" data-id="${kb.identifier}" title="删除知识库" aria-label="删除知识库">
              删除
            </button>
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
              <div class="document-meta">${doc.source_type} · ${doc.mime_type || "未知类型"}</div>
            </div>
            <div class="list-actions">
              <button class="mini-button" data-action="view-chunks" data-document-id="${doc.document_id}" data-document-title="${doc.title}">查看分块</button>
              <button class="mini-button danger-button" data-action="delete-document" data-document-id="${doc.document_id}" data-document-title="${doc.title}">删除</button>
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

function openModal(modal) {
  if (!modal) return;
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal(modal) {
  if (!modal) return;
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}

function openPreviewModal({ title, body }) {
  els.chunkModalTitle.textContent = title;
  els.chunkModalBody.innerHTML = body;
  openChunkModal();
}

function renderDocumentChunks(chunks) {
  const empty = `<div class="empty-state">没有可展示的分块。若这是较早导入的旧文档，需要重新导入一次，系统才会记录 chunk 明细。</div>`;
  if (!chunks.length) {
    openPreviewModal({
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
    state.selectedDocumentId = null;
    state.selectedDocumentTitle = null;
    return;
  }

  const documents = await api(`/api/knowledge-bases/${state.selectedKbId}/documents`);
  renderDocuments(els.knowledgeDocumentList, documents);
  if (state.selectedDocumentId && !documents.some((doc) => doc.document_id === state.selectedDocumentId)) {
    state.selectedDocumentId = null;
    state.selectedDocumentTitle = null;
  }
}

function renderChatTranscript(entries) {
  els.chatTranscript.dataset.entries = JSON.stringify(entries);
  if (!entries.length) {
    els.chatTranscript.innerHTML = `<div class="empty-state">选择知识库后直接提问。首次发送会自动创建对话上下文。</div>`;
    return;
  }

  els.chatTranscript.innerHTML = entries
    .map(
      (entry) => `
        <div class="chat-bubble ${entry.role}">
          <p class="chat-role">${entry.role === "user" ? "用户" : "助手"}</p>
          <div class="chat-answer">${formatRichText(entry.content)}</div>
          ${
            entry.citations?.length
              ? `<div class="citations">
                  <div class="citation-section-title">引用段落</div>
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
  if (els.chatKbSelect.value !== (state.selectedKbId || "")) {
    els.chatKbSelect.value = state.selectedKbId || "";
  }
}

function requireKbSelection() {
  if (!state.selectedKbId) {
    throw new Error("请先选择或创建一个知识库");
  }
}

function resetChatSession(clearTranscript = true) {
  state.chatSessionId = null;
  state.chatSessionKbId = null;
  if (clearTranscript) {
    renderChatTranscript([]);
  }
}

function updateUploadChunkControls() {
  const strategy = els.uploadDocStrategy.value;
  const showTokenSize = strategy === "fixed_tokens" || strategy === "recursive";
  const showCharSize = strategy === "fixed_chars" || strategy === "recursive";
  els.uploadDocTokenSizeGroup.classList.toggle("hidden", !showTokenSize);
  els.uploadDocCharSizeGroup.classList.toggle("hidden", !showCharSize);
}

function currentChatConfig() {
  requireKbSelection();
  return {
    vector_db_id: state.selectedKbId,
    model_id: document.getElementById("chat-model").value,
    instructions: document.getElementById("chat-instructions").value,
    mode: document.getElementById("chat-mode").value,
    query_rewrite: els.chatQueryRewrite.checked,
    max_chunks: Number(document.getElementById("chat-max-chunks").value),
    ranker_type: document.getElementById("chat-ranker").value,
    alpha: Number(document.getElementById("chat-alpha").value),
    impact_factor: Number(document.getElementById("chat-impact").value),
  };
}

async function ensureChatSession() {
  const config = currentChatConfig();
  if (state.chatSessionId && state.chatSessionKbId === config.vector_db_id) {
    return state.chatSessionId;
  }

  const session = await api("/api/chat/sessions", {
    method: "POST",
    body: JSON.stringify(config),
  });
  state.chatSessionId = session.session_id;
  state.chatSessionKbId = config.vector_db_id;
  renderChatTranscript([]);
  return session.session_id;
}

function confirmAction(message) {
  return new Promise((resolve) => {
    els.confirmModalMessage.textContent = message;
    openModal(els.confirmModal);

    const cleanup = () => {
      els.confirmModalOk.removeEventListener("click", onOk);
      els.confirmModalCancel.removeEventListener("click", onCancel);
      closeModal(els.confirmModal);
    };

    const onOk = () => {
      cleanup();
      resolve(true);
    };

    const onCancel = () => {
      cleanup();
      resolve(false);
    };

    els.confirmModalOk.addEventListener("click", onOk, { once: true });
    els.confirmModalCancel.addEventListener("click", onCancel, { once: true });
  });
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

els.openKbModal?.addEventListener("click", () => openModal(els.kbModal));
els.openUploadModal?.addEventListener("click", () => openModal(els.uploadModal));

document.querySelectorAll("[data-close-modal]").forEach((button) => {
  button.addEventListener("click", () => {
    const modalId = button.getAttribute("data-close-modal");
    if (!modalId) return;
    closeModal(document.getElementById(modalId));
  });
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
    closeModal(els.kbModal);
    fillModelSelects();
    showFeedback(`已创建知识库：${kb.vector_db_name || kb.identifier}`);
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
      resetChatSession();
      renderKnowledgeBases();
      renderHeaderContext();
      await loadDocuments();
      if (els.chatKbSelect.value !== id) {
        els.chatKbSelect.value = id;
      }
      return;
    }

    if (action === "delete-kb") {
      const confirmed = await confirmAction(`确认删除知识库 ${id}？该知识库下的文件记录也会一起移除。`);
      if (!confirmed) {
        return;
      }
      await api(`/api/knowledge-bases/${id}`, { method: "DELETE" });
      if (state.selectedKbId === id) {
        state.selectedKbId = null;
        resetChatSession();
      }
      await refreshBootstrap();
      showFeedback(`已删除知识库：${id}`);
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
        return;
      }

      if (action === "delete-document") {
        const confirmed = await confirmAction(`确认删除文件 ${documentTitle}？对应分块和检索内容会一起移除。`);
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
        showFeedback(`已删除文件：${documentTitle}`);
      }
    } catch (error) {
      showFeedback(error.message, "error");
    }
  });
}

wireDocumentListClick(els.knowledgeDocumentList);

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
    formData.append("chunking_strategy", els.uploadDocStrategy.value);
    formData.append("chunk_size_in_tokens", els.uploadDocChunkSize.value);
    formData.append("chunk_size_in_chars", els.uploadDocCharSize.value);
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
    updateUploadChunkControls();
    closeModal(els.uploadModal);
    showFeedback("文件已导入。");
  } catch (error) {
    showFeedback(error.message, "error");
  }
});

els.uploadDocTrigger.addEventListener("click", () => {
  els.uploadDocFiles.click();
});

els.uploadDocStrategy.addEventListener("change", updateUploadChunkControls);

els.uploadDocFiles.addEventListener("change", () => {
  const files = Array.from(els.uploadDocFiles.files || []);
  if (!files.length) {
    els.uploadDocSelection.textContent = "尚未选择文件";
    return;
  }
  if (files.length === 1) {
    els.uploadDocSelection.textContent = files[0].name;
    return;
  }
  els.uploadDocSelection.textContent = `已选择 ${files.length} 个文件`;
});

els.newChatSession.addEventListener("click", () => {
  resetChatSession();
});

els.chatKbSelect.addEventListener("change", async () => {
  const nextKbId = els.chatKbSelect.value || null;
  if (nextKbId === state.selectedKbId) {
    return;
  }
  state.selectedKbId = nextKbId;
  resetChatSession();
  renderKnowledgeBases();
  renderHeaderContext();
  await loadDocuments();
});

[
  document.getElementById("chat-model"),
  document.getElementById("chat-instructions"),
  document.getElementById("chat-mode"),
  document.getElementById("chat-max-chunks"),
  document.getElementById("chat-ranker"),
  document.getElementById("chat-alpha"),
  document.getElementById("chat-impact"),
  document.getElementById("chat-query-rewrite"),
].forEach((element) => {
  element.addEventListener("change", () => {
    resetChatSession();
  });
});

els.chatMessageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const sessionId = await ensureChatSession();
    const message = document.getElementById("chat-message").value;
    const entries = els.chatTranscript.dataset.entries ? JSON.parse(els.chatTranscript.dataset.entries) : [];
    entries.push({ role: "user", content: message });
    renderChatTranscript(entries);

    const response = await api(`/api/chat/sessions/${sessionId}/messages`, {
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
    updateUploadChunkControls();
    renderChatTranscript([]);
  } catch (error) {
    showFeedback(error.message, "error");
  }
});
