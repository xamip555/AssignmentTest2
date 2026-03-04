const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadMessage = document.getElementById("uploadMessage");
const documentsContainer = document.getElementById("documents");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const answerBlock = document.getElementById("answerBlock");
const answerText = document.getElementById("answerText");
const citationsContainer = document.getElementById("citations");
const sourcesContainer = document.getElementById("sources");
const sourcesPanel = document.getElementById("sourcesPanel");

async function fetchDocuments() {
  const res = await fetch("/documents");
  const docs = await res.json();

  if (!docs.length) {
    documentsContainer.innerHTML = "<div class='muted'>No documents uploaded yet.</div>";
    return;
  }

  documentsContainer.innerHTML = "";
  for (const doc of docs) {
    const row = document.createElement("div");
    row.className = "doc-row";

    const left = document.createElement("div");
    left.innerHTML = `<strong>${escapeHtml(doc.name)}</strong><br><span class='muted'>${escapeHtml(doc.upload_time)}</span>`;

    const delBtn = document.createElement("button");
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async () => {
      await fetch(`/documents/${doc.id}`, { method: "DELETE" });
      await fetchDocuments();
    });

    row.appendChild(left);
    row.appendChild(delBtn);
    documentsContainer.appendChild(row);
  }
}

async function uploadFile() {
  if (!fileInput.files.length) {
    uploadMessage.textContent = "Choose a file first.";
    return;
  }

  uploadMessage.textContent = "Uploading...";
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("/upload", {
    method: "POST",
    body: formData,
  });

  const payload = await res.json();
  if (!res.ok) {
    uploadMessage.textContent = payload.detail || "Upload failed.";
    return;
  }

  uploadMessage.textContent = `Uploaded ${payload.name} (${payload.chunks_indexed} chunks indexed).`;
  fileInput.value = "";
  await fetchDocuments();
}

function renderCitations(citations) {
  citationsContainer.innerHTML = "";

  if (!citations.length) {
    citationsContainer.innerHTML = "<div class='muted'>No citations returned.</div>";
    return;
  }

  for (const c of citations) {
    const button = document.createElement("button");
    button.className = "citation-btn";
    button.textContent = `[${c.doc_name} - chunk ${c.chunk_id}]`;
    button.dataset.target = `source-chunk-${c.chunk_id}`;
    button.addEventListener("click", () => jumpToSource(button.dataset.target));

    citationsContainer.appendChild(button);
  }
}

function renderSources(sources) {
  sourcesContainer.innerHTML = "";

  if (!sources.length) {
    sourcesContainer.innerHTML = "<div class='muted'>No source chunks retrieved.</div>";
    return;
  }

  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.id = `source-chunk-${source.chunk_id}`;
    item.innerHTML = `
      <div><strong>${escapeHtml(source.doc_name)}</strong> - chunk ${source.chunk_id} (index ${source.chunk_index})</div>
      <div class="muted">Similarity score: ${Number(source.score).toFixed(4)}</div>
      <div>${escapeHtml(source.text)}</div>
    `;
    sourcesContainer.appendChild(item);
  }
}

function jumpToSource(targetId) {
  const element = document.getElementById(targetId);
  if (!element) return;

  sourcesPanel.open = true;
  element.scrollIntoView({ behavior: "smooth", block: "start" });

  element.classList.add("highlight");
  setTimeout(() => element.classList.remove("highlight"), 1800);
}

function formatAnswer(text) {
  return (text || "").replace(/\s+/g, " ").trim();
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = "Asking...";

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const payload = await res.json();
  askBtn.disabled = false;
  askBtn.textContent = "Ask";

  if (!res.ok) {
    answerBlock.style.display = "block";
    answerText.textContent = payload.detail || "Request failed.";
    citationsContainer.innerHTML = "";
    sourcesContainer.innerHTML = "";
    return;
  }

  answerBlock.style.display = "block";
  answerText.textContent = formatAnswer(payload.answer);
  renderCitations(payload.citations || []);
  renderSources(payload.sources || []);
}

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

uploadBtn.addEventListener("click", uploadFile);
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    askQuestion();
  }
});

fetchDocuments();
