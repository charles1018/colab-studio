// ============================================================================
// Colab Studio Local Web UI — Frontend Interactive Engine (ES Module)
// ============================================================================

const ASPECT_MAP_2K = {
  "1:1": [2048, 2048],
  "16:9": [2752, 1536],
  "9:16": [1536, 2752],
  "4:3": [2400, 1792],
  "3:4": [1792, 2400],
  "3:2": [2528, 1696],
  "2:3": [1696, 2528],
};

const ASPECT_MAP_1K = {
  "1:1": [1024, 1024],
  "16:9": [1376, 768],
  "9:16": [768, 1376],
  "4:3": [1216, 896],
  "3:4": [896, 1216],
  "3:2": [1280, 832],
  "2:3": [832, 1280],
};

const SWAP_AR_MAP = {
  "1:1": "1:1",
  "16:9": "9:16",
  "9:16": "16:9",
  "4:3": "3:4",
  "3:4": "4:3",
  "3:2": "2:3",
  "2:3": "3:2",
};

const PLACEHOLDERS = {
  t2i: "例如：台北街頭的霓虹拉麵攤，雨夜地面倒影，招牌寫著「極上豚骨」，電影級光影與蒸氣細節...",
  edit: "例如：將 <image1> 主畫布中的角色換上 <image2> 的賽博龐克發光外套，背景改為夜晚的東京鐵塔，保留角色臉部特徵...",
};

// Application State
const ui = {
  mode: "t2i", // "t2i" | "edit"
  aspectRatio: "1:1",
  resolution: "1K", // "1K" | "2K"
  editSizing: "match_1k", // "match_1k" | "match_2k" | "match_orig" | "custom"
  // Reference images array: up to 10 items
  // Each item: { id: string, kind: "upload" | "artifact", file?: File, artifact_id?: string, previewUrl: string, name: string }
  refImages: [],
  galleryItems: [],
  selectedItem: null,
  lastSeed: "",
  loginFlowState: null,
  viewId: null,
  retiredViews: new Set(),
  lastRevision: -1,
  lastStatus: null,
  finishedJob: null,
  wasRunning: false,
  preparing: false,
  submitting: false,
  canCancel: false,
  runId: null,
  runSession: null,
  activeSession: null,
  remainingSec: null,
  countdownOwned: false,
};

// DOM Elements
const el = {
  authDot: document.getElementById("authDot"),
  authLabel: document.getElementById("authLabel"),
  btnToggleLogin: document.getElementById("btnToggleLogin"),
  cuValue: document.getElementById("cuValue"),
  sessionStatusPill: document.getElementById("sessionStatusPill"),
  activeSessionLabel: document.getElementById("activeSessionLabel"),
  countdownBadge: document.getElementById("countdownBadge"),
  gpuSelect: document.getElementById("gpuSelect"),
  quantSelect: document.getElementById("quantSelect"),
  btnRefreshColab: document.getElementById("btnRefreshColab"),
  colabRefreshError: document.getElementById("colabRefreshError"),
  btnKeep30: document.getElementById("btnKeep30"),
  btnKeep60: document.getElementById("btnKeep60"),
  btnCancelTop: document.getElementById("btnCancelTop"),
  btnStopTop: document.getElementById("btnStopTop"),

  loginDrawer: document.getElementById("loginDrawer"),
  btnGetLoginUrl: document.getElementById("btnGetLoginUrl"),
  loginUrlLink: document.getElementById("loginUrlLink"),
  loginCodeInput: document.getElementById("loginCodeInput"),
  btnSubmitLoginCode: document.getElementById("btnSubmitLoginCode"),
  btnCloseLoginDrawer: document.getElementById("btnCloseLoginDrawer"),

  tabT2I: document.getElementById("tabT2I"),
  tabEdit: document.getElementById("tabEdit"),
  editReferenceSection: document.getElementById("editReferenceSection"),
  refCountBadge: document.getElementById("refCountBadge"),
  imageTokenBar: document.getElementById("imageTokenBar"),
  refDropzone: document.getElementById("refDropzone"),
  refFileInput: document.getElementById("refFileInput"),
  refCardsGrid: document.getElementById("refCardsGrid"),
  btnAddRefImage: document.getElementById("btnAddRefImage"),
  btnClearRefImages: document.getElementById("btnClearRefImages"),
  editSizingSelect: document.getElementById("editSizingSelect"),

  promptSectionTitle: document.getElementById("promptSectionTitle"),
  promptInput: document.getElementById("promptInput"),
  btnClearPrompt: document.getElementById("btnClearPrompt"),
  chkTransparent: document.getElementById("chkTransparent"),
  chkRefinePrompt: document.getElementById("chkRefinePrompt"),
  peVariantRow: document.getElementById("peVariantRow"),
  peVariantSelect: document.getElementById("peVariantSelect"),

  sizeControlSection: document.getElementById("sizeControlSection"),
  btnSwapAspect: document.getElementById("btnSwapAspect"),
  aspectGrid: document.getElementById("aspectGrid"),
  res1K: document.getElementById("res1K"),
  res2K: document.getElementById("res2K"),
  pixelReadout: document.getElementById("pixelReadout"),

  seedInput: document.getElementById("seedInput"),
  btnRandomSeed: document.getElementById("btnRandomSeed"),
  btnReuseLastSeed: document.getElementById("btnReuseLastSeed"),
  sessionSelect: document.getElementById("sessionSelect"),
  idleMinutesInput: document.getElementById("idleMinutesInput"),
  batchNameInput: document.getElementById("batchNameInput"),
  chkStopAfter: document.getElementById("chkStopAfter"),

  btnGenerate: document.getElementById("btnGenerate"),
  btnCancelRun: document.getElementById("btnCancelRun"),

  toastBanner: document.getElementById("toastBanner"),
  toastMessage: document.getElementById("toastMessage"),
  btnCloseToast: document.getElementById("btnCloseToast"),

  stageEmpty: document.getElementById("stageEmpty"),
  stageImageWrap: document.getElementById("stageImageWrap"),
  stageImage: document.getElementById("stageImage"),

  progressOverlay: document.getElementById("progressOverlay"),
  progressStageText: document.getElementById("progressStageText"),
  progressMetaText: document.getElementById("progressMetaText"),
  stageStepper: document.getElementById("stageStepper"),
  progressLogs: document.getElementById("progressLogs"),

  floatingToolbar: document.getElementById("floatingToolbar"),
  metaBadgeMode: document.getElementById("metaBadgeMode"),
  metaBadgeQuant: document.getElementById("metaBadgeQuant"),
  metaBadgeSize: document.getElementById("metaBadgeSize"),
  metaBadgeSeed: document.getElementById("metaBadgeSeed"),
  btnContinueEdit: document.getElementById("btnContinueEdit"),
  btnAddAsExtraRef: document.getElementById("btnAddAsExtraRef"),
  btnRestoreRecipe: document.getElementById("btnRestoreRecipe"),
  btnTogglePromptDrawer: document.getElementById("btnTogglePromptDrawer"),
  btnDownloadPng: document.getElementById("btnDownloadPng"),

  promptDrawer: document.getElementById("promptDrawer"),
  drawerEnhancerBadge: document.getElementById("drawerEnhancerBadge"),
  drawerOriginalPrompt: document.getElementById("drawerOriginalPrompt"),
  drawerRefinedPrompt: document.getElementById("drawerRefinedPrompt"),
  btnCopyRefinedPrompt: document.getElementById("btnCopyRefinedPrompt"),
  btnUseRefinedPrompt: document.getElementById("btnUseRefinedPrompt"),
  btnClosePromptDrawer: document.getElementById("btnClosePromptDrawer"),

  galleryCount: document.getElementById("galleryCount"),
  btnRefreshGallery: document.getElementById("btnRefreshGallery"),
  galleryStrip: document.getElementById("galleryStrip"),
};

// ----------------------------------------------------------------------------
// Toast & Helpers
// ----------------------------------------------------------------------------
function showToast(msg, isInfo = false) {
  el.toastMessage.textContent = msg;
  el.toastBanner.classList.toggle("info", isInfo);
  el.toastBanner.hidden = false;
}

function hideToast() {
  el.toastBanner.hidden = true;
}

function formatCountdown(sec) {
  if (sec === null || sec === undefined) return "";
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers);
  headers.set("X-Colab-Studio-Request", "1");
  const res = await fetch(url, { ...options, headers, mode: "same-origin" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg = data.detail || data.error || `請求失敗 (${res.status})`;
    if (Array.isArray(msg)) {
      msg = msg.map((x) => x.msg || JSON.stringify(x)).join("; ");
    } else if (typeof msg === "object") {
      msg = JSON.stringify(msg);
    }
    throw new Error(msg);
  }
  return data;
}

// ----------------------------------------------------------------------------
// Mode, Sizing & Reference Image Management
// ----------------------------------------------------------------------------
function setMode(mode) {
  ui.mode = mode === "edit" ? "edit" : "t2i";
  el.tabT2I.classList.toggle("active", ui.mode === "t2i");
  el.tabEdit.classList.toggle("active", ui.mode === "edit");
  el.editReferenceSection.hidden = ui.mode !== "edit";
  el.promptSectionTitle.textContent =
    ui.mode === "edit" ? "✍️ 改圖指令 (Instruction / Prompt)" : "✍️ 提示詞 (Prompt)";
  el.promptInput.placeholder = PLACEHOLDERS[ui.mode];
  updateSizeControlVisibility();
}

function updateSizeControlVisibility() {
  if (ui.mode === "t2i") {
    el.sizeControlSection.hidden = false;
  } else {
    el.sizeControlSection.hidden = ui.editSizing !== "custom";
  }
  updatePixelReadout();
}

function updatePixelReadout() {
  const map = ui.resolution === "2K" ? ASPECT_MAP_2K : ASPECT_MAP_1K;
  const [w, h] = map[ui.aspectRatio] || [1024, 1024];
  el.pixelReadout.textContent = `${w} × ${h} px`;
}

function setAspectRatio(ar) {
  if (!ASPECT_MAP_2K[ar]) return;
  ui.aspectRatio = ar;
  for (const btn of el.aspectGrid.querySelectorAll(".aspect-btn")) {
    btn.classList.toggle("active", btn.dataset.ar === ar);
  }
  updatePixelReadout();
}

function setResolution(res) {
  ui.resolution = res === "2K" ? "2K" : "1K";
  el.res1K.classList.toggle("active", ui.resolution === "1K");
  el.res2K.classList.toggle("active", ui.resolution === "2K");
  updatePixelReadout();
}

function insertTokenAtCursor(token) {
  const ta = el.promptInput;
  const start = ta.selectionStart ?? ta.value.length;
  const end = ta.selectionEnd ?? ta.value.length;
  const before = ta.value.slice(0, start);
  const after = ta.value.slice(end);
  const needSpaceBefore = before.length > 0 && !/\s$/.test(before);
  const needSpaceAfter = after.length > 0 && !/^\s/.test(after);
  const insertText = `${needSpaceBefore ? " " : ""}${token}${needSpaceAfter ? " " : ""}`;
  ta.value = before + insertText + after;
  ta.focus();
  const pos = before.length + insertText.length;
  ta.setSelectionRange(pos, pos);
}

function renderRefImages() {
  const count = ui.refImages.length;
  el.refCountBadge.textContent = `${count} / 10 張`;
  el.btnClearRefImages.hidden = count === 0;
  el.btnAddRefImage.disabled = count >= 10;

  // Render token pills (<image1> .. <imageN>)
  el.imageTokenBar.innerHTML = "";
  const pillCount = Math.max(1, count);
  for (let i = 1; i <= pillCount; i++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "token-pill";
    const label = i === 1 ? `<image1> 主畫布` : `<image${i}>`;
    btn.textContent = `+ ${label}`;
    btn.title = `點擊將 <image${i}> 插入改圖指令游標處`;
    btn.addEventListener("click", () => insertTokenAtCursor(`<image${i}>`));
    el.imageTokenBar.appendChild(btn);
  }

  // Render reference image cards
  el.refCardsGrid.innerHTML = "";
  ui.refImages.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = `ref-card ${idx === 0 ? "primary-canvas" : ""}`;

    const header = document.createElement("div");
    header.className = "ref-card-header";

    const tag = document.createElement("span");
    tag.className = "ref-card-tag";
    tag.textContent = idx === 0 ? `參考圖片 1 (<image1> 主畫布)` : `參考圖片 ${idx + 1} (<image${idx + 1}>)`;

    const ctrls = document.createElement("div");
    ctrls.className = "ref-card-controls";

    if (idx > 0) {
      const btnUp = document.createElement("button");
      btnUp.type = "button";
      btnUp.className = "ref-ctrl-btn";
      btnUp.textContent = "←";
      btnUp.title = "往前移動順序";
      btnUp.addEventListener("click", () => {
        const tmp = ui.refImages[idx - 1];
        ui.refImages[idx - 1] = ui.refImages[idx];
        ui.refImages[idx] = tmp;
        renderRefImages();
      });
      ctrls.appendChild(btnUp);
    }

    const btnDel = document.createElement("button");
    btnDel.type = "button";
    btnDel.className = "ref-ctrl-btn";
    btnDel.textContent = "✕";
    btnDel.title = "移除此參考圖片";
    btnDel.addEventListener("click", () => {
      if (item.kind === "upload" && item.previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(item.previewUrl);
      }
      ui.refImages.splice(idx, 1);
      renderRefImages();
    });
    ctrls.appendChild(btnDel);

    header.appendChild(tag);
    header.appendChild(ctrls);

    const img = document.createElement("img");
    img.className = "ref-card-thumb checkerboard";
    img.src = item.previewUrl;
    img.alt = item.name;
    img.title = `點擊將 <image${idx + 1}> 插入提示詞`;
    img.style.cursor = "pointer";
    img.addEventListener("click", () => insertTokenAtCursor(`<image${idx + 1}>`));

    card.appendChild(header);
    card.appendChild(img);
    el.refCardsGrid.appendChild(card);
  });
}

function addUploadedFiles(fileList) {
  for (const file of fileList) {
    if (!file.type.startsWith("image/")) continue;
    if (ui.refImages.length >= 10) {
      showToast("最多支援 10 張參考圖片 (<image1> ～ <image10>)");
      break;
    }
    const url = URL.createObjectURL(file);
    ui.refImages.push({
      id: `up_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      kind: "upload",
      file,
      previewUrl: url,
      name: file.name,
    });
  }
  renderRefImages();
}

function setArtifactAsReference(item, asPrimary = true) {
  if (!item || !item.artifact_id) return;
  const refEntry = {
    id: `art_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    kind: "artifact",
    artifact_id: item.artifact_id,
    previewUrl: item.url,
    name: item.filename || "gallery.png",
  };
  setMode("edit");
  if (asPrimary) {
    if (ui.refImages.length === 0) {
      ui.refImages.push(refEntry);
    } else {
      ui.refImages[0] = refEntry;
    }
    showToast(`🎨 已將「${item.job_id}」設為圖片編輯模式的 <image1> 主畫布！`, true);
  } else {
    if (ui.refImages.length >= 10) {
      showToast("參考圖片已達 10 張上限");
      return;
    }
    ui.refImages.push(refEntry);
    showToast(`➕ 已將「${item.job_id}」加入為 <image${ui.refImages.length}> 參考圖！`, true);
  }
  renderRefImages();
  el.promptInput.focus();
}

// ----------------------------------------------------------------------------
// Stage Canvas, Recipe Restore & Gallery Strip
// ----------------------------------------------------------------------------
function selectGalleryItem(item, openDrawerIfRefined = false) {
  ui.selectedItem = item;
  if (!item) {
    el.stageEmpty.hidden = false;
    el.stageImageWrap.hidden = true;
    el.floatingToolbar.hidden = true;
    el.promptDrawer.hidden = true;
    return;
  }

  el.stageEmpty.hidden = true;
  el.stageImageWrap.hidden = false;
  el.stageImage.src = item.url;
  el.floatingToolbar.hidden = false;

  const modeLabel = item.mode === "edit" ? "Edit 改圖" : "T2I 生圖";
  const rgbaTag = item.transparent || item.has_alpha ? " · RGBA" : "";
  el.metaBadgeMode.textContent = `${modeLabel}${rgbaTag}`;
  el.metaBadgeQuant.textContent = item.quant || "INT8";
  el.metaBadgeSize.textContent =
    item.width && item.height ? `${item.width}×${item.height}` : `${item.aspect_ratio || "1:1"} (${item.resolution || "1K"})`;
  el.metaBadgeSeed.textContent = `Seed: ${item.seed ?? "--"}`;

  if (item.seed) {
    ui.lastSeed = String(item.seed);
  }

  el.btnDownloadPng.href = item.url;
  el.btnDownloadPng.download = `${item.job_dir || "cstudio"}_${item.filename || "output.png"}`;

  // Populate Prompt Drawer
  el.drawerOriginalPrompt.textContent = item.prompt || "（無原始提示詞紀錄）";
  if (item.refined_prompt) {
    el.drawerRefinedPrompt.textContent = item.refined_prompt;
    el.drawerEnhancerBadge.textContent = `✨ Qwen 3.5 9B (${(item.pe_variant || "int8").toUpperCase()}) 已擴寫`;
    el.btnCopyRefinedPrompt.disabled = false;
    el.btnUseRefinedPrompt.disabled = false;
    if (openDrawerIfRefined) {
      el.promptDrawer.hidden = false;
    }
  } else {
    el.drawerRefinedPrompt.textContent = "（此作品為原始提示詞直出，未啟用 AI 自動擴寫）";
    el.drawerEnhancerBadge.textContent = "直出模式 (未開啟擴寫)";
    el.btnCopyRefinedPrompt.disabled = true;
    el.btnUseRefinedPrompt.disabled = true;
  }

  // Highlight active thumbnail in gallery
  for (const node of el.galleryStrip.querySelectorAll(".gallery-item")) {
    node.classList.toggle("active", node.dataset.artifactId === item.artifact_id);
  }
}

function restoreRecipeFromSelected() {
  const item = ui.selectedItem;
  if (!item) return;

  setMode(item.mode === "edit" ? "edit" : "t2i");
  el.promptInput.value = item.prompt || "";
  if (item.quant) {
    el.quantSelect.value = item.quant;
  }
  if (item.gpu) {
    el.gpuSelect.value = item.gpu;
  }
  if (item.seed !== undefined && item.seed !== null) {
    el.seedInput.value = String(item.seed);
  }
  el.chkTransparent.checked = Boolean(item.transparent);
  el.chkRefinePrompt.checked = Boolean(item.refine_prompt);
  el.peVariantRow.hidden = !el.chkRefinePrompt.checked;
  if (item.pe_variant) {
    el.peVariantSelect.value = item.pe_variant;
  }
  if (item.aspect_ratio && ASPECT_MAP_2K[item.aspect_ratio]) {
    setAspectRatio(item.aspect_ratio);
  }
  if (item.resolution) {
    setResolution(item.resolution);
  }

  if (item.mode === "edit") {
    if (item.custom_size) {
      ui.editSizing = "custom";
    } else if (item.edit_resolution === 2048) {
      ui.editSizing = "match_2k";
    } else if (item.edit_resolution === 0) {
      ui.editSizing = "match_orig";
    } else {
      ui.editSizing = "match_1k";
    }
    el.editSizingSelect.value = ui.editSizing;
    updateSizeControlVisibility();

    if (Array.isArray(item.inputs) && item.inputs.length > 0) {
      ui.refImages = item.inputs.map((inp, idx) => ({
        id: `restored_${idx}_${Date.now()}`,
        kind: "artifact",
        artifact_id: inp.artifact_id,
        previewUrl: inp.url,
        name: inp.filename,
      }));
    } else {
      ui.refImages = [];
    }
    renderRefImages();
  }

  showToast(`↺ 已還原「${item.job_id}」的提示詞、權重規格 (${item.quant}) 與種子碼 (${item.seed})`, true);
}

async function loadGallery(selectNewest = false) {
  try {
    const data = await apiFetch("/api/gallery?limit=120");
    ui.galleryItems = data.items || [];
    el.galleryCount.textContent = String(ui.galleryItems.length);
    el.galleryStrip.innerHTML = "";

    if (ui.galleryItems.length === 0) {
      selectGalleryItem(null);
      const hint = document.createElement("div");
      hint.className = "gallery-empty-hint";
      hint.textContent = "尚無本地歷史圖片（生成完成後將自動同步至此）";
      el.galleryStrip.appendChild(hint);
      return;
    }

    ui.galleryItems.forEach((item) => {
      const card = document.createElement("div");
      card.className = "gallery-item checkerboard";
      card.dataset.artifactId = item.artifact_id;
      card.title = `[${item.job_dir}] ${item.prompt || ""}\n量化: ${item.quant} | Seed: ${item.seed}`;

      if (
        ui.selectedItem &&
        ui.selectedItem.artifact_id === item.artifact_id
      ) {
        card.classList.add("active");
      }

      const img = document.createElement("img");
      img.src = item.url;
      img.alt = item.job_id;
      img.loading = "lazy";
      card.appendChild(img);

      if (item.refined_prompt) {
        const peTag = document.createElement("span");
        peTag.className = "gallery-item-pe";
        peTag.textContent = "✨PE";
        card.appendChild(peTag);
      }

      const qTag = document.createElement("span");
      qTag.className = "gallery-item-tag";
      qTag.textContent = item.quant || "INT8";
      card.appendChild(qTag);

      card.addEventListener("click", () => selectGalleryItem(item, Boolean(item.refined_prompt)));
      el.galleryStrip.appendChild(card);
    });

    if (selectNewest || !ui.selectedItem) {
      selectGalleryItem(ui.galleryItems[0], Boolean(ui.galleryItems[0].refined_prompt));
    }
  } catch (err) {
    console.warn("載入畫廊失敗:", err);
  }
}

// ----------------------------------------------------------------------------
// Fast & Slow Polling (Colab Status + Run Progress)
// ----------------------------------------------------------------------------
function updateStepChips(step, done) {
  const order = ["boot", "install", "models", "generate", "sync"];
  const current = order.indexOf(step);
  el.stageStepper.querySelectorAll(".step-chip").forEach((chip) => {
    const idx = order.indexOf(chip.dataset.step);
    chip.classList.toggle("active", idx === current && !done);
    chip.classList.toggle("done", idx < current || done);
  });
}

function sessionStateLabel(row) {
  if (row.presence === "gone") return "已確認消失";
  const local = {
    release_pending: "關機待確認",
    allocating: "配置待確認",
    busy: "本機有執行紀錄",
    idle: row.owned ? "本機待命" : "外部 Session · 本機待命",
  }[row.local_state];
  const presence = row.presence === "unknown" ? "雲端待確認" : null;
  return [local || (row.blocked && !presence ? "本機識別需核對" : null), presence]
    .filter(Boolean).join(" · ") || (row.owned ? "Studio Session" : "外部 Session");
}

function renderSessionStatus(st) {
  const row = st.active_session;
  el.activeSessionLabel.textContent = row
    ? [row.name, row.hardware].filter(Boolean).join(" · ")
    : !st.token_ok ? "尚未登入" : st.cloud_status === "unknown" ? "雲端狀態待確認"
      : st.unnamed_sessions ? `雲端有 ${st.unnamed_sessions} 台未命名 Session` : "已確認無開啟的 Session";
  const executing = st.runtime_attached && row?.id === st.run_session_id && row?.presence !== "gone"
    && row?.local_state !== "release_pending";
  const label = executing ? (row.presence === "present" ? "工作台執行中" : "工作台執行中 · 雲端待確認")
    : row ? sessionStateLabel(row) : null;
  el.countdownBadge.hidden = !label;
  el.countdownBadge.textContent = label || "";
  const badgeTone = row?.presence !== "present" || row?.local_state === "release_pending" ? "warn"
    : executing || row?.local_state === "busy" ? "busy" : "";
  el.countdownBadge.className = `countdown-badge ${badgeTone}`;
  el.sessionStatusPill.title = st.confirmed_at == null ? "尚未確認 Colab Session 清單"
    : `上次確認雲端清單 ${new Date(st.confirmed_at * 1000).toLocaleTimeString()}`;
  ui.activeSession = row?.presence === "present" && row.id && !row.blocked ? row : null;
  ui.remainingSec = !executing && row?.presence === "present" && row.local_state === "idle" ? row.remaining_sec : null;
  ui.countdownOwned = row?.owned;
  renderCountdownTick();
  el.btnKeep30.disabled = el.btnKeep60.disabled = !ui.activeSession || row.local_state === "release_pending"
    || (st.running && !executing);
}

function renderCountdownTick() {
  if (ui.remainingSec === null) return;
  el.countdownBadge.className = `countdown-badge ${ui.remainingSec <= 300 ? "warn" : ""}`;
  if (ui.countdownOwned) {
    el.countdownBadge.textContent = ui.remainingSec === 0 ? "待命期滿 · 等待回收確認"
      : `⏱️ 待命 ${formatCountdown(ui.remainingSec)}`;
  } else {
    el.countdownBadge.textContent = `外部 Session · 本機保留 ${formatCountdown(ui.remainingSec)}`;
  }
}

let statusRequest = 0;
let lastStatusRequest = 0;
let fastPollInFlight = false;
let slowPoll = null;
let queuedForce = null;
let slowTransportFailed = false;

function renderRefreshStatus(st) {
  const message = slowTransportFailed ? "無法重新整理 Colab 狀態"
    : st.sessions_failed ? "無法查詢雲端 Session" : st.cu_failed ? "無法查詢剩餘 CU" : "";
  el.colabRefreshError.textContent = message;
  el.colabRefreshError.hidden = !message;
}

function renderFinishedRun(st) {
  if (ui.finishedJob === st.job_dir) return;
  ui.finishedJob = st.job_dir;
  if (st.stage === "cancelled") {
    showToast("⏹️ 已確認任務取消", true);
  } else if (["stopped", "stopped_local", "shutdown_pending"].includes(st.stage)) {
    showToast(st.stage_label, true);
  } else if (st.stage === "error" || st.error) {
    showToast(`❌ 生成失敗：${st.error || "未知錯誤"}`);
  } else if (st.stage === "done") {
    showToast("✅ 雲端生成與 SHA256 校驗下載完成！", true);
  }
  loadGallery(true);
}

function applyStatus(st, request) {
  if (!st?.view_id || typeof st.revision !== "number") return false;
  if (st.view_id !== ui.viewId) {
    if (ui.retiredViews.has(st.view_id) || request < lastStatusRequest) return false;
    if (ui.viewId) ui.retiredViews.add(ui.viewId);
    ui.viewId = st.view_id;
    ui.lastRevision = -1;
    ui.wasRunning = false;
    ui.finishedJob = null;
  }
  if (st.revision <= ui.lastRevision) return false;
  ui.lastRevision = st.revision;
  ui.lastStatus = st;
  lastStatusRequest = Math.max(request, lastStatusRequest);

  el.authDot.className = `dot ${st.token_ok ? "ok" : "err"}`;
  el.authLabel.textContent = st.token_ok ? "Google Colab 已登入" : "尚未登入 Colab";
  el.cuValue.textContent = typeof st.cu === "number" ? `${st.cu.toFixed(2)} CU`
    : !st.token_ok ? "未登入" : st.cu_failed ? "暫時查不到" : "尚無資料";
  renderRefreshStatus(st);
  renderSessionStatus(st);
  updateSessionDropdown(st.sessions || []);

  ui.runId = st.running ? st.run_id : null;
  ui.runSession = st.running ? st.session : null;
  ui.preparing = Boolean(st.preparing);
  ui.canCancel = st.running || ui.preparing ? st.can_cancel : Boolean(ui.activeSession);
  el.btnCancelRun.disabled = el.btnCancelTop.disabled = !ui.canCancel;
  el.btnCancelRun.hidden = el.progressOverlay.hidden = !st.running;
  renderGenerateButton();
  if (st.running) {
    ui.wasRunning = true;
    el.progressStageText.textContent = st.stage_label;
    const target = st.session || "待確認";
    const gpu = st.gpu ? `GPU${st.gpu_source === "colab" ? "（Colab 配置）" : ""}：${st.gpu}`
      : st.requested_gpu ? `GPU 待確認（請求 ${st.requested_gpu}）` : "GPU 待確認";
    el.progressMetaText.textContent = [
      `${st.runtime_attached ? "執行 Session" : "目標 Session"}：${target}`, gpu, `權重：${st.quant}`, `已耗時 ${st.elapsed_sec} 秒`,
    ].join(" | ");
    updateStepChips(st.progress_step, st.stage === "done");
    el.progressLogs.textContent = st.logs?.join("\n") || "正在等待 Colab 回應...";
    el.progressLogs.scrollTop = el.progressLogs.scrollHeight;
  } else if (ui.wasRunning) {
    ui.wasRunning = false;
    renderFinishedRun(st);
  }
  return true;
}

function renderGenerateButton() {
  el.btnGenerate.disabled = ui.submitting || ui.preparing || Boolean(ui.lastStatus?.running);
  el.btnGenerate.textContent = ui.lastStatus?.running ? "⏳ 雲端生成進行中..."
    : ui.preparing || ui.submitting ? "⏳ 正在準備生成..." : "🚀 開始生成 (Generate)";
}

async function pollFastRunStatus() {
  if (fastPollInFlight) return;
  fastPollInFlight = true;
  const request = ++statusRequest;
  try {
    applyStatus(await apiFetch("/api/run/status"), request);
  } catch (err) {
    // A transient polling error leaves the most recent snapshot visible.
  } finally {
    fastPollInFlight = false;
  }
}

function updateSessionDropdown(sessions) {
  const select = el.sessionSelect;
  const selected = select.value;
  const options = sessions.filter((row) => row.id && row.presence !== "gone").map((row) => ({
    id: row.id, name: row.name, disabled: !row.reusable,
    label: [row.name, row.hardware, sessionStateLabel(row)].filter(Boolean).join(" · "),
  }));
  if (selected && selected !== "__new__" && !options.some((opt) => opt.id === selected)) {
    options.push({ id: selected, name: "", disabled: true, label: "原目標已無法沿用，請重新選擇" });
  }
  const previous = [...select.options].slice(2);
  if (options.length === previous.length && options.every((row, i) => (
    previous[i].value === row.id && previous[i].textContent === row.label && previous[i].disabled === row.disabled
  ))) return;

  while (select.options.length > 2) select.remove(2);
  for (const row of options) {
    const option = document.createElement("option");
    option.value = row.id;
    option.dataset.session = row.name;
    option.textContent = row.label;
    option.disabled = row.disabled;
    select.appendChild(option);
  }
  select.value = selected;
}

function refreshSlowColabStatus(force = false, afterChange = false) {
  if (slowPoll) {
    if (force && (afterChange || !slowPoll.force)) {
      queuedForce ??= slowPoll.promise.then(() => {
        queuedForce = null;
        return refreshSlowColabStatus(true);
      });
      return queuedForce;
    }
    return slowPoll.promise;
  }
  const request = ++statusRequest;
  const promise = (async () => {
    if (force) el.btnRefreshColab.disabled = true;
    try {
      const data = await apiFetch(`/api/colab/status?force=${force}`);
      slowTransportFailed = false;
      if (applyStatus(data, request) && !data.token_ok) el.loginDrawer.hidden = false;
    } catch (err) {
      slowTransportFailed = true;
      renderRefreshStatus({});
    } finally {
      slowPoll = null;
      el.btnRefreshColab.disabled = false;
    }
  })();
  slowPoll = { force, promise };
  return promise;
}

// ----------------------------------------------------------------------------
// Start Generation Action
// ----------------------------------------------------------------------------
async function handleStartGenerate() {
  if (ui.submitting || ui.preparing || ui.lastStatus?.running) return;
  hideToast();
  const prompt = el.promptInput.value.trim();
  if (!prompt) {
    showToast("請先輸入提示詞 (Prompt)");
    el.promptInput.focus();
    return;
  }
  if (ui.mode === "edit" && ui.refImages.length === 0) {
    showToast("圖片編輯（Edit）模式需要至少 1 張參考圖片 (<image1> 主畫布)");
    return;
  }
  const sessionOption = el.sessionSelect.selectedOptions[0];
  if (sessionOption?.disabled) {
    showToast("所選 Session 目前無法沿用，請重新選擇");
    el.sessionSelect.focus();
    return;
  }

  const fd = new FormData();
  fd.append("mode", ui.mode);
  fd.append("prompt", prompt);
  fd.append("refine_prompt", el.chkRefinePrompt.checked ? "true" : "false");
  fd.append("pe_variant", el.peVariantSelect.value);
  fd.append("transparent", el.chkTransparent.checked ? "true" : "false");
  fd.append("aspect_ratio", ui.aspectRatio);
  fd.append("resolution", ui.resolution);
  fd.append("edit_sizing", el.editSizingSelect.value);
  fd.append("seed", el.seedInput.value.trim());
  fd.append("gpu", el.gpuSelect.value);
  fd.append("quant", el.quantSelect.value);
  const sessionName = sessionOption?.dataset.session;
  fd.append("session", sessionName || el.sessionSelect.value);
  if (sessionName) fd.append("session_id", el.sessionSelect.value);
  fd.append("idle_minutes", el.idleMinutesInput.value || "15");
  fd.append("stop_after", el.chkStopAfter.checked ? "true" : "false");
  fd.append("batch_name", el.batchNameInput.value.trim());

  if (ui.mode === "edit") {
    const orderSpec = [];
    for (const item of ui.refImages) {
      if (item.kind === "upload" && item.file) {
        orderSpec.push({ kind: "upload", name: item.name });
        fd.append("ref_files", item.file, item.name || "ref.png");
      } else if (item.kind === "artifact" && item.artifact_id) {
        orderSpec.push({ kind: "artifact", artifact_id: item.artifact_id });
      }
    }
    fd.append("ref_order", JSON.stringify(orderSpec));
  }

  try {
    ui.submitting = true;
    renderGenerateButton();
    const request = ++statusRequest;
    const res = await apiFetch("/api/run/start", {
      method: "POST",
      body: fd,
    });
    if (res.seed) {
      ui.lastSeed = String(res.seed);
    }
    applyStatus(res.status, request);
    if (ui.lastStatus && ui.lastStatus.job_dir === res.job_dir && !ui.lastStatus.running) {
      renderFinishedRun(ui.lastStatus);
    }
    await pollFastRunStatus();
  } catch (err) {
    showToast(`無法啟動生成任務：${err.message}`);
  } finally {
    ui.submitting = false;
    renderGenerateButton();
  }
}

// ----------------------------------------------------------------------------
// Initialization & Event Bindings
// ----------------------------------------------------------------------------
async function initStudio() {
  // Load metadata (quants, GPUs)
  try {
    const meta = await apiFetch("/api/meta");
    if (Array.isArray(meta.quants) && meta.quants.length > 0) {
      el.quantSelect.innerHTML = "";
      for (const q of meta.quants) {
        const opt = document.createElement("option");
        opt.value = q.id;
        opt.textContent = q.label;
        if (q.is_default) opt.selected = true;
        el.quantSelect.appendChild(opt);
      }
    }
  } catch (err) {
    console.warn("讀取 /api/meta 失敗:", err);
  }

  // Mode Tabs
  el.tabT2I.addEventListener("click", () => setMode("t2i"));
  el.tabEdit.addEventListener("click", () => setMode("edit"));

  // Reference Images upload & drag-drop
  el.btnAddRefImage.addEventListener("click", () => el.refFileInput.click());
  el.refFileInput.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      addUploadedFiles(e.target.files);
      el.refFileInput.value = "";
    }
  });
  el.btnClearRefImages.addEventListener("click", () => {
    ui.refImages = [];
    renderRefImages();
  });

  el.refDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.refDropzone.classList.add("dragover");
  });
  el.refDropzone.addEventListener("dragleave", () => {
    el.refDropzone.classList.remove("dragover");
  });
  el.refDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    el.refDropzone.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) {
      addUploadedFiles(e.dataTransfer.files);
    }
  });

  el.editSizingSelect.addEventListener("change", () => {
    ui.editSizing = el.editSizingSelect.value;
    updateSizeControlVisibility();
  });

  // Prompt & Enhancer Toggles
  el.btnClearPrompt.addEventListener("click", () => {
    el.promptInput.value = "";
    el.promptInput.focus();
  });
  el.chkRefinePrompt.addEventListener("change", () => {
    el.peVariantRow.hidden = !el.chkRefinePrompt.checked;
  });

  // Aspect Ratio & Resolution
  for (const btn of el.aspectGrid.querySelectorAll(".aspect-btn")) {
    btn.addEventListener("click", () => setAspectRatio(btn.dataset.ar));
  }
  el.btnSwapAspect.addEventListener("click", () => {
    setAspectRatio(SWAP_AR_MAP[ui.aspectRatio] || "1:1");
  });
  el.res1K.addEventListener("click", () => setResolution("1K"));
  el.res2K.addEventListener("click", () => setResolution("2K"));

  // Seed buttons
  el.btnRandomSeed.addEventListener("click", () => {
    el.seedInput.value = String(Math.floor(Math.random() * 2147483647) + 1);
  });
  el.btnReuseLastSeed.addEventListener("click", () => {
    if (ui.lastSeed) {
      el.seedInput.value = ui.lastSeed;
    } else if (ui.selectedItem?.seed) {
      el.seedInput.value = String(ui.selectedItem.seed);
    } else {
      showToast("目前尚無上一張作品的 Seed 紀錄");
    }
  });

  // Generate & Cancel
  el.btnGenerate.addEventListener("click", handleStartGenerate);
  const doCancel = async () => {
    if (!ui.canCancel) return;
    const target = ui.preparing ? { preparation_id: ui.lastStatus.preparation_id }
      : ui.wasRunning ? { session: ui.runSession, run_id: ui.runId }
      : { session: ui.activeSession.name, session_id: ui.activeSession.id };
    try {
      const res = await apiFetch("/api/colab/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(target),
      });
      const message = {
        preparation_pending: "⏳ 正在取消本機生成準備",
        pending: "⏳ 已要求取消，等待遠端確認",
        confirmed: "⏹️ 已確認遠端批次取消",
        idle: "目前沒有符合條件的執行中批次",
        finished: "批次已經結束",
      }[res.state];
      if (message) showToast(message, true);
      await pollFastRunStatus();
    } catch (err) {
      showToast(`中斷任務失敗：${err.message}`);
    }
  };
  el.btnCancelRun.addEventListener("click", doCancel);
  el.btnCancelTop.addEventListener("click", doCancel);

  // Top Bar Colab Controls
  el.btnRefreshColab.addEventListener("click", () => refreshSlowColabStatus(true));
  const doKeep = async (minutes) => {
    if (!ui.activeSession || el.btnKeep30.disabled) return;
    try {
      const res = await apiFetch("/api/colab/keep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: ui.activeSession.name, session_id: ui.activeSession.id, minutes }),
      });
      showToast(`⏳ 已為 ${res.session} 更新本機待命期限，剩餘 ${formatCountdown(res.remaining_sec)}`, true);
      await pollFastRunStatus();
    } catch (err) {
      showToast(`延長保留失敗：${err.message}`);
    }
  };
  el.btnKeep30.addEventListener("click", () => doKeep(30));
  el.btnKeep60.addEventListener("click", () => doKeep(60));

  el.btnStopTop.addEventListener("click", async () => {
    try {
      showToast("🛑 正在關閉 Colab GPU 並釋放資源...", true);
      const res = await apiFetch("/api/colab/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ all: true }),
      });
      const summary = [];
      if (res.preparation_cancelled) summary.push("⏳ 正在取消本機生成準備");
      if (res.generation_cancel_pending) summary.push("⏳ 等待尚在確認 Session 的批次取消");
      if (res.verified_gone?.length) summary.push(`✅ 已確認關閉 ${res.verified_gone.join(", ")}`);
      if (res.pending?.length) summary.push(`⏳ 尚待確認 ${res.pending.join(", ")}`);
      if (res.failed?.length) summary.push(`⚠️ 無法確認 ${res.failed.join(", ")}`);
      showToast(summary.join(" · ") || "⚪ 目前沒有已登記的 Colab Studio Session 需要關閉", !res.failed?.length);
      await pollFastRunStatus();
      refreshSlowColabStatus(true, true);
    } catch (err) {
      showToast(`關機失敗：${err.message}`);
    }
  });

  // Login Drawer
  el.btnToggleLogin.addEventListener("click", () => {
    el.loginDrawer.hidden = !el.loginDrawer.hidden;
  });
  el.btnCloseLoginDrawer.addEventListener("click", () => {
    el.loginDrawer.hidden = true;
  });
  el.btnGetLoginUrl.addEventListener("click", async () => {
    try {
      const res = await apiFetch("/api/colab/login/url", { method: "POST" });
      ui.loginFlowState = res.state;
      el.loginUrlLink.href = res.url;
      el.loginUrlLink.hidden = false;
      window.open(res.url, "_blank", "noopener,noreferrer");
    } catch (err) {
      showToast(`取得登入網址失敗：${err.message}`);
    }
  });
  el.btnSubmitLoginCode.addEventListener("click", async () => {
    if (!ui.loginFlowState) {
      showToast("請先在此分頁取得 Google 登入網址");
      return;
    }
    const code = el.loginCodeInput.value.trim();
    if (!code) {
      showToast("請貼上 Google 授權碼");
      return;
    }
    try {
      await apiFetch("/api/colab/login/code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, state: ui.loginFlowState }),
      });
      el.loginCodeInput.value = "";
      ui.loginFlowState = null;
      el.loginUrlLink.hidden = true;
      el.loginDrawer.hidden = true;
      showToast("✅ Google Colab OAuth 登入成功！", true);
      pollFastRunStatus();
      await refreshSlowColabStatus(true, true);
    } catch (err) {
      showToast(`登入驗證失敗：${err.message}`);
    }
  });

  // Floating Toolbar & Prompt Drawer
  el.btnContinueEdit.addEventListener("click", () => setArtifactAsReference(ui.selectedItem, true));
  el.btnAddAsExtraRef.addEventListener("click", () => setArtifactAsReference(ui.selectedItem, false));
  el.btnRestoreRecipe.addEventListener("click", restoreRecipeFromSelected);
  el.btnTogglePromptDrawer.addEventListener("click", () => {
    el.promptDrawer.hidden = !el.promptDrawer.hidden;
  });
  el.btnClosePromptDrawer.addEventListener("click", () => {
    el.promptDrawer.hidden = true;
  });
  el.btnCopyRefinedPrompt.addEventListener("click", async () => {
    const text = ui.selectedItem?.refined_prompt || ui.selectedItem?.prompt || "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast("📋 已複製擴寫後的完整提示詞", true);
    } catch (err) {
      showToast("複製失敗，請手動選取文字複製");
    }
  });
  el.btnUseRefinedPrompt.addEventListener("click", () => {
    const text = ui.selectedItem?.refined_prompt;
    if (!text) return;
    restoreRecipeFromSelected();
    el.promptInput.value = text;
    el.chkRefinePrompt.checked = false;
    el.peVariantRow.hidden = true;
    showToast("⤴️ 已還原此圖設定並填入擴寫提示詞（已關閉自動擴寫以固定重現）！", true);
  });

  el.btnCloseToast.addEventListener("click", hideToast);
  el.btnRefreshGallery.addEventListener("click", () => loadGallery(false));

  // Initial Renders & Polls
  renderRefImages();
  updatePixelReadout();
  await loadGallery(true);
  await pollFastRunStatus();
  refreshSlowColabStatus(false);

  // Fast status polling every 1.5s
  setInterval(pollFastRunStatus, 1500);
  setInterval(refreshSlowColabStatus, 60_000);
  // Local 1-second countdown ticker
  setInterval(() => {
    if (typeof ui.remainingSec === "number" && ui.remainingSec > 0) {
      ui.remainingSec -= 1;
      renderCountdownTick();
    }
  }, 1000);
}

initStudio();
