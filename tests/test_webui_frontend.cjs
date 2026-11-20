"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const app = fs.readFileSync(path.join(__dirname, "../cstudio/static/app.js"), "utf8");

function fixture(fetchStatus = async () => { throw new Error("Unexpected API call"); }, { gallery = false } = {}) {
  const elements = new Map();
  const revoked = [];
  function element() {
    const classes = new Set();
    const events = new Map();
    const children = [];
    return {
      dataset: {}, hidden: true, style: {}, children,
      appendChild: (child) => children.push(child),
      querySelectorAll: () => children,
      set innerHTML(_value) { children.length = 0; },
      addEventListener: (type, listener) => events.set(type, listener),
      fire: (type) => events.get(type)?.(),
      focus: () => {},
      getAttribute(name) { return this[name] ?? null; },
      removeAttribute(name) { delete this[name]; },
      classList: {
        add: (name) => classes.add(name),
        contains: (name) => classes.has(name),
        toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
      },
    };
  }
  function getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  }
  const select = getElementById("sessionSelect");
  select.value = "";
  select.options = [{ value: "" }, { value: "__new__" }];
  select.remove = (index) => select.options.splice(index, 1);
  select.appendChild = (option) => select.options.push(option);
  const chips = ["boot", "install", "models", "generate", "sync"].map((step) => {
    const chip = element();
    chip.dataset.step = step;
    return chip;
  });
  getElementById("stageStepper").querySelectorAll = () => chips;
  let galleryLoads = 0;
  const context = vm.createContext({
    document: { getElementById, createElement: element },
    fetchStatus,
    URLSearchParams,
    URL: { createObjectURL: (file) => `blob:${file.name}`, revokeObjectURL: (url) => revoked.push(url) },
    loadFixtureGallery: () => { galleryLoads += 1; },
  });
  const run = (code) => vm.runInContext(code, context);
  run(app.replace(/\binitStudio\(\);\s*$/, ""));
  run("apiFetch = fetchStatus;");
  if (!gallery) run("loadGallery = loadFixtureGallery;");
  return { run, select, chips, getElementById, revoked, galleryLoads: () => galleryLoads };
}

function row(id, name = "external", extra = {}) {
  return { id, name, presence: "present", hardware: "A100", reusable: true, ...extra };
}

function status(revision, sessions = [], extra = {}) {
  return {
    view_id: "workbench", revision, running: false, token_ok: true, cu: 35, sessions,
    active_session: sessions[0] || null, cloud_status: "confirmed", ...extra,
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test("a user-selected identity survives polls and cannot become a same-name replacement", () => {
  const { run, select, getElementById } = fixture();
  const apply = run("applyStatus");
  apply(status(1, [row("first"), row("second", "second")]), 1);
  select.value = "second";
  apply(status(2, [row("second", "second"), row("first")]), 2);
  assert.equal(select.value, "second");
  assert.match(getElementById("activeSessionLabel").textContent, /second · A100/);

  const unknown = row("second", "second", { presence: "unknown", reusable: false });
  apply(status(3, [unknown], { cloud_status: "unknown" }), 3);
  assert.equal(select.value, "second");
  assert.equal(select.options.find((opt) => opt.value === "second").disabled, true);
  assert.match(getElementById("countdownBadge").textContent, /雲端待確認/);

  apply(status(4, [row("replacement", "second", { hardware: "G4" })]), 4);
  assert.equal(select.value, "second");
  assert.equal(select.options.find((opt) => opt.value === "second").dataset.session, "");
  assert.equal(select.options.find((opt) => opt.value === "second").disabled, true);
  assert.equal(select.options.find((opt) => opt.value === "replacement").disabled, false);
  assert.match(getElementById("activeSessionLabel").textContent, /second · G4/);
});

test("a delayed slow response cannot overwrite a newer fast snapshot or another server instance", async () => {
  const slow = deferred();
  const { run, getElementById } = fixture((url) => url.includes("/colab/") ? slow.promise
    : Promise.resolve(status(10, [row("new", "current")], { cu: 8 })));
  const pending = run("refreshSlowColabStatus(true)");
  await run("pollFastRunStatus()");
  slow.resolve(status(9, [row("old", "past")], { cu: 99 }));
  await pending;
  assert.match(getElementById("activeSessionLabel").textContent, /current/);
  assert.equal(getElementById("cuValue").textContent, "8.00 CU");

  const apply = run("applyStatus");
  assert.equal(apply(status(1, [], { view_id: "restarted" }), 3), true);
  assert.equal(apply(status(11, [row("old")]), 4), false);
  assert.equal(apply(status(2, [], { view_id: "restarted" }), 5), true);
});

test("a forced refresh queued behind a regular fetch is sent once", async () => {
  const normal = deferred();
  const forced = deferred();
  const calls = [];
  const { run } = fixture((url) => {
    calls.push(url);
    return calls.length === 1 ? normal.promise : forced.promise;
  });
  const first = run("refreshSlowColabStatus()");
  const force = run("refreshSlowColabStatus(true)");
  assert.equal(run("refreshSlowColabStatus(true)"), force);
  assert.deepEqual(calls, ["/api/colab/status?force=false"]);
  normal.resolve(status(1));
  await first;
  assert.deepEqual(calls, ["/api/colab/status?force=false", "/api/colab/status?force=true"]);
  assert.equal(run("refreshSlowColabStatus(true)"), run("slowPoll.promise"));
  forced.resolve(status(2));
  await force;
});

test("progress shows the bound run and actual GPU, and completion refreshes only the gallery", () => {
  const { run, getElementById, chips, galleryLoads } = fixture();
  const apply = run("applyStatus");
  const base = {
    running: true, can_cancel: true, session: "target", quant: "INT8", elapsed_sec: 2,
    requested_gpu: "H100", stage: "selecting_vm", progress_step: "boot",
  };
  apply(status(1, [row("other")], base), 1);
  const meta = getElementById("progressMetaText");
  assert.match(meta.textContent, /目標 Session：target.*GPU 待確認（請求 H100）/);
  assert.doesNotMatch(meta.textContent, /external|A100/);
  apply(status(2, [row("target-id", "target")], {
    ...base, runtime_attached: true, run_session_id: "target-id", stage: "running",
    gpu: "NVIDIA L4", progress_step: "generate",
  }), 2);
  assert.match(meta.textContent, /執行 Session：target.*GPU：NVIDIA L4/);
  assert.match(getElementById("countdownBadge").textContent, /工作台執行中/);
  assert.equal(chips[3].classList.contains("active"), true);
  assert.equal(chips[0].classList.contains("done"), true);
  const done = { stage: "done", stage_label: "✅ 批次已完成" };
  apply(status(3, [], done), 3);
  apply(status(4, [], done), 4);
  assert.equal(getElementById("toastMessage").textContent, done.stage_label);
  assert.equal(galleryLoads(), 1);
  assert.equal(run("slowPoll"), null);
});

test("polling keeps submit disabled during a local upload and shows preparation cancellation", () => {
  const { run, getElementById } = fixture();
  const apply = run("applyStatus");
  const generate = getElementById("btnGenerate");
  const cancel = getElementById("btnCancelTop");
  run("ui.submitting = true;");
  apply(status(1), 1);
  assert.equal(generate.disabled, true);
  assert.match(generate.textContent, /正在準備生成/);
  apply(status(2, [], { preparing: true, preparation_id: "attempt", can_cancel: true }), 2);
  assert.equal(generate.disabled, true);
  assert.equal(cancel.disabled, false);
  apply(status(3, [], { preparing: true, preparation_id: "attempt", can_cancel: false }), 3);
  assert.equal(cancel.disabled, true);
  run("ui.submitting = false;");
  apply(status(4), 4);
  assert.equal(generate.disabled, false);
});

function galleryItem(id) {
  return {
    artifact_id: `folder/out/${id}.png`, job_dir: "folder", job_id: id, filename: `${id}.png`,
    url: `/api/jobs/folder/out/${id}.png`, thumbnail_url: `/api/gallery/thumbnail/folder/${id}.png`,
  };
}

test("gallery pages use lazy thumbnails and keep the selected original", async () => {
  const requests = [];
  const recent = galleryItem("newest");
  const older = galleryItem("older");
  const { run, getElementById } = fixture(async (url) => {
    requests.push(url);
    return url.includes("before")
      ? { items: [older], next_cursor: null }
      : { items: [recent], next_cursor: "91:folder/out/newest.png" };
  }, { gallery: true });
  const load = run("loadGallery");
  await load(true);
  const strip = getElementById("galleryStrip");
  const stage = getElementById("stageImage");
  const tile = strip.children[0].children[0];
  assert.equal(tile.loading, "lazy");
  assert.equal(tile.src, recent.thumbnail_url);
  assert.equal(stage.src, recent.url);
  assert.equal(getElementById("btnNewerGallery").disabled, true);
  assert.equal(getElementById("btnOlderGallery").disabled, false);
  await load(false, 1);
  assert.equal(requests[1], "/api/gallery?limit=32&before=91%3Afolder%2Fout%2Fnewest.png");
  assert.equal(stage.src, recent.url);
  assert.equal(getElementById("btnOlderGallery").disabled, true);
  assert.equal(getElementById("btnNewerGallery").disabled, false);
  strip.children[0].fire("click");
  assert.equal(stage.src, older.url);
  assert.equal(getElementById("btnDownloadPng").href, older.url);
});

test("refresh ignores an older page response still in flight", async () => {
  const pending = deferred();
  let refreshes = 0;
  const { run, getElementById } = fixture((url) => url.includes("before") ? pending.promise : Promise.resolve({
    items: [galleryItem(`newest-${++refreshes}`)], next_cursor: "91:folder/out/newest.png",
  }), { gallery: true });
  const load = run("loadGallery");
  await load(true);
  const olderRequest = load(false, 1);
  await load(true);
  pending.resolve({ items: [galleryItem("too-late")], next_cursor: null });
  await olderRequest;
  assert.match(getElementById("stageImage").src, /newest-2/);
  assert.match(getElementById("galleryCount").textContent, /第 1 頁/);
});

test("replacing, removing, restoring and clearing references revoke only discarded uploads", () => {
  const { run, getElementById, revoked } = fixture();
  run('setMode = () => {}; updateSizeControlVisibility = () => {};');
  const upload = run("addUploadedFiles");
  upload(["one", "two"].map((name) => ({ name, type: "image/png" })));
  run("setArtifactAsReference")(galleryItem("canvas"));
  assert.deepEqual(revoked, ["blob:one"]);
  const render = getElementById("refCardsGrid");
  render.children[1].children[0].children[1].children.at(-1).fire("click");
  assert.deepEqual(revoked, ["blob:one", "blob:two"]);
  upload([{ name: "restore", type: "image/png" }]);
  run(`ui.selectedItem = { mode: "edit", job_id: "restored", inputs: [
    { artifact_id: "folder/inputs/second.png", url: "/api/jobs/folder/inputs/second.png" },
    { artifact_id: "folder/inputs/first.png", url: "/api/jobs/folder/inputs/first.png" },
  ] }; restoreRecipeFromSelected();`);
  assert.deepEqual(revoked, ["blob:one", "blob:two", "blob:restore"]);
  assert.equal(run("ui.refImages.map(item => item.artifact_id).join(',')"),
    "folder/inputs/second.png,folder/inputs/first.png");
  upload([{ name: "clear", type: "image/png" }]);
  run("replaceReferences([]);");
  assert.deepEqual(revoked, ["blob:one", "blob:two", "blob:restore", "blob:clear"]);
});
