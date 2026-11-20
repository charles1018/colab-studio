"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const app = fs.readFileSync(path.join(__dirname, "../cstudio/static/app.js"), "utf8");

function fixture(fetchStatus = async () => { throw new Error("Unexpected API call"); }) {
  const elements = new Map();
  function element() {
    const classes = new Set();
    return {
      dataset: {}, hidden: true,
      classList: {
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
    loadFixtureGallery: () => { galleryLoads += 1; },
  });
  const run = (code) => vm.runInContext(code, context);
  run(app.replace(/\binitStudio\(\);\s*$/, ""));
  run("apiFetch = fetchStatus; loadGallery = loadFixtureGallery;");
  return { run, select, chips, getElementById, galleryLoads: () => galleryLoads };
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
  apply(status(3, [], { stage: "done" }), 3);
  apply(status(4, [], { stage: "done" }), 4);
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
