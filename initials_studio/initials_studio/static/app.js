"use strict";
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const TURN = {1: "↻90", 2: "↻180", 3: "↻270", 4: "mirror", 5: "mirror↻90", 6: "mirror↻180", 7: "mirror↻270"};
let STATUS = null, BOOKS = [], view = "library";

async function api(path, opts = {}) {
  const r = await fetch(path, opts.body && !(opts.body instanceof FormData)
    ? {...opts, headers: {"Content-Type": "application/json"}, body: JSON.stringify(opts.body)} : opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) {} throw new Error(m); }
  return r.json();
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("on"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), 2600); }
function modal(html) { $("#modal-body").innerHTML = html; $("#modal").classList.add("on"); }
$("#modal").addEventListener("click", e => { if (e.target.id === "modal" || e.target.dataset.close !== undefined) $("#modal").classList.remove("on"); });
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#modal").classList.remove("on"); });
const bookColor = {};
function colorFor(id) { if (!(id in bookColor)) bookColor[id] = `var(--b${Object.keys(bookColor).length % 12})`; return bookColor[id]; }

// ------------------------------------------------------------------ navigation
function show(v) {
  view = v;
  document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-current", b.dataset.view === v ? "page" : "false"));
  document.querySelectorAll(".view").forEach(s => s.classList.toggle("on", s.id === "v-" + v));
  try { localStorage.setItem("is-view", v); } catch (e) {}
  ({library: loadLibrary, review: loadReview, patterns: loadPatterns, settings: loadSettings, method: () => window.scrollTo(0, 0)})[v]();
}
document.querySelectorAll("nav button").forEach(b => b.onclick = () => show(b.dataset.view));

// ------------------------------------------------------------------ jobs
let lastBusy = false;
async function pollJobs() {
  try {
    const jobs = await api("/api/jobs");
    const run = jobs.find(j => j.status === "running") || jobs.find(j => j.status === "queued");
    const pill = $("#jobpill");
    pill.classList.toggle("on", !!run);
    if (run) {
      const q = jobs.filter(j => j.status === "queued").length;
      $("#jobmsg").textContent = run.message + (q && run.status === "running" ? `  (+${q} waiting)` : "");
      $("#jobbar").style.width = Math.round(run.progress * 100) + "%";
    }
    if (lastBusy && !run) { toast("Finished: " + (jobs[0]?.title || "task")); refresh(); }
    else if (run && view === "library") loadLibrary(true);
    const failed = jobs.find(j => j.status === "failed" && !j._seen);
    lastBusy = !!run;
  } catch (e) {}
  setTimeout(pollJobs, lastBusy ? 1200 : 3000);
}
$("#jobpill").onclick = async () => {
  const jobs = await api("/api/jobs");
  modal(`<h2>Background tasks</h2><div style="display:grid;gap:10px;margin-top:12px;min-width:min(520px,80vw)">${jobs.map(j => `
    <div class="card" style="padding:10px 12px;box-shadow:none"><div style="display:flex;gap:10px;align-items:center">
      <b style="flex:1">${esc(j.title)}</b><span class="chip ${j.status === "done" ? "ok" : j.status === "failed" ? "bad" : "accent"}">${j.status}</span>
      ${["queued", "running"].includes(j.status) ? `<button class="btn sm" onclick="api('/api/jobs/${j.id}/cancel',{method:'POST'}).then(()=>toast('Cancelling…'))">Cancel</button>` : ""}</div>
      <div class="small muted" style="margin-top:4px">${esc(j.message)}</div></div>`).join("") || '<p class="muted">No tasks yet.</p>'}</div>
    <div style="text-align:right;margin-top:12px"><button class="btn" data-close>Close</button></div>`);
};
function refresh() { ({library: loadLibrary, review: loadReview, patterns: loadPatterns, settings: () => {}, method: () => {}})[view](); }

// ------------------------------------------------------------------ library
let pending = [];
const drop = $("#drop");
drop.onclick = () => $("#files").click();
drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#files").click(); } };
$("#files").onchange = e => setFiles([...e.target.files]);
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFiles([...e.dataTransfer.files]));
function setFiles(fs) {
  pending = fs.filter(f => /\.(pdf|jpe?g|png|tiff?|bmp|webp)$/i.test(f.name));
  const mb = pending.reduce((a, f) => a + f.size, 0) / 1e6;
  $("#filelist").innerHTML = pending.length ? `<b>${pending.length} file${pending.length > 1 ? "s" : ""}</b> · ${mb.toFixed(1)} MB<br>` +
    pending.slice(0, 4).map(f => esc(f.name)).join("<br>") + (pending.length > 4 ? "<br>…" : "") : "";
  $("#u-go").disabled = !pending.length;
  if (pending.length === 1 && !$("#u-title").value) $("#u-title").value = pending[0].name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
}
$("#upform").onsubmit = async e => {
  e.preventDefault();
  if (!pending.length) return;
  const fd = new FormData();
  pending.forEach(f => fd.append("files", f));
  fd.append("title", $("#u-title").value); fd.append("printer", $("#u-printer").value); fd.append("date", $("#u-date").value);
  fd.append("one_book_per_pdf", $("#u-split").checked);
  $("#u-go").disabled = true; $("#u-go").textContent = "Uploading…";
  try {
    await api("/api/books", {method: "POST", body: fd});
    toast("Uploaded. Finding capitals in the background…");
    pending = []; setFiles([]); $("#u-title").value = ""; $("#files").value = "";
  } catch (err) { toast("Upload failed: " + err.message); }
  $("#u-go").textContent = "Upload and find capitals"; lastBusy = true; loadLibrary();
};

async function loadLibrary(quiet) {
  BOOKS = await api("/api/books");
  BOOKS.forEach(b => colorFor(b.id));
  $("#lib-count").textContent = `${BOOKS.length} book${BOOKS.length === 1 ? "" : "s"} · ${BOOKS.reduce((a, b) => a + b.n_capitals, 0).toLocaleString()} capitals`;
  const box = $("#books");
  if (!BOOKS.length) { box.innerHTML = `<div class="empty card"><h2>No books yet</h2>Upload a PDF above to begin.</div>`; return; }
  if (quiet && box.dataset.sig === JSON.stringify(BOOKS.map(b => [b.id, b.status, b.n_capitals, b.n_flagged, b.include_in_grouping]))) return;
  box.dataset.sig = JSON.stringify(BOOKS.map(b => [b.id, b.status, b.n_capitals, b.n_flagged, b.include_in_grouping]));
  const covers = await Promise.all(BOOKS.map(b => b.n_capitals ? api(`/api/books/${b.id}`).then(d => d.capitals.filter(c => c.visible).slice(0, 4)).catch(() => []) : []));
  box.innerHTML = BOOKS.map((b, i) => {
    const working = ["new", "importing", "detecting"].includes(b.status);
    return `<article class="card book">
      <div class="covers">${[0, 1, 2, 3].map(k => covers[i][k] ? `<img loading="lazy" alt="" src="/img/crop/${b.id}/${covers[i][k].id}?s=160">` : `<span style="background:var(--panel-2)"></span>`).join("")}</div>
      <div class="body">
        <h3><span class="dot" style="background:${colorFor(b.id)};margin-right:6px"></span>${esc(b.title)}</h3>
        <div class="meta">${esc([b.printer, b.date].filter(Boolean).join(" · ") || "Printer and date not set")}</div>
        <div class="stats">
          ${b.example ? '<span class="chip accent">Example</span>' : ""}
          ${b.crops_only ? '<span class="chip">Capitals only</span>' : `<span class="chip">${b.n_pages} pages</span>`}
          <span class="chip ok">${b.n_capitals} capitals</span>
          ${b.n_flagged ? `<span class="chip warn">${b.n_flagged} to check</span>` : ""}
        </div></div>
      ${working ? `<div class="status"><span class="spin" style="display:inline-block;vertical-align:-2px;margin-right:6px"></span>${b.status === "detecting" ? "Finding capitals…" : "Reading pages…"}</div>` : ""}
      <div class="actions">
        <button class="btn sm primary" data-act="review" ${working ? "disabled" : ""}>Review</button>
        <button class="btn sm" data-act="edit">Edit details</button>
        ${b.crops_only ? "" : `<button class="btn sm ghost" data-act="redetect" ${working ? "disabled" : ""}>Re-run detection</button>`}
        <button class="btn sm ghost bad" data-act="delete">Delete</button>
      </div></article>`;
  }).join("");
  box.querySelectorAll(".book").forEach((el, i) => el.querySelectorAll("[data-act]").forEach(btn => btn.onclick = () => bookAction(BOOKS[i], btn.dataset.act)));
}
async function bookAction(b, act) {
  if (act === "review") { reviewBook = b.id; show("review"); }
  if (act === "toggle") { await api(`/api/books/${b.id}`, {method: "PATCH", body: {include_in_grouping: b.include_in_grouping === false}}); loadLibrary(); }
  if (act === "redetect") { if (confirm("Re-run detection? Your confirmations and throw-outs for automatically found capitals will be reset; boxes you drew yourself are kept.")) { await api(`/api/books/${b.id}/detect`, {method: "POST"}); lastBusy = true; toast("Detection started"); } }
  if (act === "delete") { if (confirm(`Delete “${b.title}” and all its pages and capitals from this computer? This cannot be undone.`)) { await api(`/api/books/${b.id}`, {method: "DELETE"}); toast("Deleted"); loadLibrary(); } }
  if (act === "edit") {
    modal(`<h2>Book details</h2><form id="ed" style="display:grid;gap:10px;margin-top:12px;min-width:min(440px,80vw)">
      <label class="field">Title <input type="text" name="title" value="${esc(b.title)}"></label>
      <label class="field">Printer <input type="text" name="printer" value="${esc(b.printer)}"></label>
      <label class="field">Date / place <input type="text" name="date" value="${esc(b.date)}"></label>
      ${b.source ? `<div class="small muted">Source: ${esc(b.source)}</div>` : ""}
      <div style="display:flex;gap:8px;justify-content:end"><button type="button" class="btn" data-close>Cancel</button><button class="btn primary">Save</button></div></form>`);
    $("#ed").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target);
      await api(`/api/books/${b.id}`, {method: "PATCH", body: Object.fromEntries(f)}); $("#modal").classList.remove("on"); loadLibrary(); };
  }
}

// ------------------------------------------------------------------ review
let reviewBook = null, R = null, rMode = "grid", rSel = new Set(), rPage = null, drawing = null;
async function loadReview() {
  if (!BOOKS.length) BOOKS = await api("/api/books");
  const ready = BOOKS.filter(b => !["new", "importing", "detecting"].includes(b.status));
  if (!ready.length) { $("#r-body").innerHTML = `<div class="empty card"><h2>Nothing to review yet</h2>Upload a book in the Library first.</div>`; return; }
  if (!reviewBook || !ready.some(b => b.id === reviewBook)) reviewBook = ready[0].id;
  $("#r-book").innerHTML = ready.map(b => `<option value="${b.id}">${esc(b.title)}</option>`).join("");
  $("#r-book").value = reviewBook;
  R = await api(`/api/books/${reviewBook}`);
  const s = STATUS.settings;
  const vis = R.capitals.filter(c => c.visible);
  $("#r-title").textContent = R.book.title;
  $("#r-sub").textContent = `${vis.length} capitals at cut-off ${s.score_cutoff} · ${vis.filter(c => c.flags.length && !["kept", "added"].includes(c.status)).length} flagged for a look · ${R.capitals.filter(c => c.status === "rejected").length} thrown out`;
  $("#r-pages").disabled = !!R.book.crops_only;
  if (R.book.crops_only) rMode = "grid";
  $("#r-grid").setAttribute("aria-pressed", rMode === "grid"); $("#r-pages").setAttribute("aria-pressed", rMode === "pages");
  $("#r-keepall").hidden = $("#r-rejsel").hidden = rMode !== "grid";
  rMode === "grid" ? renderGrid() : renderPages();
}
$("#r-book").onchange = e => { reviewBook = e.target.value; rSel.clear(); rPage = null; loadReview(); };
$("#r-grid").onclick = () => { rMode = "grid"; loadReview(); };
$("#r-pages").onclick = () => { rMode = "pages"; loadReview(); };
$("#r-filter").onchange = () => rMode === "grid" ? renderGrid() : renderPages();

function filtered() {
  const f = $("#r-filter").value, cut = STATUS.settings.score_cutoff;
  return R.capitals.filter(c => f === "all" ? c.visible : f === "flagged" ? c.visible && c.flags.length && !["kept", "added"].includes(c.status)
    : f === "kept" ? ["kept", "added"].includes(c.status) : f === "rejected" ? c.status === "rejected"
    : c.status === "auto" && c.score < cut);
}
function flagChips(c) {
  return c.flags.map(f => `<span class="chip ${["unclear_block", "one_of_a_kind", "mirror"].includes(f) ? "accent" : "warn"}" title="${esc(STATUS.flag_text[f])}">${esc(STATUS.flag_text[f])}</span>`).join("");
}
function pageLabel(id) { const p = (R.book.pages || []).find(p => p.id === id); return p?.label || id; }
function renderGrid() {
  const list = filtered();
  $("#r-count").textContent = `${list.length} shown`;
  const body = $("#r-body");
  if (!list.length) { body.innerHTML = `<div class="empty card"><h2>${$("#r-filter").value === "flagged" ? "Nothing left to check" : "Nothing here"}</h2>${$("#r-filter").value === "flagged" ? "Every flagged capital in this book has been confirmed or thrown out." : ""}</div>`; return; }
  body.innerHTML = `<div class="caps">${list.map(c => `<div class="card cap ${c.status} ${rSel.has(c.id) ? "sel" : ""}" data-id="${c.id}">
      ${c.status !== "auto" ? `<span class="state ${c.status}">${{kept: "Confirmed", rejected: "Thrown out", added: "Added by you"}[c.status]}</span>` : ""}
      <input type="checkbox" class="pick" aria-label="Select" ${rSel.has(c.id) ? "checked" : ""}>
      <img loading="lazy" alt="Capital on ${esc(pageLabel(c.page))}" src="/img/crop/${R.book.id}/${c.id}?s=220">
      <div class="row"><span class="small" title="${esc(pageLabel(c.page))}" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(pageLabel(c.page))}</span><span class="score">${c.status === "added" ? "" : (c.score * 100).toFixed(0) + "%"}</span></div>
      <div class="flags">${flagChips(c)}</div>
      <div class="btns"><button class="btn ok" data-s="kept" title="Yes, a capital">✓ Keep</button><button class="btn bad" data-s="rejected" title="Not a capital">✕ Out</button></div>
    </div>`).join("")}</div>`;
  body.querySelectorAll(".cap").forEach(el => {
    const id = el.dataset.id, c = R.capitals.find(x => x.id === id);
    el.querySelectorAll("[data-s]").forEach(b => b.onclick = () => setStatus(c, c.status === b.dataset.s ? "auto" : b.dataset.s));
    el.querySelector(".pick").onchange = e => { e.target.checked ? rSel.add(id) : rSel.delete(id); el.classList.toggle("sel", e.target.checked); $("#r-rejsel").disabled = !rSel.size; };
    el.querySelector("img").onclick = () => zoomCap(c);
  });
  $("#r-rejsel").disabled = !rSel.size;
}
async function setStatus(c, st) {
  await api(`/api/books/${R.book.id}/capitals/${c.id}`, {method: "POST", body: {status: st}});
  c.status = st; c.visible = st !== "rejected" && (["kept", "added"].includes(st) || c.score >= STATUS.settings.score_cutoff);
  rMode === "grid" ? renderGrid() : renderPages();
}
$("#r-keepall").onclick = async () => {
  const ids = filtered().filter(c => c.status === "auto").map(c => c.id);
  if (!ids.length) return;
  await api(`/api/books/${R.book.id}/bulk`, {method: "POST", body: {ids, status: "kept"}}); toast(`Confirmed ${ids.length}`); loadReview();
};
$("#r-rejsel").onclick = async () => {
  await api(`/api/books/${R.book.id}/bulk`, {method: "POST", body: {ids: [...rSel], status: "rejected"}}); toast(`Threw out ${rSel.size}`); rSel.clear(); loadReview();
};
function zoomCap(c) {
  const pg = (R.book.pages || []).find(p => p.id === c.page);
  modal(`<div class="cmp"><figure><img src="/img/crop/${R.book.id}/${c.id}?s=400" alt=""><figcaption>${esc(pageLabel(c.page))} · ${c.status === "added" ? "added by you" : "confidence " + (c.score * 100).toFixed(0) + "%"}</figcaption></figure>
    ${pg ? `<figure><div class="pg" style="cursor:default"><img src="/img/page/${R.book.id}/${c.page}?s=1000" alt="" style="max-height:70vh;width:auto;max-width:min(520px,80vw)">
      <svg viewBox="0 0 ${pg.w} ${pg.h}" preserveAspectRatio="none"><rect class="bx flag" x="${c.box[0]}" y="${c.box[1]}" width="${c.box[2] - c.box[0]}" height="${c.box[3] - c.box[1]}"/></svg></div><figcaption>Where it is on the page</figcaption></figure>` : ""}</div>
    <div class="flags" style="margin:12px 0;display:flex;gap:6px;flex-wrap:wrap">${flagChips(c)}</div>
    <div style="display:flex;gap:8px;justify-content:end"><button class="btn ok" id="z-k">✓ Keep</button><button class="btn bad" id="z-r">✕ Throw out</button><button class="btn" data-close>Close</button></div>`);
  $("#z-k").onclick = () => { setStatus(c, "kept"); $("#modal").classList.remove("on"); };
  $("#z-r").onclick = () => { setStatus(c, "rejected"); $("#modal").classList.remove("on"); };
}

function renderPages() {
  const pages = R.book.pages || [];
  const cnt = {}; R.capitals.forEach(c => { if (c.visible) cnt[c.page] = (cnt[c.page] || 0) + 1; });
  const flagged = {}; R.capitals.forEach(c => { if (c.visible && c.flags.length && !["kept", "added"].includes(c.status)) flagged[c.page] = true; });
  if (!rPage || !pages.some(p => p.id === rPage)) rPage = (pages.find(p => flagged[p.id]) || pages[0])?.id;
  $("#r-count").textContent = `${pages.length} pages`;
  $("#r-body").innerHTML = `<div class="pagewrap"><div class="card pagelist" id="plist">${pages.map(p => `<button data-p="${p.id}" aria-current="${p.id === rPage}">
      <span>${esc(p.label || p.id)}</span><span>${flagged[p.id] ? '<span class="chip warn" style="padding:0 6px">!</span> ' : ""}${cnt[p.id] || ""}</span></button>`).join("")}</div>
    <div class="card stage"><div><div class="small muted" style="margin-bottom:8px">Click a box to cycle keep → throw out → automatic. Drag on the page to add a capital the model missed. <kbd>←</kbd> <kbd>→</kbd> change page.</div>
      <div class="pg" id="pg"></div></div></div></div>`;
  $("#plist").querySelectorAll("button").forEach(b => b.onclick = () => { rPage = b.dataset.p; renderPages(); });
  $("#plist").querySelector('[aria-current="true"]')?.scrollIntoView({block: "nearest"});
  drawPage();
}
function drawPage() {
  const pg = (R.book.pages || []).find(p => p.id === rPage); if (!pg) return;
  const caps = R.capitals.filter(c => c.page === pg.id && (c.visible || c.status === "rejected" || $("#r-filter").value === "below"));
  $("#pg").innerHTML = `<img src="/img/page/${R.book.id}/${pg.id}?s=1600" alt="${esc(pg.label || pg.id)}" draggable="false">
    <svg id="pgsvg" viewBox="0 0 ${pg.w} ${pg.h}" preserveAspectRatio="none">${caps.map(c => {
      const cls = c.status === "rejected" ? "no" : c.status === "added" ? "add" : c.status === "kept" ? "ok" : c.flags.length ? "flag" : "ok";
      return `<rect class="bx ${cls}" data-id="${c.id}" x="${c.box[0]}" y="${c.box[1]}" width="${c.box[2] - c.box[0]}" height="${c.box[3] - c.box[1]}"><title>${esc((c.flags.map(f => STATUS.flag_text[f]).join(", ")) || (c.status === "added" ? "added by you" : "confidence " + (c.score * 100).toFixed(0) + "%"))}</title></rect>`;
    }).join("")}</svg>`;
  const svg = $("#pgsvg");
  const toPage = e => { const b = svg.getBoundingClientRect(); return [(e.clientX - b.left) * pg.w / b.width, (e.clientY - b.top) * pg.h / b.height]; };
  svg.querySelectorAll("rect[data-id]").forEach(r => r.onclick = e => {
    e.stopPropagation(); const c = R.capitals.find(x => x.id === r.dataset.id);
    const next = {auto: "kept", kept: "rejected", rejected: "auto", added: "rejected"}[c.status];
    setStatus(c, next);
  });
  svg.onpointerdown = e => { if (e.target.dataset.id) return; drawing = {a: toPage(e), b: toPage(e)}; svg.setPointerCapture(e.pointerId); };
  svg.onpointermove = e => { if (!drawing) return; drawing.b = toPage(e); let r = svg.querySelector(".draw"); if (!r) { r = document.createElementNS("http://www.w3.org/2000/svg", "rect"); r.setAttribute("class", "bx draw"); svg.appendChild(r); }
    const [x1, y1] = drawing.a, [x2, y2] = drawing.b; r.setAttribute("x", Math.min(x1, x2)); r.setAttribute("y", Math.min(y1, y2)); r.setAttribute("width", Math.abs(x2 - x1)); r.setAttribute("height", Math.abs(y2 - y1)); };
  svg.onpointerup = async () => {
    if (!drawing) return; const [x1, y1] = drawing.a, [x2, y2] = drawing.b; drawing = null;
    const box = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
    if (box[2] - box[0] > pg.w * 0.015 && box[3] - box[1] > pg.h * 0.01) {
      await api(`/api/books/${R.book.id}/capitals`, {method: "POST", body: {page: pg.id, box}}); toast("Capital added");
      R = await api(`/api/books/${R.book.id}`);
    }
    drawPage();
  };
}
document.addEventListener("keydown", e => {
  if (view !== "review" || rMode !== "pages" || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) || $("#modal").classList.contains("on")) return;
  const pages = R?.book.pages || []; const i = pages.findIndex(p => p.id === rPage);
  if (e.key === "ArrowRight" && i < pages.length - 1) { rPage = pages[i + 1].id; renderPages(); }
  if (e.key === "ArrowLeft" && i > 0) { rPage = pages[i - 1].id; renderPages(); }
});

// ------------------------------------------------------------------ woodblocks
let G = null, gSel = new Set(), gChosen = null;
function saveChosen() { try { localStorage.setItem("is-chosen", JSON.stringify([...gChosen])); } catch (e) {} }
async function loadPatterns() {
  G = await api("/api/groups");
  Object.keys(G.books).forEach(colorFor);
  if (!gChosen) {
    let saved = null; try { saved = JSON.parse(localStorage.getItem("is-chosen")); } catch (e) {}
    gChosen = new Set(Array.isArray(saved) ? saved : Object.keys(G.books));
  }
  [...gChosen].forEach(b => { if (!G.books[b]) gChosen.delete(b); });   // deleted books
  renderPicker(); renderGroups();
}
// what still needs comparing among the chosen books: capitals never grouped, and pairs of books never compared
function pendingWork() {
  const done = new Set(G.compared_pairs || []), ids = [...gChosen].sort();
  const missingPairs = [];
  for (let i = 0; i < ids.length; i++) for (let j = i; j < ids.length; j++)
    if (!done.has(`${ids[i]}|${ids[j]}`)) missingPairs.push([ids[i], ids[j]]);
  const ungrouped = ids.reduce((a, b) => a + (G.books[b]?.ungrouped || 0), 0);
  const newBooks = ids.filter(b => !done.has(`${b}|${b}`));
  return {missingPairs, ungrouped, newBooks};
}
function renderPicker() {
  const done = new Set(G.compared_pairs || []);
  $("#g-picker").innerHTML = Object.entries(G.books).map(([id, b]) => {
    const fresh = !done.has(`${id}|${id}`) || b.ungrouped;
    return `<label class="pbook"><input type="checkbox" data-b="${id}" ${gChosen.has(id) ? "checked" : ""}>
      <span class="dot" style="background:${colorFor(id)}"></span>
      <span class="t"><b title="${esc(b.title)}">${esc(b.title)}</b><span>${esc(b.printer || "printer not set")} · ${b.capitals} capitals${fresh ? ` · <span class="warn">not compared yet</span>` : ""}</span></span></label>`;
  }).join("") || '<span class="muted">No books yet: add some in the Library.</span>';
  $("#g-picker").querySelectorAll("input").forEach(inp => inp.onchange = () => {
    inp.checked ? gChosen.add(inp.dataset.b) : gChosen.delete(inp.dataset.b); saveChosen(); renderNote(); renderGroups();
  });
  renderNote();
}
function renderNote() {
  const {missingPairs, ungrouped, newBooks} = pendingWork(), n = gChosen.size;
  const busy = lastBusy;
  let html;
  if (!n) html = `<span class="muted">Tick at least one book.</span>`;
  else if (!missingPairs.length && !ungrouped) html = `<span class="ok">✓ All ${n} selected book${n > 1 ? "s are" : " is"} compared.</span><span class="muted">Ticking and unticking books (among those already compared) is instant: nothing is recalculated.</span>`;
  else {
    const parts = [];
    if (newBooks.length) parts.push(`${newBooks.length} book${newBooks.length > 1 ? "s" : ""} not compared yet`);
    else if (missingPairs.length) parts.push(`${missingPairs.length} pair${missingPairs.length > 1 ? "s" : ""} of books not compared with each other yet`);
    if (ungrouped) parts.push(`${ungrouped} capital${ungrouped > 1 ? "s" : ""} not in a woodblock yet`);
    html = `<span class="todo">${parts.join(" · ")}.</span><span class="muted">Press <b>Compare selected books</b>: only comparisons not done before are calculated.</span>`;
  }
  $("#g-note").innerHTML = html + (G.updated ? `<span class="spacer"></span><span class="small muted">last compared ${new Date(G.updated * 1000).toLocaleString()}</span>` : "");
  const work = n && (missingPairs.length || ungrouped);
  $("#g-update").disabled = !n || busy;
  $("#g-update").classList.toggle("primary", !!work);
  $("#g-update").textContent = work ? "Compare selected books" : "Re-check selected books";
  $("#g-rebuild").disabled = !n || busy;
}
$("#g-all").onclick = () => { gChosen = new Set(Object.keys(G.books)); saveChosen(); renderPicker(); renderGroups(); };
$("#g-none").onclick = () => { gChosen = new Set(); saveChosen(); renderPicker(); renderGroups(); };
$("#g-filter").onchange = () => renderGroups();
function printerOf(bid) { return (G.books[bid]?.printer || "").trim().toLowerCase() || "unknown:" + bid; }
function renderGroups() {
  // only the chosen books' impressions; a woodblock with none of them disappears
  const groups = G.groups.map(g => ({...g, members: g.members.filter(m => gChosen.has(m.book))})).filter(g => g.members.length)
    .map(g => ({...g, books: [...new Set(g.members.map(m => m.book))]}))
    .map(g => ({...g, printers: [...new Set(g.books.map(printerOf))]}));
  const cross = groups.filter(g => g.printers.length > 1), multi = groups.filter(g => g.books.length > 1);
  const nUnclear = groups.reduce((a, g) => a + g.members.filter(m => m.verdict === "unclear").length, 0);
  $("#g-stats").innerHTML = `
    <div class="card stat"><b>${groups.length.toLocaleString()}</b><span>woodblocks found</span></div>
    <div class="card stat"><b>${multi.length}</b><span>used in more than one book</span></div>
    <div class="card stat"><b>${cross.length}</b><span>shared between printers</span></div>
    <div class="card stat"><b>${nUnclear}</b><span>unclear matches to check</span></div>
    <div class="card stat"><b>${(G.mirror_pairs || []).length}</b><span>mirror-image look-alikes (recut copies?)</span></div>`;
  const f = $("#g-filter").value;
  const show = groups.filter(g => (f === "all" || (f === "cross" && g.printers.length > 1) ||
    (f === "multi" && g.books.length > 1) || (f === "unclear" && g.members.some(m => m.verdict === "unclear"))));
  const box = $("#groups");
  $("#g-count").textContent = `${show.length} of ${groups.length} woodblocks shown`;
  if (!gChosen.size) { box.innerHTML = `<div class="empty card"><h2>No books selected</h2>Tick the books you want to compare above.</div>`; updateDock(); return; }
  if (!groups.length) { box.innerHTML = `<div class="empty card"><h2>No woodblocks yet</h2>Press <b>Compare selected books</b> to compare every capital in them with every other and group them by woodblock.</div>`; updateDock(); return; }
  if (!show.length) { box.innerHTML = `<div class="empty card"><h2>No woodblocks match this filter</h2>${f === "cross" ? (gChosen.size < 2 ? "Select books by at least two different printers." : "No block was found in books by two different printers among the selected books. Check each book's printer (Library → Edit details).") : ""}</div>`; updateDock(); return; }
  const LIM = 150;
  box.innerHTML = show.slice(0, LIM).map(g => {
    const counts = g.books.map(b => `<span class="chip"><span class="dot" style="background:${colorFor(b)}"></span>${g.members.filter(m => m.book === b).length} · ${esc(G.books[b]?.title || b)}</span>`).join("");
    return `<article class="card group ${g.printers.length > 1 ? "cross" : ""}" data-g="${g.id}">
      <div class="ghead"><h3>Woodblock ${g.id}</h3><span class="muted small">${g.members.length} impression${g.members.length > 1 ? "s" : ""}</span>
        ${g.printers.length > 1 ? '<span class="chip accent">Shared between printers</span>' : ""}${counts}
        <span class="spacer"></span><button class="btn sm" data-act="all">Select all</button><button class="btn sm primary" data-act="move">Move here</button></div>
      <div class="thumbs">${g.members.map(m => {
        const sel = gSel.has(m.key), unclear = m.verdict === "unclear";
        const tip = `${G.books[m.book]?.title || m.book} · ${m.page}` + (m.score != null ? ` · match ${m.score}` : "") + (m.orient ? ` · printed ${TURN[m.orient]}` : "") + (unclear ? " · unclear" : "");
        return `<div class="th ${sel ? "sel" : ""} ${unclear ? "unclear" : ""}" data-k="${m.key}" title="${esc(tip)}" tabindex="0">
          <span class="bk" style="background:${colorFor(m.book)}"></span>
          <img loading="lazy" alt="${esc(tip)}" src="/img/crop/${m.book}/${m.cap}?o=${m.orient}&s=160">
          ${m.orient ? `<span class="turn">${TURN[m.orient]}</span>` : ""}<div class="lab">${esc(m.page)}</div></div>`;
      }).join("")}</div></article>`;
  }).join("") + (show.length > LIM ? `<p class="muted">Showing the first ${LIM} of ${show.length}. Use the filters to narrow down.</p>` : "");
  box.querySelectorAll(".th").forEach(el => {
    el.onclick = () => { const k = el.dataset.k; gSel.has(k) ? gSel.delete(k) : gSel.add(k); el.classList.toggle("sel"); updateDock(); };
    el.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); } };
    el.ondblclick = () => compare(el.dataset.k, +el.closest(".group").dataset.g);
  });
  box.querySelectorAll(".group").forEach(el => el.querySelectorAll("[data-act]").forEach(b => b.onclick = async () => {
    const gid = +el.dataset.g;
    if (b.dataset.act === "all") { G.groups.find(g => g.id === gid).members.filter(m => gChosen.has(m.book)).forEach(m => gSel.add(m.key)); renderGroups(); }
    else if (gSel.size) { await api("/api/groups/edit", {method: "POST", body: {op: "move", keys: [...gSel], target: gid}}); toast(`Moved ${gSel.size}`); gSel.clear(); loadPatterns(); }
    else toast("Select impressions first, then press Move here");
  }));
  updateDock();
}
function compare(key, gid) {
  const g = G.groups.find(x => x.id === gid), m = g.members.find(x => x.key === key), ref = g.members.find(x => x.key !== key && x.book !== m.book) || g.members.find(x => x.key !== key) || m;
  const cap = x => `${esc(G.books[x.book]?.title || x.book)} · ${esc(x.page)}${x.orient ? " · printed " + TURN[x.orient] : ""}`;
  modal(`<div class="cmp"><figure><img src="/img/crop/${m.book}/${m.cap}?o=${m.orient}&s=400" alt=""><figcaption>${cap(m)}${m.score != null ? `<br>match score ${m.score}` : ""}</figcaption></figure>
    <figure><img src="/img/crop/${ref.book}/${ref.cap}?o=${ref.orient}&s=400" alt=""><figcaption>${cap(ref)}</figcaption></figure></div>
    <p class="small muted" style="max-width:740px">Same block: every cut of the design lines up, including small cracks and breaks. A recut copy has the same layout but the vine work differs in detail. Scores: same book ~0.92, the same block in two books by the same printer ~0.87.</p>
    <div style="text-align:right"><button class="btn" data-close>Close</button></div>`);
}
function updateDock() { $("#dock").classList.toggle("on", gSel.size > 0 && view === "patterns"); $("#dock-n").textContent = gSel.size; }
$("#dock-new").onclick = async () => { await api("/api/groups/edit", {method: "POST", body: {op: "new", keys: [...gSel]}}); toast("New woodblock made"); gSel.clear(); loadPatterns(); };
$("#dock-clear").onclick = () => { gSel.clear(); renderGroups(); };
$("#dock-rej").onclick = async () => {
  const by = {}; gSel.forEach(k => { const [b, c] = k.split("/"); (by[b] ??= []).push(c); });
  for (const [b, ids] of Object.entries(by)) await api(`/api/books/${b}/bulk`, {method: "POST", body: {ids, status: "rejected"}});
  toast(`Threw out ${gSel.size}`); gSel.clear(); loadPatterns();
};
$("#g-update").onclick = async () => {
  await api("/api/group", {method: "POST", body: {mode: "update", books: [...gChosen]}}); lastBusy = true; renderNote();
  toast(`Comparing ${gChosen.size} book${gChosen.size > 1 ? "s" : ""} in the background (only new comparisons are calculated)`);
};
$("#g-rebuild").onclick = async () => {
  if (!confirm(`Regroup the ${gChosen.size} selected book${gChosen.size > 1 ? "s" : ""} from scratch? Your manual moves for their capitals will be discarded. Match scores already calculated are reused, so this is quick.`)) return;
  await api("/api/group", {method: "POST", body: {mode: "rebuild", books: [...gChosen]}}); lastBusy = true; renderNote(); toast("Regrouping started");
};

// ------------------------------------------------------------------ settings
const PARAMS = [
  ["Finding capitals", [
    ["score_cutoff", "Confidence cut-off", 0.05, 0.99, 0.01, "Boxes the model is less sure about than this are hidden. 0.5 suits new books; 0.94 is strict (tuned on Euclid, misses more)."],
    ["flag_low_confidence", "Flag as low confidence below", 0.5, 0.999, 0.01, "Kept boxes under this confidence get a “Low confidence” flag so you look at them."],
    ["flag_size_factor", "Flag unusual size (× typical)", 1.3, 5, 0.1, "Flags capitals this many times bigger or smaller than the book's typical capital: often chapter initials, pictures or partial boxes."]]],
  ["Grouping woodblocks", [
    ["same_block", "Same-block score", 0.5, 0.95, 0.01, "How closely two impressions must match (on average across a woodblock) to count as the same block. Calibrated: same book ~0.92, different designs < 0.5."],
    ["unclear_band", "Flag as unclear above", 0.3, 0.9, 0.01, "Matches between this and the same-block score are flagged “unclear”: a worn block, or a close recut copy."],
    ["neighbours", "Candidates compared per capital", 10, 200, 5, "Higher finds slightly more matches but takes longer. 60 catches ~98% of true matches."],
    ["workers", "CPU cores to use", 1, 32, 1, "Parallel workers for matching."]]],
];
async function loadSettings() {
  STATUS = await api("/api/status");
  const s = STATUS.settings, m = STATUS.model;
  $("#settings").innerHTML = PARAMS.map(([title, ps]) => `<div class="card"><h2>${title}</h2>${ps.map(([k, label, lo, hi, step, help]) => `
      <div class="param"><div class="top-row"><label for="p-${k}">${label}</label><output id="o-${k}">${s[k]}</output></div>
      <input type="range" id="p-${k}" min="${lo}" max="${hi}" step="${step}" value="${s[k]}"><p>${help}</p></div>`).join("")}
      ${title === "Grouping woodblocks" ? `<div class="param"><label class="check" style="color:var(--ink)"><input type="checkbox" id="p-allow_rotations" ${s.allow_rotations ? "checked" : ""}> Blocks may be printed turned (90°, 180°, 270°)</label></div>` : ""}</div>`).join("") +
    `<div class="card"><h2>Model</h2><p class="small">${esc(m.description || "")}</p>
      <div class="small muted" style="display:grid;gap:4px">${(m.facts || []).map(f => `<div>${esc(f)}</div>`).join("")}</div>
      <p class="small muted">Workspace folder: ${esc(STATUS.workspace)}</p></div>`;
  document.querySelectorAll("#settings input").forEach(inp => inp.onchange = inp.oninput = async e => {
    const k = inp.id.slice(2), v = inp.type === "checkbox" ? inp.checked : +inp.value;
    if (inp.type !== "checkbox") $("#o-" + k).textContent = v;
    if (e.type === "change") { STATUS.settings = await api("/api/settings", {method: "POST", body: {[k]: v}}); toast("Saved"); }
  });
}
$("#s-reset").onclick = async () => { await api("/api/settings", {method: "POST", body: STATUS.defaults}); loadSettings(); toast("Defaults restored"); };

// ------------------------------------------------------------------ start
(async () => {
  STATUS = await api("/api/status");
  let v = "library"; try { v = localStorage.getItem("is-view") || v; } catch (e) {}
  show(["library", "review", "patterns", "settings", "method"].includes(v) ? v : "library");
  pollJobs();
})();
