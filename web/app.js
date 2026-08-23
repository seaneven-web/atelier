(() => {
const $ = id => document.getElementById(id);
const api = async (path, body) => {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
};
const S = { status: null, portfolio: null, sketch: null, last: null };

// ---------------------------------------------------------------- status / models
async function refresh() {
  S.status = await api("/api/status");
  const m = S.status.models;
  $("ddir").textContent = S.status.data_dir;
  $("pill-device").textContent = "device: " + S.status.device;
  const pm = $("pill-models");
  pm.textContent = m.ready ? "models: ready · offline ok" : (m.vgg19 && !m.sd_turbo ? "models: style only" : "models: not downloaded");
  pm.className = "pill " + (m.ready ? "ok" : "bad");
  $("models-text").innerHTML = m.ready
    ? `Both pretrained models are in the sandbox folder. The app no longer needs the network.`
    : `Atelier uses two pretrained models that have to be fetched once: <b>VGG19</b> (the style network, ~550 MB) and <b>SD-Turbo</b> (draws what you describe, ~2.5 GB). They are saved in <code>${m.folder}</code>; after that everything runs here, offline.` +
      (m.sd_lib ? "" : ` <span style='color:var(--warn)'>Drawing from words is unavailable in this build${m.sd_error ? " (" + m.sd_error + ")" : ""} — repainting sketches still works. Please report this at github.com/seaneven-web/atelier/issues with the line from the log file in the sandbox folder.</span>`);
  $("dl").hidden = m.ready;
  $("b-models").style.display = m.ready && !$("dl-log").textContent ? "none" : "";
  const sel = $("pf"); const cur = sel.value;
  sel.innerHTML = S.status.portfolios.map(p => `<option value="${p.name}">${p.name} (${p.count})</option>`).join("") + `<option value="">— new —</option>`;
  if (S.status.portfolios.length && !S.portfolio) { S.portfolio = S.status.portfolios[0].name; }
  sel.value = S.portfolio || "";
  $("pf-name").hidden = !!sel.value;
  if (S.portfolio) loadPortfolio(S.portfolio);
}
async function poll(jid, onLine, bar) {
  let seen = 0;
  for (;;) {
    const j = await api("/api/job/" + jid);
    for (; seen < j.log.length; seen++) onLine(j.log[seen]);
    if (j.status !== "running") { if (bar) bar.classList.add("done"); if (j.status === "error") throw new Error(j.error); return j.result; }
    await new Promise(r => setTimeout(r, 700));
  }
}
const logger = el => line => { el.textContent += (el.textContent ? "\n" : "") + line; el.scrollTop = el.scrollHeight; };

$("dl").addEventListener("click", async () => {
  $("dl").disabled = true; $("dl-prog").hidden = false; $("dl-log").textContent = "";
  try { const { job } = await api("/api/models/download", {}); await poll(job, logger($("dl-log")), $("dl-prog")); }
  catch (e) { logger($("dl-log"))("error: " + e.message); }
  $("dl").disabled = false; refresh();
});
$("open-folder").addEventListener("click", () => api("/api/open", { folder: "" }));
$("open-gallery").addEventListener("click", () => api("/api/open", { folder: "gallery" }));

// ---------------------------------------------------------------- portfolio
const readFile = f => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res({ name: f.name, data: r.result }); r.onerror = rej; r.readAsDataURL(f); });
async function addFiles(files) {
  const list = [...files].filter(f => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
  if (!list.length) return;
  const name = $("pf").value || $("pf-name").value.trim() || "portfolio";
  $("pf-prog").hidden = false; $("pf-prog").classList.remove("done"); $("pf-log").textContent = `reading ${list.length} file(s)…`;
  const payload = [];
  for (const f of list) payload.push(await readFile(f));
  try {
    const { job } = await api("/api/portfolio", { name, files: payload, paper: $("paper").checked });
    const info = await poll(job, logger($("pf-log")), $("pf-prog"));
    S.portfolio = name; await refresh(); showPortfolio(info);
  } catch (e) { logger($("pf-log"))("error: " + e.message); }
}
async function loadPortfolio(name) {
  try { showPortfolio(await api("/api/portfolio/" + encodeURIComponent(name))); } catch (e) { $("pf-meta").textContent = e.message; }
}
function showPortfolio(info) {
  $("thumbs").innerHTML = info.pieces.map(p => `<img src="${p.thumb}" title="${p.name}" alt="${p.name}">`).join("");
  const tags = Object.entries(info.tags || {}).sort((a, b) => b[1] - a[1]);
  $("pf-meta").innerHTML = (info.count >= 1 ? `${info.count} piece(s)` : "") +
    (info.medium ? ` · reads as: <i>${info.medium}</i>` : "") +
    (tags.length ? `<br>words it can aim for: ${tags.map(([t, c]) => `<span class="chip tag">${t} · ${c}</span>`).join("")}` : "");
  const eye = info.eye || {};
  $("eye-box").hidden = info.count < 1;
  $("eye-box").classList.toggle("trained", !!eye.trained);
  $("eye-state").textContent = eye.trained
    ? `trained · ${eye.patches} patches · ${eye.seconds}s`
    : "not trained yet — optional";
  $("train-eye").textContent = eye.trained ? "Train again" : "Train on this portfolio";
  $("l-eye").style.opacity = eye.trained ? 1 : .45;
  $("l-eye").title = eye.trained ? "" : "train the eye above to use this";
  const ok = info.count >= 1;
  $("b2").classList.toggle("off", !ok); $("b3").classList.toggle("off", !ok && !S.last);
  if (ok) $("prompt").focus();
}
const drop = $("drop");
$("files").addEventListener("change", e => addFiles(e.target.files));
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); });
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("files").click(); } });
$("pf").addEventListener("change", () => { S.portfolio = $("pf").value || null; $("pf-name").hidden = !!S.portfolio; if (S.portfolio) loadPortfolio(S.portfolio); else { $("thumbs").innerHTML = ""; $("pf-meta").textContent = ""; } });

$("train-eye").addEventListener("click", async () => {
  if (!S.portfolio) { alert("add the artist's pieces first"); return; }
  $("train-eye").disabled = true; $("eye-prog").hidden = false; $("eye-prog").classList.remove("done"); $("eye-log").textContent = "";
  try {
    const { job } = await api("/api/train_eye", { portfolio: S.portfolio, minutes: 4 });
    await poll(job, logger($("eye-log")), $("eye-prog"));
    await loadPortfolio(S.portfolio);
  } catch (e) { logger($("eye-log"))("error: " + e.message); }
  $("train-eye").disabled = false;
});

// ---------------------------------------------------------------- sketch
const sdrop = $("sdrop");
async function setSketch(file) {
  if (!file) return;
  const f = await readFile(file);
  try {
    const r = await api("/api/sketch", f);
    S.sketch = r.sketch; $("spreview").src = r.url + "?" + Date.now(); $("spreview").hidden = false; $("sdrop-text").hidden = true; $("srow").hidden = false;
  } catch (e) { alert(e.message); }
}
$("sfile").addEventListener("change", e => setSketch(e.target.files[0]));
sdrop.addEventListener("dragover", e => { e.preventDefault(); sdrop.classList.add("over"); });
sdrop.addEventListener("dragleave", () => sdrop.classList.remove("over"));
sdrop.addEventListener("drop", e => { e.preventDefault(); sdrop.classList.remove("over"); setSketch(e.dataTransfer.files[0]); });
$("sclear").addEventListener("click", e => { e.preventDefault(); S.sketch = null; $("spreview").hidden = true; $("sdrop-text").hidden = false; $("srow").hidden = true; $("sfile").value = ""; });
$("smode").addEventListener("change", () => {});

// ---------------------------------------------------------------- sliders
for (const id of ["strength", "freedom", "iters", "res", "count", "structure", "eye"]) {
  const el = $(id), out = $("v-" + id);
  const show = () => out.textContent = ["strength", "freedom", "structure", "eye"].includes(id) ? (+el.value).toFixed(2) : el.value;
  el.addEventListener("input", show); show();
}

// ---------------------------------------------------------------- paint
async function paint(extra, storedBody) {
  let body;
  if (storedBody) body = Object.assign({}, storedBody, extra || {});
  else {
    if (!S.portfolio) { alert("add the artist's pieces first"); return; }
    const prompt = $("prompt").value.trim();
    if (!prompt && !S.sketch) { $("prompt").focus(); return; }
    body = Object.assign({ portfolio: S.portfolio, prompt, count: +$("count").value, strength: +$("strength").value,
      freedom: +$("freedom").value, iters: +$("iters").value, res: +$("res").value, seed: $("seed").value.trim() || null,
      structure: +$("structure").value, eye: +$("eye").value,
      sketch: S.sketch, mode: $("smode").value, size: 768 }, extra || {});
  }
  $("viewing").hidden = true; S.viewing = null;
  $("paint").disabled = true; $("p-prog").hidden = false; $("p-prog").classList.remove("done"); $("p-log").textContent = "";
  $("b3").classList.remove("off"); $("r-hint").textContent = "painting…";
  try {
    const { job } = await api("/api/paint", body);
    const res = await poll(job, logger($("p-log")), $("p-prog"));
    S.last = body; showResult(res, body); S.viewing = res.stamp;
  } catch (e) { logger($("p-log"))("error: " + e.message); $("r-hint").textContent = ""; }
  $("paint").disabled = false; loadHistory();
}
function showResult(res, body) {
  const r = res.recipe;
  if (!r) { $("recipe").hidden = true; } else {
  $("recipe").hidden = false;
  const chips = [];
  if (r.content_prompt) chips.push(`<span class="chip">draws: “${r.content_prompt}”</span>`);
  for (const [k, v] of Object.entries(r.attributes || {})) if (Math.abs(v) > .05) chips.push(`<span class="chip ${v > 0 ? "pos" : "neg"}">${k} ${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}</span>`);
  for (const t of r.tags || []) chips.push(`<span class="chip tag">toward “${t}”</span>`);
  $("rchips").innerHTML = chips.join("") || `<span class="chip">free sample from the style</span>`;
  $("rmeta").textContent = `style from ${r.style_pieces.length} piece(s) · strength ${body.strength} · ${body.count} piece(s)`;
  $("rwhy").textContent = `Style pieces: ${r.style_pieces.slice(0, 6).join(", ")}${r.style_pieces.length > 6 ? "…" : ""}. ${body.sketch ? (body.mode === "sketch" ? `Started from your sketch (freedom ${body.freedom}).` : "Repainted your sketch as-is.") : "Mood words pick the style pieces and grade the colour; the rest of the words are drawn."}`;
  }
  $("out").innerHTML = res.files.map((f, i) => `<div class="piece ${i === 0 ? "best" : ""}">
      <img src="${f}" alt="piece ${i + 1}" data-full="${f}">
      <div class="cap"><span><b>#${i + 1}</b>${i === 0 && r ? " · pick" : ""}</span><span title="how completely it took the style">${res.info && res.info[i] && res.info[i].critic ? Math.round(res.info[i].critic * 100) + "%" : ""}</span></div>
      <div class="acts"><button data-open="${res.names[i]}">open</button><a class="quiet" href="${f}?download=1" download><button>save…</button></a><button data-vary="${res.names[i]}">variations</button></div>
    </div>`).join("");
  $("r-hint").textContent = res.sheet ? `saved in the gallery folder · sheet: ${res.sheet.split("/").pop()}` : "";
  $("more").hidden = false;
}
$("out").addEventListener("click", async e => {
  const img = e.target.closest("img[data-full]"); if (img) { lightbox(img.dataset.full); return; }
  const o = e.target.closest("[data-open]"); if (o) { api("/api/open", { name: o.dataset.open }); return; }
  const v = e.target.closest("[data-vary]");
  if (v) { // variations = repaint this result with a little freedom
    const r = await fetch(v.dataset.vary.startsWith("/") ? v.dataset.vary : "/gallery/" + v.dataset.vary); const blob = await r.blob();
    await setSketch(new File([blob], v.dataset.vary, { type: "image/png" })); $("smode").value = "sketch"; $("freedom").value = 0.35; $("freedom").dispatchEvent(new Event("input"));
    paint();
  }
});
$("more").addEventListener("click", () => paint({ seed: null }));
$("paint").addEventListener("click", () => paint());
$("prompt").addEventListener("keydown", e => { if (e.key === "Enter") paint(); });
function lightbox(src) { const d = document.createElement("div"); d.className = "lightbox"; d.innerHTML = `<img src="${src}">`; d.addEventListener("click", () => d.remove()); document.body.append(d); }
async function loadHistory() {
  try {
    const h = await api("/api/history"); S.history = h;
    $("hist-count").textContent = h.length ? `${h.length} run${h.length === 1 ? "" : "s"}` : "";
    const fmtDay = t => { const d = new Date(t * 1000), today = new Date(); const y = new Date(); y.setDate(today.getDate() - 1);
      return d.toDateString() === today.toDateString() ? "today" : d.toDateString() === y.toDateString() ? "yesterday" : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: d.getFullYear() === today.getFullYear() ? undefined : "numeric" }); };
    let day = "", html = "";
    for (const r of h) {
      const dd = fmtDay(r.time); if (dd !== day) { day = dd; html += `<div class="day">${dd}</div>`; }
      const tm = new Date(r.time * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
      html += `<button class="run ${S.viewing === r.stamp ? "active" : ""}" data-stamp="${r.stamp}"><img src="${r.thumb || ""}" alt=""><span><b>${r.title || "untitled"}</b><small>${tm} · ${r.count} piece${r.count === 1 ? "" : "s"}${r.has_sketch ? " · from a sketch" : ""}${r.portfolio ? " · " + r.portfolio : ""}</small></span></button>`;
    }
    $("hist").innerHTML = html || `<p class="muted small">Your runs will appear here.</p>`;
  } catch (e) {}
}
async function openRun(stamp) {
  try {
    const r = await api("/api/run/" + encodeURIComponent(stamp));
    S.viewing = stamp; S.viewingRun = r;
    $("b3").classList.remove("off"); showResult(r, r.settings || {});
    const when = r.time ? new Date(r.time * 1000).toLocaleString() : "";
    $("viewing").hidden = false; $("viewing-text").textContent = `viewing “${r.title || stamp}”${when ? " · " + when : ""}`;
    $("run-again").hidden = !r.settings; $("load-settings").hidden = !r.settings;
    $("r-hint").textContent = r.bare ? "an older sheet (no saved settings)" : `saved run · ${r.names.length} piece(s)`;
    document.querySelectorAll(".hist .run").forEach(b => b.classList.toggle("active", b.dataset.stamp === stamp));
    $("b3").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { alert(e.message); }
}
function applySettings(st) {
  if (!st) return;
  if (st.portfolio) { S.portfolio = st.portfolio; $("pf").value = st.portfolio; loadPortfolio(st.portfolio); }
  $("prompt").value = st.prompt || "";
  for (const k of ["strength", "freedom", "iters", "res", "count", "structure", "eye"]) if (st[k] != null) { $(k).value = st[k]; $(k).dispatchEvent(new Event("input")); }
  $("seed").value = st.seed || "";
  if (st.sketch) { S.sketch = st.sketch; $("spreview").src = "/sketches/" + st.sketch; $("spreview").hidden = false; $("sdrop-text").hidden = true; $("srow").hidden = false; $("smode").value = st.mode || "sketch"; }
  else { S.sketch = null; $("spreview").hidden = true; $("sdrop-text").hidden = false; $("srow").hidden = true; }
}
$("hist").addEventListener("click", e => { const b = e.target.closest(".run"); if (b) openRun(b.dataset.stamp); });
$("run-again").addEventListener("click", () => { const r = S.viewingRun; if (!r || !r.settings) return; applySettings(r.settings); paint({ seed: null }, Object.assign({}, r.settings, { seed: null })); });
$("load-settings").addEventListener("click", () => { if (S.viewingRun) { applySettings(S.viewingRun.settings); $("b2").scrollIntoView({ behavior: "smooth" }); } });
$("delete-run").addEventListener("click", async () => { if (!S.viewing) return; if (!confirm("Delete this run and its pieces from the gallery folder?")) return; await api("/api/run/delete", { stamp: S.viewing }); $("viewing").hidden = true; $("out").innerHTML = ""; $("r-hint").textContent = ""; S.viewing = null; loadHistory(); });

async function showLast() {
  try {
    const l = await api("/api/last"); if (!l.files.length) return;
    $("b3").classList.remove("off"); $("r-hint").textContent = `from last time · “${l.title}”`;
    $("out").innerHTML = l.files.map((f, i) => `<div class="piece"><img src="${f}" alt="piece ${i + 1}" data-full="${f}">
      <div class="cap"><span><b>#${i + 1}</b></span><span>${l.names[i]}</span></div>
      <div class="acts"><button data-open="${l.names[i]}">open</button><a class="quiet" href="${f}?download=1" download><button>save…</button></a><button data-vary="${l.names[i]}">variations</button></div></div>`).join("");
  } catch (e) {}
}
refresh().then(loadHistory).then(() => { if (S.history && S.history.length) openRun(S.history[0].stamp); }).catch(e => { $("models-text").textContent = "cannot reach the local server: " + e.message; });
})();
