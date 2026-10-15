"use strict";

const DOMS = ["dsp", "electronics", "control"];
const DOMAIN_NAMES = { dsp: "DSP", electronics: "Electronics", control: "Control" };
const MAX_ROWS = 500;
// Sequential blue ramp (light to dark) for the heatmap; text ink is picked per step.
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

const state = { summary: null, filters: { model: "", family: "", outcome: "" }, attempt: null };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const f3 = (v) => (v == null || !isFinite(v) ? "–" : v.toFixed(3));
const pct = (v) => (v == null ? "–" : `${(100 * v).toFixed(0)}%`);
const usd = (v) => (v == null ? "–" : v < 1 ? `$${v.toFixed(3)}` : `$${v.toFixed(2)}`);
const isNum = (v) => typeof v === "number" && isFinite(v);
let chartCounter = 0;

function si(v) {
  if (!isNum(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-2 && a < 1e4) return String(+v.toPrecision(4));
  const steps = [[1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"]];
  for (const [m, p] of steps) if (a >= m) return `${+(v / m).toPrecision(3)}${p}`;
  return v.toExponential(2);
}

function metricValue(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return esc(v);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return v.toExponential(3);
  return String(+v.toPrecision(5));
}

function settingsLine(s) {
  const parts = [];
  if (s.reasoning_effort) parts.push(`effort ${s.reasoning_effort}`);
  if (s.max_tokens) parts.push(`max ${s.max_tokens.toLocaleString("en-US")} tokens`);
  if (s.temperature != null) parts.push(`temperature ${s.temperature}`);
  if (s.providers && s.providers.length) parts.push(`provider ${s.providers.join(", ")}`);
  return parts.join(" · ");
}

function outcomeOf(a) {
  if (a.format_error) return { cls: "fmt", text: "Format error" };
  return a.passed ? { cls: "pass", text: "Pass" } : { cls: "fail", text: "Fail" };
}

/* ---------- Overview ---------- */

function renderLeaderboard() {
  const s = state.summary;
  if (!s.models.length) {
    $("lb").innerHTML = `<tbody><tr><td class="empty">No results yet.</td></tr></tbody>`;
    return;
  }
  const head = `<thead><tr><th class="num">#</th><th>Model</th><th>Score</th><th class="num">Pass</th>${DOMS.map((d) => `<th class="num">${DOMAIN_NAMES[d]}</th>`).join("")}<th class="num">Format errors</th><th class="num">Cost</th><th class="num">Avg output tokens</th></tr></thead>`;
  const rows = s.models.map((m, i) => {
    const [lo, hi] = m.overall.ci;
    const dom = (d) => (m.domains[d] ? f3(m.domains[d].mean) : "–");
    return `<tr tabindex="0" data-model="${esc(m.key)}" aria-label="Show attempts of ${esc(m.label)}">
      <td class="num">${i + 1}</td>
      <td class="model">${esc(m.label)}<small>${esc(settingsLine(m.settings)) || "&nbsp;"}</small></td>
      <td><div class="scorebar"><span class="mono">${f3(m.overall.mean)}</span><div class="bar"><div class="fill" style="width:${100 * m.overall.mean}%"></div><div class="ci" style="left:${100 * lo}%;width:${100 * (hi - lo)}%"></div></div></div><span class="ci-text mono">95% CI ${f3(lo)} to ${f3(hi)}</span></td>
      <td class="num">${pct(m.pass_rate)}</td>
      ${DOMS.map((d) => `<td class="num">${dom(d)}</td>`).join("")}
      <td class="num">${m.format_errors}</td>
      <td class="num">${usd(m.cost_usd)}</td>
      <td class="num">${m.mean_output_tokens == null ? "–" : Math.round(m.mean_output_tokens).toLocaleString("en-US")}</td>
    </tr>`;
  });
  $("lb").innerHTML = head + `<tbody>${rows.join("")}</tbody>`;
  $("lb").querySelectorAll("tr[data-model]").forEach((tr) => {
    const go = () => showAttempts({ model: tr.dataset.model, family: "", outcome: "" });
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
}

function rampColor(v) {
  const i = Math.max(0, Math.min(RAMP.length - 1, Math.round(v * (RAMP.length - 1))));
  return { bg: RAMP[i], ink: i >= 7 ? "#ffffff" : "#0b0b0b" };
}

function renderHeatmap() {
  const s = state.summary;
  const fams = DOMS.flatMap((d) => s.families.filter((f) => f.domain === d));
  if (!s.models.length) { $("heat").innerHTML = ""; return; }
  const domHead = DOMS.map((d) => {
    const n = fams.filter((f) => f.domain === d).length;
    return n ? `<th class="dom" colspan="${n}">${DOMAIN_NAMES[d]}</th>` : "";
  }).join("");
  const famHead = fams.map((f) => `<th class="famcol" title="${esc(f.title)}">${esc(f.id.split(".")[1])}</th>`).join("");
  const rows = s.models.map((m) => {
    const cells = fams.map((f) => {
      const r = m.families[f.id];
      if (!r) return `<td class="cell">–</td>`;
      const c = rampColor(r.mean);
      return `<td class="cell"><button type="button" data-model="${esc(m.key)}" data-family="${esc(f.id)}" style="background:${c.bg};color:${c.ink}" aria-label="${esc(m.label)}, ${esc(f.id)}: score ${f3(r.mean)}, pass rate ${pct(r.pass_rate)}" title="pass rate ${pct(r.pass_rate)}, ${r.attempts} attempts">${f3(r.mean)}</button></td>`;
    }).join("");
    return `<tr><th scope="row">${esc(m.label)}</th>${cells}</tr>`;
  }).join("");
  $("heat").innerHTML = `<thead><tr><th></th>${domHead}</tr><tr><th></th>${famHead}</tr></thead><tbody>${rows}</tbody>`;
  $("heat").querySelectorAll("button[data-model]").forEach((b) =>
    b.addEventListener("click", () => showAttempts({ model: b.dataset.model, family: b.dataset.family, outcome: "" })));
}

function renderCost() {
  const models = state.summary.models.filter((m) => isNum(m.cost_usd) && m.cost_usd > 0);
  $("cost").hidden = models.length === 0;
  if (!models.length) return;
  const costs = models.map((m) => m.cost_usd);
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...costs)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...costs))));
  renderPanel($("cost-chart"), {
    title: "", note: "",
    x: { label: "Total cost (USD)", scale: "log", range: [lo, hi === lo ? lo * 10 : hi] },
    y: { label: "Score", scale: "linear", range: [0, 1] },
    series: [], forbidden: [], envelope: null, markers: [], violations: [],
    points: models.map((m) => ({ x: m.cost_usd, y: m.overall.mean, label: m.label })),
  }, { showKeys: false });
}

function renderFamilies() {
  const s = state.summary;
  const fams = DOMS.flatMap((d) => s.families.filter((f) => f.domain === d));
  $("fam-grid").innerHTML = fams.map((f) => {
    const accepted = f.status === "accepted";
    return `<article class="panel fam">
      <header><span class="chip">${DOMAIN_NAMES[f.domain]}</span><span class="pill ${accepted ? "accepted" : "review"}">${accepted ? "Accepted" : "Awaiting review"}</span>
        <h3>${esc(f.title)}</h3><span class="mono" style="font-size:12.5px;color:var(--muted)">${esc(f.id)}</span></header>
      ${f.description.map((p) => `<p>${esc(p)}</p>`).join("")}
      <details><summary>Example prompt (dev seed 0)</summary><pre>${esc(f.example_prompt)}</pre></details>
    </article>`;
  }).join("");
}

/* ---------- Attempt list ---------- */

function fillFilters() {
  const s = state.summary;
  $("f-model").innerHTML = `<option value="">All models</option>` + s.models.map((m) => `<option value="${esc(m.key)}">${esc(m.label)}</option>`).join("");
  $("f-family").innerHTML = `<option value="">All families</option>` + s.families.map((f) => `<option value="${esc(f.id)}">${esc(f.id)}</option>`).join("");
  for (const [id, key] of [["f-model", "model"], ["f-family", "family"], ["f-outcome", "outcome"]]) {
    $(id).addEventListener("change", () => { state.filters[key] = $(id).value; renderAttemptList(); });
  }
}

function showAttempts(filters) {
  state.filters = { ...state.filters, ...filters };
  $("f-model").value = state.filters.model;
  $("f-family").value = state.filters.family;
  $("f-outcome").value = state.filters.outcome;
  renderAttemptList();
  if (location.hash !== "#attempts") location.hash = "attempts";
  else $("attempts").scrollIntoView();
}

function renderAttemptList() {
  const s = state.summary;
  const labels = Object.fromEntries(s.models.map((m) => [m.key, m.label]));
  const { model, family, outcome } = state.filters;
  const list = s.attempts.filter((a) =>
    (!model || a.model_key === model) && (!family || a.family_id === family) && (!outcome || outcomeOf(a).cls === outcome));
  $("att-count").textContent = `${list.length} attempt${list.length === 1 ? "" : "s"}${list.length > MAX_ROWS ? `, first ${MAX_ROWS} shown` : ""}`;
  const rows = list.slice(0, MAX_ROWS).map((a) => {
    const o = outcomeOf(a);
    return `<tr tabindex="0" data-id="${esc(a.id)}">
      <td>${esc(labels[a.model_key] || a.model_key)}</td><td class="mono">${esc(a.family_id)}</td>
      <td class="num">${a.seed}</td><td class="num">${a.attempt + 1}</td><td class="num">${f3(a.score)}</td>
      <td><span class="outcome ${o.cls}"><span class="dot"></span>${o.text}</span></td></tr>`;
  }).join("");
  $("att-list").innerHTML = `<thead><tr><th>Model</th><th>Task family</th><th class="num">Seed</th><th class="num">Try</th><th class="num">Score</th><th>Outcome</th></tr></thead><tbody>${rows || `<tr><td colspan="6" class="empty">No attempts match.</td></tr>`}</tbody>`;
  $("att-list").querySelectorAll("tr[data-id]").forEach((tr) => {
    const go = () => { location.hash = `attempt/${tr.dataset.id}`; };
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
}

/* ---------- Attempt viewer ---------- */

async function openAttempt(id) {
  $("main-content").hidden = true;
  document.body.classList.add("viewing");
  const v = $("viewer");
  v.hidden = false;
  v.innerHTML = `<p class="empty">Loading attempt ${esc(id)}…</p>`;
  window.scrollTo(0, 0);
  try {
    const res = await fetch(`data/attempts/${encodeURIComponent(id)}.json`);
    if (!res.ok) throw new Error(res.status);
    state.attempt = await res.json();
  } catch (e) {
    v.innerHTML = `<p class="empty">Attempt ${esc(id)} could not be loaded. <a href="#attempts">Back to the list</a></p>`;
    return;
  }
  renderViewer();
}

function closeViewer() {
  $("viewer").hidden = true;
  $("main-content").hidden = false;
  document.body.classList.remove("viewing");
  state.attempt = null;
}

const ICON = {
  pass: '<svg width="22" height="22" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="var(--good)"/><path d="M5.8 10.4l2.8 2.8 5.6-6" fill="none" stroke="var(--surface)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  fail: '<svg width="22" height="22" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="var(--critical)"/><path d="M6.8 6.8l6.4 6.4M13.2 6.8l-6.4 6.4" fill="none" stroke="var(--surface)" stroke-width="2.2" stroke-linecap="round"/></svg>',
  fmt: '<svg width="22" height="22" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="#fab219"/><path d="M10 5.5v5.5M10 14v.5" fill="none" stroke="#0b0b0b" stroke-width="2.2" stroke-linecap="round"/></svg>',
};

function renderViewer() {
  const a = state.attempt;
  const g = a.grade;
  const o = outcomeOf({ passed: g.passed, format_error: g.format_error });
  const metrics = Object.entries(g.metrics || {}).map(([k, val]) => {
    const bad = k.endsWith("_ratio") && isNum(val) && val < 1;
    return `<div><dt>${esc(k.replaceAll("_", " "))}</dt><dd class="${bad ? "bad" : ""}">${metricValue(val)}</dd></div>`;
  }).join("");
  const u = a.usage || {};
  const usage = [
    ["prompt tokens", u.prompt_tokens], ["output tokens", u.completion_tokens], ["reasoning tokens", u.reasoning_tokens],
    ["cost (USD)", a.cost_usd], ["latency (s)", a.latency_s], ["finish reason", a.finish_reason],
    ["model served", a.model_served], ["provider", a.provider_served],
  ].filter(([, val]) => val !== null && val !== undefined)
    .map(([k, val]) => `<div><dt>${esc(k)}</dt><dd>${metricValue(val)}</dd></div>`).join("");
  const spec = Object.entries(a.spec).map(([k, val]) =>
    `<div><dt>${esc(k.replaceAll("_", " "))}</dt><dd>${typeof val === "object" ? esc(JSON.stringify(val)) : metricValue(val)}</dd></div>`).join("");

  $("viewer").innerHTML = `
    <div class="viewer-head">
      <button type="button" class="back" id="back">← All attempts</button>
      <h2>${esc(a.family_id)} · seed ${a.seed} · try ${a.attempt + 1}</h2>
      <span class="chip">${esc(a.model_label)}</span>
    </div>
    <div class="summary-row">
      <div class="verdict ${o.cls === "pass" ? "" : o.cls}">
        <div class="verdict-state">${ICON[o.cls]}${o.text}</div>
        <div class="line">Score <span class="mono">${f3(g.score)}</span> of 1.000</div>
        ${g.error ? `<div class="line">${esc(g.error)}</div>` : ""}
      </div>
      <div class="panel"><dl class="kv">${metrics || `<div><dt>No measurements</dt><dd>–</dd></div>`}</dl></div>
    </div>
    <div class="charts" id="v-charts"></div>
    <div class="io">
      <div class="panel"><h3>Prompt</h3><span class="sub">Identical for every model</span><pre>${esc(a.prompt)}</pre></div>
      <div class="panel"><h3>Response</h3><span class="sub">${esc(settingsLine(a.settings)) || "&nbsp;"}</span><pre>${esc(a.response)}</pre>
        ${a.reasoning ? `<details><summary>Reasoning returned by the API</summary><pre>${esc(a.reasoning)}</pre></details>` : ""}</div>
    </div>
    <div class="io">
      <div class="panel"><h3>Spec used by the grader</h3><dl class="kv">${spec}</dl></div>
      <div class="panel"><h3>Usage</h3><dl class="kv">${usage || `<div><dt>Not recorded</dt><dd>–</dd></div>`}</dl></div>
    </div>`;
  $("back").addEventListener("click", () => { location.hash = "attempts"; });
  renderViewerCharts();
}

function renderViewerCharts() {
  const host = $("v-charts");
  if (!host || !state.attempt) return;
  const panels = (state.attempt.plot_data && state.attempt.plot_data.panels) || [];
  if (!panels.length) { host.innerHTML = `<p class="empty">No plot for this attempt.</p>`; return; }
  host.innerHTML = panels.map((p, i) => `<div class="panel chart-panel"><h3>${esc(p.title)}</h3>${p.note ? `<p class="note">${esc(p.note)}</p>` : ""}<div class="keys" id="keys-${i}"></div><div class="chart" id="chart-${i}"></div></div>`).join("");
  panels.forEach((p, i) => renderPanel($(`chart-${i}`), p, { keysHost: $(`keys-${i}`) }));
}

/* ---------- Generic chart renderer ---------- */

function niceTicks(lo, hi, target) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((st) => st >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  const decades = out.filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
  return decades.length >= 3 ? decades : out;
}

function interp(xs, ys, x) {
  if (x < xs[0] || x > xs[xs.length - 1]) return null;
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  if (xs[hi] === xs[lo]) return ys[lo];
  return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
}

function renderPanel(host, p, opts = {}) {
  const id = `c${++chartCounter}`;
  const logx = p.x.scale === "log";
  const series = p.series || [];
  const finite = (arr) => arr.filter(isNum);
  const allX = finite(series.flatMap((s) => s.x).concat((p.points || []).map((q) => q.x)));
  const allY = finite(series.flatMap((s) => s.y).concat((p.points || []).map((q) => q.y)));
  let [x0, x1] = p.x.range || [Math.min(...allX), Math.max(...allX)];
  let y0, y1;
  if (p.y.range) [y0, y1] = p.y.range;
  else {
    const extra = [];
    (p.forbidden || []).forEach((f) => f.y.forEach((v) => { if (isNum(v)) extra.push(v); }));
    if (p.envelope) extra.push(...finite(p.envelope.lo), ...finite(p.envelope.hi));
    const ys = allY.concat(extra);
    y0 = Math.min(...ys); y1 = Math.max(...ys);
    const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.06;
    y0 -= pad; y1 += pad;
  }
  if (!(x1 > x0)) x1 = x0 + 1;
  if (!(y1 > y0)) y1 = y0 + 1;

  const W = Math.max(300, host.clientWidth || 600);
  const H = Math.round(Math.max(220, Math.min(340, W * 0.56)));
  const ml = 58, mr = 14, mt = 14, mb = 42, pw = W - ml - mr, ph = H - mt - mb;
  const tx = (v) => (logx ? Math.log10(v) : v);
  const sx = (v) => ml + ((tx(v) - tx(x0)) / (tx(x1) - tx(x0))) * pw;
  const sy = (v) => mt + ((y1 - Math.max(y0 - (y1 - y0), Math.min(y1 + (y1 - y0), v))) / (y1 - y0)) * ph;

  const forb = (p.forbidden || []).map((f) => ({
    xa: isNum(f.x[0]) ? f.x[0] : -Infinity, xb: isNum(f.x[1]) ? f.x[1] : Infinity,
    ya: isNum(f.y[0]) ? f.y[0] : -Infinity, yb: isNum(f.y[1]) ? f.y[1] : Infinity,
  }));
  const env = p.envelope;
  const violates = (x, y) => {
    if (!isNum(x) || !isNum(y)) return false;
    if (forb.some((f) => x >= f.xa && x <= f.xb && y > f.ya && y < f.yb)) return true;
    if (env) {
      const lo = interp(env.x, env.lo, x), hi = interp(env.x, env.hi, x);
      if (lo !== null && hi !== null && (y < lo - 1e-12 || y > hi + 1e-12)) return true;
    }
    return false;
  };

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(p.title || "chart")}">
    <defs>
      <pattern id="hatch-${id}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="var(--forbid-fill)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--forbid-line)" stroke-width="1.2"/>
      </pattern>
      <clipPath id="clip-${id}"><rect x="${ml}" y="${mt}" width="${pw}" height="${ph}"/></clipPath>
    </defs>`;

  const xt = logx ? logTicks(x0, x1) : niceTicks(x0, x1, W < 480 ? 4 : 7);
  const yt = niceTicks(y0, y1, 6);
  yt.forEach((v) => {
    svg += `<line x1="${ml}" x2="${ml + pw}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/>`;
    svg += `<text x="${ml - 6}" y="${sy(v) + 3.5}" text-anchor="end">${si(v)}</text>`;
  });
  xt.forEach((v) => {
    svg += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${mt}" y2="${mt + ph}" stroke="var(--grid)"/>`;
    svg += `<text x="${sx(v)}" y="${mt + ph + 15}" text-anchor="middle">${si(v)}</text>`;
  });
  svg += `<text x="${ml + pw}" y="${H - 6}" text-anchor="end">${esc(p.x.label)}</text>`;
  svg += `<text x="4" y="${mt - 3}">${esc(p.y.label)}</text>`;

  svg += `<g clip-path="url(#clip-${id})">`;
  if (env && env.x.length > 1) {
    const top = env.x.map((x, i) => `${sx(x).toFixed(1)},${sy(env.hi[i]).toFixed(1)}`);
    const bot = env.x.map((x, i) => `${sx(x).toFixed(1)},${sy(env.lo[i]).toFixed(1)}`).reverse();
    svg += `<polygon points="${top.concat(bot).join(" ")}" fill="var(--band)" stroke="var(--band-edge)" stroke-width="1"/>`;
  }
  forb.forEach((f) => {
    const xa = Math.max(f.xa, x0), xb = Math.min(f.xb, x1);
    const ya = Math.max(f.ya, y0), yb = Math.min(f.yb, y1);
    if (!(xb > xa) || !(yb > ya)) return;
    svg += `<rect x="${sx(xa)}" y="${sy(yb)}" width="${sx(xb) - sx(xa)}" height="${sy(ya) - sy(yb)}" fill="url(#hatch-${id})"/>`;
    if (isNum(f.yb) && f.yb <= y1) svg += `<line x1="${sx(xa)}" x2="${sx(xb)}" y1="${sy(f.yb)}" y2="${sy(f.yb)}" stroke="var(--mask-edge)" stroke-width="1.2"/>`;
    if (isNum(f.ya) && f.ya >= y0) svg += `<line x1="${sx(xa)}" x2="${sx(xb)}" y1="${sy(f.ya)}" y2="${sy(f.ya)}" stroke="var(--mask-edge)" stroke-width="1.2"/>`;
  });

  // Markers: thin reference lines with short labels, staggered so labels do not collide.
  (p.markers || []).forEach((m, i) => {
    if (m.axis === "x" && m.value >= x0 && m.value <= x1) {
      const x = sx(m.value);
      svg += `<line x1="${x}" x2="${x}" y1="${mt}" y2="${mt + ph}" stroke="var(--axis)" stroke-width="1.2" stroke-dasharray="4 3"/>`;
      const right = x < ml + pw * 0.7;
      svg += `<text x="${right ? x + 4 : x - 4}" y="${mt + 12 + 13 * (i % 3)}" text-anchor="${right ? "start" : "end"}" style="fill:var(--ink-2)">${esc(m.label)}</text>`;
    } else if (m.axis === "y" && m.value >= y0 && m.value <= y1) {
      const y = sy(m.value);
      svg += `<line x1="${ml}" x2="${ml + pw}" y1="${y}" y2="${y}" stroke="var(--axis)" stroke-width="1.2" stroke-dasharray="4 3"/>`;
      svg += `<text x="${ml + pw - 4}" y="${y - 4}" text-anchor="end" style="fill:var(--ink-2)">${esc(m.label)}</text>`;
    }
  });

  const pathOf = (pts) => pts.map(([x, y], k) => `${k ? "L" : "M"}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join("");
  series.filter((s) => s.role === "target").forEach((s) => {
    const pts = s.x.map((x, i) => [x, s.y[i]]).filter(([x, y]) => isNum(x) && isNum(y) && (!logx || x > 0));
    svg += `<path d="${pathOf(pts)}" fill="none" stroke="var(--target)" stroke-width="1.6" stroke-dasharray="5 4"/>`;
  });
  const answers = series.filter((s) => s.role !== "target");
  let anyBad = false;
  answers.forEach((s) => {
    const pts = s.x.map((x, i) => [x, s.y[i]]).filter(([x, y]) => isNum(x) && isNum(y) && (!logx || x > 0));
    let run = [], runBad = null, prev = null;
    const flush = () => {
      if (run.length > 1) svg += `<path d="${pathOf(run)}" fill="none" stroke="${runBad ? "var(--critical)" : "var(--curve)"}" stroke-width="${runBad ? 2.6 : 2}" stroke-linejoin="round" stroke-linecap="round"/>`;
    };
    pts.forEach((pt) => {
      const bad = violates(pt[0], pt[1]);
      if (bad) anyBad = true;
      if (runBad === null || bad === runBad) run.push(pt);
      else { flush(); run = [prev, pt]; }
      runBad = bad; prev = pt;
    });
    flush();
    if (pts.length === 1) svg += `<circle cx="${sx(pts[0][0])}" cy="${sy(pts[0][1])}" r="3" fill="var(--curve)"/>`;
  });
  (p.violations || []).forEach(([x, y]) => {
    if (isNum(x) && isNum(y) && x >= x0 && x <= x1) svg += `<circle cx="${sx(x)}" cy="${sy(y)}" r="2.6" fill="var(--critical)"/>`;
  });
  svg += `</g>`;

  (p.points || []).forEach((q) => {
    svg += `<circle cx="${sx(q.x)}" cy="${sy(q.y)}" r="5" fill="var(--curve)" stroke="var(--surface)" stroke-width="2"/>`;
    svg += `<text x="${sx(q.x) + 8}" y="${sy(q.y) + 4}" style="fill:var(--ink)">${esc(q.label)}</text>`;
  });

  svg += `<line x1="${ml}" x2="${ml + pw}" y1="${mt + ph}" y2="${mt + ph}" stroke="var(--axis)"/>`;
  svg += `<line x1="${ml}" x2="${ml}" y1="${mt}" y2="${mt + ph}" stroke="var(--axis)"/>`;
  svg += `<line class="xh" x1="0" x2="0" y1="${mt}" y2="${mt + ph}" stroke="var(--ink-2)" visibility="hidden"/>`;
  svg += `<circle class="hd" r="4.5" fill="var(--curve)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>`;
  svg += `<rect class="hit" x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent" style="cursor:crosshair"/>`;
  svg += `</svg><div class="tip" hidden></div>`;
  host.innerHTML = svg;

  if (opts.keysHost) {
    const keys = [];
    if (answers.length) keys.push(`<span class="key"><i class="k-answer"></i>${esc(answers[0].name)}</span>`);
    series.filter((s) => s.role === "target").forEach((s) => keys.push(`<span class="key"><i class="k-target"></i>${esc(s.name)}</span>`));
    if (forb.length) keys.push(`<span class="key"><i class="k-forbid"></i>Forbidden by spec</span>`);
    if (env) keys.push(`<span class="key"><i class="k-band"></i>${esc(env.label || "Allowed band")}</span>`);
    if (anyBad || (p.violations || []).length) keys.push(`<span class="key"><i class="k-bad"></i>Outside the spec</span>`);
    opts.keysHost.innerHTML = keys.join("");
  }

  const main = answers[0];
  if (!main) return;
  const pts = main.x.map((x, i) => [x, main.y[i]]).filter(([x, y]) => isNum(x) && isNum(y)).sort((a, b) => a[0] - b[0]);
  if (!pts.length) return;
  const target = series.find((s) => s.role === "target");
  const el = host.querySelector("svg"), tip = host.querySelector(".tip");
  const xh = el.querySelector(".xh"), hd = el.querySelector(".hd"), hit = el.querySelector(".hit");
  const move = (ev) => {
    const rect = el.getBoundingClientRect(), scale = W / rect.width;
    const px = (ev.clientX - rect.left) * scale;
    let best = 0, bestD = Infinity;
    pts.forEach(([x], i) => { const d = Math.abs(sx(x) - px); if (d < bestD) { bestD = d; best = i; } });
    const [x, y] = pts[best];
    const cx = sx(x), cy = sy(y), bad = violates(x, y);
    xh.setAttribute("x1", cx); xh.setAttribute("x2", cx); xh.setAttribute("visibility", "visible");
    hd.setAttribute("cx", cx); hd.setAttribute("cy", cy); hd.setAttribute("fill", bad ? "var(--critical)" : "var(--curve)"); hd.setAttribute("visibility", "visible");
    let html = `${esc(p.x.label)}: ${si(x)}<br>${esc(p.y.label)}: ${metricValue(y)}`;
    if (target) { const t = interp(target.x, target.y, x); if (t !== null) html += `<br>${esc(target.name)}: ${metricValue(t)}`; }
    if (bad) html += `<br><span style="color:var(--critical)">outside the spec</span>`;
    tip.innerHTML = html;
    tip.hidden = false;
    const tw = tip.offsetWidth, hostW = host.clientWidth;
    tip.style.left = `${Math.min(hostW - tw - 4, Math.max(4, cx / scale + 12))}px`;
    tip.style.top = `${Math.max(0, cy / scale - tip.offsetHeight - 10)}px`;
  };
  const leave = () => { xh.setAttribute("visibility", "hidden"); hd.setAttribute("visibility", "hidden"); tip.hidden = true; };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", leave);
}

/* ---------- Routing and start ---------- */

function route() {
  const m = location.hash.match(/^#attempt\/([A-Za-z0-9_-]+)$/);
  if (m) openAttempt(m[1]);
  else {
    const wasOpen = !$("viewer").hidden;
    closeViewer();
    if (wasOpen && location.hash.length > 1) {
      const target = document.getElementById(location.hash.slice(1));
      if (target) target.scrollIntoView();
    }
  }
}

async function init() {
  try {
    const res = await fetch("data/summary.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    state.summary = await res.json();
  } catch (e) {
    $("main-content").innerHTML = `<p class="empty">Could not load data/summary.json. Serve the site over HTTP, for example with <code>uv run python -m http.server -d site</code>.</p>`;
    return;
  }
  const s = state.summary;
  $("demo-notice").hidden = !s.demo;
  $("gen-line").textContent = `Benchmark version ${s.benchmark_version}. Data generated ${s.generated.replace("T", " ").replace("+00:00", " UTC")} from ${s.runs.join(", ") || "no runs"}.`;
  renderLeaderboard();
  renderHeatmap();
  renderCost();
  renderFamilies();
  fillFilters();
  renderAttemptList();
  window.addEventListener("hashchange", route);
  let t = null;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (state.attempt) renderViewerCharts(); renderCost(); }, 150); });
  route();
}

init();
