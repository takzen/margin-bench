"use strict";
// margin-bench playground: replays (or follows live) one episode of a model operating a simulated plant.

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const isNum = (v) => typeof v === "number" && isFinite(v);
const fx = (v, d) => (isNum(v) ? v.toFixed(d) : "–");
const int = (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "–");
const usd = (v) => (!isNum(v) ? "–" : v === 0 ? "$0" : v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);

function clock(s) {
  if (!isNum(s)) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}
function dur(s) {
  if (!isNum(s)) return "–";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

const S = {
  ep: null, liveUrl: null, phases: [], total: 0, T: 0, playing: false, speed: 20,
  hold: true, holdUntil: 0, next: 0, cur: -1, follow: true, lastTs: 0, chartPhase: null, scene: null,
};
const COL = { t: 0, level: 1, meas: 2, out: 3, valve: 4, outlet: 5, sp: 6 };

// ---------- data ----------

function buildPhases(ep) {
  const v = ep.view || {};
  const phases = [];
  const com = v.commissioning;
  if (com && com.rows.length) {
    phases.push({ key: "commissioning", name: "Commissioning", rows: com.rows, t0: 0, dur: Math.max(1, com.rows[com.rows.length - 1][0]) });
  }
  const acc = v.acceptance;
  if (acc && acc.rows.length) {
    const t0 = phases.length ? phases[0].dur : 0;
    phases.push({ key: "acceptance", name: "Acceptance test", rows: acc.rows, t0, dur: acc.rows[acc.rows.length - 1][0], events: acc.events || [] });
  }
  return phases;
}

function anchorOf(step) {
  return step.kind === "model" ? step.sim_t : step.sim_t0;
}

function phaseAt(T) {
  let p = S.phases[0];
  for (const q of S.phases) if (T >= q.t0) p = q;
  return { p, t: Math.min(p.dur, Math.max(0, T - p.t0)) };
}

function rowAt(rows, t) {
  let lo = 0, hi = rows.length - 1;
  if (t <= rows[0][0]) return rows[0];
  if (t >= rows[hi][0]) return rows[hi];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (rows[mid][0] <= t) lo = mid; else hi = mid;
  }
  const a = rows[lo], b = rows[hi], w = (t - a[0]) / (b[0] - a[0] || 1);
  const lerp = (i) => (isNum(a[i]) && isNum(b[i]) ? a[i] + w * (b[i] - a[i]) : a[i]);
  return [t, lerp(1), a[2], lerp(3), lerp(4), a[5], a[6]];
}

// ---------- scene ----------

const TANK = { x: 200, w: 160, top: 40, bottom: 360 };
const ym = (m, span) => TANK.bottom - (m / span) * (TANK.bottom - TANK.top);

function buildScene() {
  const svg = $("scene");
  svg.innerHTML = "";
  const g = S.ep.view.geometry;
  const span = g.span_m;
  const sc = {};
  // signal lines (instrument dashes)
  el("path", { d: `M420 216 V384 H60 V330`, fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
  el("path", { d: `M60 130 V24 H98`, fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
  // inlet pipe
  el("line", { x1: 0, y1: 52, x2: TANK.x + 4, y2: 52, stroke: "var(--pipe)", "stroke-width": 11, "stroke-linecap": "butt" }, svg);
  sc.inFlow = el("line", { x1: 0, y1: 52, x2: TANK.x + 4, y2: 52, stroke: "var(--flow)", "stroke-width": 3, "stroke-dasharray": "7 9" }, svg);
  // outlet pipe
  el("line", { x1: TANK.x + TANK.w - 4, y1: 346, x2: 540, y2: 346, stroke: "var(--pipe)", "stroke-width": 11 }, svg);
  sc.outFlow = el("line", { x1: TANK.x + TANK.w - 4, y1: 346, x2: 540, y2: 346, stroke: "var(--flow)", "stroke-width": 3, "stroke-dasharray": "7 9" }, svg);
  // tank
  el("rect", { x: TANK.x, y: TANK.top, width: TANK.w, height: TANK.bottom - TANK.top, rx: 6, fill: "var(--steel-fill)" }, svg);
  sc.water = el("rect", { x: TANK.x + 2, width: TANK.w - 4, fill: "var(--water)", opacity: 0.85 }, svg);
  sc.surface = el("line", { x1: TANK.x + 2, x2: TANK.x + TANK.w - 2, stroke: "var(--water-top)", "stroke-width": 3 }, svg);
  sc.stream = el("rect", { x: TANK.x + 3, width: 6, y: 57, fill: "var(--water)", opacity: 0.7 }, svg);
  sc.band = el("rect", { x: TANK.x + 2, width: TANK.w - 4, fill: "var(--ink)", opacity: 0.1 }, svg);
  sc.sp = el("line", { x1: TANK.x - 6, x2: TANK.x + TANK.w + 6, stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "6 4" }, svg);
  sc.spLabel = el("text", { x: TANK.x + TANK.w + 10, "text-anchor": "start", class: "tag" }, svg);
  for (const [m, label] of [[g.hh_m, "HH"], [g.ll_m, "LL"]]) {
    el("line", { x1: TANK.x, x2: TANK.x + TANK.w, y1: ym(m, span), y2: ym(m, span), stroke: "var(--critical)", "stroke-width": 1.2, "stroke-dasharray": "3 3" }, svg);
    el("text", { x: TANK.x + TANK.w + 10, y: ym(m, span) + 4 }, svg).textContent = `${label} ${m.toFixed(2)} m`;
  }
  sc.shell = el("rect", { x: TANK.x, y: TANK.top, width: TANK.w, height: TANK.bottom - TANK.top, rx: 6, fill: "none", stroke: "var(--steel)", "stroke-width": 3 }, svg);
  for (let m = 0; m <= span + 1e-9; m += 0.5) {
    el("line", { x1: TANK.x - 6, x2: TANK.x, y1: ym(m, span), y2: ym(m, span), stroke: "var(--steel)" }, svg);
    el("text", { x: TANK.x - 9, y: ym(m, span) + 4, "text-anchor": "end" }, svg).textContent = m.toFixed(1);
  }
  sc.alarm = el("text", { x: TANK.x + TANK.w / 2, y: TANK.top + 22, "text-anchor": "middle", class: "tag", fill: "var(--critical)" }, svg);
  sc.levelText = el("text", { x: TANK.x + TANK.w / 2, "text-anchor": "middle", class: "big" }, svg);
  // control valve
  el("polygon", { points: "96,40 124,64 124,40 96,64", fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5, transform: "translate(0,0)" }, svg);
  el("line", { x1: 110, y1: 52, x2: 110, y2: 30, stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
  el("path", { d: "M98 30 A12 12 0 0 1 122 30 Z", fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
  el("text", { x: 130, y: 26, class: "tag" }, svg).textContent = "FV-101";
  sc.valveText = el("text", { x: 110, y: 82, "text-anchor": "middle" }, svg);
  sc.valveBar = el("rect", { x: 86, y: 88, height: 4, rx: 2, fill: "var(--output)" }, svg);
  el("rect", { x: 86, y: 88, width: 48, height: 4, rx: 2, fill: "none", stroke: "var(--axis)" }, svg);
  // outlet demand valve
  el("polygon", { points: "441,334 469,358 469,334 441,358", fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
  el("line", { x1: 455, y1: 346, x2: 455, y2: 324, stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
  el("line", { x1: 446, y1: 324, x2: 464, y2: 324, stroke: "var(--ink-2)", "stroke-width": 2 }, svg);
  el("text", { x: 455, y: 316, "text-anchor": "middle", class: "tag" }, svg).textContent = "demand";
  sc.demandText = el("text", { x: 455, y: 376, "text-anchor": "middle" }, svg);
  // level transmitter
  el("line", { x1: TANK.x + TANK.w, y1: 200, x2: 404, y2: 200, stroke: "var(--steel)", "stroke-width": 2 }, svg);
  el("circle", { cx: 420, cy: 200, r: 16, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
  el("text", { x: 420, y: 204, "text-anchor": "middle", class: "tag" }, svg).textContent = "LT";
  sc.ltText = el("text", { x: 442, y: 204 }, svg);
  // controller faceplate
  el("rect", { x: 10, y: 130, width: 150, height: 200, rx: 8, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.2 }, svg);
  el("text", { x: 20, y: 150, class: "tag" }, svg).textContent = "LIC-101";
  sc.modeBg = el("rect", { x: 100, y: 138, width: 52, height: 17, rx: 4 }, svg);
  sc.mode = el("text", { x: 126, y: 151, "text-anchor": "middle", class: "tag" }, svg);
  const rows = ["SP", "PV", "OUT", "Kc", "Ti", "Td", "PV filter", "anti-windup"];
  sc.fp = {};
  rows.forEach((r, i) => {
    el("text", { x: 20, y: 176 + i * 19 }, svg).textContent = r;
    sc.fp[r] = el("text", { x: 150, y: 176 + i * 19, "text-anchor": "end", fill: "var(--ink)" }, svg);
  });
  S.scene = sc;
}

function tuningAt(T) {
  // The tuning in force at story time T: the last set_tuning call before it (commissioning),
  // or the final tuning during the acceptance test.
  const v = S.ep.view;
  const { p } = phaseAt(T);
  if (p.key === "acceptance") return v.commissioning.tuning;
  let tuning = null;
  for (const s of S.ep.steps) {
    if (s.kind !== "tool" || s.sim_t0 > T) continue;
    if (s.result && s.result.tuning) tuning = s.result.tuning;
  }
  return tuning || (S.ep.steps.find((s) => s.result && s.result.tuning) || {}).result?.tuning || null;
}

let dashIn = 0, dashOut = 0;
function drawScene(T, dt) {
  const sc = S.scene, g = S.ep.view.geometry, span = g.span_m;
  const { p, t } = phaseAt(T);
  const r = rowAt(p.rows, t);
  const h = r[COL.level], valve = r[COL.valve], out = r[COL.out], outlet = r[COL.outlet], sp = r[COL.sp], meas = r[COL.meas];
  const y = ym(h, span);
  sc.water.setAttribute("y", y);
  sc.water.setAttribute("height", Math.max(0, TANK.bottom - 2 - y));
  sc.surface.setAttribute("y1", y); sc.surface.setAttribute("y2", y);
  sc.levelText.setAttribute("y", Math.min(TANK.bottom - 12, Math.max(TANK.top + 40, y + 24)));
  sc.levelText.textContent = `${h.toFixed(3)} m`;
  const qIn = Math.max(0, valve) / 100, qOut = (outlet / 100) * Math.sqrt(Math.max(0, h) / 1.2) * 1.1;
  sc.stream.setAttribute("height", Math.max(0, y - 57));
  sc.stream.setAttribute("opacity", qIn > 0.005 ? 0.25 + 0.6 * Math.min(1, qIn) : 0);
  sc.stream.setAttribute("width", 2 + 7 * Math.min(1, qIn));
  dashIn -= dt * 60 * qIn; dashOut -= dt * 60 * qOut;
  sc.inFlow.setAttribute("stroke-dashoffset", dashIn);
  sc.inFlow.setAttribute("opacity", qIn > 0.005 ? 0.9 : 0.15);
  sc.outFlow.setAttribute("stroke-dashoffset", dashOut);
  if (isNum(sp)) {
    for (const e of [sc.sp, sc.spLabel, sc.band]) e.removeAttribute("visibility");
    sc.sp.setAttribute("y1", ym(sp, span)); sc.sp.setAttribute("y2", ym(sp, span));
    sc.spLabel.setAttribute("y", ym(sp, span) - 6);
    sc.spLabel.textContent = `SP ${sp.toFixed(2)} m`;
    sc.band.setAttribute("y", ym(sp + g.band_m, span));
    sc.band.setAttribute("height", ym(sp - g.band_m, span) - ym(sp + g.band_m, span));
  } else {
    for (const e of [sc.sp, sc.spLabel, sc.band]) e.setAttribute("visibility", "hidden");
  }
  const alarm = h >= g.hh_m ? "HIGH-HIGH: tank spills" : h <= g.ll_m ? "LOW-LOW: pump runs dry" : "";
  sc.alarm.textContent = alarm;
  sc.shell.setAttribute("stroke", alarm ? "var(--critical)" : "var(--steel)");
  sc.valveText.textContent = `valve ${valve.toFixed(1)} %`;
  sc.valveBar.setAttribute("width", (48 * Math.min(100, Math.max(0, valve))) / 100);
  sc.demandText.textContent = `${outlet.toFixed(0)} %`;
  sc.ltText.textContent = `${meas.toFixed(3)} m`;
  const auto = isNum(sp);
  sc.mode.textContent = auto ? "AUTO" : "MAN";
  sc.modeBg.setAttribute("fill", auto ? "var(--good-wash)" : "var(--warn-wash)");
  sc.mode.setAttribute("fill", auto ? "var(--good-ink)" : "var(--warn-ink)");
  const tu = tuningAt(T) || {};
  const f = sc.fp;
  f.SP.textContent = auto ? `${sp.toFixed(2)} m` : "–";
  f.PV.textContent = `${meas.toFixed(3)} m`;
  f.OUT.textContent = `${out.toFixed(1)} %`;
  f.Kc.textContent = fx(tu.kc, 3);
  f.Ti.textContent = isNum(tu.ti_s) ? (tu.ti_s > 0 ? `${tu.ti_s.toFixed(1)} s` : "off") : "–";
  f.Td.textContent = isNum(tu.td_s) ? `${tu.td_s.toFixed(1)} s` : "–";
  f["PV filter"].textContent = isNum(tu.pv_filter_s) ? `${tu.pv_filter_s.toFixed(1)} s` : "–";
  f["anti-windup"].textContent = tu.anti_windup == null ? "–" : tu.anti_windup ? "on" : "off";
}

// ---------- trend chart ----------

const CH = { W: 1000, L: 54, R: 16, top1: 22, bot1: 262, top2: 300, bot2: 410, H: 440 };

function niceStep(span, target) {
  const steps = [10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
  return steps.find((s) => span / s <= target) || 7200;
}

function buildChart(p) {
  const svg = $("trend");
  svg.setAttribute("viewBox", `0 0 ${CH.W} ${CH.H}`);
  svg.innerHTML = "";
  const g = S.ep.view.geometry;
  const xs = (t) => CH.L + (t / p.dur) * (CH.W - CH.L - CH.R);
  const y1 = (m) => CH.bot1 - (m / g.span_m) * (CH.bot1 - CH.top1);
  const y2 = (pct) => CH.bot2 - (pct / 100) * (CH.bot2 - CH.top2);
  const grid = el("g", {}, svg);
  for (const m of [0, 0.5, 1, 1.5, 2]) {
    el("line", { x1: CH.L, x2: CH.W - CH.R, y1: y1(m), y2: y1(m), stroke: "var(--grid)" }, grid);
    el("text", { x: CH.L - 8, y: y1(m) + 4, "text-anchor": "end" }, grid).textContent = `${m.toFixed(1)} m`;
  }
  for (const pct of [0, 50, 100]) {
    el("line", { x1: CH.L, x2: CH.W - CH.R, y1: y2(pct), y2: y2(pct), stroke: "var(--grid)" }, grid);
    el("text", { x: CH.L - 8, y: y2(pct) + 4, "text-anchor": "end" }, grid).textContent = `${pct} %`;
  }
  const step = niceStep(p.dur, 8);
  for (let t = 0; t <= p.dur + 1e-9; t += step) {
    el("line", { x1: xs(t), x2: xs(t), y1: CH.bot2, y2: CH.bot2 + 5, stroke: "var(--axis)" }, grid);
    el("text", { x: xs(t), y: CH.bot2 + 18, "text-anchor": "middle" }, grid).textContent = clock(t);
  }
  el("text", { x: CH.L, y: 13, class: "title" }, grid).textContent = "Level";
  el("text", { x: CH.L, y: CH.top2 - 9, class: "title" }, grid).textContent = "Controller output and outlet demand";
  for (const m of [g.hh_m, g.ll_m]) {
    el("line", { x1: CH.L, x2: CH.W - CH.R, y1: y1(m), y2: y1(m), stroke: "var(--critical)", "stroke-dasharray": "3 3", "stroke-width": 1.2 }, grid);
  }
  // acceptance: band, events and limits
  const lim = S.ep.view.limits || {};
  if (p.key === "acceptance") {
    el("rect", { x: CH.L, width: CH.W - CH.L - CH.R, y: y1(g.setpoint_m + g.band_m), height: y1(g.setpoint_m - g.band_m) - y1(g.setpoint_m + g.band_m), fill: "var(--band)" }, grid);
    for (const e of p.events) {
      el("line", { x1: xs(e.t_s), x2: xs(e.t_s), y1: CH.top1, y2: CH.bot2, stroke: "var(--ink-2)", "stroke-width": 1 }, grid);
      el("text", { x: xs(e.t_s) + 4, y: CH.top1 + 10 }, grid).textContent = e.label;
    }
    const marks = [];
    if (p.events[0] && isNum(lim.start_s)) marks.push([p.events[0].t_s + lim.start_s, "start limit"]);
    for (const e of p.events.slice(1)) if (isNum(lim.recovery_s)) marks.push([e.t_s + lim.recovery_s, "recovery limit"]);
    for (const [t, label] of marks) {
      if (t > p.dur) continue;
      el("line", { x1: xs(t), x2: xs(t), y1: CH.top1 + 16, y2: CH.bot1, stroke: "var(--critical)", "stroke-dasharray": "2 3", "stroke-width": 1 }, grid);
      el("text", { x: xs(t) + 3, y: CH.bot1 - 6, fill: "var(--critical)" }, grid).textContent = label;
    }
  }
  const clip = el("clipPath", { id: "reveal" }, el("defs", {}, svg));
  const clipRect = el("rect", { x: 0, y: 0, height: CH.H, width: CH.L }, clip);
  const data = el("g", { "clip-path": "url(#reveal)" }, svg);
  const rows = p.rows;
  const stride = Math.max(1, Math.floor(rows.length / 2500));
  const path = (fy, col, from = 0) => {
    let d = "", pen = false;
    for (let i = from; i < rows.length; i += stride) {
      const v = rows[i][col];
      if (!isNum(v)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${xs(rows[i][0]).toFixed(1)} ${fy(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  el("path", { d: path(y1, COL.meas), fill: "none", stroke: "var(--measured)", "stroke-width": 1 }, data);
  el("path", { d: path(y1, COL.sp), fill: "none", stroke: "var(--ink)", "stroke-width": 1.4, "stroke-dasharray": "6 4" }, data);
  el("path", { d: path(y1, COL.level), fill: "none", stroke: "var(--curve)", "stroke-width": 2 }, data);
  el("path", { d: path(y2, COL.outlet), fill: "none", stroke: "var(--demand)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, data);
  el("path", { d: path(y2, COL.out), fill: "none", stroke: "var(--output)", "stroke-width": 1.5 }, data);
  const cursor = el("line", { y1: CH.top1, y2: CH.bot2, stroke: "var(--ink)", "stroke-width": 1, opacity: 0.6 }, svg);
  const hover = el("line", { y1: CH.top1, y2: CH.bot2, stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "2 2", visibility: "hidden" }, svg);
  S.chart = { p, xs, clipRect, cursor, hover };
  $("chart-title").textContent = p.key === "acceptance" ? "Acceptance test" : "Commissioning";
  $("legend").innerHTML = [
    ["var(--curve)", "", "level (true)"], ["var(--measured)", "", "measured"], ["var(--ink)", "dash", "setpoint"],
    ...(p.key === "acceptance" ? [["var(--band)", "box", "±3 cm band"]] : []),
    ["var(--output)", "", "controller output"], ["var(--demand)", "dash", "outlet demand"], ["var(--critical)", "dash", "alarm limits"],
  ].map(([c, k, label]) => `<span><i class="${k}" style="${k === "box" ? "background" : "border-color"}:${c}"></i>${label}</span>`).join("");
  S.chartPhase = p.key;
}

function drawChart(T) {
  const { p, t } = phaseAt(T);
  if (S.chartPhase !== p.key || S.chart.p !== p) buildChart(p);
  const x = S.chart.xs(t);
  S.chart.clipRect.setAttribute("width", x);
  S.chart.cursor.setAttribute("x1", x); S.chart.cursor.setAttribute("x2", x);
}

function chartHover(ev) {
  const svg = $("trend"), box = svg.getBoundingClientRect(), c = S.chart;
  if (!c) return;
  const x = ((ev.clientX - box.left) / box.width) * CH.W;
  const tMax = Math.max(0, S.T - c.p.t0);
  const t = Math.min(tMax, Math.max(0, ((x - CH.L) / (CH.W - CH.L - CH.R)) * c.p.dur));
  if (x < CH.L || x > CH.W - CH.R) return chartLeave();
  const r = rowAt(c.p.rows, t);
  c.hover.setAttribute("x1", c.xs(t)); c.hover.setAttribute("x2", c.xs(t)); c.hover.removeAttribute("visibility");
  const ro = $("readout");
  ro.innerHTML = `t ${clock(t)}<br>level ${fx(r[1], 3)} m · measured ${fx(r[2], 3)} m${isNum(r[6]) ? ` · SP ${fx(r[6], 2)} m` : ""}<br>output ${fx(r[3], 1)} % · demand ${fx(r[5], 0)} %`;
  ro.hidden = false;
  const host = $("trend").parentElement.getBoundingClientRect();
  const left = ev.clientX - host.left + 14;
  ro.style.left = `${Math.min(left, host.width - ro.offsetWidth - 8)}px`;
  ro.style.top = `${ev.clientY - host.top + 14}px`;
}
function chartLeave() {
  $("readout").hidden = true;
  if (S.chart) S.chart.hover.setAttribute("visibility", "hidden");
}

// ---------- log ----------

function argText(args) {
  return Object.entries(args || {}).map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v) : v}`).join(", ");
}

function resultText(s) {
  const r = s.result || {};
  if (s.error) return `<div class="res err">refused: ${esc(s.error)}</div>`;
  const tu = r.tuning || {};
  switch (s.name) {
    case "set_manual": return `<div class="res">manual, valve ${fx(r.output_pct, 1)} %</div>`;
    case "set_auto": return `<div class="res">automatic, setpoint ${fx(r.setpoint_m, 2)} m</div>`;
    case "set_outlet_demand": return `<div class="res">outlet demand ${fx(r.outlet_pct, 0)} %</div>`;
    case "set_tuning":
      return `<div class="res">Kc ${fx(tu.kc, 3)} · Ti ${tu.ti_s > 0 ? fx(tu.ti_s, 1) + " s" : "off"} · Td ${fx(tu.td_s, 1)} s · filter ${fx(tu.pv_filter_s, 1)} s · anti-windup ${tu.anti_windup ? "on" : "off"}</div>`;
    case "wait": {
      const rows = (r.trend && r.trend.rows) || [];
      const lv = rows.map((x) => x[1]);
      const alarms = (r.alarms || []).map((a) => `<div class="res err">${esc(a.alarm)} alarm at ${clock(a.t_s)} (${fx(a.level_m, 3)} m)</div>`).join("");
      return `<div class="res">plant ran ${dur(s.sim_t1 - s.sim_t0)}: level ${fx(lv[0], 3)} → ${fx(lv[lv.length - 1], 3)} m (min ${fx(Math.min(...lv), 3)}, max ${fx(Math.max(...lv), 3)}), output ${fx(r.output_pct, 1)} %, ${rows.length} trend rows returned</div>${alarms}`;
    }
    case "submit": return `<div class="res">handed over for the acceptance test</div>`;
    default: return "";
  }
}

function renderLog() {
  const ep = S.ep, list = $("log");
  const scripted = ep.agent.kind === "scripted";
  list.innerHTML = ep.steps.map((s, i) => {
    if (s.kind === "model") {
      const meta = scripted ? [] : [
        isNum(s.latency_s) ? `${s.latency_s.toFixed(1)} s` : null,
        s.tokens && isNum(s.tokens.completion) ? `${int(s.tokens.completion)} tokens out` : null,
        isNum(s.cost_usd) ? usd(s.cost_usd) : null,
      ].filter(Boolean);
      const think = s.reasoning ? `<details><summary>Thinking${s.tokens && isNum(s.tokens.reasoning) ? ` (${int(s.tokens.reasoning)} tokens)` : ""}</summary><pre>${esc(s.reasoning)}</pre></details>` : "";
      return `<li class="model" data-i="${i}"><div class="who"><b>${scripted ? "Scripted engineer" : "Model"}</b><span>plant ${clock(s.sim_t)}</span>${meta.map((m) => `<span>${esc(m)}</span>`).join("")}</div><div class="text">${esc((s.text || "").trim())}</div>${think}</li>`;
    }
    const data = s.result ? `<details><summary>Data returned to the model</summary><pre>${esc(JSON.stringify(s.result, null, 1))}</pre></details>` : "";
    return `<li class="tool" data-i="${i}"><div class="call"><span class="fn">${esc(s.name)}</span>(${esc(argText(s.args))})</div>${resultText(s)}${s.name === "wait" || s.name === "submit" || s.error ? data : ""}</li>`;
  }).join("");
  list.querySelectorAll("li").forEach((li) => li.addEventListener("click", (ev) => {
    if (ev.target.closest("details")) return;
    seekStep(+li.dataset.i);
  }));
  const marks = $("marks");
  marks.innerHTML = ep.steps.map((s) => {
    const left = (100 * anchorOf(s)) / (S.total || 1);
    return `<span class="${s.kind === "model" ? "model" : s.error ? "err" : ""}" style="left:${left}%"></span>`;
  }).join("") + (S.phases[1] ? `<span class="phase" style="left:${(100 * S.phases[1].t0) / S.total}%"></span>` : "");
  S.logCount = ep.steps.length;
  markLog();
}

function markLog() {
  const list = $("log");
  let current = null;
  list.querySelectorAll("li").forEach((li) => {
    const i = +li.dataset.i;
    li.classList.toggle("current", i === S.cur);
    li.classList.toggle("future", i > S.cur);
    if (i === S.cur) current = li;
  });
  if (current) {
    const top = current.offsetTop, bottom = top + current.offsetHeight;
    if (top < list.scrollTop || bottom > list.scrollTop + list.clientHeight) list.scrollTop = Math.max(0, top - 12);
  }
}

// ---------- header and result ----------

function renderHead() {
  const ep = S.ep, r = ep.result, u = ep.usage || {};
  const name = ep.agent.kind === "scripted" ? `Scripted ${ep.agent.name} agent` : ep.agent.name;
  $("ep-name").textContent = name;
  const live = ep.status === "running";
  const badge = live ? `<span class="badge live">live</span>` : r ? `<span class="badge ${r.passed ? "pass" : "fail"}">${r.passed ? "✓ passed" : "✗ failed"}</span>` : ep.error ? `<span class="badge fail">API error</span>` : "";
  const settings = ep.agent.settings ? Object.entries(ep.agent.settings).filter(([k, v]) => v != null && !(Array.isArray(v) && !v.length) && k !== "model").map(([k, v]) => `${k} ${v}`).join(", ") : "";
  $("ep-sub").innerHTML = `${esc(ep.env.title)} · seed ${ep.env.seed}${settings ? ` · ${esc(settings)}` : ""} ${badge}`;
  const tools = ep.steps.filter((s) => s.kind === "tool").length;
  const stats = [
    ["Score", r ? r.score.toFixed(3) : "–"],
    ["Wall time", dur(ep.wall_s)],
    ["Model time", ep.agent.kind === "scripted" ? "–" : dur(ep.model_s)],
    ["Tool calls", tools],
    ["Tokens", ep.agent.kind === "scripted" ? "–" : int((u.prompt_tokens || 0) + (u.completion_tokens || 0))],
    ["Cost", ep.agent.kind === "scripted" ? "–" : usd(u.cost_usd)],
  ];
  $("ep-stats").innerHTML = stats.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
  $("prompt").textContent = ep.prompt;
}

function renderResult() {
  const ep = S.ep, r = ep.result, box = $("result");
  if (!r && !ep.error) { box.hidden = true; return; }
  box.hidden = false;
  if (ep.error) {
    box.innerHTML = `<h2>The run stopped on an API error</h2><p>${esc(ep.error)}</p><p>API errors are not counted against the model; this run has no score.</p>`;
    return;
  }
  const m = r.metrics, lim = ep.view.limits;
  const yes = (ok) => `<td class="${ok ? "ok" : "no"}">${ok ? "✓ met" : "✗ missed"}</td>`;
  const rec = [m.recovery1_s, m.recovery2_s];
  const recWorst = rec.some((v) => v == null) ? null : Math.max(...rec);
  const rows = [
    ["No alarm during the acceptance test", m.acceptance_alarm ? "alarm" : "none", "none", !m.acceptance_alarm],
    ["Inside 1.20 ± 0.03 m after the switch to automatic", m.start_s == null ? "never" : `${m.start_s} s`, `${lim.start_s} s`, m.start_s != null && m.start_s <= lim.start_s],
    ["Overshoot during the start", `${fx(m.overshoot_m, 3)} m`, `${fx(lim.overshoot_m, 3)} m`, m.overshoot_m <= lim.overshoot_m],
    ["Back inside the band after each demand step (worst)", recWorst == null ? "never" : `${recWorst} s`, `${lim.recovery_s} s`, recWorst != null && recWorst <= lim.recovery_s],
    ["Valve travel", `${int(m.travel_pct)} %`, `${int(lim.travel_pct)} %`, m.travel_pct <= lim.travel_pct],
    ["Alarms while commissioning", String(m.commissioning_alarms), "0", m.commissioning_alarms === 0],
  ];
  const hp = ep.view.hidden_plant || {};
  const ft = m.final_tuning || {}, rt = hp.reference_tuning || {};
  const tuneRow = (label, k, d, unit = "") => `<dt>${label}</dt><dd>${fx(ft[k], d)}${unit} <span style="color:var(--muted)">(SIMC ${fx(rt[k], d)}${unit})</span></dd>`;
  box.innerHTML = `
    <div class="verdict-line"><h2>Acceptance test</h2><span class="score">${r.score.toFixed(3)}</span>
      <span class="badge ${r.passed ? "pass" : "fail"}">${r.passed ? "✓ passed" : "✗ failed"}</span>
      <span style="color:var(--ink-2)">${m.submitted ? "submitted by the agent" : "not submitted: " + esc(ep.stop_reason)}</span></div>
    <div class="scroll-x"><table>
      <thead><tr><th>Requirement</th><th class="num">Measured</th><th class="num">Limit</th><th></th></tr></thead>
      <tbody>${rows.map(([a, b, c, ok]) => `<tr><td>${a}</td><td class="num">${esc(b)}</td><td class="num">${esc(c)}</td>${yes(ok)}</tr>`).join("")}</tbody>
    </table></div>
    <div class="reveal">
      <div><h3>The hidden plant</h3><dl>
        <dt>Time constant</dt><dd>${fx(hp.time_constant_s, 0)} s</dd>
        <dt>Gain</dt><dd>${fx(hp.gain_pct_per_pct, 2)} %/%</dd>
        <dt>Transmitter delay</dt><dd>${fx(hp.transmitter_delay_s, 1)} s</dd>
        <dt>Valve lag</dt><dd>${fx(hp.valve_lag_s, 1)} s</dd>
        <dt>Transmitter noise</dt><dd>${fx(1000 * hp.noise_m, 1)} mm</dd>
        <dt>Tank area</dt><dd>${fx(hp.area_m2, 2)} m²</dd></dl></div>
      <div><h3>Final tuning</h3><dl>
        ${tuneRow("Kc", "kc", 2)}${tuneRow("Ti", "ti_s", 0, " s")}${tuneRow("Td", "td_s", 1, " s")}${tuneRow("PV filter", "pv_filter_s", 1, " s")}
        <dt>Anti-windup</dt><dd>${ft.anti_windup ? "on" : "off"} <span style="color:var(--muted)">(SIMC on)</span></dd></dl></div>
    </div>`;
}

// ---------- playback ----------

function setPlaying(on) {
  S.playing = on;
  $("btn-play").textContent = on ? "Pause" : S.T >= S.total - 1e-6 && S.ep.status !== "running" ? "Replay" : "Play";
  $("btn-play").setAttribute("aria-label", on ? "Pause" : "Play");
}

function holdMs(step) {
  const n = (step.text || "").length;
  return Math.min(5000, 1300 + 14 * n);
}

function seek(T, { resetNext = true } = {}) {
  S.T = Math.max(0, Math.min(S.total, T));
  S.holdUntil = 0;
  if (resetNext) {
    const steps = S.ep.steps;
    let i = 0;
    while (i < steps.length && anchorOf(steps[i]) <= S.T + 1e-6) i++;
    S.next = i;
    S.cur = i - 1;
  }
  S.follow = S.T >= S.total - 1;
  render(0);
}

function seekStep(i) {
  const s = S.ep.steps[i];
  S.T = anchorOf(s);
  S.cur = i;
  S.next = i + 1;
  S.holdUntil = 0;
  S.follow = false;
  render(0);
}

function render(dt) {
  drawScene(S.T, dt);
  drawChart(S.T);
  const { p, t } = phaseAt(S.T);
  $("phase").textContent = p.key === "acceptance" ? "Acceptance test: the tuning the agent left, on a fresh start" : "Commissioning: the agent works on the plant";
  $("clock").textContent = `${p.name.toLowerCase()} ${clock(t)} / ${clock(p.dur)}`;
  $("scrub").value = String(Math.round((1000 * S.T) / (S.total || 1)));
  markLog();
}

function tick(ts) {
  const dt = S.lastTs ? Math.min(0.1, (ts - S.lastTs) / 1000) : 0;
  S.lastTs = ts;
  if (S.playing && ts >= S.holdUntil) {
    const live = S.ep.status === "running";
    let speed = S.speed;
    if (live && S.follow) speed = Math.max(speed, (S.total - S.T) / 3);
    let target = S.T + dt * speed;
    const steps = S.ep.steps;
    while (S.next < steps.length && anchorOf(steps[S.next]) <= target) {
      const s = steps[S.next];
      S.cur = S.next++;
      if (s.kind === "model" && S.hold && (s.text || "").trim()) {
        target = anchorOf(s);
        S.holdUntil = ts + holdMs(s);
        break;
      }
    }
    S.T = Math.min(S.total, target);
    if (S.T >= S.total && !live && S.next >= steps.length) setPlaying(false);
  }
  render(dt);
  requestAnimationFrame(tick);
}

// ---------- loading ----------

function adopt(ep) {
  const first = !S.ep;
  const keepT = S.T;
  if (first) {
    $("load-error").hidden = true;
    $("episode").hidden = false;
  }
  S.ep = ep;
  S.phases = buildPhases(ep);
  const last = S.phases[S.phases.length - 1];
  S.total = last ? last.t0 + last.dur : 0;
  if (first) {
    buildScene();
    const speed = [5, 20, 60, 200].find((s) => S.total / s <= 120) || 200;
    S.speed = speed;
    $("speed").value = String(speed);
  }
  if (first || S.chartPhase) S.chartPhase = null;
  renderHead();
  if (first || S.logCount !== ep.steps.length) renderLog();
  renderResult();
  if (first) {
    seek(ep.status === "running" ? S.total : 0);
    S.follow = ep.status === "running";
    setPlaying(true);
    requestAnimationFrame(tick);
  } else {
    S.T = Math.min(keepT, S.total);
  }
}

async function poll() {
  try {
    const res = await fetch(`${S.liveUrl}${S.liveUrl.includes("?") ? "&" : "?"}t=${Date.now()}`, { cache: "no-store" });
    if (res.ok) adopt(await res.json());
  } catch (e) {
    // the writer may be replacing the file; try again
  }
  if (!S.ep || S.ep.status === "running") setTimeout(poll, 1500);
}

async function showList() {
  $("list").hidden = false;
  let index;
  try {
    const res = await fetch("data/play/index.json", { cache: "no-store" });
    index = res.ok ? await res.json() : { episodes: [] };
  } catch (e) {
    index = { episodes: [] };
  }
  const grid = $("ep-grid");
  if (!index.episodes.length) {
    grid.innerHTML = `<p class="notice">No runs published yet.</p>`;
    return;
  }
  grid.innerHTML = index.episodes.map((e) => {
    const scripted = e.agent.kind === "scripted";
    const verdict = e.passed == null ? "" : `<span class="badge ${e.passed ? "pass" : "fail"}">${e.passed ? "✓ passed" : "✗ failed"} · ${fx(e.score, 3)}</span>`;
    return `<a class="ep-card" href="?ep=${encodeURIComponent(e.id)}">
      <span class="meta">${esc(e.env.title)} · seed ${e.env.seed}</span>
      <span class="who">${scripted ? `Scripted ${esc(e.agent.name)} agent` : esc(e.agent.name)}</span>
      <span>${verdict}</span>
      <span class="nums"><span>${e.tool_calls} tool calls</span><span>wall ${dur(e.wall_s)}</span>${scripted ? "" : `<span>${usd(e.usage && e.usage.cost_usd)}</span>`}</span>
    </a>`;
  }).join("");
}

function wire() {
  $("btn-play").addEventListener("click", () => {
    if (!S.playing && S.T >= S.total - 1e-6 && S.ep.status !== "running") seek(0);
    setPlaying(!S.playing);
  });
  $("btn-next").addEventListener("click", () => { if (S.ep && S.cur + 1 < S.ep.steps.length) seekStep(S.cur + 1); });
  $("btn-prev").addEventListener("click", () => { if (S.ep && S.cur > 0) seekStep(S.cur - 1); });
  $("speed").addEventListener("change", (e) => { S.speed = +e.target.value; });
  $("hold").addEventListener("change", (e) => { S.hold = e.target.checked; S.holdUntil = 0; });
  $("scrub").addEventListener("input", (e) => seek((+e.target.value / 1000) * S.total));
  $("trend").addEventListener("pointermove", chartHover);
  $("trend").addEventListener("pointerleave", chartLeave);
  document.addEventListener("keydown", (e) => {
    if (!S.ep || e.target.closest("input, select, textarea")) return;
    if (e.key === " ") { e.preventDefault(); $("btn-play").click(); }
    if (e.key === "ArrowRight") $("btn-next").click();
    if (e.key === "ArrowLeft") $("btn-prev").click();
  });
}

async function init() {
  const q = new URLSearchParams(location.search);
  const id = q.get("ep"), live = q.get("live");
  if (!id && !live) return showList();
  wire();
  try {
    if (live) {
      // The run may not have written its first record yet: keep polling until it has.
      S.liveUrl = live;
      $("load-error").hidden = false;
      $("load-error").textContent = "Waiting for the run to start…";
      await poll();
    } else {
      const res = await fetch(`data/play/${encodeURIComponent(id)}.json`);
      if (!res.ok) throw new Error(`run ${id} not found`);
      adopt(await res.json());
    }
  } catch (e) {
    $("episode").hidden = true;
    $("load-error").hidden = false;
    $("load-error").textContent = String(e.message || e);
  }
}

init();
