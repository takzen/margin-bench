"use strict";
// margin-bench playground: replays (or follows live) one episode of a model operating a simulated plant.
// The chart, log and result are generic (driven by episode.view); each plant kind has its own scene.

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const isNum = (v) => typeof v === "number" && isFinite(v);
const fx = (v, d) => (isNum(v) ? v.toFixed(d) : "–");
const int = (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "–");
const usd = (v) => (!isNum(v) ? "–" : v === 0 ? "$0" : v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
const COLORS = {
  curve: "var(--curve)", measured: "var(--measured)", ink: "var(--ink)", output: "var(--output)",
  demand: "var(--demand)", critical: "var(--critical)", muted: "var(--muted)", good: "var(--good)",
};

function clock(s) {
  if (!isNum(s)) return "–";
  if (s < 60 && s % 1) return `${s.toFixed(1)} s`;
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
function txt(parent, x, y, text, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = text;
  return t;
}

const S = {
  ep: null, liveUrl: null, phases: [], total: 0, T: 0, playing: false, speed: 20,
  hold: true, holdUntil: 0, next: 0, cur: -1, follow: true, lastTs: 0, chart: null, scene: null, kind: null,
};

// ---------- data ----------

function buildPhases(ep) {
  const phases = [];
  let t0 = 0;
  for (const p of (ep.view && ep.view.phases) || []) {
    if (!p.rows.length) continue;
    const dur = Math.max(1, p.rows[p.rows.length - 1][0]);
    phases.push({ ...p, t0, dur });
    t0 += dur;
  }
  return phases;
}

const anchorOf = (step) => (step.kind === "model" ? step.sim_t : step.sim_t0);

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
  if (!S.ep.view.smooth) return rows[lo];
  const a = rows[lo], b = rows[hi], w = (t - a[0]) / (b[0] - a[0] || 1);
  return a.map((v, i) => (i === 0 ? t : isNum(v) && isNum(b[i]) ? v + w * (b[i] - v) : v));
}

// The newest tool result (or visual) of a given kind at or before the current step.
function latestStep(test) {
  const steps = S.ep.steps;
  for (let i = Math.min(S.cur, steps.length - 1); i >= 0; i--) if (test(steps[i])) return steps[i];
  return null;
}

// ---------- scene: tank ----------

const TANK = { x: 200, w: 160, top: 40, bottom: 360 };

const tankScene = {
  build(svg, scene) {
    const span = scene.span_m;
    const ym = (m) => TANK.bottom - (m / span) * (TANK.bottom - TANK.top);
    const sc = { ym };
    el("path", { d: "M420 216 V384 H60 V330", fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
    el("path", { d: "M60 130 V24 H98", fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
    el("line", { x1: 0, y1: 52, x2: TANK.x + 4, y2: 52, stroke: "var(--pipe)", "stroke-width": 11 }, svg);
    sc.inFlow = el("line", { x1: 0, y1: 52, x2: TANK.x + 4, y2: 52, stroke: "var(--flow)", "stroke-width": 3, "stroke-dasharray": "7 9" }, svg);
    el("line", { x1: TANK.x + TANK.w - 4, y1: 346, x2: 540, y2: 346, stroke: "var(--pipe)", "stroke-width": 11 }, svg);
    sc.outFlow = el("line", { x1: TANK.x + TANK.w - 4, y1: 346, x2: 540, y2: 346, stroke: "var(--flow)", "stroke-width": 3, "stroke-dasharray": "7 9" }, svg);
    el("rect", { x: TANK.x, y: TANK.top, width: TANK.w, height: TANK.bottom - TANK.top, rx: 6, fill: "var(--steel-fill)" }, svg);
    sc.water = el("rect", { x: TANK.x + 2, width: TANK.w - 4, fill: "var(--water)", opacity: 0.85 }, svg);
    sc.surface = el("line", { x1: TANK.x + 2, x2: TANK.x + TANK.w - 2, stroke: "var(--water-top)", "stroke-width": 3 }, svg);
    sc.stream = el("rect", { x: TANK.x + 3, width: 6, y: 57, fill: "var(--water)", opacity: 0.7 }, svg);
    sc.band = el("rect", { x: TANK.x + 2, width: TANK.w - 4, fill: "var(--ink)", opacity: 0.1 }, svg);
    sc.sp = el("line", { x1: TANK.x - 6, x2: TANK.x + TANK.w + 6, stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "6 4" }, svg);
    sc.spLabel = el("text", { x: TANK.x + TANK.w + 10, class: "tag" }, svg);
    for (const [m, label] of [[scene.hh_m, "HH"], [scene.ll_m, "LL"]]) {
      el("line", { x1: TANK.x, x2: TANK.x + TANK.w, y1: ym(m), y2: ym(m), stroke: "var(--critical)", "stroke-width": 1.2, "stroke-dasharray": "3 3" }, svg);
      txt(svg, TANK.x + TANK.w + 10, ym(m) + 4, `${label} ${m.toFixed(2)} m`);
    }
    sc.shell = el("rect", { x: TANK.x, y: TANK.top, width: TANK.w, height: TANK.bottom - TANK.top, rx: 6, fill: "none", stroke: "var(--steel)", "stroke-width": 3 }, svg);
    for (let m = 0; m <= span + 1e-9; m += 0.5) {
      el("line", { x1: TANK.x - 6, x2: TANK.x, y1: ym(m), y2: ym(m), stroke: "var(--steel)" }, svg);
      txt(svg, TANK.x - 9, ym(m) + 4, m.toFixed(1), { "text-anchor": "end" });
    }
    sc.alarm = el("text", { x: TANK.x + TANK.w / 2, y: TANK.top + 22, "text-anchor": "middle", class: "tag", fill: "var(--critical)" }, svg);
    sc.levelText = el("text", { x: TANK.x + TANK.w / 2, "text-anchor": "middle", class: "big" }, svg);
    el("polygon", { points: "96,40 124,64 124,40 96,64", fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    el("line", { x1: 110, y1: 52, x2: 110, y2: 30, stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    el("path", { d: "M98 30 A12 12 0 0 1 122 30 Z", fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    txt(svg, 130, 26, "FV-101", { class: "tag" });
    sc.valveText = el("text", { x: 110, y: 82, "text-anchor": "middle" }, svg);
    sc.valveBar = el("rect", { x: 86, y: 88, height: 4, rx: 2, fill: "var(--output)" }, svg);
    el("rect", { x: 86, y: 88, width: 48, height: 4, rx: 2, fill: "none", stroke: "var(--axis)" }, svg);
    el("polygon", { points: "441,334 469,358 469,334 441,358", fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    el("line", { x1: 455, y1: 346, x2: 455, y2: 324, stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    el("line", { x1: 446, y1: 324, x2: 464, y2: 324, stroke: "var(--ink-2)", "stroke-width": 2 }, svg);
    txt(svg, 455, 316, "demand", { "text-anchor": "middle", class: "tag" });
    sc.demandText = el("text", { x: 455, y: 376, "text-anchor": "middle" }, svg);
    el("line", { x1: TANK.x + TANK.w, y1: 200, x2: 404, y2: 200, stroke: "var(--steel)", "stroke-width": 2 }, svg);
    el("circle", { cx: 420, cy: 200, r: 16, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    txt(svg, 420, 204, "LT", { "text-anchor": "middle", class: "tag" });
    sc.ltText = el("text", { x: 442, y: 204 }, svg);
    el("rect", { x: 10, y: 130, width: 150, height: 200, rx: 8, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.2 }, svg);
    txt(svg, 20, 150, "LIC-101", { class: "tag" });
    sc.modeBg = el("rect", { x: 100, y: 138, width: 52, height: 17, rx: 4 }, svg);
    sc.mode = el("text", { x: 126, y: 151, "text-anchor": "middle", class: "tag" }, svg);
    sc.fp = {};
    ["SP", "PV", "OUT", "Kc", "Ti", "Td", "PV filter", "anti-windup"].forEach((r, i) => {
      txt(svg, 20, 176 + i * 19, r);
      sc.fp[r] = el("text", { x: 150, y: 176 + i * 19, "text-anchor": "end", fill: "var(--ink)" }, svg);
    });
    sc.dashIn = 0; sc.dashOut = 0;
    return sc;
  },
  draw(sc, scene, row, phase, dt) {
    const [, h, meas, out, valve, outlet, sp] = row;
    const y = sc.ym(h);
    sc.water.setAttribute("y", y);
    sc.water.setAttribute("height", Math.max(0, TANK.bottom - 2 - y));
    sc.surface.setAttribute("y1", y); sc.surface.setAttribute("y2", y);
    sc.levelText.setAttribute("y", Math.min(TANK.bottom - 12, Math.max(TANK.top + 40, y + 24)));
    sc.levelText.textContent = `${h.toFixed(3)} m`;
    const qIn = Math.max(0, valve) / 100, qOut = (outlet / 100) * Math.sqrt(Math.max(0, h) / 1.2) * 1.1;
    sc.stream.setAttribute("height", Math.max(0, y - 57));
    sc.stream.setAttribute("opacity", qIn > 0.005 ? 0.25 + 0.6 * Math.min(1, qIn) : 0);
    sc.stream.setAttribute("width", 2 + 7 * Math.min(1, qIn));
    sc.dashIn -= dt * 60 * qIn; sc.dashOut -= dt * 60 * qOut;
    sc.inFlow.setAttribute("stroke-dashoffset", sc.dashIn);
    sc.inFlow.setAttribute("opacity", qIn > 0.005 ? 0.9 : 0.15);
    sc.outFlow.setAttribute("stroke-dashoffset", sc.dashOut);
    const auto = isNum(sp);
    for (const e of [sc.sp, sc.spLabel, sc.band]) e.setAttribute("visibility", auto ? "visible" : "hidden");
    if (auto) {
      sc.sp.setAttribute("y1", sc.ym(sp)); sc.sp.setAttribute("y2", sc.ym(sp));
      sc.spLabel.setAttribute("y", sc.ym(sp) - 6);
      sc.spLabel.textContent = `SP ${sp.toFixed(2)} m`;
      sc.band.setAttribute("y", sc.ym(sp + scene.band_m));
      sc.band.setAttribute("height", sc.ym(sp - scene.band_m) - sc.ym(sp + scene.band_m));
    }
    const alarm = h >= scene.hh_m ? "HIGH-HIGH: tank spills" : h <= scene.ll_m ? "LOW-LOW: pump runs dry" : "";
    sc.alarm.textContent = alarm;
    sc.shell.setAttribute("stroke", alarm ? "var(--critical)" : "var(--steel)");
    sc.valveText.textContent = `valve ${valve.toFixed(1)} %`;
    sc.valveBar.setAttribute("width", (48 * Math.min(100, Math.max(0, valve))) / 100);
    sc.demandText.textContent = `${outlet.toFixed(0)} %`;
    sc.ltText.textContent = `${meas.toFixed(3)} m`;
    sc.mode.textContent = auto ? "AUTO" : "MAN";
    sc.modeBg.setAttribute("fill", auto ? "var(--good-wash)" : "var(--warn-wash)");
    sc.mode.setAttribute("fill", auto ? "var(--good-ink)" : "var(--warn-ink)");
    let tu = scene.final_tuning;
    if (phase.key !== "acceptance") {
      const s = latestStep((st) => st.kind === "tool" && st.result && st.result.tuning);
      tu = s ? s.result.tuning : (S.ep.steps.find((st) => st.result && st.result.tuning) || {}).result?.tuning || {};
    }
    const f = sc.fp;
    f.SP.textContent = auto ? `${sp.toFixed(2)} m` : "–";
    f.PV.textContent = `${meas.toFixed(3)} m`;
    f.OUT.textContent = `${out.toFixed(1)} %`;
    f.Kc.textContent = fx(tu.kc, 3);
    f.Ti.textContent = isNum(tu.ti_s) ? (tu.ti_s > 0 ? `${tu.ti_s.toFixed(1)} s` : "off") : "–";
    f.Td.textContent = isNum(tu.td_s) ? `${tu.td_s.toFixed(1)} s` : "–";
    f["PV filter"].textContent = isNum(tu.pv_filter_s) ? `${tu.pv_filter_s.toFixed(1)} s` : "–";
    f["anti-windup"].textContent = tu.anti_windup == null ? "–" : tu.anti_windup ? "on" : "off";
  },
};

// ---------- scene: weighing scale ----------

const SCOPE = { x: 10, y: 236, w: 255, h: 150 };
const SPEC = { x: 275, y: 236, w: 255, h: 150 };

function blockLabel(b) {
  switch (b.type) {
    case "moving_average": return [`MA`, `n ${b.n}`];
    case "notch": return [`notch`, `${+b.f0_hz.toFixed(2)} Hz Q${+b.q.toFixed(1)}`];
    case "lowpass": return [`${b.kind} ${b.order}`, `${+b.fc_hz.toFixed(2)} Hz`];
    case "fir_lowpass": return [`FIR ${b.taps}`, `${+b.fc_hz.toFixed(2)} Hz`];
    case "decimate": return [`↓ ${b.factor}`, `to ${+(b.rate_in_hz / b.factor).toFixed(1)} Hz`];
    default: return [b.type, ""];
  }
}

function screen(svg, box, title) {
  el("rect", { x: box.x, y: box.y, width: box.w, height: box.h, rx: 6, fill: "var(--screen)" }, svg);
  for (let i = 1; i < 8; i++) el("line", { x1: box.x + (box.w * i) / 8, x2: box.x + (box.w * i) / 8, y1: box.y, y2: box.y + box.h, stroke: "var(--screen-grid)" }, svg);
  for (let i = 1; i < 6; i++) el("line", { x1: box.x, x2: box.x + box.w, y1: box.y + (box.h * i) / 6, y2: box.y + (box.h * i) / 6, stroke: "var(--screen-grid)" }, svg);
  txt(svg, box.x, box.y - 6, title, { class: "tag" });
  const label = txt(svg, box.x + 6, box.y + 14, "", { class: "screen-text" });
  const path = el("path", { fill: "none", stroke: "var(--trace)", "stroke-width": 1.4 }, svg);
  return { label, path };
}

const scaleScene = {
  build(svg, scene) {
    const sc = {};
    // platform scale
    el("rect", { x: 22, y: 128, width: 150, height: 12, rx: 3, fill: "var(--steel)" }, svg);
    el("rect", { x: 86, y: 108, width: 22, height: 20, rx: 2, fill: "var(--steel-fill)", stroke: "var(--steel)", "stroke-width": 1.5 }, svg);
    txt(svg, 97, 122, "LC", { "text-anchor": "middle", class: "tag" });
    el("rect", { x: 16, y: 100, width: 162, height: 8, rx: 2, fill: "var(--steel)" }, svg);
    sc.weight = el("g", {}, svg);
    sc.weightRect = el("rect", { x: 52, width: 90, rx: 4, fill: "var(--weight)", stroke: "var(--ink-2)", "stroke-width": 1 }, sc.weight);
    sc.weightText = el("text", { x: 97, "text-anchor": "middle", class: "tag", fill: "var(--surface)" }, sc.weight);
    txt(svg, 97, 156, `platform 0..${scene.full_scale_kg} kg`, { "text-anchor": "middle" });
    // cable and ADC
    el("path", { d: "M108 118 H190", fill: "none", stroke: "var(--signal)", "stroke-width": 1.5 }, svg);
    el("rect", { x: 190, y: 100, width: 64, height: 36, rx: 5, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.2 }, svg);
    txt(svg, 222, 116, "ADC", { "text-anchor": "middle", class: "tag" });
    txt(svg, 222, 130, `${scene.fs} Hz`, { "text-anchor": "middle" });
    // display
    el("rect", { x: 300, y: 18, width: 228, height: 118, rx: 10, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.2 }, svg);
    el("rect", { x: 312, y: 30, width: 204, height: 62, rx: 6, fill: "var(--lcd)" }, svg);
    sc.reading = el("text", { x: 504, y: 76, "text-anchor": "end", class: "lcd" }, svg);
    txt(svg, 300, 12, "Display, 10 readings/s", { class: "tag" });
    sc.truth = el("text", { x: 516, y: 108, "text-anchor": "end" }, svg);
    sc.err = el("text", { x: 516, y: 126, "text-anchor": "end" }, svg);
    el("path", { d: "M254 118 H270 V60 H300", fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
    // chain row
    txt(svg, 10, 172, "Processing chain", { class: "tag" });
    sc.chain = el("g", {}, svg);
    sc.cost = el("text", { x: 530, y: 172, "text-anchor": "end" }, svg);
    sc.scope = screen(svg, SCOPE, "Oscilloscope");
    sc.spec = screen(svg, SPEC, "Spectrum analyzer");
    sc.chainKey = null;
    sc.lastLoad = null;
    sc.dropAt = 0;
    return sc;
  },
  draw(sc, scene, row, phase) {
    const [, load, reading, err] = row;
    // weight drops in when the load changes
    if (load !== sc.lastLoad) { sc.lastLoad = load; sc.dropAt = performance.now(); }
    const k = Math.min(1, (performance.now() - sc.dropAt) / 350);
    const h = load > 0 ? 10 + (52 * load) / scene.full_scale_kg : 0;
    sc.weight.setAttribute("visibility", load > 0 ? "visible" : "hidden");
    sc.weight.setAttribute("transform", `translate(0 ${-(1 - k) * (1 - k) * 60})`);
    sc.weightRect.setAttribute("y", 100 - h); sc.weightRect.setAttribute("height", h);
    sc.weightText.setAttribute("y", 100 - h / 2 + 4);
    sc.weightText.textContent = h >= 14 ? `${+load.toFixed(2)} kg` : "";
    sc.reading.textContent = `${reading.toFixed(3)}`;
    sc.truth.textContent = `true load ${load.toFixed(3)} kg`;
    const bad = Math.abs(err) > scene.band_kg;
    sc.err.textContent = `error ${err >= 0 ? "+" : ""}${err.toFixed(3)} kg`;
    sc.err.setAttribute("fill", bad ? "var(--critical)" : "var(--good-ink)");
    // chain in force now
    let chain = scene.final_chain, cost = scene.final_cost;
    if (phase.key !== "acceptance") {
      const s = latestStep((st) => st.kind === "tool" && st.name === "set_chain" && !st.error);
      chain = s ? s.result.blocks : [];
      cost = s ? s.result.cost_per_adc_sample : 0;
    }
    const key = JSON.stringify(chain);
    if (key !== sc.chainKey) {
      sc.chainKey = key;
      sc.chain.innerHTML = "";
      const items = [["ADC", `${scene.fs} Hz`], ...chain.map(blockLabel), ["display", "10/s"]];
      const gap = 10, w = Math.min(92, (520 - gap * (items.length - 1)) / items.length);
      items.forEach(([a, b], i) => {
        const x = 10 + i * (w + gap);
        const end = i === 0 || i === items.length - 1;
        el("rect", { x, y: 180, width: w, height: 38, rx: 5, fill: end ? "var(--chip)" : "var(--surface)", stroke: end ? "var(--axis)" : "var(--curve)", "stroke-width": 1.2 }, sc.chain);
        txt(sc.chain, x + w / 2, 196, a, { "text-anchor": "middle", class: "tag" });
        txt(sc.chain, x + w / 2, 210, b, { "text-anchor": "middle" });
        if (i) el("path", { d: `M${x - gap} 199 h${gap - 2}`, stroke: "var(--ink-2)", "stroke-width": 1.2, "marker-end": "" }, sc.chain);
      });
      if (!chain.length) txt(sc.chain, 270, 230, "no chain: the display shows raw ADC samples", { "text-anchor": "middle", fill: "var(--critical)" });
      sc.cost.textContent = `cost ${fx(cost, 2)} / ${scene.budget} per ADC sample`;
    }
    // instruments: the newest capture so far
    const sv = latestStep((st) => st.viz && st.viz.scope);
    if (sv !== sc.lastScope) {
      sc.lastScope = sv;
      if (sv) {
        const ys = sv.viz.scope.y, lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.1 || 0.01;
        sc.scope.path.setAttribute("d", ys.map((v, i) => `${i ? "L" : "M"}${(SCOPE.x + (i / Math.max(1, ys.length - 1)) * SCOPE.w).toFixed(1)} ${(SCOPE.y + SCOPE.h - ((v - lo + pad) / (hi - lo + 2 * pad)) * SCOPE.h).toFixed(1)}`).join(""));
        sc.scope.label.textContent = `${sv.viz.scope.source}, ${(sv.viz.scope.dt_ms * ys.length).toFixed(0)} ms, ${lo.toFixed(2)}..${hi.toFixed(2)} kg`;
      } else {
        sc.scope.path.setAttribute("d", "");
        sc.scope.label.textContent = "no capture yet";
      }
    }
    const pv = latestStep((st) => st.viz && st.viz.spectrum);
    if (pv !== sc.lastSpec) {
      sc.lastSpec = pv;
      if (pv) {
        const { f_hz: f, db, source } = pv.viz.spectrum;
        const fmin = Math.max(f[0], 0.1), fmax = f[f.length - 1], top = Math.max(...db) + 5, bottom = top - 110;
        const X = (v) => SPEC.x + (Math.log(Math.max(v, fmin) / fmin) / Math.log(fmax / fmin)) * SPEC.w;
        const Y = (d) => SPEC.y + SPEC.h * Math.min(1, Math.max(0, (top - d) / (top - bottom)));
        sc.spec.path.setAttribute("d", f.map((v, i) => `${i ? "L" : "M"}${X(v).toFixed(1)} ${Y(db[i]).toFixed(1)}`).join(""));
        sc.spec.label.textContent = `${source}, ${fmin.toFixed(2)}..${fmax.toFixed(0)} Hz (log), ${pv.result.window} x${pv.result.averages}`;
      } else {
        sc.spec.path.setAttribute("d", "");
        sc.spec.label.textContent = "no spectrum yet";
      }
    }
  },
};

// ---------- scene: inline heater with a long pipe ----------

const PIPE_PTS = [[200, 70], [520, 70], [520, 150], [40, 150], [40, 230], [440, 230]];
const PIPE_SEGS = 72;

function pipePoints() {
  // Evenly spaced points along the serpentine, from the heater outlet (0) to the sensor (1).
  const lens = PIPE_PTS.slice(1).map((p, i) => Math.hypot(p[0] - PIPE_PTS[i][0], p[1] - PIPE_PTS[i][1]));
  const total = lens.reduce((a, b) => a + b, 0);
  const pts = [];
  for (let k = 0; k <= PIPE_SEGS; k++) {
    let d = (k / PIPE_SEGS) * total, i = 0;
    while (i < lens.length - 1 && d > lens[i]) d -= lens[i++];
    const [a, b] = [PIPE_PTS[i], PIPE_PTS[i + 1]], w = Math.min(1, d / lens[i]);
    pts.push([a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])]);
  }
  return pts;
}

function tempColor(t, lo, hi) {
  const w = Math.min(1, Math.max(0, (t - lo) / (hi - lo)));
  const c0 = [63, 143, 220], c1 = [226, 85, 63];
  return `rgb(${c0.map((v, i) => Math.round(v + w * (c1[i] - v))).join(",")})`;
}

// Heater outlet temperature that sits at fraction x along the pipe at frame time t.
function pipeTemp(rows, t, x, volume) {
  const now = rowAt(rows, t), target = now[8] - x * volume;
  if (target <= rows[0][8]) return rows[0][7];
  let lo = 0, hi = rows.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (rows[mid][8] <= target) lo = mid; else hi = mid;
  }
  const a = rows[lo], b = rows[hi], w = (target - a[8]) / (b[8] - a[8] || 1);
  return a[7] + Math.min(1, Math.max(0, w)) * (b[7] - a[7]);
}

const heaterScene = {
  build(svg, scene) {
    const sc = { pts: pipePoints() };
    const d = PIPE_PTS.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join("");
    el("path", { d: "M0 70 H120", stroke: "var(--pipe)", "stroke-width": 13, fill: "none" }, svg);
    sc.inlet = el("path", { d: "M0 70 H120", stroke: "rgb(63,143,220)", "stroke-width": 7, fill: "none" }, svg);
    el("path", { d, stroke: "var(--pipe)", "stroke-width": 13, fill: "none", "stroke-linejoin": "round" }, svg);
    sc.segs = [];
    for (let k = 0; k < PIPE_SEGS; k++) {
      const [a, b] = [sc.pts[k], sc.pts[k + 1]];
      sc.segs.push(el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], "stroke-width": 7, "stroke-linecap": "round" }, svg));
    }
    sc.flowDash = el("path", { d, stroke: "rgba(255,255,255,0.55)", "stroke-width": 2, fill: "none", "stroke-dasharray": "4 14", "stroke-linejoin": "round" }, svg);
    el("path", { d: "M440 230 H540", stroke: "var(--pipe)", "stroke-width": 13, fill: "none" }, svg);
    sc.outlet = el("path", { d: "M452 230 H540", stroke: "rgb(63,143,220)", "stroke-width": 7, fill: "none" }, svg);
    txt(svg, 540, 254, "to process", { "text-anchor": "end" });
    // heater
    sc.heaterBox = el("rect", { x: 120, y: 44, width: 80, height: 52, rx: 6, stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    el("path", { d: "M128 70 l6 -12 l8 24 l8 -24 l8 24 l8 -24 l8 24 l8 -24 l6 12", stroke: "var(--ink)", "stroke-width": 1.6, fill: "none", opacity: 0.7 }, svg);
    txt(svg, 160, 36, "EH-101", { "text-anchor": "middle", class: "tag" });
    sc.powerText = el("text", { x: 160, y: 114, "text-anchor": "middle" }, svg);
    txt(svg, 4, 52, "inlet", { class: "tag" });
    sc.inText = el("text", { x: 4, y: 96 }, svg);
    sc.flowText = el("text", { x: 4, y: 112 }, svg);
    // transmitter
    el("line", { x1: 470, y1: 230, x2: 470, y2: 262, stroke: "var(--steel)", "stroke-width": 2 }, svg);
    el("circle", { cx: 470, cy: 278, r: 16, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    txt(svg, 470, 282, "TT", { "text-anchor": "middle", class: "tag" });
    sc.ttText = el("text", { x: 448, y: 276, "text-anchor": "end", fill: "var(--ink)" }, svg);
    sc.trueText = el("text", { x: 448, y: 292, "text-anchor": "end" }, svg);
    // controller
    el("path", { d: "M470 294 V392 H210 V350", fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
    el("path", { d: "M106 270 V84 H120", fill: "none", stroke: "var(--signal)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
    el("rect", { x: 10, y: 270, width: 200, height: 124, rx: 8, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 1.2 }, svg);
    txt(svg, 20, 288, "TIC-101", { class: "tag" });
    sc.modeBg = el("rect", { x: 150, y: 277, width: 52, height: 16, rx: 4 }, svg);
    sc.mode = el("text", { x: 176, y: 289, "text-anchor": "middle", class: "tag" }, svg);
    sc.fp = {};
    [["SP", 20, 306], ["PV", 20, 322], ["OUT", 20, 338], ["Kc", 112, 306], ["Ti", 112, 322], ["Smith", 112, 338], ["flow FF", 20, 360], ["AW", 112, 360]].forEach(([k, x, y]) => {
      txt(svg, x, y, k);
      sc.fp[k] = el("text", { x: x + 88, y, "text-anchor": "end", fill: "var(--ink)" }, svg);
    });
    // colour key
    const lo = scene.t_cold, hi = scene.t_hot;
    for (let i = 0; i < 20; i++) el("rect", { x: 250 + i * 9, y: 352, width: 9, height: 8, fill: tempColor(lo + ((hi - lo) * i) / 19, lo, hi) }, svg);
    txt(svg, 250, 376, `${lo.toFixed(0)} °C`);
    txt(svg, 430, 376, `${hi.toFixed(0)} °C`, { "text-anchor": "end" });
    txt(svg, 250, 344, "fluid temperature along the pipe", { class: "tag" });
    sc.dash = 0;
    return sc;
  },
  draw(sc, scene, row, phase, dt) {
    const [t, outT, meas, power, sp, flow, tin] = row;
    const lo = scene.t_cold, hi = scene.t_hot, rows = phase.rows;
    for (let k = 0; k < PIPE_SEGS; k++) {
      const temp = pipeTemp(rows, t, (k + 0.5) / PIPE_SEGS, scene.pipe_volume);
      sc.segs[k].setAttribute("stroke", tempColor(temp, lo, hi));
    }
    sc.inlet.setAttribute("stroke", tempColor(tin, lo, hi));
    sc.outlet.setAttribute("stroke", tempColor(outT, lo, hi));
    sc.dash -= dt * 40 * (flow / 100);
    sc.flowDash.setAttribute("stroke-dashoffset", sc.dash);
    sc.heaterBox.setAttribute("fill", `rgba(240,120,40,${(0.1 + 0.75 * Math.min(1, power / 100)).toFixed(2)})`);
    sc.powerText.textContent = `power ${power.toFixed(1)} %`;
    sc.inText.textContent = `${tin.toFixed(1)} °C`;
    sc.flowText.textContent = `flow ${flow.toFixed(0)} %`;
    sc.ttText.textContent = `TT ${meas.toFixed(2)} °C`;
    sc.trueText.textContent = `true ${outT.toFixed(2)} °C`;
    const auto = isNum(sp);
    sc.mode.textContent = auto ? "AUTO" : "MAN";
    sc.modeBg.setAttribute("fill", auto ? "var(--good-wash)" : "var(--warn-wash)");
    sc.mode.setAttribute("fill", auto ? "var(--good-ink)" : "var(--warn-ink)");
    let tu = scene.final_tuning;
    if (phase.key !== "acceptance") {
      const s = latestStep((st) => st.kind === "tool" && st.result && st.result.tuning);
      tu = s ? s.result.tuning : (S.ep.steps.find((st) => st.result && st.result.tuning) || {}).result?.tuning || {};
    }
    const f = sc.fp;
    f.SP.textContent = auto ? `${sp.toFixed(1)} °C` : "–";
    f.PV.textContent = `${meas.toFixed(2)} °C`;
    f.OUT.textContent = `${power.toFixed(1)} %`;
    f.Kc.textContent = fx(tu.kc, 3);
    f.Ti.textContent = isNum(tu.ti_s) ? (tu.ti_s > 0 ? `${tu.ti_s.toFixed(0)} s` : "off") : "–";
    f.Smith.textContent = tu.smith_gain > 0 ? `${tu.smith_delay_s.toFixed(0)} s` : "off";
    f["flow FF"].textContent = tu.flow_feedforward == null ? "–" : tu.flow_feedforward ? "on" : "off";
    f.AW.textContent = tu.anti_windup == null ? "–" : tu.anti_windup ? "on" : "off";
  },
};

// ---------- scene: DC motor drive ----------

function bar(svg, x, y, w, label) {
  el("rect", { x, y, width: w, height: 14, rx: 3, fill: "var(--chip)" }, svg);
  const fill = el("rect", { x, y, height: 14, rx: 3, fill: "var(--curve)" }, svg);
  txt(svg, x, y - 5, label, { class: "tag" });
  const value = el("text", { x: x + w, y: y + 27, "text-anchor": "end", fill: "var(--ink)" }, svg);
  return { x, y, w, fill, value };
}

function block(svg, x, y, w, a, b) {
  const r = el("rect", { x, y, width: w, height: 34, rx: 5, fill: "var(--surface)", stroke: "var(--curve)", "stroke-width": 1.2 }, svg);
  txt(svg, x + w / 2, y + 14, a, { "text-anchor": "middle", class: "tag" });
  const v = txt(svg, x + w / 2, y + 27, b, { "text-anchor": "middle" });
  return { r, v };
}

const motorScene = {
  build(svg, scene) {
    const sc = { angle: 0 };
    // motor, shaft, flywheel, dynamometer
    el("rect", { x: 40, y: 58, width: 170, height: 104, rx: 16, fill: "var(--steel-fill)", stroke: "var(--steel)", "stroke-width": 3 }, svg);
    for (let i = 0; i < 6; i++) el("line", { x1: 60 + i * 26, x2: 60 + i * 26, y1: 66, y2: 154, stroke: "var(--steel)", opacity: 0.5 }, svg);
    el("rect", { x: 100, y: 38, width: 50, height: 20, rx: 3, fill: "var(--steel-fill)", stroke: "var(--steel)", "stroke-width": 2 }, svg);
    txt(svg, 125, 116, "M", { "text-anchor": "middle", class: "big" });
    el("rect", { x: 210, y: 104, width: 120, height: 12, fill: "var(--steel)" }, svg);
    sc.wheel = el("g", { transform: "translate(262 110)" }, svg);
    el("circle", { r: 42, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 3 }, sc.wheel);
    for (let k = 0; k < 4; k++) el("line", { x1: 0, y1: 0, x2: 40 * Math.cos((k * Math.PI) / 2), y2: 40 * Math.sin((k * Math.PI) / 2), stroke: k ? "var(--ink-2)" : "var(--output)", "stroke-width": k ? 3 : 5, "stroke-linecap": "round" }, sc.wheel);
    el("circle", { r: 6, fill: "var(--ink-2)" }, sc.wheel);
    sc.lock = txt(svg, 262, 170, "", { "text-anchor": "middle", class: "tag", fill: "var(--warn-ink)" });
    txt(svg, 262, 186, "rotation shown 100x slower", { "text-anchor": "middle" });
    sc.dyno = el("rect", { x: 330, y: 70, width: 110, height: 80, rx: 8, stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
    txt(svg, 385, 98, "dynamometer", { "text-anchor": "middle", class: "tag" });
    sc.loadText = el("text", { x: 385, y: 120, "text-anchor": "middle", class: "big" }, svg);
    sc.mode = el("text", { x: 125, y: 30, "text-anchor": "middle", class: "tag" }, svg);
    sc.trip = el("text", { x: 125, y: 186, "text-anchor": "middle", class: "tag", fill: "var(--critical)" }, svg);
    // gauges
    sc.speed = bar(svg, 456, 58, 80, "speed");
    sc.speedRef = el("line", { y1: 54, y2: 76, stroke: "var(--ink)", "stroke-width": 2 }, svg);
    sc.cur = bar(svg, 456, 110, 80, "current");
    sc.volt = bar(svg, 456, 162, 80, "voltage");
    // cascade
    txt(svg, 10, 222, "Cascade", { class: "tag" });
    const y = 234;
    sc.bRef = block(svg, 10, y, 74, "speed ref", "");
    sc.bSpd = block(svg, 104, y, 84, "speed PI", "");
    sc.bLim = block(svg, 208, y, 80, "limit", `±${scene.i_max} A`);
    sc.bCur = block(svg, 308, y, 84, "current PI", "");
    sc.bBr = block(svg, 412, y, 64, "bridge", "");
    txt(svg, 496, y + 22, "M", { class: "tag" });
    for (const x of [84, 188, 288, 392, 476]) el("path", { d: `M${x} ${y + 17} h18`, stroke: "var(--ink-2)", "stroke-width": 1.4 }, svg);
    el("path", { d: `M500 ${y + 34} V${y + 72} H146 V${y + 34}`, fill: "none", stroke: "var(--signal)", "stroke-dasharray": "5 4" }, svg);
    txt(svg, 320, y + 66, "encoder speed", { "text-anchor": "middle" });
    el("path", { d: `M444 ${y + 34} V${y + 54} H350 V${y + 34}`, fill: "none", stroke: "var(--signal)", "stroke-dasharray": "3 3" }, svg);
    txt(svg, 397, y + 50, "current", { "text-anchor": "middle" });
    // gains
    sc.gains = txt(svg, 10, 336, "", {});
    sc.gains2 = txt(svg, 10, 354, "", {});
    return sc;
  },
  draw(sc, scene, row, phase, dt) {
    const [t, speed, meas, cur, iref, volt, load, sref] = row;
    sc.angle = (sc.angle + dt * speed * 0.06) % 360;
    sc.wheel.setAttribute("transform", `translate(262 110) rotate(${sc.angle.toFixed(1)})`);
    sc.loadText.textContent = `${load.toFixed(3)} Nm`;
    sc.dyno.setAttribute("fill", load > 0 ? `rgba(240,120,40,${(0.15 + 0.6 * Math.min(1, load / scene.load_max)).toFixed(2)})` : "var(--surface)");
    const top = scene.no_load_rpm * 1.1;
    const setBar = (b, v, lo, hi, text, warn) => {
      const z = b.x + (b.w * (0 - lo)) / (hi - lo), x = b.x + (b.w * (Math.min(hi, Math.max(lo, v)) - lo)) / (hi - lo);
      b.fill.setAttribute("x", Math.min(z, x)); b.fill.setAttribute("width", Math.abs(x - z));
      b.fill.setAttribute("fill", warn ? "var(--critical)" : "var(--curve)");
      b.value.textContent = text;
    };
    setBar(sc.speed, speed, 0, top, `${speed.toFixed(0)} rpm`, false);
    if (isNum(sref)) {
      const x = sc.speed.x + (sc.speed.w * Math.min(top, sref)) / top;
      sc.speedRef.setAttribute("x1", x); sc.speedRef.setAttribute("x2", x); sc.speedRef.setAttribute("visibility", "visible");
    } else sc.speedRef.setAttribute("visibility", "hidden");
    setBar(sc.cur, cur, -scene.trip_a, scene.trip_a, `${cur.toFixed(2)} A`, Math.abs(cur) > scene.i_max * 1.02);
    setBar(sc.volt, volt, -scene.v_supply, scene.v_supply, `${volt.toFixed(1)} V`, false);
    // mode and trip, from the log
    let mode = "speed", tripped = false;
    if (phase.key !== "acceptance") {
      const s = latestStep((st) => st.kind === "tool" && st.result && st.result.mode);
      mode = s ? s.result.mode : "off";
      for (const st of S.ep.steps) for (const a of (st.result && st.result.alarms) || []) if (Math.abs(a.t_ms / 1000 - t) < 0.15) tripped = true;
      const lk = latestStep((st) => st.kind === "tool" && st.name === "lock_shaft");
      sc.lock.textContent = lk && lk.result.shaft_locked ? "SHAFT LOCKED" : "";
    } else sc.lock.textContent = "";
    sc.mode.textContent = `mode: ${mode}`;
    sc.trip.textContent = tripped ? "OVERCURRENT TRIP" : "";
    const sat = isNum(iref) && Math.abs(iref) >= scene.i_max * 0.999;
    sc.bLim.r.setAttribute("stroke", sat ? "var(--critical)" : "var(--curve)");
    sc.bLim.v.textContent = sat ? "at the limit" : `±${scene.i_max} A`;
    sc.bRef.v.textContent = isNum(sref) ? `${sref.toFixed(0)} rpm` : "–";
    sc.bSpd.v.textContent = isNum(iref) && mode === "speed" ? `${iref.toFixed(2)} A` : "–";
    sc.bCur.v.textContent = isNum(iref) ? `ref ${iref.toFixed(2)} A` : "–";
    sc.bBr.v.textContent = `${volt.toFixed(1)} V`;
    let g = scene.final_gains;
    if (phase.key !== "acceptance") {
      const s = latestStep((st) => st.kind === "tool" && st.result && st.result.gains);
      g = s ? s.result.gains : null;
    }
    if (g) {
      sc.gains.textContent = `current PI: Kp ${(+g.current_kp).toPrecision(3)} V/A, Ki ${(+g.current_ki).toPrecision(3)} V/(A s)`;
      sc.gains2.textContent = `speed PI: Kp ${(+g.speed_kp).toPrecision(3)} A/rpm, Ki ${(+g.speed_ki).toPrecision(3)} A/(rpm s), anti-windup ${g.anti_windup ? "on" : "off"}, filter ${(+g.filter_ms).toFixed(2)} ms`;
    } else {
      sc.gains.textContent = "factory gains";
      sc.gains2.textContent = "";
    }
  },
};

// ---------- scene: inverted pendulum on a cart ----------

const pendulumScene = {
  build(svg, scene) {
    const sc = {};
    sc.px = (x) => 270 + (x / scene.track_m) * 240;
    el("line", { x1: 20, x2: 520, y1: 318, y2: 318, stroke: "var(--steel)", "stroke-width": 4 }, svg);
    for (const s of [-1, 1]) {
      el("rect", { x: s < 0 ? 14 : 516, y: 288, width: 10, height: 34, fill: "var(--critical)", opacity: 0.8 }, svg);
    }
    for (let x = -0.4; x <= 0.401; x += 0.1) {
      el("line", { x1: sc.px(x), x2: sc.px(x), y1: 322, y2: 330, stroke: "var(--axis)" }, svg);
      txt(svg, sc.px(x), 344, `${x.toFixed(1)}`, { "text-anchor": "middle" });
    }
    txt(svg, 520, 362, "m", { "text-anchor": "end" });
    sc.refMark = el("polygon", { points: "0,0 -7,12 7,12", fill: "var(--ink)" }, svg);
    sc.cart = el("g", {}, svg);
    el("rect", { x: -40, y: -34, width: 80, height: 26, rx: 5, fill: "var(--steel-fill)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, sc.cart);
    for (const wx of [-24, 24]) el("circle", { cx: wx, cy: -4, r: 6, fill: "var(--ink-2)" }, sc.cart);
    sc.force = el("line", { y1: -21, y2: -21, stroke: "var(--output)", "stroke-width": 4, "stroke-linecap": "round" }, sc.cart);
    sc.forceHead = el("polygon", { fill: "var(--output)" }, sc.cart);
    sc.rodLen = Math.min(240, scene.rod_m * 300);
    sc.rod = el("line", { stroke: "var(--curve)", "stroke-width": 7, "stroke-linecap": "round" }, svg);
    sc.pivot = el("circle", { r: 5, fill: "var(--ink)" }, svg);
    sc.angle = txt(svg, 20, 30, "", { class: "big" });
    sc.pos = txt(svg, 20, 50, "", {});
    sc.fText = txt(svg, 20, 68, "", {});
    sc.status = txt(svg, 520, 30, "", { "text-anchor": "end", class: "tag" });
    sc.gains = txt(svg, 20, 384, "", {});
    return sc;
  },
  draw(sc, scene, row, phase) {
    const [, x, th, f, ref] = row;
    const cx = sc.px(x), py = 284;
    sc.cart.setAttribute("transform", `translate(${cx.toFixed(1)} 318)`);
    const fl = (60 * f) / scene.f_max;
    sc.force.setAttribute("x1", 0); sc.force.setAttribute("x2", fl.toFixed(1));
    const s = Math.sign(fl) || 1;
    sc.forceHead.setAttribute("points", Math.abs(fl) > 4 ? `${fl + s * 8},-21 ${fl},-27 ${fl},-15` : "");
    const a = (th * Math.PI) / 180;
    const tx = cx + sc.rodLen * Math.sin(a), ty = py - sc.rodLen * Math.cos(a);
    sc.rod.setAttribute("x1", cx); sc.rod.setAttribute("y1", py); sc.rod.setAttribute("x2", tx.toFixed(1)); sc.rod.setAttribute("y2", ty.toFixed(1));
    sc.pivot.setAttribute("cx", cx); sc.pivot.setAttribute("cy", py);
    const fallen = Math.abs(th) > scene.fall_deg;
    sc.rod.setAttribute("stroke", fallen ? "var(--critical)" : "var(--curve)");
    sc.angle.textContent = `θ ${th >= 0 ? "+" : ""}${th.toFixed(2)}°`;
    sc.pos.textContent = `cart ${x >= 0 ? "+" : ""}${x.toFixed(3)} m`;
    sc.fText.textContent = `force ${f.toFixed(2)} N`;
    if (isNum(ref)) {
      sc.refMark.setAttribute("transform", `translate(${sc.px(ref).toFixed(1)} 324)`);
      sc.refMark.setAttribute("visibility", "visible");
    } else sc.refMark.setAttribute("visibility", "hidden");
    const atStop = Math.abs(x) >= scene.track_m - 1e-4;
    sc.status.textContent = atStop ? "END STOP" : fallen ? (Math.abs(th) > 170 ? "rod hanging" : "ROD FALLING") : isNum(ref) ? "controller on" : "controller off";
    sc.status.setAttribute("fill", atStop || (fallen && Math.abs(th) <= 170) ? "var(--critical)" : "var(--ink-2)");
    let g = scene.final_gains;
    if (phase.key !== "acceptance") {
      const st = latestStep((s2) => s2.kind === "tool" && s2.result && s2.result.gains);
      g = st ? st.result.gains : null;
    }
    sc.gains.textContent = g ? `K = [${[g.k_x, g.k_v, g.k_theta, g.k_omega].map((v) => (+v).toPrecision(4)).join(", ")}], filter ${(+g.filter_ms).toFixed(1)} ms` : "";
  },
};

// ---------- scene: amplifier with a fault ----------

const circuitScene = {
  build(svg, scene) {
    const sc = { parts: {}, nodes: {} };
    const wire = (d) => el("path", { d, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.6 }, svg);
    const res = (name, x, y, vertical) => {
      const g = el("g", {}, svg);
      const r = vertical ? el("rect", { x: x - 8, y: y - 24, width: 16, height: 48, rx: 2, fill: "var(--surface)", "stroke-width": 1.8 }, g)
        : el("rect", { x: x - 24, y: y - 8, width: 48, height: 16, rx: 2, fill: "var(--surface)", "stroke-width": 1.8 }, g);
      const lab = txt(svg, vertical ? x + 14 : x, vertical ? y - 4 : y - 14, name, { class: "tag", "text-anchor": vertical ? "start" : "middle" });
      txt(svg, vertical ? x + 14 : x, vertical ? y + 10 : y + 26, scene.nominal[name], { "text-anchor": vertical ? "start" : "middle" });
      const note = txt(svg, vertical ? x + 14 : x, vertical ? y + 24 : y + 40, "", { "text-anchor": vertical ? "start" : "middle" });
      sc.parts[name] = { shape: r, lab, note };
    };
    const cap = (name, x, y, vertical) => {
      const g = el("g", {}, svg);
      const plates = vertical
        ? [el("line", { x1: x - 12, x2: x + 12, y1: y - 4, y2: y - 4, "stroke-width": 3 }, g), el("line", { x1: x - 12, x2: x + 12, y1: y + 4, y2: y + 4, "stroke-width": 3 }, g)]
        : [el("line", { x1: x - 4, x2: x - 4, y1: y - 12, y2: y + 12, "stroke-width": 3 }, g), el("line", { x1: x + 4, x2: x + 4, y1: y - 12, y2: y + 12, "stroke-width": 3 }, g)];
      const lab = txt(svg, vertical ? x + 16 : x, vertical ? y - 2 : y - 18, name, { class: "tag", "text-anchor": vertical ? "start" : "middle" });
      txt(svg, vertical ? x + 16 : x, vertical ? y + 12 : y + 28, scene.nominal[name], { "text-anchor": vertical ? "start" : "middle" });
      const note = txt(svg, vertical ? x + 16 : x, vertical ? y + 26 : y + 42, "", { "text-anchor": vertical ? "start" : "middle" });
      sc.parts[name] = { shape: plates, lab, note };
    };
    // rails
    wire("M60 40 H460"); wire("M30 360 H460");
    txt(svg, 60, 30, `VCC ${scene.vcc} V`, { class: "tag" });
    for (let x = 30; x <= 460; x += 430) el("path", { d: `M${x} 360 v6 m-8 0 h16 m-12 4 h8`, stroke: "var(--ink-2)", fill: "none" }, svg);
    // bias divider and base
    wire("M150 40 V86"); res("R1", 150, 110, true); wire("M150 134 V226"); res("R2", 150, 250, true); wire("M150 274 V360");
    wire("M150 180 H232");
    wire("M30 180 H83"); cap("Cin", 87, 180, false); wire("M91 180 H150");
    el("circle", { cx: 30, cy: 214, r: 14, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.6 }, svg);
    el("path", { d: "M22 214 q4 -8 8 0 t8 0", fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.4 }, svg);
    wire("M30 180 V200"); wire("M30 228 V360");
    // transistor
    sc.parts.Q1 = { shape: [el("line", { x1: 236, x2: 236, y1: 158, y2: 202, "stroke-width": 4 }, svg)], lab: txt(svg, 196, 152, "Q1", { class: "tag" }), note: txt(svg, 300, 206, "", {}) };
    sc.parts.Q1.shape.push(el("path", { d: "M236 170 L275 148 V120", fill: "none", "stroke-width": 1.8 }, svg));
    sc.parts.Q1.shape.push(el("path", { d: "M236 190 L275 212 V250", fill: "none", "stroke-width": 1.8 }, svg));
    el("polygon", { points: "268,203 275,212 263,212", fill: "var(--ink-2)" }, svg);
    // collector and emitter
    wire("M275 40 V62"); res("RC", 275, 86, true); wire("M275 110 V120");
    res("RE", 275, 290, true); wire("M275 314 V360");
    wire("M275 250 V266"); wire("M275 250 H360 V276"); cap("CE", 360, 280, true); wire("M360 284 V360");
    // output
    wire("M275 120 H340"); cap("Cout", 344, 120, false); wire("M348 120 H440 V226"); res("RL", 440, 250, true); wire("M440 274 V360");
    // node badges
    const badge = (node, x, y) => {
      el("circle", { cx: x, cy: y, r: 3.5, fill: "var(--ink)" }, svg);
      sc.nodes[node] = txt(svg, x + 6, y - 6, "", { class: "tag", fill: "var(--curve)" });
    };
    badge("vcc", 460, 40); badge("b", 150, 180); badge("c", 275, 120); badge("e", 275, 250); badge("out", 440, 120); badge("in", 30, 180);
    sc.gain = txt(svg, 330, 30, "", { class: "tag" });
    sc.verdict = txt(svg, 480, 390, "", { class: "tag", "text-anchor": "end" });
    return sc;
  },
  draw(sc, scene, row, phase) {
    // latest measurements up to the current step
    const meas = {}, tests = {}, replaced = new Set();
    let gain = null;
    const upto = Math.min(S.cur, S.ep.steps.length - 1);
    for (let i = 0; i <= upto; i++) {
      const s = S.ep.steps[i];
      if (s.kind !== "tool" || s.error) continue;
      if (s.name === "measure_dc") meas[s.result.node] = s.result.volts;
      if (s.name === "measure_gain") gain = s.result;
      if (s.name === "test_part") tests[s.result.part] = s.summary.split(": ").slice(1).join(": ") || s.summary;
      if (s.name === "replace_part") replaced.add(s.result.part);
    }
    for (const [node, t] of Object.entries(sc.nodes)) t.textContent = node in meas ? `${node} ${meas[node].toFixed(3)} V` : "";
    sc.gain.textContent = gain ? `gain at ${gain.frequency_hz} Hz: ${gain.gain.toFixed(2)}` : "";
    const fault = phase.key === "acceptance" && scene.fault ? scene.fault[0] : null;
    for (const [name, p] of Object.entries(sc.parts)) {
      const color = name === fault ? "var(--critical)" : replaced.has(name) ? "var(--good)" : name in tests ? "var(--output)" : "var(--ink-2)";
      for (const sh of [].concat(p.shape)) sh.setAttribute("stroke", color);
      p.lab.setAttribute("fill", color === "var(--ink-2)" ? "var(--ink)" : color);
      p.note.textContent = name === fault ? `fault: ${scene.fault[1].replace("_", " ")}${replaced.has(name) ? ", replaced" : ""}`
        : replaced.has(name) ? "replaced" : name in tests ? tests[name] : "";
    }
    sc.verdict.textContent = phase.key === "acceptance" ? (S.ep.result && S.ep.result.passed ? "board repaired" : "board not repaired") : "";
    sc.verdict.setAttribute("fill", S.ep.result && S.ep.result.passed ? "var(--good-ink)" : "var(--critical)");
  },
};

const SCENES = { tank: tankScene, scale: scaleScene, heater: heaterScene, motor: motorScene, pendulum: pendulumScene, circuit: circuitScene };

function buildScene() {
  const svg = $("scene");
  svg.innerHTML = "";
  S.kind = S.ep.view.kind;
  S.scene = SCENES[S.kind].build(svg, S.ep.view.scene);
}

function drawScene(T, dt) {
  const { p, t } = phaseAt(T);
  SCENES[S.kind].draw(S.scene, S.ep.view.scene, rowAt(p.rows, t), p, dt);
}

// ---------- trend chart (generic, from view.panels) ----------

const CH = { W: 1000, L: 74, R: 16, top: 22, gap: 40, bottom: 30 };

function niceStep(span, target) {
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
  return steps.find((s) => span / s <= target) || 7200;
}
const tickLabel = (v) => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(v) : String(+v.toFixed(3)));

function buildChart(p) {
  const svg = $("trend"), v = S.ep.view, panels = v.panels;
  const units = panels.reduce((a, q) => a + (q.height || 1), 0);
  const plotH = 110 * units;
  const H = CH.top + plotH + CH.gap * (panels.length - 1) + CH.bottom;
  svg.setAttribute("viewBox", `0 0 ${CH.W} ${H}`);
  svg.innerHTML = "";
  const xs = (t) => CH.L + (t / p.dur) * (CH.W - CH.L - CH.R);
  const grid = el("g", {}, svg);
  const clip = el("clipPath", { id: "reveal" }, el("defs", {}, svg));
  const clipRect = el("rect", { x: 0, y: 0, height: H, width: CH.L }, clip);
  const data = el("g", { "clip-path": "url(#reveal)" }, svg);
  const rows = p.rows, stride = Math.max(1, Math.floor(rows.length / 2500));
  let y0 = CH.top;
  const boxes = [];
  for (const q of panels) {
    const h = 110 * (q.height || 1), [lo, hi] = q.range;
    const Y = (val) => y0 + h - ((Math.min(hi, Math.max(lo, val)) - lo) / (hi - lo)) * h;
    txt(grid, CH.L, y0 - 8, q.title, { class: "title" });
    for (const tk of q.ticks || [lo, hi]) {
      el("line", { x1: CH.L, x2: CH.W - CH.R, y1: Y(tk), y2: Y(tk), stroke: "var(--grid)" }, grid);
      txt(grid, CH.L - 8, Y(tk) + 4, `${tickLabel(tk)} ${q.unit}`, { "text-anchor": "end" });
    }
    const bands = (q.bands && (q.bands[p.key] || q.bands["*"])) || [];
    for (const b of bands) el("rect", { x: CH.L, width: CH.W - CH.L - CH.R, y: Y(b.hi), height: Y(b.lo) - Y(b.hi), fill: "var(--band)" }, grid);
    for (const l of q.hlines || []) el("line", { x1: CH.L, x2: CH.W - CH.R, y1: Y(l.y), y2: Y(l.y), stroke: COLORS[l.color] || "var(--axis)", "stroke-dasharray": "3 3", "stroke-width": 1.2 }, grid);
    for (const s of q.series) {
      let d = "", pen = false;
      for (let i = 0; i < rows.length; i += stride) {
        const val = rows[i][s.col];
        if (!isNum(val)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${xs(rows[i][0]).toFixed(1)} ${Y(val).toFixed(1)}`;
        pen = true;
      }
      el("path", { d, fill: "none", stroke: COLORS[s.color] || "var(--curve)", "stroke-width": s.width || 1.5, ...(s.dash ? { "stroke-dasharray": "6 4" } : {}) }, data);
    }
    boxes.push({ q, y0, h });
    y0 += h + CH.gap;
  }
  const yTop = CH.top, yBot = y0 - CH.gap;
  const step = niceStep(p.dur, 8);
  for (let t = 0; t <= p.dur + 1e-9; t += step) {
    el("line", { x1: xs(t), x2: xs(t), y1: yBot, y2: yBot + 5, stroke: "var(--axis)" }, grid);
    const tl = S.ep.view.time_unit === "step" ? `step ${+t.toFixed(0)}` : p.dur < 60 ? `${+t.toFixed(2)} s` : clock(t);
    txt(grid, xs(t), yBot + 18, tl, { "text-anchor": "middle" });
  }
  for (const e of p.events || []) {
    el("line", { x1: xs(e.t_s), x2: xs(e.t_s), y1: yTop, y2: yBot, stroke: "var(--ink-2)", "stroke-width": 1 }, grid);
    txt(grid, xs(e.t_s) + 4, yTop + 10, e.label);
  }
  for (const m of p.marks || []) {
    if (m.t_s > p.dur) continue;
    el("line", { x1: xs(m.t_s), x2: xs(m.t_s), y1: yTop + 16, y2: yBot, stroke: "var(--critical)", "stroke-dasharray": "2 3", "stroke-width": 1 }, grid);
    txt(grid, xs(m.t_s) + 3, boxes[0].y0 + boxes[0].h - 6, m.label, { fill: "var(--critical)" });
  }
  const cursor = el("line", { y1: yTop, y2: yBot, stroke: "var(--ink)", "stroke-width": 1, opacity: 0.6 }, svg);
  const hover = el("line", { y1: yTop, y2: yBot, stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "2 2", visibility: "hidden" }, svg);
  S.chart = { p, xs, clipRect, cursor, hover };
  $("chart-title").textContent = p.name;
  const seen = new Set(), legend = [];
  for (const q of panels) {
    for (const s of q.series) if (!seen.has(s.label)) { seen.add(s.label); legend.push([COLORS[s.color], s.dash ? "dash" : "", s.label]); }
    for (const b of (q.bands && (q.bands[p.key] || q.bands["*"])) || []) if (b.label && !seen.has(b.label)) { seen.add(b.label); legend.push(["var(--band)", "box", b.label]); }
    for (const l of q.hlines || []) if (l.label && !seen.has(l.label)) { seen.add(l.label); legend.push([COLORS[l.color], "dash", l.label]); }
  }
  $("legend").innerHTML = legend.map(([c, k, label]) => `<span><i class="${k}" style="${k === "box" ? "background" : "border-color"}:${c}"></i>${esc(label)}</span>`).join("");
}

function drawChart(T) {
  const { p, t } = phaseAt(T);
  if (!S.chart || S.chart.p !== p) buildChart(p);
  const x = S.chart.xs(t);
  S.chart.clipRect.setAttribute("width", x);
  S.chart.cursor.setAttribute("x1", x); S.chart.cursor.setAttribute("x2", x);
}

function chartHover(ev) {
  const c = S.chart;
  if (!c) return;
  const box = $("trend").getBoundingClientRect();
  const x = ((ev.clientX - box.left) / box.width) * CH.W;
  if (x < CH.L || x > CH.W - CH.R) return chartLeave();
  const t = Math.min(Math.max(0, S.T - c.p.t0), Math.max(0, ((x - CH.L) / (CH.W - CH.L - CH.R)) * c.p.dur));
  const r = rowAt(c.p.rows, t);
  c.hover.setAttribute("x1", c.xs(t)); c.hover.setAttribute("x2", c.xs(t)); c.hover.removeAttribute("visibility");
  const parts = [`t ${clock(t)}`];
  for (const q of S.ep.view.panels) for (const s of q.series) if (isNum(r[s.col])) parts.push(`${s.label} ${+r[s.col].toFixed(4)} ${q.unit}`);
  const ro = $("readout");
  ro.innerHTML = parts.map(esc).join("<br>");
  ro.hidden = false;
  const host = $("trend").parentElement.getBoundingClientRect();
  ro.style.left = `${Math.min(ev.clientX - host.left + 14, host.width - ro.offsetWidth - 8)}px`;
  ro.style.top = `${ev.clientY - host.top + 14}px`;
}
function chartLeave() {
  $("readout").hidden = true;
  if (S.chart) S.chart.hover.setAttribute("visibility", "hidden");
}

// ---------- log ----------

function argText(args) {
  return Object.entries(args || {}).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : typeof v === "string" ? JSON.stringify(v) : v}`).join(", ");
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
      return `<li class="model" data-i="${i}"><div class="who"><b>${scripted ? "Scripted engineer" : "Model"}</b><span>${ep.view.time_unit === "step" ? `step ${Math.round(s.sim_t)}` : `plant ${clock(s.sim_t)}`}</span>${meta.map((m) => `<span>${esc(m)}</span>`).join("")}</div><div class="text">${esc((s.text || "").trim())}</div>${think}</li>`;
    }
    const data = s.result ? `<details><summary>Data returned to the model</summary><pre>${esc(JSON.stringify(s.result, null, 1))}</pre></details>` : "";
    return `<li class="tool" data-i="${i}"><div class="call"><span class="fn">${esc(s.name)}</span>(${esc(argText(s.args))})</div><div class="res${s.alert ? " err" : ""}">${esc(s.summary || s.error || "")}</div>${data}</li>`;
  }).join("");
  list.querySelectorAll("li").forEach((li) => li.addEventListener("click", (ev) => {
    if (ev.target.closest("details")) return;
    seekStep(+li.dataset.i);
  }));
  $("marks").innerHTML = ep.steps.map((s) => {
    const left = (100 * anchorOf(s)) / (S.total || 1);
    return `<span class="${s.kind === "model" ? "model" : s.alert ? "err" : ""}" style="left:${left}%"></span>`;
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
  if (current && current !== S.lastCurrent) {
    S.lastCurrent = current;
    const top = current.offsetTop, bottom = top + current.offsetHeight;
    if (top < list.scrollTop || bottom > list.scrollTop + list.clientHeight) list.scrollTop = Math.max(0, top - 12);
  }
}

// ---------- header and result ----------

function renderHead() {
  const ep = S.ep, r = ep.result, u = ep.usage || {};
  const scripted = ep.agent.kind === "scripted";
  $("ep-name").textContent = scripted ? `Scripted ${ep.agent.name} agent` : ep.agent.name;
  const badge = ep.status === "running" ? `<span class="badge live">live</span>`
    : r ? `<span class="badge ${r.passed ? "pass" : "fail"}">${r.passed ? "✓ passed" : "✗ failed"}</span>`
    : ep.error ? `<span class="badge fail">API error</span>` : "";
  const settings = ep.agent.settings ? Object.entries(ep.agent.settings).filter(([k, v]) => v != null && !(Array.isArray(v) && !v.length) && k !== "model").map(([k, v]) => `${k} ${v}`).join(", ") : "";
  $("ep-sub").innerHTML = `${esc(ep.env.title)} · seed ${ep.env.seed}${settings ? ` · ${esc(settings)}` : ""} ${badge}`;
  const stats = [
    ["Score", r ? r.score.toFixed(3) : "–"],
    ["Wall time", dur(ep.wall_s)],
    ["Model time", scripted ? "–" : dur(ep.model_s)],
    ["Tool calls", ep.steps.filter((s) => s.kind === "tool").length],
    ["Tokens", scripted ? "–" : int((u.prompt_tokens || 0) + (u.completion_tokens || 0))],
    ["Cost", scripted ? "–" : usd(u.cost_usd)],
  ];
  $("ep-stats").innerHTML = stats.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
  $("prompt").textContent = ep.prompt;
}

function renderResult() {
  const ep = S.ep, r = ep.result, box = $("result"), res = ep.view.result;
  if (!r && !ep.error) { box.hidden = true; return; }
  box.hidden = false;
  if (ep.error) {
    box.innerHTML = `<h2>The run stopped on an API error</h2><p>${esc(ep.error)}</p><p>API errors are not counted against the model; this run has no score.</p>`;
    return;
  }
  const reqs = (res && res.requirements) || [];
  const fin = res && res.final;
  box.innerHTML = `
    <div class="verdict-line"><h2>Acceptance test</h2><span class="score">${r.score.toFixed(3)}</span>
      <span class="badge ${r.passed ? "pass" : "fail"}">${r.passed ? "✓ passed" : "✗ failed"}</span>
      <span style="color:var(--ink-2)">${r.metrics.submitted ? "submitted by the agent" : "not submitted: " + esc(ep.stop_reason)}</span></div>
    <div class="scroll-x"><table>
      <thead><tr><th>Requirement</th><th class="num">Measured</th><th class="num">Limit</th><th></th></tr></thead>
      <tbody>${reqs.map((q) => `<tr><td>${esc(q.label)}</td><td class="num">${esc(q.measured)}</td><td class="num">${esc(q.limit)}</td><td class="${q.ok ? "ok" : "no"}">${q.ok ? "✓ met" : "✗ missed"}</td></tr>`).join("")}</tbody>
    </table></div>
    <div class="reveal">
      <div><h3>What the agent could not see</h3><dl>${((res && res.hidden) || []).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></div>
      ${fin ? `<div><h3>${esc(fin.title)}</h3><div class="scroll-x"><table class="final"><thead><tr>${fin.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
        <tbody>${fin.rows.map((row) => `<tr>${row.map((c, i) => `<td${i ? ' class="mono"' : ""}>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div>` : ""}
    </div>`;
}

// ---------- playback ----------

function setPlaying(on) {
  S.playing = on;
  $("btn-play").textContent = on ? "Pause" : S.T >= S.total - 1e-6 && S.ep.status !== "running" ? "Replay" : "Play";
  $("btn-play").setAttribute("aria-label", on ? "Pause" : "Play");
}

const holdMs = (step) => Math.min(5000, 1300 + 14 * (step.text || "").length);

function seek(T) {
  S.T = Math.max(0, Math.min(S.total, T));
  S.holdUntil = 0;
  const steps = S.ep.steps;
  let i = 0;
  while (i < steps.length && anchorOf(steps[i]) <= S.T + 1e-6) i++;
  S.next = i;
  S.cur = i - 1;
  S.follow = S.T >= S.total - 1;
  render(0);
}

function seekStep(i) {
  S.T = anchorOf(S.ep.steps[i]);
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
  $("phase").textContent = p.key === "acceptance" ? "Acceptance test: what the agent left, on a fresh start" : "Commissioning: the agent works on the plant";
  $("clock").textContent = S.ep.view.time_unit === "step" ? `${p.name.toLowerCase()}: step ${Math.floor(t)} of ${p.dur}` : `${p.name.toLowerCase()} ${clock(t)} / ${clock(p.dur)}`;
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
  S.chart = null;
  if (first) {
    buildScene();
    const speed = ep.view.speed || [0.1, 0.25, 1, 5, 20, 60, 200].find((s) => S.total / s <= 120) || 200;
    S.speed = speed;
    $("speed").value = String(speed);
  }
  renderHead();
  if (first || S.logCount !== ep.steps.length) renderLog();
  renderResult();
  if (first) {
    // ?t=<seconds of story time> opens a run at a given moment; &pause keeps it still (for links and screenshots)
    const q = new URLSearchParams(location.search);
    const t0 = q.has("t") ? Number(q.get("t")) : null;
    seek(ep.status === "running" ? S.total : isNum(t0) ? t0 : 0);
    S.follow = ep.status === "running";
    setPlaying(!q.has("pause"));
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
  const byEnv = new Map();
  for (const e of index.episodes) {
    if (!byEnv.has(e.env.id)) byEnv.set(e.env.id, []);
    byEnv.get(e.env.id).push(e);
  }
  grid.innerHTML = [...byEnv.values()].map((eps) => `<h2 class="env-head">${esc(eps[0].env.title)} <span class="chip">${esc(eps[0].env.id)}</span></h2>` + eps.map((e) => {
    const scripted = e.agent.kind === "scripted";
    const verdict = e.passed == null ? "" : `<span class="badge ${e.passed ? "pass" : "fail"}">${e.passed ? "✓ passed" : "✗ failed"} · ${fx(e.score, 3)}</span>`;
    return `<a class="ep-card" href="?ep=${encodeURIComponent(e.id)}">
      <span class="meta">seed ${e.env.seed}</span>
      <span class="who">${scripted ? `Scripted ${esc(e.agent.name)} agent` : esc(e.agent.name)}</span>
      <span>${verdict}</span>
      <span class="nums"><span>${e.tool_calls} tool calls</span><span>wall ${dur(e.wall_s)}</span>${scripted ? "" : `<span>${usd(e.usage && e.usage.cost_usd)}</span>`}</span>
    </a>`;
  }).join("")).join("");
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
