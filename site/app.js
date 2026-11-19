/* DiagRare-Bench leaderboard. Vanilla JS, no dependencies. Data: window.DIAGRARE (results.js), built by build_results.py. */
(function () {
  'use strict';
  var D = window.DIAGRARE, M = D.models, S = D.meta.stats;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var NS = 'http://www.w3.org/2000/svg';
  var f3 = function (v) { return v == null ? '–' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(3); };
  var pct = function (v, d) { return v == null ? '–' : (100 * v).toFixed(d == null ? 1 : d) + '%'; };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

  /* ---------- derived ---------- */
  var core = M.filter(function (m) { return m.tier === 'core'; }).sort(function (a, b) { return b.ge - a.ge; });
  core.forEach(function (m, i) { m.rank = i + 1; });
  var KEYS = {
    ge: { label: 'γ evidence', get: function (m) { return m.ge; }, lo: function (m) { return m.ge_lo; }, hi: function (m) { return m.ge_hi; }, fmt: f3 },
    gp: { label: 'γ prior', get: function (m) { return m.gp; }, lo: function (m) { return m.gp_lo; }, hi: function (m) { return m.gp_hi; }, fmt: f3 },
    acc1: { label: 'Top-1 accuracy', get: function (m) { return m.acc1; }, fmt: pct, scale: 100, unit: '%' },
    acc3: { label: 'Top-3 accuracy', get: function (m) { return m.acc3; }, fmt: pct, scale: 100, unit: '%' },
    rare1: { label: 'Rare-disease top-1', get: function (m) { return m.rare1; }, fmt: pct, scale: 100, unit: '%' },
    dE: { label: 'ΔE (randomized evidence response)', get: function (m) { return m.dE; }, lo: function (m) { return m.dE_lo; }, hi: function (m) { return m.dE_hi; }, fmt: f3 },
    rec: { label: 'Sequential recovery', get: function (m) { return m.rec; }, lo: function (m) { return m.rec_lo; }, hi: function (m) { return m.rec_hi; }, fmt: f3 },
    cup: { label: 'γ evidence on CUPCase', get: function (m) { return m.cup; }, fmt: function (v) { return v == null ? '–' : v.toFixed(1); } },
    rare_ext: { label: 'γ evidence on RareArena', get: function (m) { return m.rare_ext; }, fmt: function (v) { return v == null ? '–' : v.toFixed(1); } },
    sct: { label: 'Script-concordance score', get: function (m) { return m.sct; }, fmt: f3 },
    params: { label: 'Parameters (B)', get: function (m) { return m.params; }, fmt: function (v) { return v == null ? '–' : v + 'B'; }, log: true },
    used: { label: 'Usable outputs', get: function (m) { return m.used; }, fmt: pct, scale: 100, unit: '%' }
  };
  function corr(a, b) {
    var n = a.length; if (n < 3) return null;
    var ma = a.reduce(function (s, v) { return s + v; }, 0) / n, mb = b.reduce(function (s, v) { return s + v; }, 0) / n, sab = 0, saa = 0, sbb = 0;
    for (var i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb); }
    return sab / Math.sqrt(saa * sbb);
  }
  function ranks(v) { var idx = v.map(function (x, i) { return [x, i]; }).sort(function (p, q) { return p[0] - q[0]; }), r = new Array(v.length), i = 0; while (i < idx.length) { var j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (var k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; }
  function stats(pts) { var a = pts.map(function (p) { return p.x; }), b = pts.map(function (p) { return p.y; }); var r = corr(a, b); return r == null ? null : { n: pts.length, r: r, rho: corr(ranks(a), ranks(b)) }; }
  function statText(s) { return s ? 'n = ' + s.n + ' · r = ' + s.r.toFixed(3) + ' · ρ = ' + s.rho.toFixed(3) : 'too few points'; }

  /* ---------- theme / hero ---------- */
  $('#theme').addEventListener('click', function () {
    var t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = t; try { localStorage.setItem('dr-theme', t); } catch (e) { }
    renderCharts();
  });
  $('#kpis').innerHTML = [
    [S.n_checkpoints, 'checkpoints evaluated'], ['984', 'diagnostic cases'], ['82', 'diseases'], ['9', 'organ systems'], ['r = ' + S.r_acc_ge_original21.toFixed(2), 'γ evidence vs top-1, ' + S.n_original_identifiable + ' original models']
  ].map(function (k) { return '<div class="kpi"><b>' + k[0] + '</b><span>' + k[1] + '</span></div>'; }).join('');

  /* ---------- tabs ---------- */
  var tabEls = document.querySelectorAll('#tabs button'), rendered = {};
  function showTab(id, push) {
    tabEls.forEach(function (b) { b.setAttribute('aria-selected', b.dataset.tab === id); });
    document.querySelectorAll('.panel').forEach(function (p) { p.classList.toggle('on', p.id === 'p-' + id); });
    if (push) { try { history.replaceState(null, '', '#' + id); } catch (e) { } }
    if (id !== 'table') renderCharts();
  }
  tabEls.forEach(function (b) { b.addEventListener('click', function () { showTab(b.dataset.tab, true); }); });

  /* ---------- leaderboard table ---------- */
  var st = { scope: 'core', q: '', gen: true, med: true, key: 'ge', dir: -1 };
  var COLS = [
    { k: 'rank', t: '#', l: 1, nosort: 1 },
    { k: 'name', t: 'Model', l: 1 },
    { k: 'params', t: 'Size' },
    { k: 'ge', t: 'γ evidence', tip: 'Evidence coefficient with 95% interval' },
    { k: 'acc1', t: 'Top-1' },
    { k: 'gp', t: 'γ prior' },
    { k: 'dE', t: 'ΔE', tip: 'Response to a stronger evidence contrast (randomized experiment)' },
    { k: 'rec', t: 'Recovery', tip: 'Revision after supporting findings (sequential task)' },
    { k: 'used', t: 'Usable', tip: 'Share of outputs that could be parsed into a ranking' }
  ];
  function head() {
    $('#lb').innerHTML = '<thead><tr>' + COLS.map(function (c) {
      var a = st.key === c.k ? (st.dir < 0 ? '▼' : '▲') : '';
      return '<th class="' + (c.l ? 'l' : '') + '" data-k="' + c.k + '"' + (c.tip ? ' title="' + esc(c.tip) + '"' : '') + '>' + c.t + '<span class="arr">' + a + '</span></th>';
    }).join('') + '</tr></thead><tbody></tbody>';
    document.querySelectorAll('#lb th').forEach(function (th) {
      th.addEventListener('click', function () {
        var k = th.dataset.k; if (k === 'rank') k = 'ge'; if (st.key === k) st.dir *= -1; else { st.key = k; st.dir = k === 'name' ? 1 : -1; } body(); head2();
      });
    });
  }
  function head2() { document.querySelectorAll('#lb th').forEach(function (th) { var k = th.dataset.k === 'rank' ? 'ge' : th.dataset.k; th.querySelector('.arr').textContent = st.key === k ? (st.dir < 0 ? '▼' : '▲') : ''; }); }
  var DOM = [-0.1, 0.55];
  function gcell(m) {
    if (m.ge == null) return '<span class="na-dash">not estimated</span>';
    var sc = function (v) { return (Math.min(Math.max(v, DOM[0]), DOM[1]) - DOM[0]) / (DOM[1] - DOM[0]) * 100; }, z = sc(0), e = sc(m.ge), neg = m.ge < 0;
    var bar = '<span class="fill" style="left:' + Math.min(z, e) + '%;width:' + Math.abs(e - z) + '%"></span>';
    var wh = m.ge_lo != null ? '<span class="whisk" style="left:' + sc(m.ge_lo) + '%;width:' + (sc(m.ge_hi) - sc(m.ge_lo)) + '%"></span><span class="cap" style="left:' + sc(m.ge_lo) + '%"></span><span class="cap" style="left:' + sc(m.ge_hi) + '%"></span>' : '';
    return '<div class="gcell"><div class="gtrack' + (neg ? ' neg' : '') + '"><span class="zero" style="left:' + z + '%"></span>' + bar + wh + '</div><span class="v mono">' + f3(m.ge) + '</span></div>';
  }
  function rows() {
    var q = st.q.toLowerCase();
    var r = M.filter(function (m) {
      if (st.scope === 'core' && m.tier !== 'core') return false;
      if (st.scope === 'orig' && m.added) return false;
      if (m.medical ? !st.med : !st.gen) return false;
      return !q || (m.name + ' ' + m.org + ' ' + m.family + ' ' + m.id).toLowerCase().indexOf(q) >= 0;
    });
    var key = st.key, get = key === 'name' ? function (m) { return m.name.toLowerCase(); } : KEYS[key].get;
    r.sort(function (a, b) {
      var x = get(a), y = get(b); if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
      if (st.key === 'ge' && a.tier !== b.tier) return (a.tier === 'core' ? -1 : (b.tier === 'core' ? 1 : (a.tier === 'low' ? -1 : 1)));
      return (x < y ? -1 : x > y ? 1 : 0) * st.dir;
    });
    return r;
  }
  function body() {
    var r = rows();
    $('#lb tbody').innerHTML = r.map(function (m) {
      var medal = m.rank && m.rank <= 3 ? '<span class="medal m' + m.rank + '">' + m.rank + '</span>' : (m.rank || '–');
      var pill = '<span class="pill ' + (m.medical ? 'med' : 'gen') + '">' + (m.medical ? 'Medical' : 'General') + '</span>' + (m.added ? ' <span class="pill new">New</span>' : '') + (m.tier !== 'core' ? ' <span class="pill na" title="' + esc(m.reason) + '">' + (m.tier === 'low' ? 'Reported' : 'Not estimated') + '</span>' : '');
      var ci = function (v, lo, hi, fmt) { return v == null ? '<span class="na-dash">–</span>' : '<span class="mono">' + fmt(v) + '</span>'; };
      var u = m.used == null ? '–' : '<span class="usable ' + (m.used < .5 ? 'low' : '') + '"><span class="u"><i style="width:' + Math.round(100 * m.used) + '%"></i></span><span class="mono">' + Math.round(100 * m.used) + '%</span></span>';
      return '<tr class="' + (m.tier !== 'core' ? 'dim' : '') + '" data-id="' + m.id + '"><td class="rank ' + (m.rank && m.rank <= 3 ? 'top' : '') + '">' + medal + '</td>' +
        '<td class="l"><div class="mname">' + esc(m.name) + ' ' + pill + '</div><div class="msub">' + esc(m.org) + ' · ' + esc(m.family) + '</div></td>' +
        '<td class="mono">' + (m.params == null ? '–' : m.params + 'B') + '</td><td>' + gcell(m) + '</td>' +
        '<td>' + ci(m.acc1, 0, 0, pct) + '</td><td>' + ci(m.gp, 0, 0, f3) + '</td><td>' + ci(m.dE, 0, 0, f3) + '</td><td>' + ci(m.rec, 0, 0, f3) + '</td><td>' + u + '</td></tr>';
    }).join('') || '<tr><td colspan="9" class="l" style="padding:28px;color:var(--muted)">No checkpoints match.</td></tr>';
    $('#count').textContent = r.length + ' of ' + M.length + ' checkpoints';
    document.querySelectorAll('#lb tbody tr[data-id]').forEach(function (tr) { tr.addEventListener('click', function () { openDrawer(tr.dataset.id); }); });
  }
  head(); body();
  $('#q').addEventListener('input', function (e) { st.q = e.target.value; body(); });
  document.querySelectorAll('#scope button').forEach(function (b) { b.addEventListener('click', function () { st.scope = b.dataset.v; document.querySelectorAll('#scope button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); body(); }); });
  [['#f-gen', 'gen'], ['#f-med', 'med']].forEach(function (p) { $(p[0]).addEventListener('click', function (e) { st[p[1]] = !st[p[1]]; e.currentTarget.setAttribute('aria-pressed', st[p[1]]); body(); }); });

  /* ---------- drawer ---------- */
  function openDrawer(id) {
    var m = M.filter(function (x) { return x.id === id; })[0]; if (!m) return;
    var ci = function (lo, hi) { return lo == null ? '' : '<small>[' + f3(lo) + ', ' + f3(hi) + ']</small>'; };
    var cell = function (lab, v, extra) { return '<div class="dstat"><span>' + lab + '</span><b>' + v + '</b>' + (extra || '') + '</div>'; };
    var html = '<button class="icon-btn x" id="dx" aria-label="Close"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg></button>' +
      '<h3>' + esc(m.name) + '</h3><div class="msub" style="margin-bottom:8px">' + esc(m.org) + ' · ' + esc(m.family) + (m.params ? ' · ' + m.params + 'B' : '') + '</div>' +
      '<span class="pill ' + (m.medical ? 'med' : 'gen') + '">' + (m.medical ? 'Medical' : 'General') + '</span>' + (m.added ? ' <span class="pill new">Added in review</span>' : '') + (m.rank ? ' <span class="pill new">Rank ' + m.rank + ' of ' + core.length + '</span>' : '') +
      (m.tier !== 'core' ? '<div class="note-box">Not ranked: ' + esc(m.reason) + '.</div>' : '') +
      '<div class="dgrid">' + cell('γ evidence', f3(m.ge), ci(m.ge_lo, m.ge_hi)) + cell('γ prior', f3(m.gp), ci(m.gp_lo, m.gp_hi)) +
      cell('Top-1 accuracy', pct(m.acc1)) + cell('Top-3 accuracy', pct(m.acc3)) + cell('Rare-disease top-1', pct(m.rare1)) + cell('Usable outputs', pct(m.used, 0)) +
      cell('ΔE (randomized)', f3(m.dE), ci(m.dE_lo, m.dE_hi) + (m.dE_n ? '<small>' + m.dE_n + ' disease pairs</small>' : '')) + cell('Sequential recovery', f3(m.rec), ci(m.rec_lo, m.rec_hi) + (m.rec_n ? '<small>n = ' + m.rec_n + '</small>' : '')) +
      cell('γ evidence, CUPCase', m.cup == null ? '–' : m.cup.toFixed(1), m.cup_cov != null && m.cup_cov < .5 ? '<small>under half of cases usable</small>' : '') + cell('γ evidence, RareArena', m.rare_ext == null ? '–' : m.rare_ext.toFixed(1), m.rare_cov != null && m.rare_cov < .5 ? '<small>under half of cases usable</small>' : '') + '</div>' +
      (m.added ? '<div class="note-box">Added with the same 984-case baseline protocol. Interval, randomized, sequential and transfer results were not run for this checkpoint.</div>' : '') +
      '<a class="btn" href="https://huggingface.co/' + esc(m.hf) + '" target="_blank" rel="noopener">Checkpoint on Hugging Face ↗</a>';
    $('#drawer').innerHTML = html; $('#drawer').classList.add('on'); $('#dbg').classList.add('on'); $('#drawer').setAttribute('aria-hidden', 'false');
    $('#dx').addEventListener('click', closeDrawer);
  }
  function closeDrawer() { $('#drawer').classList.remove('on'); $('#dbg').classList.remove('on'); $('#drawer').setAttribute('aria-hidden', 'true'); }
  $('#dbg').addEventListener('click', closeDrawer); document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeDrawer(); });

  /* ---------- scatter ---------- */
  var tip = $('#tip');
  function showTip(e, m, xk, yk) {
    var X = KEYS[xk], Y = KEYS[yk];
    tip.innerHTML = '<b>' + esc(m.name) + '</b><br>' + X.label + ': ' + X.fmt(X.get(m)) + '<br>' + Y.label + ': ' + Y.fmt(Y.get(m)) + (m.tier !== 'core' ? '<br><span style="opacity:.7">' + esc(m.reason) + '</span>' : '');
    tip.classList.add('on'); var w = tip.offsetWidth; tip.style.left = Math.min(e.clientX + 14, innerWidth - w - 10) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  }
  function el(n, a, p) { var e = document.createElementNS(NS, n); for (var k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
  function nice(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step; step *= err >= 7 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var a = Math.floor(lo / step) * step, t = [], v = a; for (var i = 0; i < 40; i++) { t.push(+v.toFixed(10)); if (v >= hi - step * 1e-6) break; v += step; } return { lo: a, hi: t[t.length - 1], ticks: t };
  }
  function scatter(host, xk, yk, o) {
    o = o || {}; var X = KEYS[xk], Y = KEYS[yk], W = o.w || 640, H = o.h || 420, L = 58, R = 16, T = 14, B = 50;
    var all = M.filter(function (m) { return X.get(m) != null && Y.get(m) != null && (!o.orig || !m.added) && (o.showall || m.tier === 'core'); });
    var P = all.map(function (m) { var xv = X.get(m), yv = Y.get(m); return { m: m, x: X.log ? Math.log10(xv) : xv * (X.scale || 1), y: yv * (Y.scale || 1), xr: xv, yr: yv }; });
    var cp = P, s = stats(cp);
    var xs = P.map(function (p) { return p.x; }), ys = P.map(function (p) { return p.y; });
    if (Y.lo) P.forEach(function (p) { var lo = Y.lo(p.m), hi = Y.hi(p.m); if (lo != null) { ys.push(lo * (Y.scale || 1)); ys.push(hi * (Y.scale || 1)); } });
    host.innerHTML = ''; if (!P.length) { host.innerHTML = '<div class="cap-note">No overlapping data.</div>'; return s; }
    var xa = nice(Math.min.apply(null, xs) - (Math.max.apply(null, xs) - Math.min.apply(null, xs)) * .05, Math.max.apply(null, xs) + (Math.max.apply(null, xs) - Math.min.apply(null, xs)) * .05 || 1, 6);
    var ya = nice(Math.min.apply(null, ys) - (Math.max.apply(null, ys) - Math.min.apply(null, ys)) * .05, Math.max.apply(null, ys) + (Math.max.apply(null, ys) - Math.min.apply(null, ys)) * .05 || 1, 6);
    var sx = function (v) { return L + (v - xa.lo) / (xa.hi - xa.lo) * (W - L - R); }, sy = function (v) { return H - B - (v - ya.lo) / (ya.hi - ya.lo) * (H - T - B); };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img' }); host.appendChild(svg);
    ya.ticks.forEach(function (t) { el('line', { x1: L, x2: W - R, y1: sy(t), y2: sy(t), class: 'gl' }, svg); var tx = el('text', { x: L - 9, y: sy(t) + 4, 'text-anchor': 'end' }, svg); tx.textContent = Y.unit ? t.toFixed(0) + Y.unit : (+t.toFixed(3)); });
    xa.ticks.forEach(function (t) { var tx = el('text', { x: sx(t), y: H - B + 20, 'text-anchor': 'middle' }, svg); tx.textContent = X.log ? (Math.pow(10, t) >= 1 ? +Math.pow(10, t).toFixed(1) + 'B' : +Math.pow(10, t).toFixed(2) + 'B') : (X.unit ? t.toFixed(0) + X.unit : +t.toFixed(3)); });
    el('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, class: 'ax' }, svg); el('line', { x1: L, x2: L, y1: T, y2: H - B, class: 'ax' }, svg);
    var xl = el('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle', style: 'font-weight:600;fill:var(--ink)' }, svg); xl.textContent = X.label;
    var yl = el('text', { transform: 'translate(15,' + (T + (H - T - B) / 2) + ') rotate(-90)', 'text-anchor': 'middle', style: 'font-weight:600;fill:var(--ink)' }, svg); yl.textContent = Y.label;
    if (s && cp.length >= 3) { // least-squares line over ranked points
      var n = cp.length, mx = cp.reduce(function (a, p) { return a + p.x; }, 0) / n, my = cp.reduce(function (a, p) { return a + p.y; }, 0) / n, sxy = 0, sxx = 0; cp.forEach(function (p) { sxy += (p.x - mx) * (p.y - my); sxx += (p.x - mx) * (p.x - mx); });
      var b = sxy / sxx, a0 = my - b * mx, x1 = Math.min.apply(null, cp.map(function (p) { return p.x; })), x2 = Math.max.apply(null, cp.map(function (p) { return p.x; }));
      el('line', { x1: sx(x1), x2: sx(x2), y1: sy(a0 + b * x1), y2: sy(a0 + b * x2), stroke: css('--blue-d'), 'stroke-width': 2, 'stroke-opacity': .55, 'stroke-linecap': 'round' }, svg);
    }
    var cb = css('--blue'), co = css('--orange'), surf = css('--surface'), lab = {}, placed = [];
    cp.slice().sort(function (a, b) { return b.y - a.y; }).slice(0, o.nlab == null ? 4 : o.nlab).forEach(function (p) { lab[p.m.id] = 1; });
    P.forEach(function (p) {
      var col = p.m.medical ? co : cb, core1 = p.m.tier === 'core', cx = sx(p.x), cy = sy(p.y);
      if (Y.lo && Y.lo(p.m) != null) el('line', { x1: cx, x2: cx, y1: sy(Y.lo(p.m) * (Y.scale || 1)), y2: sy(Y.hi(p.m) * (Y.scale || 1)), stroke: css('--muted'), 'stroke-opacity': .6, 'stroke-width': 1.3 }, svg);
      var c = el('circle', { cx: cx, cy: cy, r: 6.5, class: 'pt', fill: core1 ? col : surf, stroke: core1 ? surf : col, 'stroke-width': core1 ? 1.6 : 2 }, svg);
      c.addEventListener('mousemove', function (e) { c.setAttribute('r', 8.5); showTip(e, p.m, xk, yk); }); c.addEventListener('mouseleave', function () { c.setAttribute('r', 6.5); tip.classList.remove('on'); }); c.addEventListener('click', function () { openDrawer(p.m.id); });
    });
    P.forEach(function (p) {   // labels last, greedy collision avoidance
      if (!lab[p.m.id]) return; var cx = sx(p.x), cy = sy(p.y), w = p.m.name.length * 6.1, left = cx + w + 12 > W - 2, x0 = left ? cx - 10 - w : cx + 10, r = { x0: x0, x1: x0 + w, y0: cy - 20, y1: cy - 6 };
      if (placed.some(function (q) { return r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0; })) return; placed.push(r);
      var t = el('text', { x: left ? cx - 10 : cx + 10, y: cy - 8, class: 'lab', 'text-anchor': left ? 'end' : 'start' }, svg); t.textContent = p.m.name;
    });
    return s;
  }

  /* ---------- explorer ---------- */
  var XO = ['ge', 'gp', 'acc1', 'acc3', 'rare1', 'dE', 'rec', 'cup', 'rare_ext', 'sct', 'params', 'used'];
  $('#xs').innerHTML = XO.map(function (k) { return '<option value="' + k + '">' + KEYS[k].label + '</option>'; }).join(''); $('#ys').innerHTML = $('#xs').innerHTML;
  $('#xs').value = 'acc1'; $('#ys').value = 'ge';
  ['#xs', '#ys', '#showall'].forEach(function (s) { $(s).addEventListener('change', renderCharts); });

  function renderCharts() {
    var on = document.querySelector('.panel.on').id;
    if (on === 'p-explore') $('#xstat').textContent = statText(scatter($('#xchart'), $('#xs').value, $('#ys').value, { showall: $('#showall').checked, nlab: 8, w: 980, h: 500 }));
    if (on === 'p-interv') {
      $('#s-dE').textContent = statText(scatter($('#c-dE'), 'ge', 'dE', { orig: true, showall: true, nlab: 3 }));
      $('#s-rec').textContent = statText(scatter($('#c-rec'), 'ge', 'rec', { orig: true, showall: true, nlab: 3 }));
    }
    if (on === 'p-transfer') {
      $('#s-cup').textContent = statText(scatter($('#c-cup'), 'ge', 'cup', { orig: true, showall: true, nlab: 3 }));
      $('#s-rare').textContent = statText(scatter($('#c-rare'), 'ge', 'rare_ext', { orig: true, showall: true, nlab: 3 }));
    }
    if (on === 'p-context') renderCtx();
  }

  /* ---------- referral context ---------- */
  function renderCtx() {
    var C = D.context, lo = -0.13, hi = 0.19, w = 230, sx = function (v) { return (v - lo) / (hi - lo) * w; };
    var clo = -0.1, chi = 0.16, sx2 = function (v) { return (v - clo) / (chi - clo) * w; };
    function dumb(a, b) {
      return '<svg class="chart" viewBox="0 0 ' + (w + 10) + ' 22" style="width:' + (w + 10) + 'px"><line x1="' + sx(0) + '" x2="' + sx(0) + '" y1="0" y2="22" class="ax"/><line x1="' + sx(a) + '" x2="' + sx(b) + '" y1="11" y2="11" stroke="' + css('--muted') + '" stroke-width="2.4" stroke-linecap="round"/>' +
        '<circle cx="' + sx(a) + '" cy="11" r="6" fill="' + css('--muted') + '"/><circle cx="' + sx(b) + '" cy="11" r="6" fill="' + css('--orange') + '"/></svg>';
    }
    function forest(e, l, h) {
      return '<svg class="chart" viewBox="0 0 ' + (w + 10) + ' 22" style="width:' + (w + 10) + 'px"><line x1="' + sx2(0) + '" x2="' + sx2(0) + '" y1="0" y2="22" class="ax"/><line x1="' + sx2(l) + '" x2="' + sx2(h) + '" y1="11" y2="11" stroke="' + css('--ink2') + '" stroke-width="2" stroke-linecap="round"/><circle cx="' + sx2(e) + '" cy="11" r="5.5" fill="' + css('--blue') + '"/></svg>';
    }
    $('#ctxt').innerHTML = '<thead><tr><th>Model</th><th>γ prior: primary care → rare clinic</th><th>Change in γ prior</th><th>Change in top-1 accuracy</th><th>Top-1 answers that changed</th><th>Changes toward less prevalent</th></tr></thead><tbody>' +
      C.map(function (c) {
        return '<tr><td class="nm">' + esc(c.name) + '</td><td>' + dumb(c.gp_prim, c.gp_rare) + '</td><td class="mono">' + f3(c.dgp) + '<br><small style="color:var(--muted)">[' + f3(c.dgp_lo) + ', ' + f3(c.dgp_hi) + ']</small></td>' +
          '<td>' + forest(c.dacc, c.dacc_lo, c.dacc_hi) + '<div class="mono" style="font-size:12px;color:var(--ink2)">' + (c.dacc >= 0 ? '+' : '−') + Math.abs(100 * c.dacc).toFixed(1) + ' pts [' + (100 * c.dacc_lo).toFixed(1) + ', ' + (100 * c.dacc_hi).toFixed(1) + ']</div></td>' +
          '<td class="mono">' + pct(c.switch, 0) + '</td><td class="mono">' + pct(c.lower, 0) + '</td></tr>';
      }).join('') + '</tbody>';
  }

  /* ---------- misc ---------- */
  $('#copy').addEventListener('click', function (e) { var t = $('#bib').textContent.replace('Copy', '').trim(); try { navigator.clipboard.writeText(t); e.target.textContent = 'Copied'; setTimeout(function () { e.target.textContent = 'Copy'; }, 1500); } catch (x) { } });
  var h = (location.hash || '').replace('#', ''); if (['explore', 'interv', 'transfer', 'context', 'about'].indexOf(h) >= 0) showTab(h);
  window.addEventListener('hashchange', function () { var k = location.hash.replace('#', ''); if (['table', 'explore', 'interv', 'transfer', 'context', 'about'].indexOf(k) >= 0) showTab(k); });
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(renderCharts, 150); });
})();
