document.querySelectorAll(".tabs-k").forEach(function (bar) {
  bar.querySelectorAll("button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      bar.querySelectorAll("button").forEach(function (b) {
        b.classList.remove("active");
        var p = document.getElementById(b.dataset.pane);
        if (!p) return;
        p.classList.remove("active");
        p.querySelectorAll("video").forEach(function (v) { v.pause(); });
      });
      btn.classList.add("active");
      var pane = document.getElementById(btn.dataset.pane);
      if (!pane) return;
      pane.classList.add("active");
      pane.querySelectorAll("video").forEach(function (v) { v.play(); });
    });
  });
});

function copyBibTeX() {
  var code = document.getElementById("bibtex-code").textContent;
  var btn = document.querySelector(".copy-bibtex-btn .copy-text");
  navigator.clipboard.writeText(code).then(function () {
    btn.textContent = "Copied";
    setTimeout(function () { btn.textContent = "Copy"; }, 1800);
  });
}

(function () {
  var v = document.getElementById("hero-video"), r = document.getElementById("tscrub"), tv = document.getElementById("tval");
  if (!v) return;
  var HOLD = 24, STEPS = 36, FPS = 24;
  function frameOf(t) {
    if (t <= 0) return HOLD / 2;
    if (t >= 1) return HOLD + STEPS + HOLD;
    return HOLD + Math.acos(1 - 2 * t) / Math.PI * (STEPS - 1);
  }
  function tOf(k) {
    if (k < HOLD) return 0;
    if (k >= HOLD + STEPS) return 1;
    return 0.5 - 0.5 * Math.cos(Math.PI * (k - HOLD) / (STEPS - 1));
  }
  r.addEventListener("input", function () {
    var t = r.value / 1000;
    v.pause();
    if (v.readyState >= 1) v.currentTime = Math.min((frameOf(t) + 0.5) / FPS, v.duration - 0.01);
    tv.textContent = "t = " + t.toFixed(2);
  });
  v.addEventListener("timeupdate", function () {
    if (v.paused) return;
    var t = tOf(Math.floor(v.currentTime * FPS));
    r.value = Math.round(t * 1000); tv.textContent = "t = " + t.toFixed(2);
  });
  v.addEventListener("click", function () { if (v.paused) v.play(); else v.pause(); });
})();

(function () {
  var cv = document.getElementById("toycv"), ctx = cv.getContext("2d");
  var G = 220, S = 16, NOISE = 0.06, MAXP = 40;
  var grid = []; for (var i = 0; i < G; i++) grid.push(i / (G - 1));
  var pts = [[0.06, 0.75], [0.15, 0.95], [0.27, 0.45], [0.38, -0.55], [0.47, -1.0], [0.85, 0.1], [0.94, 0.3]];
  var ell = 0.10, tcur = 1, seed = 7;
  function rng() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
  function gauss() { var u = rng() || 1e-12, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  var Z = [], E = [];
  function newDraws() { Z = []; E = []; for (var s = 0; s < S; s++) { var z = [], e = []; for (var i = 0; i < G + MAXP; i++) z.push(gauss()); for (i = 0; i < MAXP; i++) e.push(gauss()); Z.push(z); E.push(e); } }
  function k(a, b) { var r = Math.sqrt(5) * Math.abs(a - b) / ell; return (1 + r + r * r / 3) * Math.exp(-r); }
  function chol(A, n) { var L = new Float64Array(n * n); for (var i = 0; i < n; i++) for (var j = 0; j <= i; j++) { var s = A[i * n + j]; for (var q = 0; q < j; q++) s -= L[i * n + q] * L[j * n + q]; L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j]; } return L; }
  function solve(L, n, b) { var y = b.slice(); for (var i = 0; i < n; i++) { for (var q = 0; q < i; q++) y[i] -= L[i * n + q] * y[q]; y[i] /= L[i * n + i]; } for (i = n - 1; i >= 0; i--) { for (q = i + 1; q < n; q++) y[i] -= L[q * n + i] * y[q]; y[i] /= L[i * n + i]; } return y; }
  var cache = null;
  function compute() {
    var n = pts.length, X = pts.map(function (p) { return p[0]; }), Y = pts.map(function (p) { return p[1]; });
    var all = grid.concat(X), m = all.length, A = new Float64Array(m * m);
    for (var i = 0; i < m; i++) for (var j = 0; j < m; j++) A[i * m + j] = k(all[i], all[j]) + (i === j ? 1e-7 : 0);
    var Lp = chol(A, m), K = new Float64Array(n * n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) K[i * n + j] = k(X[i], X[j]) + (i === j ? NOISE * NOISE : 0);
    var Lk = chol(K, n), kx = grid.map(function (g) { return X.map(function (x) { return k(g, x); }); });
    var alpha = n ? solve(Lk, n, Y) : [], mean = [], sd = [];
    for (i = 0; i < G; i++) { var mu = 0, v = 1; for (j = 0; j < n; j++) mu += kx[i][j] * alpha[j]; if (n) { var w = solve(Lk, n, kx[i]); for (j = 0; j < n; j++) v -= kx[i][j] * w[j]; } mean.push(mu); sd.push(Math.sqrt(Math.max(v, 0))); }
    var draws = Z.map(function (z, s) {
      var f = new Float64Array(m); for (var i = 0; i < m; i++) { var a = 0; for (var q = 0; q <= i; q++) a += Lp[i * m + q] * z[q]; f[i] = a; }
      var r = []; for (var j = 0; j < n; j++) r.push(Y[j] - f[G + j] - NOISE * E[s][j]);
      var b = n ? solve(Lk, n, r) : [], c = [];
      for (i = 0; i < G; i++) { var cc = 0; for (j = 0; j < n; j++) cc += kx[i][j] * b[j]; c.push(cc); }
      return { f: Array.prototype.slice.call(f, 0, G), fX: Array.prototype.slice.call(f, G), r: r, c: c };
    });
    cache = { X: X, Y: Y, mean: mean, sd: sd, draws: draws };
  }
  var PAL = {"--ink": "#1f2430", "--prussian": "#3b5bdb", "--correction": "#e8590c", "--rule": "#e5e7eb", "--prussian-soft": "#dbe4ff", "--sheet": "#ffffff", "--body": '"Inter", sans-serif'};
  function css(name) { return PAL[name]; }
  var W, H, Y0 = -2.8, Y1 = 2.6;
  function sx(x) { return 18 + x * (W - 36); }
  function sy(y) { return 12 + (Y1 - y) / (Y1 - Y0) * (H - 24); }
  function draw() {
    var dpr = window.devicePixelRatio || 1; W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    var ink = css("--ink"), blue = css("--prussian"), orange = css("--correction"), rule = css("--rule"), soft = css("--prussian-soft");
    ctx.strokeStyle = rule; ctx.lineWidth = 1;
    for (var g = -2; g <= 2; g++) { ctx.beginPath(); ctx.moveTo(sx(0), sy(g)); ctx.lineTo(sx(1), sy(g)); ctx.stroke(); }
    var c = cache, t = tcur;
    if (document.getElementById("toyband").checked && t > 0.999) {
      ctx.fillStyle = soft; ctx.beginPath();
      for (var i = 0; i < G; i++) ctx.lineTo(sx(grid[i]), sy(c.mean[i] + 1.96 * c.sd[i]));
      for (i = G - 1; i >= 0; i--) ctx.lineTo(sx(grid[i]), sy(c.mean[i] - 1.96 * c.sd[i]));
      ctx.fill();
    }
    c.draws.forEach(function (d, s) {
      ctx.strokeStyle = s === 0 ? blue : (t > 0.5 ? ink : blue); ctx.globalAlpha = s === 0 ? 1 : 0.28; ctx.lineWidth = s === 0 ? 2.2 : 1;
      ctx.beginPath(); for (var i = 0; i < G; i++) ctx.lineTo(sx(grid[i]), sy(d.f[i] + t * d.c[i])); ctx.stroke();
    });
    ctx.globalAlpha = 1;
    if (document.getElementById("toyres").checked && c.X.length) {
      var d0 = c.draws[0]; ctx.strokeStyle = orange; ctx.lineWidth = 2.4; ctx.lineCap = "round";
      c.X.forEach(function (x, j) { var a = d0.fX[j] + t * (c.Y[j] - d0.fX[j]); ctx.beginPath(); ctx.moveTo(sx(x), sy(a)); ctx.lineTo(sx(x), sy(d0.fX[j] + d0.r[j])); ctx.stroke(); });
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6; ctx.globalAlpha = 0.85; ctx.beginPath();
      for (i = 0; i < G; i++) ctx.lineTo(sx(grid[i]), sy(d0.c[i] - 2.2)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = orange; ctx.font = "12px " + css("--body"); ctx.fillText("correction k(·, X) K⁻¹ r  (shifted down)", sx(0.01), sy(-2.55));
    }
    c.X.forEach(function (x, j) { ctx.beginPath(); ctx.arc(sx(x), sy(c.Y[j]), 5.5, 0, 2 * Math.PI); ctx.fillStyle = ink; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = css("--sheet"); ctx.stroke(); });
  }
  function update() { compute(); draw(); }
  function toData(ev) { var b = cv.getBoundingClientRect(); var x = (ev.clientX - b.left - 18) / (b.width - 36); var y = Y1 - (ev.clientY - b.top - 12) / (b.height - 24) * (Y1 - Y0); return [Math.min(1, Math.max(0, x)), y]; }
  function nearest(p) { var best = -1, bd = 1e9; pts.forEach(function (q, i) { var dx = (q[0] - p[0]) * W, dy = (q[1] - p[1]) / (Y1 - Y0) * H, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }); return bd < 225 ? best : -1; }
  var drag = -1;
  cv.addEventListener("pointerdown", function (ev) { var p = toData(ev), i = nearest(p); if (i >= 0) { drag = i; } else if (pts.length < MAXP) { pts.push(p); drag = pts.length - 1; update(); } cv.setPointerCapture(ev.pointerId); });
  cv.addEventListener("pointermove", function (ev) { if (drag < 0) return; pts[drag] = toData(ev); update(); });
  cv.addEventListener("pointerup", function () { drag = -1; });
  cv.addEventListener("dblclick", function (ev) { var i = nearest(toData(ev)); if (i >= 0) { pts.splice(i, 1); update(); } });
  document.getElementById("toyt").addEventListener("input", function (e) { tcur = e.target.value / 100; document.getElementById("toytv").textContent = "t = " + tcur.toFixed(2); draw(); });
  document.getElementById("toyl").addEventListener("input", function (e) { ell = e.target.value / 100; document.getElementById("toylv").textContent = ell.toFixed(2); update(); });
  document.getElementById("toyres").addEventListener("change", draw);
  document.getElementById("toyband").addEventListener("change", draw);
  document.getElementById("toynew").addEventListener("click", function () { newDraws(); update(); });
  window.addEventListener("resize", draw);
  newDraws(); update();
})();

