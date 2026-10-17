/* Multiple linear regression pipeline: schema inference, preprocessing, train/test split,
   feature selection (SelectKBest / RFE / Sequential Forward, 5-fold CV), OLS with CI/PI,
   and assumption diagnostics. Pure functions, no DOM: usable in the browser (window.MLR)
   and in Node (tests compare it with statsmodels). */
(function (root) {
  "use strict";
  const S = typeof module !== "undefined" && module.exports ? require("./stats.js") : root.Stats;

  const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "?", "-"]);
  const ID_NAME = /^(id|index|row|no|unnamed: ?0|.*[_\s-]id)$/i;
  const UNIQUE_CAP = 100000;   // stop counting distinct values after this many

  function isMissing(v) {
    return v === null || v === undefined || (typeof v === "number" && isNaN(v)) ||
      (typeof v === "string" && MISSING.has(v.trim().toLowerCase()));
  }

  function toNumber(v) {
    if (typeof v === "number") return isFinite(v) ? v : null;
    if (typeof v !== "string") return null;
    const s = v.trim().replace(/,/g, "");
    if (s === "" || !/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return null;
    return Number(s);
  }

  // ---------------------------------------------------------------- schema
  /** Decide, for every column, whether it is numeric or categorical and whether it can be used. */
  function inferSchema(rows, columns) {
    const n = rows.length;
    return columns.map((name) => {
      let nonMissing = 0, numeric = 0, allInt = true;
      const uniq = new Set();
      for (const r of rows) {
        const v = r[name];
        if (isMissing(v)) continue;
        nonMissing++;
        const x = toNumber(v);
        if (x !== null) { numeric++; if (!Number.isInteger(x)) allInt = false; }
        if (uniq.size <= UNIQUE_CAP) uniq.add(typeof v === "string" ? v.trim() : v);
      }
      const col = { name, missing: n - nonMissing, unique: uniq.size, kind: "numeric", excluded: null };
      const counted = Math.min(nonMissing, UNIQUE_CAP + 1);   // rows the unique count can speak for
      if (nonMissing === 0) { col.excluded = "全部缺失"; return col; }
      col.kind = numeric / nonMissing >= 0.95 ? "numeric" : "categorical";
      if (uniq.size === 1) col.excluded = "常數欄";
      else if (col.kind === "numeric" && allInt && ID_NAME.test(name.trim()) && uniq.size >= 0.95 * counted)
        col.excluded = "ID 欄";
      else if (col.kind === "categorical" && (uniq.size > 50 || uniq.size > 0.5 * counted))
        col.excluded = `類別太多（${uniq.size}）`;
      return col;
    });
  }

  function targetCandidates(schema) {
    return schema.filter((c) => c.kind === "numeric" && !c.excluded).map((c) => c.name);
  }

  // ---------------------------------------------------------------- encoder
  function median(xs) {
    const s = Float64Array.from(xs).sort();
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /** Fit imputation + one-hot (drop first level) on training rows only. */
  function fitEncoder(trainRows, schema, features) {
    const cols = [];        // encoded columns: {name, feature, kind, level?}
    const fill = {};
    const means = {};
    const levels = {};
    for (const f of features) {
      const c = schema.find((s) => s.name === f);
      if (c.kind === "numeric") {
        const xs = trainRows.map((r) => toNumber(r[f])).filter((x) => x !== null);
        fill[f] = xs.length ? median(xs) : 0;
        means[f] = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
        cols.push({ name: f, feature: f, kind: "numeric" });
      } else {
        const counts = new Map();
        for (const r of trainRows) {
          const v = r[f];
          if (!isMissing(v)) { const k = String(v).trim(); counts.set(k, (counts.get(k) || 0) + 1); }
        }
        const lv = [...counts.keys()].sort();
        fill[f] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
        levels[f] = lv;
        for (const l of lv.slice(1)) cols.push({ name: `${f} = ${l}`, feature: f, kind: "dummy", level: l });
      }
    }
    return { cols, fill, means, levels, features };
  }

  function encodeRow(enc, r) {
    const out = new Float64Array(enc.cols.length);
    enc.cols.forEach((c, j) => {
      const v = r[c.feature];
      if (c.kind === "numeric") {
        const x = toNumber(v);
        out[j] = x === null ? enc.fill[c.feature] : x;
      } else {
        const s = isMissing(v) ? enc.fill[c.feature] : String(v).trim();
        out[j] = s === c.level ? 1 : 0;
      }
    });
    return out;
  }

  // ---------------------------------------------------------------- Gram matrices
  /** Gram matrix of [1, Z] plus X'y and y'y for the given row indices. */
  function gram(Z, y, idx) {
    const p = Z[0].length + 1;
    const G = Array.from({ length: p }, () => new Float64Array(p));
    const Xty = new Float64Array(p);
    let yty = 0;
    const x = new Float64Array(p);
    x[0] = 1;
    for (const i of idx) {
      x.set(Z[i], 1);
      const yi = y[i];
      yty += yi * yi;
      for (let a = 0; a < p; a++) {
        const xa = x[a];
        if (xa === 0) continue;
        Xty[a] += xa * yi;
        const Ga = G[a];
        for (let b = a; b < p; b++) Ga[b] += xa * x[b];
      }
    }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
    return { G, Xty, yty, n: idx.length };
  }

  function subtract(A, B) {
    return {
      G: A.G.map((row, i) => row.map((v, j) => v - B.G[i][j])),
      Xty: A.Xty.map((v, i) => v - B.Xty[i]),
      yty: A.yty - B.yty,
      n: A.n - B.n,
    };
  }

  /** OLS on a column subset (indices into Z), solved from Gram matrices. ridge > 0 only for ranking/CV. */
  function solve(g, subset, ridge = 0) {
    const ix = [0, ...subset.map((j) => j + 1)];
    const A = ix.map((a, r) => ix.map((b, c) => g.G[a][b] + (r === c && r > 0 ? ridge * g.n : 0)));
    const inv = S.invert(A);
    if (!inv) return null;
    const xty = ix.map((a) => g.Xty[a]);
    return { ix, inv, b: S.matVec(inv, xty) };
  }

  function sse(g, ix, b) {
    let s = g.yty;
    for (let r = 0; r < ix.length; r++) {
      s -= 2 * b[r] * g.Xty[ix[r]];
      for (let c = 0; c < ix.length; c++) s += b[r] * b[c] * g.G[ix[r]][ix[c]];
    }
    return Math.max(s, 0);
  }

  function cvRmse(folds, subset) {
    let total = 0, n = 0;
    for (const f of folds) {
      const fit = solve(f.train, subset, 1e-10);
      if (!fit) return Infinity;
      total += sse(f.val, fit.ix, fit.b);
      n += f.val.n;
    }
    return Math.sqrt(total / n);
  }

  // ---------------------------------------------------------------- feature ranking
  function rankKBest(full, p) {
    // |corr(z_j, y)| is proportional to |sum z_j y| because z is centred and scaled
    return [...Array(p).keys()].sort((a, b) => Math.abs(full.Xty[b + 1]) - Math.abs(full.Xty[a + 1]));
  }

  function rankRFE(full, p) {
    let active = [...Array(p).keys()];
    const removed = [];
    while (active.length > 1) {
      const fit = solve(full, active, 1e-10);
      let worst = 0;
      for (let j = 1; j < active.length; j++) if (Math.abs(fit.b[j + 1]) < Math.abs(fit.b[worst + 1])) worst = j;
      removed.push(active[worst]);
      active = active.filter((_, j) => j !== worst);
    }
    return [...active, ...removed.reverse()];
  }

  function rankForward(folds, p, maxSteps, fallback) {
    const chosen = [];
    let left = [...Array(p).keys()];
    while (left.length && chosen.length < maxSteps) {
      let best = null, bestScore = Infinity;
      for (const j of left) {
        const s = cvRmse(folds, [...chosen, j]);
        if (s < bestScore) { bestScore = s; best = j; }
      }
      if (best === null) break;
      chosen.push(best);
      left = left.filter((j) => j !== best);
    }
    return [...chosen, ...fallback.filter((j) => !chosen.includes(j))];
  }

  // ---------------------------------------------------------------- main pipeline
  const METHODS = { kbest: "SelectKBest (|r|)", rfe: "RFE", forward: "Sequential Forward" };

  /**
   * Run the whole pipeline.
   * opts: {seed, testRatio, maxRows, conf, method: 'auto'|'kbest'|'rfe'|'forward', k: null|number, tol,
   *        features: null|string[] (candidate features; default = every usable column)}
   */
  function run(rows, schema, target, opts = {}) {
    const o = Object.assign({ seed: 42, testRatio: 0.2, maxRows: 50000, conf: 0.95, method: "auto", k: null, tol: 0.005, features: null }, opts);
    const t0 = Date.now();

    // 1. rows with a numeric target; sample very large data
    let data = rows.filter((r) => toNumber(r[target]) !== null);
    const droppedTarget = rows.length - data.length;
    let sampled = false;
    if (data.length > o.maxRows) {
      data = S.shuffledIndex(data.length, o.seed + 1).slice(0, o.maxRows).map((i) => data[i]);
      sampled = true;
    }
    const n = data.length;
    if (n < 20) throw new Error(`可用的資料只有 ${n} 筆，至少需要 20 筆`);

    // 2. train/test split (shuffled, seeded)
    const order = S.shuffledIndex(n, o.seed);
    const nTest = Math.max(5, Math.round(n * o.testRatio));
    const testIdx = order.slice(0, nTest), trainIdx = order.slice(nTest);
    const train = trainIdx.map((i) => data[i]), test = testIdx.map((i) => data[i]);

    // 3. preprocessing fitted on train only
    const features = schema.filter((c) => !c.excluded && c.name !== target && (!o.features || o.features.includes(c.name))).map((c) => c.name);
    if (!features.length) throw new Error("沒有可用的特徵欄位");
    const enc = fitEncoder(train, schema, features);
    let Xtr = train.map((r) => encodeRow(enc, r));
    let Xte = test.map((r) => encodeRow(enc, r));
    const ytr = train.map((r) => toNumber(r[target]));
    const yte = test.map((r) => toNumber(r[target]));

    // standardize with train mean / population sd; drop zero-variance columns
    let mean = [], sd = [], keep = [];
    enc.cols.forEach((c, j) => {
      let m = 0;
      for (const x of Xtr) m += x[j];
      m /= Xtr.length;
      let v = 0;
      for (const x of Xtr) v += (x[j] - m) ** 2;
      const s = Math.sqrt(v / Xtr.length);
      if (s > 1e-12) { keep.push(j); mean.push(m); sd.push(s); }
    });
    const cols = keep.map((j) => enc.cols[j]);
    const z = (x) => Float64Array.from(keep, (j, k) => (x[j] - mean[k]) / sd[k]);
    const Ztr = Xtr.map(z), Zte = Xte.map(z);
    const p = cols.length;
    if (p === 0) throw new Error("所有特徵在訓練集中都是常數");
    if (Ztr.length <= p + 1) throw new Error(`訓練資料 ${Ztr.length} 筆不足以估計 ${p} 個係數`);

    // 4. Gram matrices: full train + 5 CV folds
    const all = [...Array(Ztr.length).keys()];
    const full = gram(Ztr, ytr, all);
    const folds = [0, 1, 2, 3, 4].map((f) => {
      const val = gram(Ztr, ytr, all.filter((i) => i % 5 === f));
      return { val, train: subtract(full, val) };
    });

    // 5. feature selection
    const ranks = {};
    ranks.kbest = rankKBest(full, p);
    ranks.rfe = rankRFE(full, p);
    ranks.forward = rankForward(folds, p, 40, ranks.rfe);
    const curves = {};
    for (const m of Object.keys(ranks)) curves[m] = ranks[m].map((_, k) => cvRmse(folds, ranks[m].slice(0, k + 1)));

    let best = Infinity;
    for (const m in curves) for (const v of curves[m]) best = Math.min(best, v);
    let chosen;
    if (o.method === "auto") {
      chosen = null;
      for (let k = 1; k <= p && !chosen; k++)
        for (const m of ["forward", "rfe", "kbest"])
          if (curves[m][k - 1] <= best * (1 + o.tol)) { chosen = { method: m, k }; break; }
    } else {
      const c = curves[o.method];
      const bestM = Math.min(...c);
      const k = o.k ? Math.min(o.k, p) : c.findIndex((v) => v <= bestM * (1 + o.tol)) + 1;
      chosen = { method: o.method, k };
    }
    if (o.method === "auto" && o.k) chosen.k = Math.min(o.k, p);
    const subset = ranks[chosen.method].slice(0, chosen.k);

    // 6. final OLS on the full training split
    const fit = solve(full, subset, 0);
    if (!fit) throw new Error("選出的特徵完全共線，無法估計 OLS（請換特徵數或方法）");
    const k = subset.length, dfResid = Ztr.length - k - 1;
    const s2 = sse(full, fit.ix, fit.b) / dfResid;
    const cov = fit.inv.map((row) => row.map((v) => v * s2));
    const tq = S.tQuantile(1 - (1 - o.conf) / 2, dfResid);

    // coefficients back in original units: beta = A b, Cov = A C A'
    const kk = k + 1;
    const A = Array.from({ length: kk }, () => new Float64Array(kk));
    A[0][0] = 1;
    subset.forEach((j, r) => { A[0][r + 1] = -mean[j] / sd[j]; A[r + 1][r + 1] = 1 / sd[j]; });
    const beta = S.matVec(A, fit.b);
    const AC = A.map((row) => Array.from({ length: kk }, (_, c) => row.reduce((s, a, t) => s + a * cov[t][c], 0)));
    const covO = AC.map((row) => Array.from({ length: kk }, (_, c) => row.reduce((s, a, t) => s + a * A[c][t], 0)));
    const coefs = beta.map((b, r) => {
      const se = Math.sqrt(Math.max(covO[r][r], 0));
      const t = se > 0 ? b / se : Infinity;
      return { name: r === 0 ? "(Intercept)" : cols[subset[r - 1]].name, coef: b, se, t,
        p: se > 0 ? S.tPValue(t, dfResid) : 0, lo: b - tq * se, hi: b + tq * se };
    });

    const predictZ = (zrow) => {
      const x0 = [1, ...subset.map((j) => zrow[j])];
      const yhat = x0.reduce((s, v, i) => s + v * fit.b[i], 0);
      const seMean = Math.sqrt(Math.max(S.quadForm(cov, x0), 0));
      const seObs = Math.sqrt(s2 + seMean * seMean);
      return { pred: yhat, ciLo: yhat - tq * seMean, ciHi: yhat + tq * seMean, piLo: yhat - tq * seObs, piHi: yhat + tq * seObs };
    };

    // 7. evaluation on the test split
    const testPred = Zte.map((zr, i) => Object.assign(predictZ(zr), { actual: yte[i] }));
    const metrics = evaluate(yte, testPred.map((d) => d.pred), k);
    metrics.coverage = testPred.filter((d) => d.actual >= d.piLo && d.actual <= d.piHi).length / testPred.length;
    const cvAll = curves.rfe[p - 1];

    // 8. diagnostics on training residuals
    const fitted = Ztr.map((zr) => [1, ...subset.map((j) => zr[j])].reduce((s, v, i) => s + v * fit.b[i], 0));
    const resid = ytr.map((y, i) => y - fitted[i]);
    const diag = diagnostics(Ztr, subset, resid);

    return {
      target, n, nTrain: Ztr.length, nTest: Zte.length, sampled, droppedTarget,
      features, encoded: cols.map((c) => c.name), p,
      methods: METHODS, ranks: Object.fromEntries(Object.entries(ranks).map(([m, r]) => [m, r.map((j) => cols[j].name)])),
      curves, chosen, selected: subset.map((j) => cols[j].name),
      coefs, stdCoefs: subset.map((j, r) => ({ name: cols[j].name, coef: fit.b[r + 1] })), s2, dfResid, conf: o.conf, metrics, cvRmseAll: cvAll, cvRmseSelected: curves[chosen.method][chosen.k - 1],
      testPred, train: { fitted, resid }, diag,
      predictRow: (raw) => predictZ(z(encodeRow(enc, raw))),
      inputs: features.map((f) => {
        const c = schema.find((s) => s.name === f);
        return c.kind === "numeric" ? { name: f, kind: "numeric", default: enc.means[f] }
          : { name: f, kind: "categorical", levels: enc.levels[f], default: enc.fill[f] };
      }),
      ms: Date.now() - t0,
    };
  }

  function evaluate(y, yhat, k) {
    const n = y.length;
    const ybar = y.reduce((a, b) => a + b, 0) / n;
    let mae = 0, mse = 0, sst = 0;
    for (let i = 0; i < n; i++) {
      const e = y[i] - yhat[i];
      mae += Math.abs(e); mse += e * e; sst += (y[i] - ybar) ** 2;
    }
    mae /= n; mse /= n;
    const r2 = 1 - (mse * n) / sst;
    return { MAE: mae, MSE: mse, RMSE: Math.sqrt(mse), R2: r2, AdjR2: 1 - ((1 - r2) * (n - 1)) / (n - k - 1), k };
  }

  function diagnostics(Z, subset, e) {
    const n = e.length;
    // Durbin-Watson
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { den += e[i] * e[i]; if (i) num += (e[i] - e[i - 1]) ** 2; }
    const dw = num / den;
    // Jarque-Bera
    const m = e.reduce((a, b) => a + b, 0) / n;
    let m2 = 0, m3 = 0, m4 = 0;
    for (const v of e) { const d = v - m; m2 += d * d; m3 += d ** 3; m4 += d ** 4; }
    m2 /= n; m3 /= n; m4 /= n;
    const skew = m3 / m2 ** 1.5, kurt = m4 / (m2 * m2);
    const jb = (n / 6) * (skew * skew + ((kurt - 3) ** 2) / 4);
    // Breusch-Pagan (studentized, Koenker): n * R^2 of e^2 regressed on the selected features
    const e2 = e.map((v) => v * v);
    const g = gram(Z.map((zr) => Float64Array.from(subset, (j) => zr[j])), e2, [...Array(n).keys()]);
    const all = subset.map((_, i) => i);
    const fit = solve(g, all, 1e-10);
    const mean2 = e2.reduce((a, b) => a + b, 0) / n;
    const sst = e2.reduce((s, v) => s + (v - mean2) ** 2, 0);
    const bpR2 = fit ? 1 - sse(g, fit.ix, fit.b) / sst : 0;
    const bp = n * bpR2;
    // VIF from the inverse correlation matrix (Z is standardized, so Z'Z / n = R)
    let vif = subset.map(() => 1);
    if (subset.length > 1) {
      const R = subset.map((_, a) => subset.map((__, b) => g.G[a + 1][b + 1] / n));
      const inv = S.invert(R);
      vif = inv ? inv.map((row, i) => row[i]) : subset.map(() => Infinity);
    }
    return {
      dw, jb, jbP: S.chi2PValue(jb, 2), skew, kurt,
      bp, bpP: S.chi2PValue(bp, subset.length), vif, maxVif: Math.max(...vif),
    };
  }

  const MLR = { inferSchema, targetCandidates, run, evaluate, isMissing, toNumber, METHODS };
  if (typeof module !== "undefined" && module.exports) module.exports = MLR;
  else root.MLR = MLR;
})(typeof window !== "undefined" ? window : globalThis);
