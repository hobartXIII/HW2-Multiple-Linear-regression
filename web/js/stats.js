/* Statistical distributions and small dense linear algebra (no dependencies).
   Works as a browser global (window.Stats) and as a Node module (for tests). */
(function (root) {
  "use strict";

  // ---------- special functions ----------
  function logGamma(x) {
    // Lanczos approximation (g = 7, n = 9)
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function betacf(a, b, x) {
    // continued fraction for the incomplete beta function (Numerical Recipes)
    const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  /** Regularized incomplete beta I_x(a, b). */
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
  }

  /** Regularized lower incomplete gamma P(a, x). */
  function gammaP(a, x) {
    if (x <= 0) return 0;
    const gln = logGamma(a);
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 500; n++) {
        ap += 1; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 3e-16) break;
      }
      return sum * Math.exp(-x + a * Math.log(x) - gln);
    }
    let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (let i = 1; i < 500; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-16) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - gln) * h;
  }

  // ---------- distributions ----------
  /** Two-sided p-value of a t statistic. */
  function tPValue(t, df) {
    if (!isFinite(t)) return 0;
    return ibeta(df / (df + t * t), df / 2, 0.5);
  }

  function tCdf(t, df) {
    const p = 0.5 * ibeta(df / (df + t * t), df / 2, 0.5);
    return t >= 0 ? 1 - p : p;
  }

  function normalQuantile(p) {
    // Acklam's rational approximation (|error| < 1.2e-9)
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    let q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - pl) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5; r = q * q;
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  /** Quantile of Student's t (bisection on the CDF, started from the normal quantile). */
  function tQuantile(p, df) {
    if (df > 1e6) return normalQuantile(p);
    let lo = -1e3, hi = 1e3;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tCdf(mid, df) < p) lo = mid; else hi = mid;
      if (hi - lo < 1e-12) break;
    }
    return (lo + hi) / 2;
  }

  /** Upper-tail p-value of a chi-square statistic. */
  function chi2PValue(x, df) {
    return 1 - gammaP(df / 2, x / 2);
  }

  // ---------- linear algebra ----------
  /** Inverse of a symmetric positive (semi)definite matrix by Gauss-Jordan with partial pivoting.
      Returns null when the matrix is numerically singular. */
  function invert(A) {
    const n = A.length;
    const M = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j];
      r[n + i] = 1;
      return r;
    });
    let scale = 0;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(A[i][i]));
    const tol = Math.max(scale, 1) * 1e-12;
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      if (Math.abs(M[piv][col]) < tol) return null;
      [M[col], M[piv]] = [M[piv], M[col]];
      const pv = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= pv;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return M.map((r) => Array.from(r.subarray(n)));
  }

  function matVec(A, v) {
    return A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
  }

  function quadForm(A, v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) {
      let t = 0;
      for (let j = 0; j < v.length; j++) t += A[i][j] * v[j];
      s += v[i] * t;
    }
    return s;
  }

  /** Small seeded PRNG (mulberry32) so splits and samples are reproducible. */
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffledIndex(n, seed) {
    const r = rng(seed), idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  const Stats = { logGamma, ibeta, gammaP, tPValue, tCdf, tQuantile, normalQuantile, chi2PValue,
    invert, matVec, quadForm, rng, shuffledIndex };
  if (typeof module !== "undefined" && module.exports) module.exports = Stats;
  else root.Stats = Stats;
})(typeof window !== "undefined" ? window : globalThis);
