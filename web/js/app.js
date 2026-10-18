/* 介面：讀取資料 → 呼叫 mlr.js 的分析流程 → 繪製四個頁面（資料總覽、模型結果、評估、預測）。
   回歸運算都在 mlr.js / stats.js，這裡只負責畫面。 */
(function () {
  "use strict";

  // ------------------------------------------------------------ 題目設定
  const PRESETS = {
    flood: {
      label: "題目 A：洪水機率",
      url: "data/flood_s4e5_sample.csv",
      target: "FloodProbability",
      desc: "依季風強度、都市化、排水系統等 20 項 0–18 的風險評分，預測地區的洪水機率（0–1）。" +
        "資料為 Kaggle Playground S4E5 訓練集（1,117,957 筆）隨機抽樣 20,000 筆；完整資料的分析與 Kaggle 提交結果見 notebook。",
      source: "https://www.kaggle.com/competitions/playground-series-s4e5",
    },
    student: {
      label: "題目 B：學生考試成績",
      url: "data/student_performance_factors.csv",
      target: "Exam_Score",
      desc: "依讀書時數、出席率、家長參與、家庭收入、師資品質等 19 項因素（6 個數值、13 個類別，含少量缺失值）預測期末考成績。",
      source: "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors",
    },
  };
  const PAGES = ["overview", "model", "evaluation", "predict"];

  const $ = (id) => document.getElementById(id);
  const state = {
    rows: null, columns: null, schema: null, meta: null, result: null,
    xSelected: new Set(), page: "overview", dirty: {}, problem: "flood", loadId: 0,
  };
  const charts = {};

  // ------------------------------------------------------------ 格式化
  function fmt(x, digits = 4) {
    if (x === null || x === undefined || Number.isNaN(x)) return "–";
    if (!isFinite(x)) return x > 0 ? "∞" : "-∞";
    const a = Math.abs(x);
    if (a < 1e-12) return "0";   // 浮點誤差（例如 -1.39e-17）一律顯示為 0
    if (a < 1e-4 || a >= 1e7) return x.toExponential(2);
    if (a >= 1000) return x.toLocaleString("en-US", { maximumFractionDigits: 1 });
    return Number(x.toPrecision(digits)).toString();
  }
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const fmtP = (p) => (p < 1e-4 ? "< 0.0001" : p.toFixed(4));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const short = (s, n = 16) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

  // ------------------------------------------------------------ 提示訊息
  function showAlert(msg) {
    $("alert-text").textContent = msg;
    $("alert").classList.remove("hidden");
    window.scrollTo({ top: 0 });
  }
  const hideAlert = () => $("alert").classList.add("hidden");
  const setStatus = (msg) => { $("status").textContent = msg; };

  /** 控制「有資料才顯示」與「有模型才顯示」的區塊（對應 CSS 的 .no-data / .no-model） */
  function setFlags(hasData, hasModel) {
    document.body.classList.toggle("no-data", !hasData);
    document.body.classList.toggle("no-model", !hasModel);
  }

  /** 清空目前的資料與圖表（切換到「自訂資料集」時使用） */
  function clearData() {
    state.loadId++;   // 讓進行中的讀取作廢
    state.rows = null;
    state.result = null;
    for (const id of Object.keys(charts)) { charts[id].destroy(); delete charts[id]; }
    setFlags(false, false);
    hideAlert();
    setStatus("");
    $("dataset-desc").textContent = "請在下方貼上 CSV 網址（已預填 50_Startups 範例）或上傳檔案；載入後才會顯示欄位與分析結果。";
  }

  function setProblemLabel(key) {
    state.problem = key;
    $("problem-label").textContent = key === "custom" ? "自訂資料集" : PRESETS[key].label;
    document.querySelectorAll(".problem-item").forEach((b) => b.classList.toggle("current", b.dataset.problem === key));
    $("custom-card").classList.toggle("hidden", key !== "custom");
  }

  // ------------------------------------------------------------ 讀取資料
  /** GitHub 一般網址自動轉成 raw 網址 */
  function toRawUrl(url) {
    const m = url.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    return m ? `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}` : url;
  }

  function parseCsv(src, meta) {
    hideAlert();
    setStatus("讀取資料中…");
    const loadId = ++state.loadId;   // 只採用最新一次讀取的結果
    Papa.parse(src, {
      download: typeof src === "string",
      header: true,
      skipEmptyLines: "greedy",
      dynamicTyping: false,
      transformHeader: (h) => h.trim(),
      complete: (res) => {
        if (loadId !== state.loadId) return;
        const columns = (res.meta.fields || []).filter((f) => f !== "");
        if (!res.data.length || columns.length < 2) {
          setStatus("");
          showAlert("讀不到表格資料：請確認是逗號分隔、第一列為欄位名稱的 CSV。");
          return;
        }
        setData(res.data, columns, meta);
      },
      error: (err) => {
        if (loadId !== state.loadId) return;
        setStatus("");
        showAlert(`無法讀取資料（${err.message || err}）。網址可能不存在，或該網站不允許跨網域讀取（CORS）；` +
          "請改用 GitHub raw 連結，或下載後用拖曳上傳。");
      },
    });
  }

  function loadUrl(url) {
    url = toRawUrl(url.trim());
    if (!url) return;
    $("csv-url").value = url;
    setQuery({ url });
    parseCsv(url, { label: decodeURIComponent(url.split("/").pop().split("?")[0]), desc: "自訂資料集。", source: url });
  }

  function selectProblem(key) {
    setProblemLabel(key);
    try { FlowbiteInstances.getInstance("Dropdown", "problem-menu").hide(); } catch (e) { /* 尚未初始化時略過 */ }
    setQuery({ problem: key });
    if (key === "custom") {
      // 自訂資料集：先清空，等使用者載入網址或上傳檔案
      clearData();
      showPage("overview");
      return;
    }
    const p = PRESETS[key];
    parseCsv(p.url, { label: p.url.split("/").pop(), desc: p.desc, source: p.source, target: p.target });
  }

  function setData(rows, columns, meta) {
    state.rows = rows;
    state.columns = columns;
    state.meta = meta;
    state.schema = MLR.inferSchema(rows, columns);
    const candidates = MLR.targetCandidates(state.schema);
    if (!candidates.length) {
      showAlert("找不到可當作目標變數的數值欄位。");
      return;
    }
    const target = candidates.includes(meta.target) ? meta.target : candidates[candidates.length - 1];
    $("target").innerHTML = candidates.map((c) => `<option${c === target ? " selected" : ""}>${esc(c)}</option>`).join("");
    resetX();
    state.desc = describe(rows, state.schema);
    $("k").value = "";
    setFlags(true, false);
    runModel();
  }

  /** 候選特徵預設為「除了目標以外所有可用欄位」 */
  function resetX() {
    const target = $("target").value;
    state.xSelected = new Set(state.schema.filter((c) => !c.excluded && c.name !== target).map((c) => c.name));
  }

  // ------------------------------------------------------------ 敘述統計與相關係數
  function describe(rows, schema) {
    return schema.filter((c) => c.kind === "numeric" && !c.excluded).map((c) => {
      const xs = [];
      for (const r of rows) { const v = MLR.toNumber(r[c.name]); if (v !== null) xs.push(v); }
      const n = xs.length, mean = xs.reduce((a, b) => a + b, 0) / n;
      const sd = Math.sqrt(xs.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
      let min = Infinity, max = -Infinity;
      for (const v of xs) { if (v < min) min = v; if (v > max) max = v; }
      return { name: c.name, n, mean, sd, min, max };
    });
  }

  function correlation(rows, cols) {
    const X = cols.map((c) => rows.map((r) => MLR.toNumber(r[c])));
    return cols.map((_, a) => cols.map((__, b) => {
      let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
      for (let i = 0; i < rows.length; i++) {
        const x = X[a][i], y = X[b][i];
        if (x === null || y === null) continue;
        n++; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
      }
      const cov = sab - (sa * sb) / n, va = saa - (sa * sa) / n, vb = sbb - (sb * sb) / n;
      return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
    }));
  }

  // ------------------------------------------------------------ 建模
  function runModel() {
    if (!state.rows) return;
    hideAlert();
    const features = [...state.xSelected];
    if (!features.length) {
      state.result = null;
      setFlags(true, false);
      renderOverview();
      state.dirty.overview = true;
      renderVisibleCharts();
      showAlert("請至少選擇一個候選特徵。");
      return;
    }
    setStatus("計算中…");
    // 先讓狀態文字顯示，再進行同步計算
    setTimeout(() => {
      try {
        const method = $("method").value;
        const res = MLR.run(state.rows, state.schema, $("target").value, {
          method: method === "none" ? "rfe" : method,
          k: method === "none" ? 1e9 : $("k").value ? Number($("k").value) : null,
          conf: Number($("conf").value),
          features,
        });
        res.noSelection = method === "none";
        state.result = res;
        setFlags(true, true);   // 先顯示區塊，圖表才量得到寬度
        updateKOptions(res);
        renderOverview();
        renderModel(res);
        renderEvaluation(res);
        renderPredictor(res);
        for (const p of PAGES) state.dirty[p] = true;
        renderVisibleCharts();
        const sampled = res.sampled ? `；資料超過 50,000 筆，已隨機抽樣 ${res.n.toLocaleString()} 筆` : "";
        const dropped = res.droppedTarget ? `；${res.droppedTarget} 筆目標值缺失已略過` : "";
        setStatus(`${state.meta.label} · 訓練 ${res.nTrain.toLocaleString()} 筆、測試 ${res.nTest.toLocaleString()} 筆 · 計算 ${res.ms} ms${sampled}${dropped}`);
      } catch (e) {
        console.error(e);
        state.result = null;
        setFlags(true, false);
        renderOverview();
        setStatus("");
        showAlert(`無法建立模型：${e.message}`);
      }
    }, 20);
  }

  function updateKOptions(res) {
    const sel = $("k"), cur = sel.value;
    sel.innerHTML = '<option value="">自動</option>' +
      Array.from({ length: res.p }, (_, i) => `<option value="${i + 1}">${i + 1}</option>`).join("");
    sel.value = cur && Number(cur) <= res.p ? cur : "";
    sel.disabled = res.noSelection;
  }

  // ------------------------------------------------------------ 一、資料總覽
  function kpi(label, value, sub = "", tone = "") {
    return `<div class="kpi"><div class="kpi-bar ${tone}"></div><div class="kpi-label">${label}</div>` +
      `<div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
  }

  function renderOverview() {
    const { meta, rows, columns, schema } = state;
    const target = $("target").value;
    $("dataset-desc").innerHTML = esc(meta.desc) +
      (meta.source ? ` <a class="text-ai underline underline-offset-4" href="${esc(meta.source)}" target="_blank" rel="noopener">資料來源</a>` : "");

    const usable = schema.filter((c) => !c.excluded && c.name !== target);
    const excluded = schema.filter((c) => c.excluded);
    const missing = schema.reduce((s, c) => s + c.missing, 0);
    $("kpi-data").innerHTML = [
      kpi("資料筆數", rows.length.toLocaleString(), esc(meta.label)),
      kpi("欄位數", columns.length, `數值 ${schema.filter((c) => c.kind === "numeric").length} · 類別 ${schema.filter((c) => c.kind !== "numeric").length}`, "matcha"),
      kpi("缺失值", missing.toLocaleString(), missing ? `分布在 ${schema.filter((c) => c.missing).length} 個欄位` : "無缺失", "sakura"),
      kpi("可用特徵", usable.length, excluded.length ? `自動排除 ${excluded.length} 欄` : "無排除欄位"),
    ].join("");

    // 候選特徵多選
    $("x-list").innerHTML = usable.map((c) =>
      `<label class="chip"><input type="checkbox" value="${esc(c.name)}"${state.xSelected.has(c.name) ? " checked" : ""}>` +
      `${esc(c.name)}<span class="kind">${c.kind === "numeric" ? "數值" : "類別"}</span></label>`).join("");

    // 資料表（前 100 筆、最多 40 欄）
    const head = columns.slice(0, 40);
    $("table-sub").textContent = `前 ${Math.min(100, rows.length)} 筆${columns.length > 40 ? `、前 40 欄` : ""}`;
    $("data-table").innerHTML = `<thead><tr>${head.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      rows.slice(0, 100).map((r) => `<tr>${head.map((c) => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("") + "</tbody>";

    // 敘述統計
    $("desc-table").innerHTML = "<thead><tr><th>欄位</th><th class=num>筆數</th><th class=num>平均數</th><th class=num>標準差</th><th class=num>最小值</th><th class=num>最大值</th></tr></thead><tbody>" +
      state.desc.map((d) => `<tr><td>${esc(d.name)}${d.name === target ? ' <span class="tag info">y</span>' : ""}</td><td class=num>${d.n.toLocaleString()}</td>` +
        `<td class=num>${fmt(d.mean)}</td><td class=num>${fmt(d.sd)}</td><td class=num>${fmt(d.min)}</td><td class=num>${fmt(d.max)}</td></tr>`).join("") + "</tbody>";

    // 欄位檢查
    $("schema-table").innerHTML = "<thead><tr><th>欄位</th><th>類型</th><th class=num>缺失</th><th class=num>唯一值</th><th>用途</th></tr></thead><tbody>" +
      schema.map((c) => {
        const role = c.name === target ? '<span class="tag info">目標 y</span>'
          : c.excluded ? `<span class="tag warn">排除：${esc(c.excluded)}</span>`
          : state.xSelected.has(c.name) ? '<span class="tag good">候選 X</span>' : '<span class="tag plain">未選</span>';
        return `<tr><td>${esc(c.name)}</td><td>${c.kind === "numeric" ? "數值" : "類別"}</td><td class=num>${c.missing}</td>` +
          `<td class=num>${c.unique > 100000 ? "> 100,000" : c.unique.toLocaleString()}</td><td>${role}</td></tr>`;
      }).join("") + "</tbody>";
    state.dirty.overview = true;
  }

  // ------------------------------------------------------------ 二、模型結果
  function methodName(res) {
    return res.noSelection ? "不篩選（全部使用）" : res.methods[res.chosen.method];
  }

  function renderModel(res) {
    const m = res.metrics;
    $("kpi-model").innerHTML = [
      kpi("R²（測試集）", fmt(m.R2, 4), `訓練 ${res.nTrain.toLocaleString()} · 測試 ${res.nTest.toLocaleString()}`),
      kpi("Adjusted R²", fmt(m.AdjR2, 4), `k = ${res.chosen.k} 個特徵`, "matcha"),
      kpi("RMSE", fmt(m.RMSE), `MSE ${fmt(m.MSE)}`, "sakura"),
      kpi("截距", fmt(res.coefs[0].coef, 5), "原始單位"),
    ].join("");

    const dropped = res.encoded.filter((c) => !res.selected.includes(c));
    $("selection-note").innerHTML = res.noSelection
      ? `不做特徵選擇，使用全部 ${res.p} 個欄位（one-hot 後）。`
      : `選用 <b>${esc(methodName(res))}</b> 的前 <b>${res.chosen.k}</b> / ${res.p} 個特徵（自動規則：CV RMSE 在最佳值 0.5% 以內的最少特徵）。` +
        (dropped.length ? `未選入：${dropped.map(esc).join("、")}。` : "全部特徵皆保留。");

    const confPct = `${Math.round(res.conf * 100)}%`;
    const stars = (p) => (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "");
    const sorted = [res.coefs[0], ...res.coefs.slice(1).sort((a, b) => Math.abs(b.t) - Math.abs(a.t))];
    $("coef-table").innerHTML = `<thead><tr><th>變數</th><th class=num>係數</th><th class=num>標準誤</th><th class=num>t 值</th><th class=num>p 值</th>` +
      `<th class=num>${confPct} CI 下界</th><th class=num>${confPct} CI 上界</th></tr></thead><tbody>` +
      sorted.map((c) => `<tr><td>${esc(c.name)}</td><td class=num>${fmt(c.coef, 5)}</td><td class=num>${fmt(c.se, 4)}</td>` +
        `<td class=num>${fmt(c.t, 4)}</td><td class=num>${fmtP(c.p)} ${stars(c.p)}</td>` +
        `<td class=num>${fmt(c.lo, 5)}</td><td class=num>${fmt(c.hi, 5)}</td></tr>`).join("") + "</tbody>";

    const ms = Object.keys(res.methods);
    $("rank-table").innerHTML = `<thead><tr><th>排名</th>${ms.map((x) => `<th>${esc(res.methods[x])}</th>`).join("")}</tr></thead><tbody>` +
      res.ranks[ms[0]].map((_, i) => `<tr${i >= res.chosen.k ? ' class="dim"' : ""}><td>${i + 1}</td>` +
        ms.map((x) => `<td>${esc(res.ranks[x][i])}</td>`).join("") + "</tr>").join("") + "</tbody>";
  }

  // ------------------------------------------------------------ 三、評估
  function renderEvaluation(res) {
    const m = res.metrics, d = res.diag, confPct = `${Math.round(res.conf * 100)}%`;
    $("kpi-eval").innerHTML = [
      kpi("MAE", fmt(m.MAE), "平均絕對誤差"),
      kpi(`${confPct} PI 覆蓋率`, pct(m.coverage), `名目值 ${confPct}`, "matcha"),
      kpi("CV RMSE（訓練）", fmt(res.cvRmseSelected), `全部特徵 ${fmt(res.cvRmseAll)}`, "sakura"),
      kpi("測試集 R²", fmt(m.R2, 4), `RMSE ${fmt(m.RMSE)}`),
    ].join("");
    $("pred-sub").textContent = `測試集隨機 ${Math.min(120, res.nTest)} 筆，依預測值排序`;
    $("pred-note").textContent = `深色帶 = ${confPct} 信賴區間；淺色帶 = ${confPct} 預測區間。` +
      `測試集 ${res.nTest.toLocaleString()} 筆中有 ${pct(m.coverage)} 落在預測區間內（名目值 ${confPct}）。`;

    const tag = (ok, warn) => (ok ? '<span class="tag good">通過</span>' : warn ? '<span class="tag warn">注意</span>' : '<span class="tag warn">不通過</span>');
    const big = res.nTrain > 5000 ? "；樣本數大時微小偏差也會顯著" : "";
    const rows = [
      ["一、線性 Linearity", "殘差圖", "—", "殘差隨機分布於 0 兩側", '<span class="tag plain">看圖</span>'],
      // 資料是隨機切分的橫斷面資料，列的順序沒有時間意義，DW 只能當參考
      ["二、獨立性 Independence", "Durbin-Watson", fmt(d.dw, 4), "列的順序無時間意義時僅供參考；獨立性由隨機抽樣保證", '<span class="tag plain">參考</span>'],
      ["三、同質變異 Homoscedasticity", "Breusch-Pagan p 值", fmtP(d.bpP), `> 0.05${big}`, tag(d.bpP > 0.05, true)],
      ["四、常態性 Normality", `Jarque-Bera p 值（偏態 ${fmt(d.skew, 3)}、峰度 ${fmt(d.kurt, 3)}）`, fmtP(d.jbP), `> 0.05${big}`, tag(d.jbP > 0.05, true)],
      ["五、無多重共線性", "最大 VIF", fmt(d.maxVif, 4), "< 5（< 10 尚可）", tag(d.maxVif < 5, d.maxVif < 10)],
    ];
    $("assump-table").innerHTML = "<thead><tr><th>假設</th><th>檢查方式</th><th class=num>數值</th><th>判斷標準</th><th>結果</th></tr></thead><tbody>" +
      rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class=num>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`).join("") + "</tbody>";
  }

  // ------------------------------------------------------------ 四、預測
  function renderPredictor(res) {
    const used = new Set(res.selected.map((s) => res.features.find((f) => s === f || s.startsWith(`${f} = `))));
    $("predict-form").innerHTML = res.inputs.map((inp, i) => {
      const control = inp.kind === "numeric"
        ? `<input id="inp-${i}" class="input" type="number" step="any" data-name="${esc(inp.name)}" value="${Number(inp.default.toPrecision(6))}">`
        : `<select id="inp-${i}" class="input" data-name="${esc(inp.name)}">${inp.levels.map((l) => `<option${l === inp.default ? " selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
      return `<label for="inp-${i}" class="block${used.has(inp.name) ? "" : " opacity-50"}"><span class="field-label">${esc(inp.name)}</span>${control}</label>`;
    }).join("");
    updatePrediction();
  }

  function updatePrediction() {
    const res = state.result;
    if (!res) return;
    const raw = {};
    for (const el of $("predict-form").querySelectorAll("[data-name]")) raw[el.dataset.name] = el.value;
    const p = res.predictRow(raw), confPct = `${Math.round(res.conf * 100)}%`;
    // 區間示意條：以預測區間為範圍，左右各留 10%
    const span = p.piHi - p.piLo || 1, lo = p.piLo - span * 0.1, w = span * 1.2;
    const pos = (v) => `${((v - lo) / w) * 100}%`;
    $("predict-out").innerHTML =
      `<p class="eyebrow">預測 ${esc(res.target)}</p>` +
      `<p class="font-serif text-4xl mt-3 tabular-nums break-all">${fmt(p.pred, 5)}</p>` +
      `<div class="range-track" aria-hidden="true">` +
      `<div class="range-pi" style="left:${pos(p.piLo)};width:${(span / w) * 100}%"></div>` +
      `<div class="range-ci" style="left:${pos(p.ciLo)};width:${Math.max(((p.ciHi - p.ciLo) / w) * 100, 0.6)}%"></div>` +
      `<div class="range-dot" style="left:${pos(p.pred)}"></div></div>` +
      `<dl class="mt-5 space-y-3 text-sm">` +
      `<div><dt class="text-xs text-muted">${confPct} 信賴區間（平均值）</dt><dd class="tabular-nums">${fmt(p.ciLo, 5)} ～ ${fmt(p.ciHi, 5)}</dd></div>` +
      `<div><dt class="text-xs text-muted">${confPct} 預測區間（單一樣本）</dt><dd class="tabular-nums">${fmt(p.piLo, 5)} ～ ${fmt(p.piHi, 5)}</dd></div>` +
      `</dl><p class="note mt-5">模型：${esc(methodName(res))}，k = ${res.chosen.k}；測試集 R² ${fmt(res.metrics.R2, 4)}。</p>`;
  }

  // ------------------------------------------------------------ 圖表（ApexCharts）
  const isDark = () => document.documentElement.classList.contains("dark");
  /** 讀取 CSS 色票變數 → rgb() 字串 */
  function color(name, alpha = 1) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim().split(/\s+/).join(",");
    return alpha === 1 ? `rgb(${v})` : `rgba(${v},${alpha})`;
  }
  /** 兩個色票依比例混色（給熱力圖的色階用） */
  function mix(a, b, t) {
    const pa = getComputedStyle(document.documentElement).getPropertyValue(a).trim().split(/\s+/).map(Number);
    const pb = getComputedStyle(document.documentElement).getPropertyValue(b).trim().split(/\s+/).map(Number);
    return `rgb(${pa.map((v, i) => Math.round(v * t + pb[i] * (1 - t))).join(",")})`;
  }

  function base(height) {
    return {
      chart: {
        height, background: "transparent", foreColor: color("--muted"), fontFamily: '"Noto Sans TC", sans-serif',
        toolbar: { show: false }, zoom: { enabled: false }, animations: { enabled: false }, parentHeightOffset: 0,
      },
      theme: { mode: isDark() ? "dark" : "light" },
      tooltip: { theme: isDark() ? "dark" : "light" },
      grid: { borderColor: color("--line"), strokeDashArray: 3 },
      legend: { position: "top", horizontalAlign: "left", fontSize: "12px", markers: { radius: 2 }, itemMargin: { horizontal: 10 } },
      dataLabels: { enabled: false },
    };
  }

  function draw(id, options) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new ApexCharts($(id), options);
    charts[id].render();
  }

  function sampleIdx(n, max, seed) {
    return n <= max ? [...Array(n).keys()] : Stats.shuffledIndex(n, seed).slice(0, max);
  }
  const axisTitle = (text) => ({ text, style: { fontWeight: 400, color: color("--muted") } });

  /** 座標軸取整：回傳以 1、2、2.5、5 × 10ⁿ 為間距的 {min, max, tickAmount} */
  function niceAxis(lo, hi, ticks = 6) {
    if (!(hi > lo)) { hi = lo + 1; lo -= 1; }
    const raw = (hi - lo) / ticks, exp = 10 ** Math.floor(Math.log10(raw)), f = raw / exp;
    const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * exp;
    const min = Math.floor(lo / step) * step, max = Math.ceil(hi / step) * step;
    return { min, max, tickAmount: Math.round((max - min) / step) };
  }

  function drawCorr() {
    const target = $("target").value;
    let cols = state.desc.map((d) => d.name);
    let mat = correlation(state.rows, cols);
    if (cols.length > 16) {
      // 欄位太多時只保留與目標相關最高的 15 個
      const ti = cols.indexOf(target);
      const keep = cols.map((_, i) => i).filter((i) => i !== ti).sort((a, b) => Math.abs(mat[ti][b]) - Math.abs(mat[ti][a])).slice(0, 15);
      const idx = [ti, ...keep];
      cols = idx.map((i) => cols[i]);
      mat = idx.map((a) => idx.map((b) => mat[a][b]));
      $("corr-sub").textContent = "數值欄位（目標與相關最高的 15 個）";
    } else {
      $("corr-sub").textContent = "數值欄位 Pearson r";
    }
    const n = cols.length;
    // 每格至少 36px 以容納數值；容器不夠寬時圖表加寬、由外層左右捲動
    const width = Math.max($("corr-scroll").clientWidth, 130 + n * 36);
    const o = base(Math.max(280, n * 32 + 110));
    o.chart.width = width;
    Object.assign(o, {
      chart: { ...o.chart, type: "heatmap" },
      // ApexCharts 熱力圖由下往上畫，反轉後第一個欄位在最上方
      series: cols.map((c, a) => ({ name: short(c), data: cols.map((c2, b) => ({ x: short(c2), y: Number(mat[a][b].toFixed(3)) })) })).reverse(),
      plotOptions: {
        heatmap: {
          radius: 0, enableShades: false,
          colorScale: {
            ranges: [
              { from: -1, to: -0.6, color: mix("--beni", "--card", 1) },
              { from: -0.6, to: -0.3, color: mix("--sakura", "--card", 0.9) },
              { from: -0.3, to: -0.1, color: mix("--sakura", "--card", 0.45) },
              { from: -0.1, to: 0.1, color: color("--soft") },
              { from: 0.1, to: 0.3, color: mix("--ai", "--card", 0.35) },
              { from: 0.3, to: 0.6, color: mix("--ai", "--card", 0.65) },
              { from: 0.6, to: 1, color: color("--ai") },
            ],
          },
        },
      },
      stroke: { width: 1, colors: [color("--card")] },
      dataLabels: {
        enabled: true,
        formatter: (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2),   // 避免出現 -0.00
        style: { fontSize: n > 12 ? "9px" : "10px", fontWeight: 400, colors: [color("--ink")] },
      },
      legend: { show: false },
      xaxis: { labels: { rotate: -45, style: { fontSize: "10px" } }, tooltip: { enabled: false } },
      yaxis: { labels: { style: { fontSize: "10px" } } },
      tooltip: { ...o.tooltip, y: { formatter: (v) => `r = ${v.toFixed(3)}` } },
    });
    draw("chart-corr", o);
  }

  function drawCoef() {
    const res = state.result;
    const items = [...res.stdCoefs].sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));
    const o = base(Math.max(220, items.length * 26 + 70));
    Object.assign(o, {
      chart: { ...o.chart, type: "bar" },
      series: [{ name: "標準化係數", data: items.map((d) => Number(d.coef.toPrecision(5))) }],
      colors: items.map((d) => (d.coef >= 0 ? color("--ai") : color("--beni"))),
      plotOptions: { bar: { horizontal: true, distributed: true, barHeight: "55%", borderRadius: 1 } },
      legend: { show: false },
      xaxis: { categories: items.map((d) => short(d.name, 24)), labels: { formatter: (v) => fmt(Number(v), 3) } },
      yaxis: { labels: { style: { fontSize: "11px" }, maxWidth: 220 } },
      grid: { ...o.grid, yaxis: { lines: { show: false } } },
    });
    draw("chart-coef", o);
  }

  function drawK() {
    const res = state.result;
    const colors = { forward: color("--ai"), rfe: color("--matcha"), kbest: color("--beni") };
    const ms = Object.keys(res.methods);
    const o = base(320);
    Object.assign(o, {
      chart: { ...o.chart, type: "line" },
      series: ms.map((m) => ({ name: res.methods[m], data: res.curves[m].map((v, i) => ({ x: i + 1, y: v })) })),
      colors: ms.map((m) => colors[m]),
      stroke: { width: 1.6, curve: "straight" },
      markers: { size: 2.5, strokeWidth: 0 },
      xaxis: { type: "numeric", tickAmount: Math.min(res.p - 1, 10) || 1, title: axisTitle("特徵數 k"), labels: { formatter: (v) => Math.round(v) } },
      yaxis: { title: axisTitle("CV RMSE"), labels: { formatter: (v) => fmt(v, 3) } },
      annotations: {
        points: [{
          x: res.chosen.k, y: res.cvRmseSelected,
          marker: { size: 6, fillColor: color("--card"), strokeColor: color("--ink"), strokeWidth: 1.5 },
          label: { text: `選用 k = ${res.chosen.k}`, borderColor: color("--line"), style: { background: color("--card"), color: color("--ink"), fontSize: "11px" } },
        }],
      },
      tooltip: { ...o.tooltip, shared: true, x: { formatter: (v) => `k = ${v}` }, y: { formatter: (v) => fmt(v, 5) } },
    });
    draw("chart-k", o);
  }

  function drawPred() {
    const res = state.result, confPct = `${Math.round(res.conf * 100)}%`;
    const pts = sampleIdx(res.testPred.length, 120, 7).map((i) => res.testPred[i]).sort((a, b) => a.pred - b.pred);
    // y 軸範圍自己算（rangeArea 預設會從 0 開始，色帶會被壓扁）
    const yAxis = niceAxis(Math.min(...pts.map((d) => Math.min(d.piLo, d.actual))), Math.max(...pts.map((d) => Math.max(d.piHi, d.actual))));
    const o = base(380);
    Object.assign(o, {
      chart: { ...o.chart, type: "rangeArea" },
      series: [
        { type: "rangeArea", name: `${confPct} 預測區間`, data: pts.map((d, i) => ({ x: i + 1, y: [d.piLo, d.piHi] })) },
        { type: "rangeArea", name: `${confPct} 信賴區間`, data: pts.map((d, i) => ({ x: i + 1, y: [d.ciLo, d.ciHi] })) },
        { type: "line", name: "預測值", data: pts.map((d, i) => ({ x: i + 1, y: d.pred })) },
        { type: "line", name: "實際值", data: pts.map((d, i) => ({ x: i + 1, y: d.actual })) },
      ],
      // 色帶用帶透明度的顏色（rangeArea 不吃 fill.opacity 陣列）
      colors: [color("--ai", 0.18), color("--ai", 0.5), color("--ai"), color("--beni")],
      fill: { opacity: 1 },
      stroke: { width: [0, 0, 2, 0], curve: "straight" },
      markers: { size: [0, 0, 0, 3.5], strokeWidth: 0, hover: { sizeOffset: 2 } },
      xaxis: { type: "numeric", tickAmount: 6, title: axisTitle("測試樣本（依預測值排序）"), labels: { formatter: (v) => Math.round(v) } },
      yaxis: { ...yAxis, title: axisTitle(res.target), labels: { formatter: (v) => fmt(v, 4) } },
      tooltip: { ...o.tooltip, shared: true, x: { show: false }, y: { formatter: (v) => (Array.isArray(v) ? v.map((x) => fmt(x, 5)).join(" ～ ") : fmt(v, 5)) } },
    });
    draw("chart-pred", o);
  }

  function drawScatter() {
    const res = state.result;
    const sc = sampleIdx(res.testPred.length, 1500, 11).map((i) => res.testPred[i]);
    const lo = Math.min(...sc.map((d) => Math.min(d.actual, d.pred))), hi = Math.max(...sc.map((d) => Math.max(d.actual, d.pred)));
    const ax = niceAxis(lo, hi);
    const o = base(340);
    Object.assign(o, {
      chart: { ...o.chart, type: "line" },
      series: [
        { type: "scatter", name: "測試樣本", data: sc.map((d) => [d.actual, d.pred]) },
        { type: "line", name: "y = x", data: [[lo, lo], [hi, hi]] },
      ],
      colors: [color("--ai"), color("--beni")],
      stroke: { width: [0, 1.2] },
      markers: { size: [3, 0], strokeWidth: 0, fillOpacity: 0.55 },
      xaxis: { type: "numeric", ...ax, title: axisTitle(`實際 ${res.target}`), labels: { formatter: (v) => fmt(v, 4) } },
      yaxis: { ...ax, title: axisTitle(`預測 ${res.target}`), labels: { formatter: (v) => fmt(v, 4) } },
      tooltip: { ...o.tooltip, x: { formatter: (v) => `實際 ${fmt(v, 5)}` }, y: { formatter: (v) => fmt(v, 5) } },
    });
    draw("chart-scatter", o);
  }

  function drawResid() {
    const res = state.result;
    const ri = sampleIdx(res.train.fitted.length, 1500, 13);
    const fx = ri.map((i) => res.train.fitted[i]);
    const rMax = Math.max(...ri.map((i) => Math.abs(res.train.resid[i])));
    const o = base(340);
    Object.assign(o, {
      chart: { ...o.chart, type: "scatter" },
      series: [{ name: "訓練樣本", data: ri.map((i) => [res.train.fitted[i], res.train.resid[i]]) }],
      colors: [color("--beni")],
      markers: { size: 3, strokeWidth: 0, fillOpacity: 0.55 },
      legend: { show: false },
      xaxis: { type: "numeric", ...niceAxis(Math.min(...fx), Math.max(...fx)), title: axisTitle("預測值"), labels: { formatter: (v) => fmt(v, 4) } },
      yaxis: { ...niceAxis(-rMax, rMax), title: axisTitle("殘差"), labels: { formatter: (v) => fmt(v, 3) } },
      annotations: { yaxis: [{ y: 0, borderColor: color("--muted"), strokeDashArray: 0 }] },
      tooltip: { ...o.tooltip, x: { formatter: (v) => `預測 ${fmt(v, 5)}` }, y: { formatter: (v) => fmt(v, 5) } },
    });
    draw("chart-resid", o);
  }

  const CHARTS = { overview: [drawCorr], model: [drawCoef, drawK], evaluation: [drawPred, drawScatter, drawResid], predict: [] };

  /** 隱藏中的頁面無法量測寬度，所以圖表只在頁面顯示時才繪製 */
  function renderVisibleCharts() {
    const p = state.page;
    // 總覽頁的熱力圖只需要資料；其他頁需要模型結果
    if (!state.dirty[p] || (p === "overview" ? !state.rows : !state.result)) return;
    state.dirty[p] = false;
    for (const f of CHARTS[p]) f();
  }

  // ------------------------------------------------------------ 頁面切換（單頁應用）
  function showPage(page) {
    if (!PAGES.includes(page)) page = "overview";
    state.page = page;
    for (const p of PAGES) $(`page-${p}`).classList.toggle("hidden", p !== page);
    document.querySelectorAll(".nav-link").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
    if (location.hash !== `#${page}`) history.replaceState(null, "", `${location.search}#${page}`);
    window.scrollTo({ top: 0 });
    renderVisibleCharts();
    // 手機版：切換後收起側邊欄
    if (window.innerWidth < 640) {
      try { FlowbiteInstances.getInstance("Drawer", "sidebar").hide(); } catch (e) { /* 略過 */ }
    }
  }

  function setQuery(params) {
    const q = new URLSearchParams(params).toString();
    history.replaceState(null, "", `${q ? `?${q}` : location.pathname}#${state.page}`);
  }

  // ------------------------------------------------------------ 日／夜間模式
  function applyTheme(dark, save) {
    document.documentElement.classList.toggle("dark", dark);
    if (save) { try { localStorage.setItem("mlr-theme", dark ? "dark" : "light"); } catch (e) { /* 無法儲存時略過 */ } }
    for (const p of PAGES) state.dirty[p] = true;
    renderVisibleCharts();
  }
  $("theme-toggle").addEventListener("click", () => applyTheme(!isDark(), true));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let saved = null;
    try { saved = localStorage.getItem("mlr-theme"); } catch (err) { /* 略過 */ }
    if (!saved) applyTheme(e.matches, false);
  });

  // ------------------------------------------------------------ 事件
  document.querySelectorAll(".problem-item").forEach((b) => b.addEventListener("click", () => selectProblem(b.dataset.problem)));
  document.querySelectorAll(".nav-link").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showPage(a.dataset.page); }));
  $("alert-close").addEventListener("click", hideAlert);

  $("url-form").addEventListener("submit", (e) => {
    e.preventDefault();
    loadUrl($("csv-url").value);
  });

  function loadFile(file) {
    if (!file) return;
    setQuery({ problem: "custom" });
    parseCsv(file, { label: file.name, desc: "上傳的資料集。" });
  }
  $("csv-file").addEventListener("change", (e) => { loadFile(e.target.files[0]); e.target.value = ""; });
  // 拖曳上傳
  const dz = $("dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("bg-soft"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("bg-soft"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("bg-soft"); loadFile(e.dataTransfer.files[0]); });

  $("target").addEventListener("change", () => { resetX(); $("k").value = ""; runModel(); });
  $("x-list").addEventListener("change", (e) => {
    if (e.target.type !== "checkbox") return;
    e.target.checked ? state.xSelected.add(e.target.value) : state.xSelected.delete(e.target.value);
    $("k").value = "";
    runModel();
  });
  $("x-all").addEventListener("click", () => { resetX(); $("k").value = ""; runModel(); });
  $("x-none").addEventListener("click", () => { state.xSelected.clear(); runModel(); });
  for (const id of ["method", "k", "conf"]) $(id).addEventListener("change", runModel);
  $("predict-form").addEventListener("input", updatePrediction);
  $("predict-form").addEventListener("submit", (e) => e.preventDefault());

  // 熱力圖寬度是繪製時計算的，視窗大小改變後重畫（其他圖表由 ApexCharts 自動調整）
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      state.dirty.overview = true;
      renderVisibleCharts();
    }, 200);
  });

  // Tailwind（Play CDN）的樣式可能晚於第一次繪圖才套用；整頁載入後重畫一次，確保圖表寬度正確
  window.addEventListener("load", () => {
    for (const p of PAGES) state.dirty[p] = true;
    renderVisibleCharts();
  });

  // ------------------------------------------------------------ 啟動
  const q = new URLSearchParams(location.search);
  showPage(location.hash.slice(1) || "overview");
  if (q.get("url")) {
    // 網址帶 ?url=：直接以自訂資料集載入
    setProblemLabel("custom");
    clearData();
    loadUrl(q.get("url"));
  } else {
    const key = q.get("problem");
    selectProblem(key === "custom" || PRESETS[key] ? key : "flood");
  }
})();
