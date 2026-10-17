// Run the browser pipeline in Node and dump what Python needs to re-check it.
// usage: node run_mlr.js <csv> <target> <method> <k|auto> <out.json>
const fs = require("fs");
const MLR = require("../js/mlr.js");
const Stats = require("../js/stats.js");

const [csv, target, method, kArg, out] = process.argv.slice(2);
const lines = fs.readFileSync(csv, "utf8").trim().split(/\r?\n/);
const columns = lines[0].split(",");
const rows = lines.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [columns[i], v])));

const schema = MLR.inferSchema(rows, columns);
const res = MLR.run(rows, schema, target, { method, k: kArg === "auto" ? null : Number(kArg) });
const order = Stats.shuffledIndex(res.n, 42);
const nTest = res.nTest;

fs.writeFileSync(out, JSON.stringify({
  schema, chosen: res.chosen, selected: res.selected, encoded: res.encoded,
  testIdx: order.slice(0, nTest), trainIdx: order.slice(nTest),
  coefs: res.coefs, metrics: res.metrics, diag: res.diag, s2: res.s2,
  testPred: res.testPred.slice(0, 50), curves: res.curves, ms: res.ms,
}));
console.log(`${csv.split(/[\\/]/).pop()}: method=${res.chosen.method} k=${res.chosen.k} R2=${res.metrics.R2.toFixed(5)} ` +
  `coverage=${(res.metrics.coverage * 100).toFixed(1)}% excluded=${schema.filter((c) => c.excluded).map((c) => c.name + ":" + c.excluded).join("|") || "-"} (${res.ms} ms)`);
