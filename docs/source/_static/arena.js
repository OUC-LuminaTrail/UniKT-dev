// UniKT model-zoo leaderboard.
// Framework-free rewrite of the original VuePress client (docs/src/.vuepress/client.ts):
// no VuePress/router deps, runs on DOMContentLoaded, follows the system colour scheme
// for the Tabulator theme. Data is fetched from _static/arena/leaderboard.json.
(function () {
  "use strict";

  var TABULATOR_JS = "https://unpkg.com/tabulator-tables@6.5.0/dist/js/tabulator.min.js";
  var TABULATOR_LIGHT = "https://unpkg.com/tabulator-tables@6.5.0/dist/css/tabulator_site.min.css";
  var TABULATOR_DARK = "https://unpkg.com/tabulator-tables@6.5.0/dist/css/tabulator_midnight.min.css";

  function loadScript(src, onload) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onload;
    document.head.appendChild(s);
  }

  function applyTabulatorTheme() {
    var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    var want = dark ? TABULATOR_DARK : TABULATOR_LIGHT;
    var link = document.querySelector('link[data-arena="tabulator-css"]');
    if (!link) {
      link = document.createElement("link");
      link.rel = "stylesheet";
      link.dataset.arena = "tabulator-css";
      document.head.appendChild(link);
    }
    if (link.href !== want) link.href = want;
  }

  var allData = [];
  var table = null;

  var columns = [
    { title: "#", width: 40, hozAlign: "center", headerSort: false, formatter: "rownum" },
    { title: "模型", field: "model", sorter: "string", width: 120 },
    { title: "数据集", field: "dataset", sorter: "string", width: 110 },
    {
      title: "AUC", field: "auc", sorter: "number", hozAlign: "center", width: 100, tooltip: true,
      formatter: function (cell) {
        var row = cell.getRow().getData();
        var folds = (row.auc_folds || []).map(function (v) { return v.toFixed(4); }).join(", ");
        return '<span title="Folds: ' + folds + '">' + cell.getValue().toFixed(4) + "</span>";
      },
    },
    {
      title: "AUC标准差", field: "auc_std", sorter: "number", hozAlign: "center", width: 90,
      formatter: function (cell) { var v = cell.getValue(); return v != null ? v.toFixed(6) : "—"; },
    },
    {
      title: "ACC", field: "acc", sorter: "number", hozAlign: "center", width: 100, tooltip: true,
      formatter: function (cell) {
        var row = cell.getRow().getData();
        var folds = (row.acc_folds || []).map(function (v) { return v.toFixed(4); }).join(", ");
        return '<span title="Folds: ' + folds + '">' + cell.getValue().toFixed(4) + "</span>";
      },
    },
    {
      title: "ACC标准差", field: "acc_std", sorter: "number", hozAlign: "center", width: 90,
      formatter: function (cell) { var v = cell.getValue(); return v != null ? v.toFixed(6) : "—"; },
    },
    {
      title: "备注", field: "notes", sorter: "string", width: 160,
      formatter: function (cell) { var v = cell.getValue(); return v ? '<span class="arena-note">' + v + "</span>" : "—"; },
    },
  ];

  function updateCount(n) {
    var el = document.getElementById("arena-count");
    if (el) el.textContent = "共 " + n + " 条记录";
  }

  function buildTable(data) {
    if (table) {
      table.setData(data);
    } else {
      table = new Tabulator("#arena-table", {
        data: data,
        layout: "fitDataFill",
        height: "500px",
        placeholder: "暂无数据",
        initialSort: [{ column: "auc", dir: "desc" }],
        columns: columns,
      });
      var search = document.getElementById("arena-filter");
      if (search) search.addEventListener("input", function () { table.setFilter("model", "like", search.value); });
    }
    updateCount(data.length);
  }

  function createTab(text, active) {
    var b = document.createElement("button");
    b.textContent = text;
    b.className = active ? "arena-tab active" : "arena-tab";
    return b;
  }

  function setActiveTab(active) {
    var tabs = document.querySelectorAll(".arena-tab");
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove("active");
    active.classList.add("active");
  }

  function buildTabs(datasets) {
    var el = document.getElementById("arena-tabs");
    if (!el) return;
    el.innerHTML = "";
    var all = createTab("全部", true);
    all.onclick = function () { setActiveTab(all); buildTable(allData); };
    el.appendChild(all);
    datasets.forEach(function (ds) {
      var b = createTab(ds, false);
      b.onclick = function () { setActiveTab(b); buildTable(allData.filter(function (r) { return r.dataset === ds; })); };
      el.appendChild(b);
    });
  }

  function initArena() {
    var el = document.getElementById("arena-table");
    if (!el || el.dataset.initialized) return;
    el.dataset.initialized = "true";

    applyTabulatorTheme();
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTabulatorTheme);
    }

    fetch("_static/arena/leaderboard.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        allData = data;
        var datasets = [];
        var seen = {};
        data.forEach(function (r) { if (!seen[r.dataset]) { seen[r.dataset] = true; datasets.push(r.dataset); } });
        datasets.sort();
        buildTabs(datasets);
        buildTable(data);
      })
      .catch(function () {
        el.innerHTML = '<p class="arena-error">加载排行榜数据失败。</p>';
      });
  }

  function boot() {
    if (window.Tabulator) { initArena(); return; }
    loadScript(TABULATOR_JS, initArena);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
