const state = {
  page: 'overview',
  recordsPage: 1,
  recordsPages: 0,
  recordsQuery: '',
  charts: {},
  options: null,
  modelResults: null,
};

const palette = ['#2f6b50', '#c49a5a', '#799b7c', '#a3b4a5', '#8b9d91'];
const numberFmt = new Intl.NumberFormat('en-PH', { maximumFractionDigits: 0 });
const decimalFmt = new Intl.NumberFormat('en-PH', { maximumFractionDigits: 2 });

function fmtNumber(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return new Intl.NumberFormat('en-PH', { maximumFractionDigits: digits }).format(n);
}
function fmtPercent(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : '—';
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`Server returned an unreadable response (${response.status}).`); }
  if (!response.ok || data.ok === false) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function toast(message, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.style.background = type === 'error' ? '#8e5148' : '#234a3b';
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), 3200);
}

function showNotice(message, isError = false) {
  const notice = document.getElementById('globalNotice');
  if (!message) {
    notice.classList.add('hidden');
    notice.textContent = '';
    return;
  }
  notice.textContent = message;
  notice.classList.toggle('error', isError);
  notice.classList.remove('hidden');
}

function navigate(page) {
  state.page = page;
  document.querySelectorAll('.page-section').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
  const labels = {
    overview: 'Overview', analytics: 'Data analytics', models: 'Model evaluation',
    predictions: 'Predictions', records: 'Data records'
  };
  document.getElementById('pageHeading').textContent = labels[page] || 'Overview';
  document.getElementById('sidebar').classList.remove('open');
  if (page === 'analytics') loadAnalytics();
  if (page === 'records') loadRecords();
  if (page === 'models') loadModelResults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function makeChart(id, config) {
  const canvas = document.getElementById(id);
  if (!canvas || typeof Chart === 'undefined') return;
  if (state.charts[id]) state.charts[id].destroy();
  state.charts[id] = new Chart(canvas, config);
}

function chartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.font.family = "'DM Sans', sans-serif";
  Chart.defaults.font.size = 10;
  Chart.defaults.color = '#829087';
  Chart.defaults.borderColor = '#edf1ed';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 7;
  Chart.defaults.plugins.legend.labels.padding = 16;
}

async function loadStatus() {
  try {
    const status = await api('/api/status');
    const dot = document.getElementById('sidebarStatusDot');
    const text = document.getElementById('sidebarStatusText');
    const meta = document.getElementById('sidebarStatusMeta');
    dot.classList.toggle('online', status.loaded);
    text.textContent = status.loaded ? 'Dataset loaded' : 'Dataset required';
    meta.textContent = status.loaded ? `${fmtNumber(status.records)} cleaned observations` : 'Upload ZIP, ARFF, or CSV';
    if (status.loaded) {
      showNotice('Dataset loaded successfully. The dashboard uses cleaned province-level observations from 2010–2025.');
      await loadOptions();
      await loadOverview();
      await loadAnalytics();
      await loadRecords();
      if (status.models_ready) await loadModelResults();
    } else {
      showNotice(status.error || 'Upload the GitHub repository ZIP or the original ARFF file to begin.', true);
      document.getElementById('profileSource').textContent = 'Not loaded';
    }
  } catch (error) {
    showNotice(error.message, true);
  }
}

async function loadOptions() {
  const opts = await api('/api/options');
  state.options = opts;
  fillSelect('animalFilter', opts.animals, 'All animal types', 'all');
  fillSelect('provinceFilter', opts.provinces, 'All provinces', 'all');
  fillSelect('yearFilter', opts.years.slice().reverse(), 'All years', 'all');
  fillSelect('predictionAnimal', opts.animals, 'Select animal type', '');
  fillSelect('predictionProvince', opts.provinces, 'Select province', '');
}

function fillSelect(id, values, placeholder, defaultValue) {
  const el = document.getElementById(id);
  if (!el) return;
  const previous = el.value;
  el.innerHTML = '';
  const first = document.createElement('option');
  first.value = defaultValue;
  first.textContent = placeholder;
  el.appendChild(first);
  values.forEach(value => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    el.appendChild(option);
  });
  if ([...el.options].some(option => option.value === previous)) el.value = previous;
}

async function loadOverview() {
  const data = await api('/api/overview');
  document.getElementById('kpiObservations').textContent = fmtNumber(data.kpis.observations);
  document.getElementById('kpiProvinces').textContent = fmtNumber(data.kpis.provinces);
  document.getElementById('kpiAnimals').textContent = fmtNumber(data.kpis.animal_types);
  document.getElementById('kpiProduction').textContent = fmtNumber(data.kpis.total_production / 1000000, 2) + 'M';
  document.getElementById('profileLatestYear').textContent = data.kpis.latest_year;
  document.getElementById('profileSource').textContent = data.source || 'Uploaded file';
  document.getElementById('sidebarStatusText').textContent = 'Dataset loaded';
  document.getElementById('sidebarStatusMeta').textContent = `${fmtNumber(data.kpis.observations)} cleaned observations`;

  makeChart('overviewShareChart', {
    type: 'doughnut',
    data: {
      labels: data.animals.map(row => row.animal),
      datasets: [{ data: data.animals.map(row => row.production), backgroundColor: palette, borderWidth: 3, borderColor: '#fff', hoverOffset: 5 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '67%',
      plugins: {
        legend: { position: 'right' },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${fmtNumber(ctx.raw, 2)} metric tons (${data.animals[ctx.dataIndex].share.toFixed(2)}%)` } }
      }
    }
  });
}

function filterParams() {
  const params = new URLSearchParams();
  ['animal', 'province', 'year'].forEach(key => {
    const id = `${key}Filter`;
    const value = document.getElementById(id)?.value || 'all';
    params.set(key, value);
  });
  return params.toString();
}

async function loadAnalytics() {
  try {
    const data = await api(`/api/analytics?${filterParams()}`);
    document.getElementById('filteredRecords').textContent = fmtNumber(data.summary.records);
    document.getElementById('filteredProduction').textContent = fmtNumber(data.summary.production, 2);
    document.getElementById('filteredMedian').textContent = fmtNumber(data.summary.median, 2);
    document.getElementById('filteredCoverage').textContent = `${data.summary.first_year}–${data.summary.last_year}`;
    document.getElementById('topProvinceCaption').textContent = `Year: ${document.getElementById('yearFilter').value === 'all' ? 'latest' : document.getElementById('yearFilter').value}`;
    renderTrend(data.annual_trend);
    renderAnimalShare(data.animal_totals);
    renderTopProvinces(data.top_provinces);
    renderDistribution(data.distribution_2025);
    renderDescriptive(data.descriptive);
  } catch (error) {
    if (state.options) toast(error.message, 'error');
  }
}

function renderTrend(rows) {
  const years = [...new Set(rows.map(row => row.year))].sort((a, b) => a - b);
  const animals = [...new Set(rows.map(row => row.animal))];
  const colors = { Carabao: '#c49a5a', Cattle: '#789b7c', Goat: '#a5b99f', Hog: '#2f6b50' };
  const datasets = animals.map(animal => ({
    label: animal,
    data: years.map(year => {
      const item = rows.find(row => row.year === year && row.animal === animal);
      return item ? item.production : null;
    }),
    borderColor: colors[animal] || palette[animals.indexOf(animal) % palette.length],
    backgroundColor: colors[animal] || palette[animals.indexOf(animal) % palette.length],
    borderWidth: 2, pointRadius: 2.5, pointHoverRadius: 5, tension: .28, spanGaps: true
  }));
  makeChart('trendChart', {
    type: 'line', data: { labels: years, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: 'Year' } },
        y: { beginAtZero: true, ticks: { callback: value => compact(value) }, title: { display: true, text: 'Production (metric tons)' } }
      },
      plugins: { tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmtNumber(ctx.raw, 2)} metric tons` } } }
    }
  });
}

function renderAnimalShare(rows) {
  makeChart('animalShareChart', {
    type: 'bar',
    data: { labels: rows.map(row => row.animal), datasets: [{
      data: rows.map(row => row.production), backgroundColor: palette, borderRadius: 6, maxBarThickness: 35
    }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${fmtNumber(ctx.raw, 2)} metric tons` } } },
      scales: { x: { beginAtZero: true, ticks: { callback: value => compact(value) } }, y: { grid: { display: false } } }
    }
  });
}

function renderTopProvinces(rows) {
  makeChart('provinceChart', {
    type: 'bar',
    data: { labels: rows.map(row => row.province), datasets: [{
      data: rows.map(row => row.production), backgroundColor: '#568269', borderRadius: 5, maxBarThickness: 22
    }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${fmtNumber(ctx.raw, 2)} metric tons` } } },
      scales: { x: { beginAtZero: true, ticks: { callback: value => compact(value) } }, y: { grid: { display: false } } }
    }
  });
}

function renderDistribution(rows) {
  makeChart('distributionChart', {
    type: 'bar',
    data: { labels: rows.map(row => row.animal), datasets: [{
      label: 'Median provincial production', data: rows.map(row => row.production),
      backgroundColor: palette, borderRadius: 6, maxBarThickness: 35
    }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `Median: ${fmtNumber(ctx.raw, 2)} metric tons` } } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, ticks: { callback: value => compact(value) } } }
    }
  });
}

function renderDescriptive(rows) {
  const body = document.getElementById('descriptiveTable');
  if (!rows.length) { body.innerHTML = '<tr><td colspan="7" class="empty-cell">No statistics available.</td></tr>'; return; }
  body.innerHTML = rows.map(row => `<tr>
    <td>${escapeHtml(row['Animal Type'])}</td><td>${fmtNumber(row.Count)}</td>
    <td>${fmtNumber(row.Mean, 2)}</td><td>${fmtNumber(row.Median, 2)}</td>
    <td>${fmtNumber(row.Std_Dev, 2)}</td><td>${fmtNumber(row.Minimum, 2)}</td><td>${fmtNumber(row.Maximum, 2)}</td>
  </tr>`).join('');
}

function compact(value) {
  const n = Number(value);
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}K`;
  return String(n);
}

async function loadRecords() {
  try {
    const params = new URLSearchParams({ page: state.recordsPage, page_size: 25, q: state.recordsQuery });
    const data = await api(`/api/records?${params}`);
    state.recordsPages = data.pages;
    document.getElementById('recordsCount').textContent = `${fmtNumber(data.total)} matching records`;
    document.getElementById('paginationInfo').textContent = `Page ${data.pages ? data.page : 0} of ${data.pages}`;
    document.getElementById('prevPage').disabled = data.page <= 1;
    document.getElementById('nextPage').disabled = data.page >= data.pages;
    const tbody = document.getElementById('recordsTable');
    tbody.innerHTML = data.rows.length ? data.rows.map(row => `<tr>
      <td>${row.year}</td><td>${escapeHtml(row.animal)}</td><td>${escapeHtml(row.province)}</td>
      <td class="align-right">${fmtNumber(row.production, 3)}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty-cell">No records found for this search.</td></tr>';
  } catch (error) {
    document.getElementById('recordsTable').innerHTML = `<tr><td colspan="4" class="empty-cell">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadModelResults() {
  try {
    const data = await api('/api/models');
    if (data.ready) renderModelResults(data);
  } catch (error) {
    console.warn(error);
  }
}

async function trainModels() {
  const button = document.getElementById('trainModelsButton');
  const title = document.getElementById('trainingStatusTitle');
  const text = document.getElementById('trainingStatusText');
  const tag = document.getElementById('modelStatusTag');
  button.disabled = true;
  button.textContent = 'Training models…';
  title.textContent = 'Training in progress';
  text.textContent = 'Fitting Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Conditional SMOTENC augmentation is checked on validation data only.';
  tag.textContent = 'RUNNING';
  tag.classList.remove('ready');
  try {
    const data = await api('/api/train', { method: 'POST' });
    renderModelResults(data);
    toast('All four classification models have been trained.');
  } catch (error) {
    title.textContent = 'Model training could not be completed';
    text.textContent = error.message;
    tag.textContent = 'ERROR';
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.innerHTML = '▶ Train four models';
  }
}

function renderModelResults(data) {
  state.modelResults = data;
  document.getElementById('trainingStatusTitle').textContent = `Model evaluation complete · Best test accuracy: ${data.best_model}`;
  document.getElementById('trainingStatusText').textContent = `${fmtNumber(data.record_count)} records evaluated. The model comparison is sorted by test accuracy, then ROC-AUC and F1-score.`;
  const tag = document.getElementById('modelStatusTag');
  tag.textContent = 'TRAINED'; tag.classList.add('ready');
  document.getElementById('modelSummary').classList.remove('hidden');
  document.getElementById('modelMedian').textContent = `${fmtNumber(data.median_threshold, 2)} MT`;
  document.getElementById('modelTrainCount').textContent = fmtNumber(data.train_count);
  document.getElementById('modelValidationCount').textContent = fmtNumber(data.validation_count);
  document.getElementById('modelTestCount').textContent = fmtNumber(data.test_count);
  renderModelMetrics(data.metrics);
  renderAugmentation(data.augmentation_log);
  renderComparisonChart(data.metrics);
  renderClassChart(data.class_counts);
  renderRoc(data.roc);
  renderConfusion(data.confusion);
}

function renderModelMetrics(rows) {
  const body = document.getElementById('modelMetricsTable');
  body.innerHTML = rows.map((row, index) => `<tr>
    <td>${index === 0 ? '<span class="metric-badge">BEST</span> ' : ''}${escapeHtml(row.model)}</td>
    <td><strong>${fmtPercent(row.accuracy)}</strong></td><td>${fmtPercent(row.precision)}</td>
    <td>${fmtPercent(row.recall)}</td><td>${fmtPercent(row.f1)}</td><td>${Number(row.roc_auc).toFixed(4)}</td>
    <td><span class="augmentation-tag">${row.augmentation_triggered ? 'Attempted' : 'Not needed'}</span></td>
  </tr>`).join('');
}

function renderAugmentation(rows) {
  const body = document.getElementById('augmentationTable');
  body.innerHTML = rows.map(row => `<tr>
    <td>${escapeHtml(row.model)}</td><td>${fmtPercent(row.validation_before)}</td>
    <td>${row.triggered ? 'Yes' : 'No'}</td><td>${fmtNumber(row.training_rows_before)}</td>
    <td>${fmtNumber(row.synthetic_rows_created)}</td><td>${row.validation_after == null ? '—' : fmtPercent(row.validation_after)}</td>
    <td>${row.augmented_model_selected ? 'Yes' : 'No'}</td><td>${escapeHtml(row.decision)}</td>
  </tr>`).join('');
}

function renderComparisonChart(rows) {
  const metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
  const labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'];
  const datasets = rows.map((row, index) => ({
    label: row.model,
    data: metrics.map(key => row[key]),
    backgroundColor: palette[index % palette.length],
    borderRadius: 5,
    maxBarThickness: 23,
  }));
  makeChart('modelComparisonChart', {
    type: 'bar', data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${Number(ctx.raw).toFixed(4)}` } } },
      scales: { x: { grid: { display: false } }, y: { min: 0, max: 1, ticks: { callback: value => `${Math.round(value * 100)}%` } } }
    }
  });
}

function renderClassChart(counts) {
  makeChart('classDistributionChart', {
    type: 'doughnut',
    data: { labels: Object.keys(counts), datasets: [{ data: Object.values(counts), backgroundColor: ['#c49a5a', '#4e8a68'], borderColor: '#fff', borderWidth: 3 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: '66%', plugins: { legend: { position: 'bottom' } } }
  });
}

function renderRoc(roc) {
  const datasets = Object.entries(roc).map(([name, result], index) => ({
    label: `${name} (AUC ${Number(result.auc).toFixed(3)})`,
    data: result.points,
    borderColor: palette[index % palette.length],
    backgroundColor: palette[index % palette.length],
    borderWidth: 2, pointRadius: 0, showLine: true, tension: 0,
  }));
  datasets.push({ label: 'No-skill baseline', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#b7c0ba', borderDash: [5, 5], pointRadius: 0, showLine: true, borderWidth: 1.5 });
  makeChart('rocChart', {
    type: 'scatter', data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate' } },
        y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate' } },
      },
      plugins: { tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: (${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})` } } }
    }
  });
}

function renderConfusion(confusion) {
  const grid = document.getElementById('confusionGrid');
  grid.innerHTML = Object.entries(confusion).map(([name, matrix]) => {
    const [low, high] = matrix;
    return `<div class="confusion-card">
      <h4>${escapeHtml(name)}</h4>
      <div class="confusion-matrix">
        <div class="confusion-axis">Actual ↓<br>Predicted →</div>
        <div class="confusion-axis">Low</div><div class="confusion-axis">High</div>
        <div class="confusion-axis">Low</div>
        <div class="cm-cell"><strong>${fmtNumber(low[0])}</strong><small>True Low</small></div>
        <div class="cm-cell error"><strong>${fmtNumber(low[1])}</strong><small>False High</small></div>
        <div class="confusion-axis">High</div>
        <div class="cm-cell error"><strong>${fmtNumber(high[0])}</strong><small>False Low</small></div>
        <div class="cm-cell"><strong>${fmtNumber(high[1])}</strong><small>True High</small></div>
      </div>
    </div>`;
  }).join('');
}

async function makePrediction(event) {
  event.preventDefault();
  const button = document.getElementById('predictButton');
  const resultPanel = document.getElementById('predictionResult');
  const animal = document.getElementById('predictionAnimal').value;
  const province = document.getElementById('predictionProvince').value;
  const year = document.getElementById('predictionYear').value;
  if (!animal || !province || !year) { toast('Complete all prediction fields.', 'error'); return; }
  button.disabled = true; button.textContent = 'Generating…';
  try {
    const data = await api('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ animal_type: animal, province, year: Number(year) })
    });
    const best = data.best_prediction;
    resultPanel.innerHTML = `
      <div class="prediction-result-head">
        <div><div class="section-kicker">MODEL CONSENSUS COMPARISON</div><h3>${escapeHtml(data.animal_type)} · ${escapeHtml(data.province)} · ${data.year}</h3></div>
        <span class="result-class ${best.startsWith('Low') ? 'low' : ''}">${escapeHtml(best)}</span>
      </div>
      <div class="prediction-results">
        ${data.results.map(row => `<div class="prediction-model-row">
          <strong>${escapeHtml(row.model)}</strong>
          <div class="probability-track"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, row.probability_high * 100))}%"></div></div>
          <span class="model-prob">${(row.probability_high * 100).toFixed(1)}%</span>
        </div>`).join('')}
      </div>
      <div class="prediction-disclaimer"><strong>Best evaluated model:</strong> ${escapeHtml(data.best_model)}. Its class result is shown above. The bars show each model's estimated probability of High Production. ${escapeHtml(data.notice)}</div>`;
    toast('Prediction generated.');
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false; button.innerHTML = 'Generate prediction <span>→</span>';
  }
}

async function uploadDataset(event) {
  event.preventDefault();
  const input = document.getElementById('datasetFile');
  const file = input.files[0];
  if (!file) { toast('Choose a dataset file first.', 'error'); return; }
  const button = document.getElementById('uploadButton');
  const status = document.getElementById('uploadStatus');
  const form = new FormData(); form.append('dataset', file);
  button.disabled = true; button.textContent = 'Uploading…';
  status.textContent = 'Uploading and validating the dataset…';
  try {
    const result = await api('/api/upload', { method: 'POST', body: form });
    status.textContent = `${fmtNumber(result.records)} cleaned provincial observations loaded from ${result.source}.`;
    toast('Dataset uploaded and loaded successfully.');
    state.recordsPage = 1; state.recordsQuery = '';
    document.getElementById('recordsSearch').value = '';
    showNotice(`${result.message} Train the four classification models from Model evaluation when ready.`);
    await loadStatus();
  } catch (error) {
    status.textContent = error.message;
    toast(error.message, 'error');
  } finally {
    button.disabled = false; button.textContent = 'Upload dataset';
  }
}

function bindEvents() {
  document.querySelectorAll('.nav-item').forEach(button => button.addEventListener('click', () => navigate(button.dataset.page)));
  document.querySelectorAll('[data-goto]').forEach(button => button.addEventListener('click', () => navigate(button.dataset.goto)));
  document.getElementById('mobileMenu').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
  document.getElementById('refreshButton').addEventListener('click', () => loadStatus().then(() => toast('Dashboard refreshed.')).catch(error => toast(error.message, 'error')));
  ['animalFilter', 'provinceFilter', 'yearFilter'].forEach(id => document.getElementById(id).addEventListener('change', loadAnalytics));
  document.getElementById('resetFilters').addEventListener('click', () => {
    ['animalFilter', 'provinceFilter', 'yearFilter'].forEach(id => document.getElementById(id).value = 'all');
    loadAnalytics();
  });
  document.getElementById('trainModelsButton').addEventListener('click', trainModels);
  document.getElementById('predictionForm').addEventListener('submit', makePrediction);
  document.getElementById('uploadForm').addEventListener('submit', uploadDataset);
  document.getElementById('datasetFile').addEventListener('change', event => {
    document.getElementById('selectedFileName').textContent = event.target.files[0]?.name || 'No file selected';
  });
  document.getElementById('recordsSearchButton').addEventListener('click', () => {
    state.recordsQuery = document.getElementById('recordsSearch').value.trim();
    state.recordsPage = 1; loadRecords();
  });
  document.getElementById('recordsSearch').addEventListener('keydown', event => {
    if (event.key === 'Enter') document.getElementById('recordsSearchButton').click();
  });
  document.getElementById('prevPage').addEventListener('click', () => {
    if (state.recordsPage > 1) { state.recordsPage--; loadRecords(); }
  });
  document.getElementById('nextPage').addEventListener('click', () => {
    if (state.recordsPage < state.recordsPages) { state.recordsPage++; loadRecords(); }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  chartDefaults();
  bindEvents();
  loadStatus();
});
