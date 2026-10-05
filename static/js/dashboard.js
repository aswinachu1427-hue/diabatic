/**
 * dashboard.js - Clinical Assessment & Visualization Controller
 * Manages two-way input synchronization, AJAX inference, SVG donut animation,
 * horizontal factor contribution bars, and Chart.js trend tracking.
 */

let trendChart = null;

document.addEventListener('DOMContentLoaded', () => {
  initTrendChart();
  initInitialResultState();
});

/* 1. Synchronize Slider & Numeric Inputs */
function syncSliderAndNumber(sourceId, targetId) {
  const source = document.getElementById(sourceId);
  const target = document.getElementById(targetId);
  if (!source || !target) return;
  target.value = source.value;
}

/* 2. Segmented Toggle Controllers */
function selectGender(val, btn) {
  document.getElementById('genderInput').value = val;
  const parent = btn.parentElement;
  parent.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function selectBinaryToggle(field, val, btn) {
  const inputId = field === 'hypertension' ? 'hypertensionInput' : 'heartDiseaseInput';
  document.getElementById(inputId).value = val;
  const parent = btn.parentElement;
  parent.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

/* 3. Form Reset */
function resetForm() {
  document.getElementById('predictionForm').reset();
  selectGender('Female', document.querySelector('[data-value="Female"]'));
  selectBinaryToggle('hypertension', 0, document.querySelectorAll('.segmented-control button[data-val="0"]')[0]);
  selectBinaryToggle('heart_disease', 0, document.querySelectorAll('.segmented-control button[data-val="0"]')[1]);
  syncSliderAndNumber('ageSlider', 'ageNumber');
  syncSliderAndNumber('glucoseSlider', 'glucoseInput');
  syncSliderAndNumber('hba1cSlider', 'hba1cInput');
  syncSliderAndNumber('bmiSlider', 'bmiInput');
  showToast('Form reset to default baseline indicators.', 'info');
}

/* 4. AJAX Prediction Handler */
async function handlePredictionSubmit(event) {
  event.preventDefault();

  const btn = document.getElementById('predictBtn');
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="loading-spinner"></span> <span>Analyzing Biomarkers...</span>`;

  const payload = {
    gender: document.getElementById('genderInput').value,
    age: parseFloat(document.getElementById('ageNumber').value),
    hypertension: parseInt(document.getElementById('hypertensionInput').value),
    heart_disease: parseInt(document.getElementById('heartDiseaseInput').value),
    smoking_history: document.getElementById('smokingInput').value,
    bmi: parseFloat(document.getElementById('bmiInput').value),
    HbA1c_level: parseFloat(document.getElementById('hba1cInput').value),
    blood_glucose_level: parseFloat(document.getElementById('glucoseInput').value)
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(result.error || result.message || 'Error executing clinical inference.');
    }

    // Reveal and animate result
    renderPredictionResult(result);
    showToast(`Assessment complete: ${result.risk_level} Risk (${result.risk_probability}%)`, 
              result.risk_level === 'High' ? 'danger' : result.risk_level === 'Moderate' ? 'warning' : 'success');

  } catch (error) {
    showToast(error.message, 'danger');
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
    lucide.createIcons();
  }
}

/* 5. Render Animated Results & Explainability Bars */
function renderPredictionResult(res) {
  const prob = res.risk_probability;
  const level = res.risk_level;
  const color = res.level_color;

  // Animate Donut Ring
  // Circumference: 2 * Math.PI * 70 = ~439.82
  const circumference = 2 * Math.PI * 70;
  const offset = circumference - (prob / 100) * circumference;
  
  const ring = document.getElementById('donutProgressRing');
  ring.style.stroke = color;
  ring.style.strokeDashoffset = offset;

  // Number Count-Up in Center
  animateScoreNumber('riskPercentageNum', prob);

  // Risk Badge
  const badge = document.getElementById('riskBadge');
  const badgeText = document.getElementById('riskBadgeText');
  badge.className = `risk-badge risk-badge-${level.toLowerCase()}`;
  badgeText.textContent = `${level} Risk Priority`;

  // Summary Text
  document.getElementById('riskExplanation').textContent = res.summary_statement;

  // Top Contributing Factors
  const factorsContainer = document.getElementById('factorsListContainer');
  if (res.top_factors && res.top_factors.length > 0) {
    factorsContainer.innerHTML = '';
    res.top_factors.forEach((f) => {
      const barClass = f.severity === 'danger' ? 'factor-bar-danger' :
                       f.severity === 'warning' ? 'factor-bar-warning' : 'factor-bar-success';
      const item = document.createElement('div');
      item.className = 'factor-item';
      item.innerHTML = `
        <div class="factor-header">
          <span>${f.name}</span>
          <span class="factor-impact-badge" style="color: ${f.severity === 'danger' ? 'var(--danger)' : f.severity === 'warning' ? 'var(--warning)' : 'var(--success)'};">${f.impact}% Impact</span>
        </div>
        <div class="factor-track">
          <div class="factor-bar ${barClass}" style="width: 0%;" data-target-width="${f.impact}%"></div>
        </div>
        <div class="factor-note">${f.note}</div>
      `;
      factorsContainer.appendChild(item);
    });

    // Animate bars expanding
    setTimeout(() => {
      document.querySelectorAll('.factor-bar').forEach(b => {
        b.style.width = b.getAttribute('data-target-width');
      });
    }, 100);
  }

  // Personalized Lifestyle Recommendations
  const tipsContainer = document.getElementById('tipsListContainer');
  if (res.suggestions && res.suggestions.length > 0) {
    tipsContainer.innerHTML = '';
    res.suggestions.forEach(s => {
      const tipItem = document.createElement('div');
      tipItem.className = 'tip-item';
      tipItem.innerHTML = `
        <div class="tip-icon-box"><i data-lucide="${s.icon}" style="width: 18px; height: 18px;"></i></div>
        <div class="tip-content">
          <h4>${s.title}</h4>
          <p>${s.desc}</p>
        </div>
      `;
      tipsContainer.appendChild(tipItem);
    });
    lucide.createIcons({ root: tipsContainer });
  }

  // Update Top Stats Strip
  const totalKpi = document.getElementById('kpi-total-checks');
  if (totalKpi) {
    const currentTotal = parseInt(totalKpi.textContent) || 0;
    totalKpi.textContent = (currentTotal + 1).toString();
  }

  const todayScoreVal = document.getElementById('today-score-val');
  if (todayScoreVal) {
    todayScoreVal.textContent = `${prob}%`;
  }

  // Add data point to Trend Chart
  if (trendChart) {
    const todayLabel = new Date().toISOString().slice(0, 10);
    trendChart.data.labels.push(todayLabel);
    trendChart.data.datasets[0].data.push(prob);
    if (trendChart.data.labels.length > 10) {
      trendChart.data.labels.shift();
      trendChart.data.datasets[0].data.shift();
    }
    trendChart.update();
  }
}

function animateScoreNumber(elId, targetVal) {
  const el = document.getElementById(elId);
  const duration = 1200;
  const start = 0;
  const startTime = performance.now();

  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = start + (targetVal - start) * ease;
    el.textContent = current.toFixed(1) + '%';
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = targetVal.toFixed(1) + '%';
  }
  requestAnimationFrame(step);
}

/* 6. Initial Result State (e.g. from existing database record) */
function initInitialResultState() {
  const todayValEl = document.getElementById('today-score-val');
  if (todayValEl && todayValEl.textContent.trim() !== 'N/A') {
    const score = parseFloat(todayValEl.textContent);
    if (!isNaN(score)) {
      const ring = document.getElementById('donutProgressRing');
      const circumference = 2 * Math.PI * 70;
      const offset = circumference - (score / 100) * circumference;
      const color = score < 30 ? '#10B981' : score <= 60 ? '#F59E0B' : '#EF4444';
      ring.style.stroke = color;
      ring.style.strokeDashoffset = offset;
      document.getElementById('riskPercentageNum').textContent = `${score}%`;

      const level = score < 30 ? 'Low' : score <= 60 ? 'Moderate' : 'High';
      const badge = document.getElementById('riskBadge');
      badge.className = `risk-badge risk-badge-${level.toLowerCase()}`;
      document.getElementById('riskBadgeText').textContent = `${level} Risk Priority`;

      // Animate placeholder factor bars
      setTimeout(() => {
        document.getElementById('factor-bar-1').style.width = '42%';
        document.getElementById('factor-bar-2').style.width = '26%';
        document.getElementById('factor-bar-3').style.width = '18%';
        document.getElementById('factor-impact-1').textContent = '42% Impact';
        document.getElementById('factor-impact-2').textContent = '26% Impact';
        document.getElementById('factor-impact-3').textContent = '18% Impact';
      }, 300);
    }
  }
}

/* 7. Longitudinal Trend Chart Initialization (Chart.js) */
function initTrendChart() {
  const canvas = document.getElementById('trendChartCanvas');
  if (!canvas) return;

  const dataSeries = window.trendSeriesData || [];
  const labels = dataSeries.map(d => d.date);
  const values = dataSeries.map(d => d.score);

  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 260);
  gradient.addColorStop(0, 'rgba(37, 99, 235, 0.35)');
  gradient.addColorStop(1, 'rgba(37, 99, 235, 0.01)');

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels.length > 0 ? labels : ['Aug 14', 'Sep 02', 'Sep 21', 'Oct 02'],
      datasets: [{
        label: 'Diabetic Risk Probability (%)',
        data: values.length > 0 ? values : [4.2, 11.5, 48.6, 34.0],
        borderColor: '#2563EB',
        borderWidth: 2.5,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointBackgroundColor: '#FFFFFF',
        pointBorderColor: '#2563EB',
        pointBorderWidth: 2,
        pointRadius: 4.5,
        pointHoverRadius: 7
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0F172A',
          titleFont: { size: 12, weight: 'bold' },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (ctx) => ` Predicted Risk: ${ctx.parsed.y.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#64748B', font: { size: 11 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: '#F1F5F9' },
          ticks: {
            color: '#64748B',
            font: { size: 11 },
            callback: (v) => `${v}%`
          }
        }
      }
    }
  });
}
