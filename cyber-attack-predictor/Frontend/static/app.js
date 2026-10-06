/**
 * app.js
 * Frontend client logic connecting to the FastAPI REST Backend.
 */

const API_BASE = window.location.origin;

// Presets data
const PRESETS = {
  benign: {
    pkt_rate: 45.2,
    byte_rate: 12500.0,
    fwd_pkt_rate: 22.5,
    bwd_pkt_rate: 22.7,
    fwd_bwd_pkt_ratio: 1.0,
    syn_flag_ratio: 0.05,
    ack_flag_ratio: 0.85,
    rst_flag_ratio: 0.0,
  },
  dos: {
    pkt_rate: 1850.0,
    byte_rate: 980000.0,
    fwd_pkt_rate: 1820.0,
    bwd_pkt_rate: 30.0,
    fwd_bwd_pkt_ratio: 60.6,
    syn_flag_ratio: 0.95,
    ack_flag_ratio: 0.02,
    rst_flag_ratio: 0.0,
  },
  portscan: {
    pkt_rate: 320.0,
    byte_rate: 28000.0,
    fwd_pkt_rate: 310.0,
    bwd_pkt_rate: 10.0,
    fwd_bwd_pkt_ratio: 31.0,
    syn_flag_ratio: 0.88,
    ack_flag_ratio: 0.08,
    rst_flag_ratio: 0.75,
  },
};

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  for (const [k, v] of Object.entries(p)) {
    const el = document.getElementById(k);
    if (el) el.value = v;
  }
}

// Fetch Initial Metrics & Health
async function fetchSystemMetrics() {
  try {
    const [healthRes, statsRes] = await Promise.all([
      fetch(`${API_BASE}/health`),
      fetch(`${API_BASE}/api/v1/stats`),
    ]);

    if (healthRes.ok) {
      const health = await healthRes.json();
      const dbEngine = health.database?.engine || 'sqlite';
      document.getElementById('db-badge').textContent = `DB: ${dbEngine.toUpperCase()}`;
      document.getElementById('kpi-engine').textContent = dbEngine.toUpperCase();
    }

    if (statsRes.ok) {
      const stats = await statsRes.json();
      document.getElementById('kpi-flows').textContent = stats.total_flows_analyzed || 0;
      document.getElementById('kpi-risk').textContent = stats.average_risk_score || 0;
      document.getElementById('kpi-alerts').textContent = stats.open_alerts_count || 0;
    }
  } catch (err) {
    console.warn('Could not fetch initial stats:', err);
    document.getElementById('db-badge').textContent = 'DB: LOCAL';
  }
}

// Fetch Alerts
async function loadAlerts(filter = 'All') {
  const tbody = document.getElementById('alerts-tbody');
  tbody.innerHTML = '<tr><td colspan="7" class="loading-td">Loading alerts...</td></tr>';

  // Update active button
  document.querySelectorAll('.filter-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.textContent.includes(filter));
  });

  let url = `${API_BASE}/api/v1/alerts?limit=20`;
  if (filter === 'Critical' || filter === 'High') {
    url += `&threat_level=${filter}`;
  } else if (filter === 'Open') {
    url += `&status=Open`;
  }

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Alert fetch failed');
    const alerts = await res.json();

    if (!alerts.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="loading-td">No alerts in this category.</td></tr>';
      return;
    }

    tbody.innerHTML = alerts
      .map((a) => {
        const tierColor =
          a.threat_level === 'Critical'
            ? 'var(--tier-critical)'
            : a.threat_level === 'High'
            ? 'var(--tier-high)'
            : a.threat_level === 'Medium'
            ? 'var(--tier-medium)'
            : 'var(--tier-low)';

        const statusClass =
          a.status === 'Open'
            ? 'status-open'
            : a.status === 'Investigating'
            ? 'status-investigating'
            : 'status-resolved';

        const nextStatus = a.status === 'Open' ? 'Investigating' : a.status === 'Investigating' ? 'Resolved' : 'Open';

        return `
          <tr>
            <td>#${a.id}</td>
            <td style="font-family: var(--font-mono); font-size: 11px;">${a.timestamp ? a.timestamp.slice(0, 19).replace('T', ' ') : 'N/A'}</td>
            <td>${a.source}</td>
            <td style="font-weight: 700; color: ${tierColor}">${a.risk_score.toFixed(1)}</td>
            <td><span style="color: ${tierColor}; font-weight: 700;">${a.threat_level}</span></td>
            <td><span class="status-pill ${statusClass}">${a.status}</span></td>
            <td>
              <button class="triage-btn" onclick="updateAlert(${a.id}, '${nextStatus}')">Mark ${nextStatus}</button>
            </td>
          </tr>
        `;
      })
      .join('');
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="7" class="loading-td">Error loading alerts.</td></tr>';
  }
}

// Update Alert
async function updateAlert(alertId, newStatus) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/alerts/${alertId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus, resolved_by: 'soc_analyst', notes: 'Triage via HTML client' }),
    });
    if (res.ok) {
      loadAlerts();
      fetchSystemMetrics();
    }
  } catch (e) {
    alert('Failed to update alert status.');
  }
}

// Prediction Form Submission
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn = document.getElementById('btn-predict');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');

  btn.disabled = true;
  btnText.textContent = 'Analyzing Telemetry...';
  btnSpinner.style.display = 'inline-block';

  // Gather fields
  const fields = ['pkt_rate', 'byte_rate', 'fwd_pkt_rate', 'bwd_pkt_rate', 'fwd_bwd_pkt_ratio', 'syn_flag_ratio', 'ack_flag_ratio', 'rst_flag_ratio'];
  const features = {};
  for (const f of fields) {
    const el = document.getElementById(f);
    if (el) features[f] = parseFloat(el.value) || 0.0;
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        features: features,
        source_ip: 'Client-192.168.1.100',
        alert_threshold: 60.0,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference request failed');
    }

    const data = await res.json();
    renderPredictionResult(data);
    fetchSystemMetrics();
    loadAlerts();
  } catch (err) {
    alert(`Prediction Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btnText.textContent = '⚡ Run Attack & Risk Assessment';
    btnSpinner.style.display = 'none';
  }
});

function renderPredictionResult(data) {
  document.getElementById('results-placeholder').style.display = 'none';
  document.getElementById('results-content').style.display = 'block';

  // Risk Score & Tier
  const score = Math.round(data.risk_score);
  const tier = data.threat_level;
  document.getElementById('res-score').textContent = score;
  document.getElementById('res-threat-tier').textContent = `${tier.toUpperCase()} THREAT`;

  const circle = document.getElementById('score-circle');
  let color = 'var(--tier-low)';
  if (tier === 'Medium') color = 'var(--tier-medium)';
  if (tier === 'High') color = 'var(--tier-high)';
  if (tier === 'Critical') color = 'var(--tier-critical)';

  circle.style.borderColor = color;
  document.getElementById('res-threat-tier').style.color = color;

  // Badge
  const badge = document.getElementById('alert-status-badge');
  if (data.alert) {
    badge.textContent = '🚨 EARLY WARNING ACTIVE';
    badge.style.background = 'rgba(239, 68, 68, 0.2)';
    badge.style.color = '#fca5a5';
    badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
  } else {
    badge.textContent = '✅ TELEMETRY NORMAL';
    badge.style.background = 'rgba(16, 185, 129, 0.2)';
    badge.style.color = '#6ee7b7';
    badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
  }

  // Metrics
  document.getElementById('res-attack-type').textContent = data.predicted_attack_type;
  document.getElementById('res-prob').textContent = `${Math.round(data.attack_probability * 100)}%`;
  document.getElementById('res-anomaly').textContent = `${Math.round(data.anomaly_score * 100)}%`;
  document.getElementById('res-priority').textContent = data.priority;

  // Indicators
  const indList = document.getElementById('res-indicators');
  indList.innerHTML = '';
  if (data.top_indicators && data.top_indicators.length > 0) {
    data.top_indicators.forEach((ind) => {
      const li = document.createElement('li');
      li.textContent = ind;
      indList.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.textContent = 'All flow behavioral features within normal baseline parameters.';
    indList.appendChild(li);
  }
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  fetchSystemMetrics();
  loadAlerts();
});
