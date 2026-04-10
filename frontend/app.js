const API_BASE = window.location.origin;

let playerHistory = [];

// --- Utility ---
function $(id) { return document.getElementById(id); }

function formatCurrency(val) {
    return '$' + Math.round(val).toLocaleString();
}

function getFormValues() {
    return {
        sport: 'basketball',
        school: $('school').value,
        conference: $('conference').value,
        program_tier: parseInt($('program-tier').value),
        ppg: parseFloat($('ppg').value),
        apg: parseFloat($('apg').value),
        rpg: parseFloat($('rpg').value),
        spg: parseFloat($('spg').value),
        bpg: parseFloat($('bpg').value),
        mpg: parseFloat($('mpg').value),
        fg_pct: parseFloat($('fg-pct').value),
        three_pt_pct: parseFloat($('three-pt-pct').value),
        ft_pct: parseFloat($('ft-pct').value),
        injury_flag: $('injury-flag').checked,
        games_played: parseInt($('games-played').value),
        snapshot_week: playerHistory.length + 1,
    };
}

function adjustStat(id, delta) {
    const input = $(id);
    const newVal = Math.max(0, parseFloat(input.value) + delta);
    input.value = newVal.toFixed(1);
}

// --- Slider displays ---
$('program-tier').addEventListener('input', function() {
    $('tier-display').textContent = this.value;
});
$('games-played').addEventListener('input', function() {
    $('games-display').textContent = this.value;
});

// --- History management ---
function addWeek() {
    const snapshot = getFormValues();
    playerHistory.push(snapshot);
    $('week-count').textContent = `Weeks added: ${playerHistory.length}`;
}

function clearHistory() {
    playerHistory = [];
    $('week-count').textContent = 'Weeks added: 0';
    $('results').style.display = 'none';
}

// --- Scenarios ---
const SCENARIOS = {
    breakout: {
        school: 'Duke', conference: 'ACC',
        program_tier: 1, ppg: 22.5, apg: 5.2, rpg: 7.8,
        spg: 1.5, bpg: 1.0, mpg: 34.0,
        fg_pct: 0.48, three_pt_pct: 0.40, ft_pct: 0.85,
        injury_flag: false, games_played: 12,
    },
    injury: {
        school: 'Alabama', conference: 'SEC',
        program_tier: 1, ppg: 12.0, apg: 3.0, rpg: 4.0,
        spg: 0.8, bpg: 0.5, mpg: 22.0,
        fg_pct: 0.40, three_pt_pct: 0.30, ft_pct: 0.70,
        injury_flag: true, games_played: 4,
    },
    transfer: {
        school: 'Oregon', conference: 'Pac-12',
        program_tier: 2, ppg: 16.0, apg: 3.5, rpg: 5.0,
        spg: 1.2, bpg: 0.6, mpg: 30.0,
        fg_pct: 0.44, three_pt_pct: 0.36, ft_pct: 0.80,
        injury_flag: false, games_played: 8,
    },
};

function loadScenario(name) {
    const s = SCENARIOS[name];
    if (!s) return;
    clearHistory();
    $('school').value = s.school;
    $('conference').value = s.conference;
    $('program-tier').value = s.program_tier;
    $('tier-display').textContent = s.program_tier;
    $('ppg').value = s.ppg;
    $('apg').value = s.apg;
    $('rpg').value = s.rpg;
    $('spg').value = s.spg;
    $('bpg').value = s.bpg;
    $('mpg').value = s.mpg;
    $('fg-pct').value = s.fg_pct;
    $('three-pt-pct').value = s.three_pt_pct;
    $('ft-pct').value = s.ft_pct;
    $('injury-flag').checked = s.injury_flag;
    $('games-played').value = s.games_played;
    $('games-display').textContent = s.games_played;
}

// --- Simulation ---
async function runSimulation() {
    const btn = $('simulate-btn');
    btn.textContent = 'Simulating...';
    btn.classList.add('loading');

    const newSnapshot = getFormValues();

    const payload = {
        player_history: playerHistory,
        new_snapshot: newSnapshot,
        simulate_weeks_ahead: 8,
    };

    try {
        const resp = await fetch(`${API_BASE}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json();
            alert(`Error: ${err.detail || resp.statusText}`);
            return;
        }

        const data = await resp.json();
        renderResults(data);
    } catch (e) {
        alert(`Connection error: ${e.message}`);
    } finally {
        btn.textContent = 'Simulate';
        btn.classList.remove('loading');
    }
}

// --- Rendering ---
function renderResults(data) {
    $('results').style.display = 'block';

    // NIL Value
    $('nil-value').textContent = formatCurrency(data.nil_valuation_estimate);

    // Direction
    const dirEl = $('direction');
    const arrows = { up: '\u2191 Up', down: '\u2193 Down', stable: '\u2192 Stable' };
    dirEl.textContent = arrows[data.direction] || data.direction;
    dirEl.className = `value direction-${data.direction}`;

    // Tier gauge
    renderTierGauge(data.nil_tier_probs);

    // Timeline
    renderTimeline(data.timeline);

    // Cohort
    renderCohort(data.cohort_comparison);
}

function renderTierGauge(probs) {
    const tierLabels = ['Tier 1\n<$50K', 'Tier 2\n$50-200K', 'Tier 3\n$200-500K', 'Tier 4\n$500K-1M', 'Tier 5\n>$1M'];
    const colors = ['#94a3b8', '#3b82f6', '#22c55e', '#eab308', '#ef4444'];

    const trace = {
        type: 'bar',
        x: tierLabels,
        y: probs.map(p => (p * 100).toFixed(1)),
        marker: { color: colors },
        text: probs.map(p => (p * 100).toFixed(1) + '%'),
        textposition: 'outside',
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#f1f5f9' },
        yaxis: { title: 'Probability (%)', range: [0, 100], gridcolor: '#334155' },
        xaxis: { tickfont: { size: 11 } },
        margin: { t: 20, b: 60, l: 50, r: 20 },
        height: 250,
    };

    Plotly.newPlot('tier-gauge', [trace], layout, { displayModeBar: false });
}

function renderTimeline(timeline) {
    const weeks = timeline.map(t => `Week ${t.week}`);
    const values = timeline.map(t => t.nil_valuation);

    // Compute std band from cohort-like variation
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
    const upper = values.map(v => v + std);
    const lower = values.map(v => Math.max(0, v - std));

    const bandTrace = {
        x: [...weeks, ...weeks.slice().reverse()],
        y: [...upper, ...lower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(59,130,246,0.15)',
        line: { color: 'transparent' },
        type: 'scatter',
        showlegend: false,
    };

    const lineTrace = {
        x: weeks,
        y: values,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3b82f6', width: 3 },
        marker: { size: 8 },
        name: 'NIL Valuation',
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#f1f5f9' },
        yaxis: {
            title: 'NIL Valuation ($)',
            gridcolor: '#334155',
            tickformat: '$,.0f',
        },
        xaxis: { gridcolor: '#334155' },
        margin: { t: 20, b: 40, l: 80, r: 20 },
        height: 300,
        showlegend: false,
    };

    Plotly.newPlot('timeline-chart', [bandTrace, lineTrace], layout, { displayModeBar: false });
}

function renderCohort(cohort) {
    $('cohort-median').textContent = formatCurrency(cohort.cohort_median_nil || 0);
    $('percentile').textContent = (cohort.percentile_rank || 0).toFixed(0) + 'th';
    const residual = cohort.residual || 0;
    const resEl = $('residual');
    resEl.textContent = (residual >= 0 ? '+' : '') + formatCurrency(residual);
    resEl.style.color = residual >= 0 ? '#22c55e' : '#ef4444';

    const tbody = $('cohort-body');
    tbody.innerHTML = '';
    (cohort.similar_players || []).forEach(p => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${p.player_id || '-'}</td>
            <td>${p.school || '-'}</td>
            <td>${formatCurrency(p.nil_valuation || 0)}</td>
            <td>${(p.ppg || 0).toFixed(1)}</td>
            <td>${((p.similarity || 0) * 100).toFixed(0)}%</td>
        `;
        tbody.appendChild(tr);
    });
}
