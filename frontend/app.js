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
    renderWeekChips();
}

function removeWeek(index) {
    playerHistory.splice(index, 1);
    renderWeekChips();
}

function clearHistory() {
    playerHistory = [];
    renderWeekChips();
    $('results').style.display = 'none';
}

function renderWeekChips() {
    const container = $('week-chips');
    const sub = $('week-count');
    container.innerHTML = '';

    if (playerHistory.length === 0) {
        container.hidden = true;
        sub.textContent = 'No weeks tracked yet — start by adjusting stats above';
        return;
    }

    container.hidden = false;
    sub.textContent = `${playerHistory.length} week${playerHistory.length === 1 ? '' : 's'} tracked · click any to remove`;

    playerHistory.forEach((snap, idx) => {
        const chip = document.createElement('div');
        chip.className = 'week-chip';
        chip.innerHTML = `
            <span class="chip-week">Wk ${idx + 1}</span>
            <span class="chip-stat">${(snap.ppg || 0).toFixed(1)} ppg</span>
            <button type="button" class="chip-remove" aria-label="Remove week ${idx + 1}" title="Remove">×</button>
        `;
        chip.querySelector('.chip-remove').addEventListener('click', () => removeWeek(idx));
        container.appendChild(chip);
    });
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
const TIER_LABELS = ['developmental', 'low', 'mid', 'high', 'elite'];
const TIER_DISPLAY = ['Developmental', 'Low', 'Mid', 'High', 'Elite'];

function renderResults(data) {
    $('results').style.display = 'block';

    const probs = data.nil_tier_probs || [];
    const topIdx = probs.length
        ? probs.indexOf(Math.max(...probs))
        : 2;
    const tierKey = TIER_LABELS[topIdx] || 'mid';
    const tierLabel = TIER_DISPLAY[topIdx] || 'Mid';

    // Tier badge
    const badge = $('tier-badge');
    badge.className = `tier-badge tier-${tierKey}`;
    badge.textContent = `${tierLabel} Tier \u00b7 ${(probs[topIdx] * 100 || 0).toFixed(0)}% confidence`;

    // Animated count-up for NIL value
    animateCount($('nil-value'), data.nil_valuation_estimate, formatCurrency);

    // Direction
    const wrap = $('direction-wrap');
    const arrowEl = $('direction-arrow');
    const dirEl = $('direction');
    const arrows = { up: '\u2191', down: '\u2193', stable: '\u2192' };
    const dirLabel = { up: 'Up', down: 'Down', stable: 'Stable' };
    arrowEl.textContent = arrows[data.direction] || '-';
    dirEl.textContent = dirLabel[data.direction] || data.direction;
    wrap.className = `direction-${data.direction}`;
    wrap.style.cssText = 'display:inline-flex;align-items:center;gap:0.55rem;';

    // Tier strip
    renderTierStrip(probs, topIdx);

    // Tier gauge (Plotly bar)
    renderTierGauge(probs);

    // Timeline
    renderTimeline(data.timeline);
}

function animateCount(el, target, formatter, duration = 900) {
    const startVal = parseFloat((el.dataset.value || '0')) || 0;
    const startTime = performance.now();
    function frame(now) {
        const t = Math.min(1, (now - startTime) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        const current = startVal + (target - startVal) * eased;
        el.textContent = formatter(current);
        if (t < 1) requestAnimationFrame(frame);
        else el.dataset.value = String(target);
    }
    requestAnimationFrame(frame);
}

function renderTierStrip(probs, activeIdx) {
    const strip = $('tier-strip');
    if (!strip) return;
    strip.innerHTML = '';
    TIER_LABELS.forEach((tier, i) => {
        const seg = document.createElement('div');
        seg.className = `seg tier-${tier}` + (i === activeIdx ? ' active' : '');
        const pct = ((probs[i] || 0) * 100).toFixed(1);
        seg.innerHTML = `
            <div class="seg-label">${TIER_DISPLAY[i]}</div>
            <div class="seg-pct">${pct}%</div>
        `;
        strip.appendChild(seg);
    });
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
        margin: { t: 20, b: 90, l: 50, r: 20 },
        height: 290,
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

