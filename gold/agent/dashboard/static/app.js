// ─── Gold Trading Bot Dashboard — 5s Polling ───

const API = '';
let botRunning = false;
let cachedCandles = [];  // shared candle data for all mini-charts

// ─── Helpers ───
function $(id) { return document.getElementById(id); }
function kv(key, val, cls) {
  return `<div class="kv"><span class="k">${key}</span><span class="v ${cls || ''}">${val}</span></div>`;
}
function dirClass(d) { return d === 'LONG' || d === 'long' ? 'dir-long' : 'dir-short'; }
function signClass(v) { return v > 0 ? 'pos' : v < 0 ? 'neg' : ''; }
function fmt(n, d) { return typeof n === 'number' ? n.toFixed(d ?? 2) : n || '--'; }
function pct(n) { return typeof n === 'number' ? (n * 100).toFixed(0) + '%' : '--'; }

// ─── Mini Chart Drawing ───
function drawMiniChart(canvasId, entry, stop, target, direction, barsHeld, horizonBars) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !cachedCandles.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * 2;  // retina
  const H = canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  const w = W / 2;
  const h = H / 2;

  // Use last N candles based on horizon
  const n = Math.min(cachedCandles.length, Math.max(horizonBars || 20, 30));
  const bars = cachedCandles.slice(-n);
  if (bars.length < 3) return;

  // Price range including entry/stop/target
  const allPrices = bars.flatMap(b => [b.h, b.l]);
  if (entry) allPrices.push(entry);
  if (stop) allPrices.push(stop);
  if (target) allPrices.push(target);
  const pMin = Math.min(...allPrices) - 1;
  const pMax = Math.max(...allPrices) + 1;
  const pRange = pMax - pMin || 1;

  const px = (i) => (i / (bars.length - 1)) * (w - 8) + 4;
  const py = (p) => h - 4 - ((p - pMin) / pRange) * (h - 8);

  // Clear
  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = '#fafaf5';
  ctx.fillRect(0, 0, w, h);

  // Target zone (green band)
  if (target) {
    const tY = py(target);
    ctx.fillStyle = 'rgba(22,163,74,0.08)';
    if (direction === 'LONG' || direction === 'long') {
      ctx.fillRect(0, 0, w, tY);
    } else {
      ctx.fillRect(0, tY, w, h - tY);
    }
  }

  // Stop zone (red band)
  if (stop) {
    const sY = py(stop);
    ctx.fillStyle = 'rgba(220,38,38,0.08)';
    if (direction === 'LONG' || direction === 'long') {
      ctx.fillRect(0, sY, w, h - sY);
    } else {
      ctx.fillRect(0, 0, w, sY);
    }
  }

  // Entry line (blue dashed)
  if (entry) {
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 0.7;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, py(entry));
    ctx.lineTo(w, py(entry));
    ctx.stroke();
    ctx.setLineDash([]);
    // Label
    ctx.fillStyle = '#2563eb';
    ctx.font = '7px monospace';
    ctx.fillText('Entry ' + entry.toFixed(0), 2, py(entry) - 2);
  }

  // Target line (green)
  if (target) {
    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 0.7;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(0, py(target));
    ctx.lineTo(w, py(target));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#16a34a';
    ctx.font = '7px monospace';
    ctx.fillText('TP ' + target.toFixed(0), 2, py(target) - 2);
  }

  // Stop line (red)
  if (stop) {
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 0.7;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(0, py(stop));
    ctx.lineTo(w, py(stop));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#dc2626';
    ctx.font = '7px monospace';
    ctx.fillText('SL ' + stop.toFixed(0), 2, py(stop) + 8);
  }

  // Draw candles
  const barW = Math.max(1, (w - 8) / bars.length * 0.6);
  bars.forEach((bar, i) => {
    const x = px(i);
    const oY = py(bar.o);
    const cY = py(bar.c);
    const hY = py(bar.h);
    const lY = py(bar.l);
    const bull = bar.c >= bar.o;

    // Wick
    ctx.strokeStyle = bull ? '#16a34a' : '#dc2626';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, hY);
    ctx.lineTo(x, lY);
    ctx.stroke();

    // Body
    ctx.fillStyle = bull ? '#16a34a' : '#dc2626';
    const bodyTop = Math.min(oY, cY);
    const bodyH = Math.max(Math.abs(cY - oY), 0.5);
    ctx.fillRect(x - barW / 2, bodyTop, barW, bodyH);
  });

  // Current price dot
  const lastPrice = bars[bars.length - 1].c;
  ctx.fillStyle = '#2563eb';
  ctx.beginPath();
  ctx.arc(px(bars.length - 1), py(lastPrice), 2.5, 0, Math.PI * 2);
  ctx.fill();

  // P&L label
  if (entry) {
    const pnl = direction === 'LONG' || direction === 'long'
      ? lastPrice - entry : entry - lastPrice;
    const pnlColor = pnl >= 0 ? '#16a34a' : '#dc2626';
    ctx.fillStyle = pnlColor;
    ctx.font = 'bold 9px monospace';
    ctx.fillText((pnl >= 0 ? '+' : '') + pnl.toFixed(1) + ' pts', w - 55, 10);
  }
}

// Schedule chart drawing after DOM updates
function drawAllCharts() {
  document.querySelectorAll('[data-mini-chart]').forEach(canvas => {
    const d = JSON.parse(canvas.dataset.miniChart);
    drawMiniChart(canvas.id, d.entry, d.stop, d.target, d.direction, d.barsHeld, d.horizon);
  });
}

let chartCounter = 0;
function chartId() { return 'mc-' + (++chartCounter); }

function outcomeTag(outcome) {
  if (!outcome) return '<span class="tag tag-pending">PENDING</span>';
  if (outcome === 'CORRECT') return '<span class="tag tag-correct">CORRECT</span>';
  if (outcome === 'WRONG') return '<span class="tag tag-wrong">WRONG</span>';
  if (outcome === 'PARTIAL') return '<span class="tag tag-partial">PARTIAL</span>';
  return `<span class="tag">${outcome}</span>`;
}

// ─── Fetch all data ───
async function fetchAll() {
  try {
    const [status, knowledge, strategies, predictions, trades, market, intel, logs, risk, mental, candles, paper] = await Promise.all([
      fetch(API + '/api/status').then(r => r.json()),
      fetch(API + '/api/knowledge').then(r => r.json()),
      fetch(API + '/api/strategies').then(r => r.json()),
      fetch(API + '/api/predictions').then(r => r.json()),
      fetch(API + '/api/trades').then(r => r.json()),
      fetch(API + '/api/market').then(r => r.json()),
      fetch(API + '/api/intelligence').then(r => r.json()),
      fetch(API + '/api/logs').then(r => r.json()),
      fetch(API + '/api/risk').then(r => r.json()),
      fetch(API + '/api/mental').then(r => r.json()),
      fetch(API + '/api/candles?bars=60&tf=5m').then(r => r.json()),
      fetch(API + '/api/paper').then(r => r.json()),
    ]);
    cachedCandles = (candles || {}).candles || [];
    renderStatus(status);
    renderRisk(paper, status);
    renderMental(mental, status);
    renderKnowledge(knowledge);
    renderStrategies(strategies);
    renderPredictions(predictions, status);
    renderTrades(trades);
    renderMarket(market);
    renderIntel(intel, knowledge);
    renderLogs(logs);
  } catch (e) {
    $('status-text').textContent = 'Dashboard error';
    console.error(e);
  }
}

// ─── Renderers ───

function renderStatus(d) {
  botRunning = d.running;
  $('status-dot').className = 'dot ' + (d.running ? 'on' : 'off');
  $('status-text').textContent = d.running ? 'Learning' : 'Stopped';

  const btn = $('bot-toggle');
  btn.textContent = d.running ? 'Stop' : 'Start';
  btn.className = d.running ? 'running' : '';

  $('tick-count').textContent = (d.tick_count || 0).toLocaleString() + ' ticks';

  if (d.latest_tick) {
    $('live-price').textContent = '$' + parseFloat(d.latest_tick.m).toFixed(2);
    const spread = (parseFloat(d.latest_tick.a) - parseFloat(d.latest_tick.b)).toFixed(2);
    $('live-spread').textContent = 'spread $' + spread;
  }
}

function renderRisk(d, status) {
  // Use OANDA live account data from status endpoint
  const oanda = (status || {}).oanda;

  if (oanda) {
    $('oanda-id').textContent = oanda.account_id || '';

    $('rm-balance').textContent = '$' + oanda.balance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

    const upl = oanda.unrealized_pnl || 0;
    $('rm-unrealized').textContent = (upl >= 0 ? '+' : '') + '$' + upl.toFixed(2);
    $('rm-unrealized').className = 'risk-value ' + signClass(upl);

    $('rm-nav').textContent = '$' + (oanda.nav || oanda.balance).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

    const pl = oanda.pl || 0;
    $('rm-pnl').textContent = (pl >= 0 ? '+' : '') + '$' + pl.toFixed(2);
    $('rm-pnl').className = 'risk-value ' + signClass(pl);

    $('rm-trades').textContent = oanda.open_trades || 0;

    const margin = oanda.margin_used || 0;
    $('rm-margin').textContent = '$' + margin.toFixed(2);

    $('rm-status').textContent = oanda.open_trades > 0 ? 'IN TRADE' : 'READY';
    $('rm-status').className = 'risk-value ' + (oanda.open_trades > 0 ? 'pos' : '');
  } else {
    // Fallback to local risk manager data
    const bal = d && d.balance ? d.balance : 100000;
    $('rm-balance').textContent = '$' + bal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    $('rm-unrealized').textContent = '$0.00';
    $('rm-nav').textContent = '$' + bal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    $('rm-pnl').textContent = '$0.00';
    $('rm-trades').textContent = '0';
    $('rm-margin').textContent = '$0.00';
    $('rm-status').textContent = 'READY';
    $('rm-status').className = 'risk-value';
  }

  // Paper trading progress bar
  const paper = d; // paper trader state
  const paperState = paper || {};
  const totalTrades = paperState.total_trades || 0;
  const totalWins = paperState.total_wins || 0;
  const wr = totalTrades > 0 ? (totalWins / totalTrades * 100).toFixed(0) : '0';
  const pct = Math.min(100, (totalTrades / 200 * 100)).toFixed(0);
  const wrOk = totalTrades > 0 && (totalWins / totalTrades) >= 0.70;

  $('paper-progress').innerHTML = `
    <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--sub);margin-top:8px">
      <span>Paper: ${totalTrades}/200 trades | ${totalWins}W-${(paperState.total_losses||0)}L | WR ${wr}% (target 70%)</span>
      <span>${wrOk && totalTrades >= 200 ? 'READY FOR REAL MONEY' : pct + '% to milestone'}</span>
    </div>
    <div style="height:4px;background:#e8e6de;border-radius:2px;margin-top:4px;overflow:hidden">
      <div style="height:100%;width:${pct}%;background:${wrOk ? 'var(--green)' : 'var(--accent)'};border-radius:2px;transition:width 0.5s"></div>
    </div>
  `;
}

function renderMental(d, status) {
  const open = d.open_trades || [];
  const resolved = (d.resolved_trades || []).slice().reverse();
  const total = resolved.length;
  const wins = resolved.filter(t => t.outcome === 'WON').length;
  const losses = resolved.filter(t => t.outcome === 'LOST').length;
  const expired = resolved.filter(t => t.outcome === 'EXPIRED').length;
  const totalPnl = resolved.reduce((s, t) => s + (t.pnl_dollars || 0), 0);
  const wr = total > 0 ? ((wins / total) * 100).toFixed(0) : '--';

  $('mental-stats').textContent = total > 0
    ? `${wins}W-${losses}L-${expired}E | WR ${wr}% | P&L $${totalPnl.toFixed(2)} | ${open.length} open`
    : `${open.length} open`;

  // Current price for unrealized P&L
  const curPrice = status.latest_tick ? parseFloat(status.latest_tick.m) : 0;

  // Open positions
  if (open.length === 0) {
    $('mental-open').innerHTML = '<p class="sub">No open positions — waiting for high-confidence setup</p>';
  } else {
    $('mental-open').innerHTML = open.map(t => {
      const unrealized = t.direction === 'LONG'
        ? curPrice - t.entry_price
        : t.entry_price - curPrice;
      const unrealizedDollars = (t.lot_size || 0) * unrealized;
      const barPct = ((t.bars_elapsed || 0) / (t.horizon_bars || 20) * 100).toFixed(0);
      const totalRange = Math.abs(t.target_price - t.stop_price) || 1;
      const distToStop = t.direction === 'LONG' ? curPrice - t.stop_price : t.stop_price - curPrice;
      const progressPct = Math.max(0, Math.min(100, (distToStop / totalRange * 100)));
      const cid = chartId();
      const chartData = JSON.stringify({
        entry: t.entry_price, stop: t.stop_price,
        target: t.target_price, direction: t.direction,
        barsHeld: t.bars_elapsed, horizon: t.horizon_bars
      });

      return `
      <div class="mental-card mental-open-card">
        <div class="pred-chart-row">
          <div class="pred-info-col">
            <div class="mental-header">
              <span>
                <span class="${dirClass(t.direction)}" style="font-size:14px">${t.direction}</span>
                <strong>${t.strategy}</strong>
              </span>
              <span class="mental-live-pnl ${signClass(unrealized)}">${unrealized >= 0 ? '+' : ''}${unrealized.toFixed(1)} pts ($${unrealizedDollars.toFixed(2)})</span>
            </div>
            <div class="pred-levels-bar" style="margin:6px 0">
              <span class="pls">SL $${t.stop_price.toFixed(0)}</span>
              <div class="pred-progress-track">
                <div class="pred-progress-fill ${unrealized >= 0 ? 'fill-green' : 'fill-red'}" style="width:${progressPct}%"></div>
                <div class="pred-progress-dot" style="left:${progressPct}%"></div>
              </div>
              <span class="plt">TP $${t.target_price.toFixed(0)}</span>
            </div>
            <div class="mental-meta">
              <span>R:R 1:${t.rr_ratio?.toFixed(1) || '2.5'}</span>
              <span>Risk $${t.risk_dollars?.toFixed(2) || '20'}</span>
              <span>${t.lot_size?.toFixed(2) || '?'} lots</span>
              <span>Conf ${(t.confidence * 100).toFixed(0)}%</span>
              <span>Bar ${t.bars_elapsed}/${t.horizon_bars}</span>
              <span>MFE +${t.max_favorable?.toFixed(1) || '0'}</span>
              <span>MAE ${t.max_adverse?.toFixed(1) || '0'}</span>
            </div>
            <div class="mental-reasons">
              ${(t.reasons || []).map(r => `<div class="reason-line">${r}</div>`).join('')}
            </div>
          </div>
          <div class="pred-chart-col">
            <canvas id="${cid}" class="mini-chart" data-mini-chart='${chartData}'></canvas>
          </div>
        </div>
        <div class="mental-progress">
          <div class="mental-progress-bar" style="width:${barPct}%"></div>
        </div>
      </div>`;
    }).join('');
    setTimeout(drawAllCharts, 50);
  }

  // Resolved trades
  if (resolved.length === 0) {
    $('mental-resolved').innerHTML = '<p class="sub">No resolved trades yet — first trade is being tracked</p>';
  } else {
    $('mental-resolved').innerHTML = resolved.slice(0, 20).map(t => {
      const pnlCls = t.outcome === 'WON' ? 'pred-correct' : t.outcome === 'LOST' ? 'pred-wrong' : '';
      return `
      <div class="mental-card ${pnlCls}">
        <div class="mental-header">
          <span>
            ${outcomeTag(t.outcome)}
            <span class="${dirClass(t.direction)}">${t.direction}</span>
            <strong>${t.strategy}</strong>
          </span>
          <span class="${signClass(t.pnl_dollars)}" style="font-family:var(--mono);font-weight:700">
            ${t.pnl_dollars >= 0 ? '+' : ''}$${t.pnl_dollars?.toFixed(2)} (${t.pnl_points >= 0 ? '+' : ''}${t.pnl_points?.toFixed(1)} pts)
          </span>
        </div>
        <div class="mental-meta">
          <span>Entry $${t.entry_price?.toFixed(2)}</span>
          <span>Exit $${t.exit_price?.toFixed(2)}</span>
          <span>Stop $${t.stop_price?.toFixed(2)}</span>
          <span>Target $${t.target_price?.toFixed(2)}</span>
          <span>R:R 1:${t.rr_ratio?.toFixed(1)}</span>
          <span>${t.bars_held} bars</span>
          <span>MFE +${t.max_favorable?.toFixed(1)}</span>
          <span>MAE ${t.max_adverse?.toFixed(1)}</span>
        </div>
        <div class="mental-reasons">
          ${(t.reasons || []).map(r => `<div class="reason-line">${r}</div>`).join('')}
        </div>
        <div class="mental-meta sub">
          ${t.market_context?.regime ? `regime: ${t.market_context.regime}` : ''}
          ${t.market_context?.gold_pressure ? ` | pressure: ${t.market_context.gold_pressure}` : ''}
        </div>
      </div>`;
    }).join('');
  }
}

function renderKnowledge(d) {
  $('obs-count').textContent = (d.total_observations || 0) + ' observations';
  const pc = d.pattern_counts || {};
  const sorted = Object.entries(pc).sort((a, b) => b[1] - a[1]);
  $('patterns-list').innerHTML = sorted
    .map(([k, v]) => kv(k.replace(/_/g, ' '), v.toLocaleString()))
    .join('');
}

function renderStrategies(d) {
  const tbody = $('strategies-table').querySelector('tbody');
  tbody.innerHTML = (d.hypotheses || []).map(h => {
    const pnlCls = h.total_pnl > 0 ? 'pos' : h.total_pnl < 0 ? 'neg' : '';
    return `<tr>
      <td style="font-family:var(--sans)">${h.name.replace(/_/g, ' ')}</td>
      <td class="${dirClass(h.direction)}">${h.direction}</td>
      <td>${h.signals_generated}</td>
      <td class="pos">${h.correct}</td>
      <td class="neg">${h.wrong}</td>
      <td class="${pnlCls}">${h.total_pnl >= 0 ? '+' : ''}${fmt(h.total_pnl, 1)}</td>
      <td>${h.status}</td>
    </tr>`;
  }).join('');
}

function renderPredictions(d, status) {
  // Accuracy scoreboard
  const acc = d.pattern_accuracy || {};
  $('pred-accuracy').innerHTML = Object.entries(acc).map(([pat, data]) => {
    const total = (data.correct || 0) + (data.wrong || 0) + (data.partial || 0);
    const wr = total > 0 ? ((data.correct / total) * 100).toFixed(0) + '%' : '--';
    const cls = data.correct > data.wrong ? 'pos' : data.wrong > 0 ? 'neg' : '';
    return kv(pat.replace(/_/g, ' '), `${data.correct}W ${data.wrong}L (${wr})`, cls);
  }).join('');

  // Active predictions with mini-charts
  const active = d.active || [];
  const curPrice = status && status.latest_tick ? parseFloat(status.latest_tick.m) : 0;

  $('pred-active').innerHTML = active.length === 0
    ? '<p class="sub">No active predictions</p>'
    : active.map(p => {
      const reasons = (p.reasons || []);
      const cid = chartId();
      const barsHeld = p.bars_held || 0;
      const horizon = p.horizon_bars || 20;
      const barPct = (barsHeld / horizon * 100).toFixed(0);
      const unrealized = p.direction === 'LONG'
        ? curPrice - p.entry_price
        : p.entry_price - curPrice;
      const distToTarget = p.direction === 'LONG'
        ? p.target_price - curPrice
        : curPrice - p.target_price;
      const distToStop = p.direction === 'LONG'
        ? curPrice - p.stop_price
        : p.stop_price - curPrice;
      const totalRange = Math.abs(p.target_price - p.stop_price) || 1;
      const progressPct = Math.max(0, Math.min(100, ((distToStop / totalRange) * 100)));

      const chartData = JSON.stringify({
        entry: p.entry_price, stop: p.stop_price,
        target: p.target_price, direction: p.direction,
        barsHeld: barsHeld, horizon: horizon
      });

      return `
      <div class="pred-card pred-with-chart">
        <div class="pred-chart-row">
          <div class="pred-info-col">
            <div class="pred-header">
              <span class="${dirClass(p.direction)}" style="font-size:14px">${p.direction} @ $${fmt(p.entry_price)}</span>
              <span class="${signClass(unrealized)}" style="font-family:var(--mono);font-weight:700">
                ${unrealized >= 0 ? '+' : ''}${unrealized.toFixed(1)} pts
              </span>
            </div>
            <div class="pred-meta">
              ${p.timeframe} | conf ${pct(p.confidence)} | bar ${barsHeld}/${horizon} | ${p.strategy_hypothesis || ''}
            </div>
            <div class="pred-levels-bar">
              <span class="pls">SL $${fmt(p.stop_price, 0)}</span>
              <div class="pred-progress-track">
                <div class="pred-progress-fill ${unrealized >= 0 ? 'fill-green' : 'fill-red'}" style="width:${progressPct}%"></div>
                <div class="pred-progress-dot" style="left:${progressPct}%"></div>
              </div>
              <span class="plt">TP $${fmt(p.target_price, 0)}</span>
            </div>
            <div class="pred-meta">
              MFE +${fmt(p.max_favorable || 0, 1)} | MAE ${fmt(p.max_adverse || 0, 1)} | To target: ${distToTarget.toFixed(1)} pts | To stop: ${distToStop.toFixed(1)} pts
            </div>
            <div class="pred-reasons">
              ${reasons.map(r => `<div class="reason-line">${r}</div>`).join('')}
            </div>
          </div>
          <div class="pred-chart-col">
            <canvas id="${cid}" class="mini-chart" data-mini-chart='${chartData}'></canvas>
          </div>
        </div>
        <div class="pred-bar-progress">
          <div class="pred-bar-fill" style="width:${barPct}%"></div>
        </div>
      </div>`;
    }).join('');

  // Draw charts after DOM update
  setTimeout(drawAllCharts, 50);

  // Resolved predictions — with outcomes
  const resolved = (d.resolved || []).slice().reverse();
  $('pred-resolved').innerHTML = resolved.length === 0
    ? '<p class="sub">No resolved predictions yet</p>'
    : resolved.map(p => {
      const reasons = (p.reasons || []);
      const pnl = p.pnl_points;
      const pnlStr = typeof pnl === 'number' ? (pnl >= 0 ? '+' : '') + pnl.toFixed(1) + ' pts' : '';
      return `
      <div class="pred-card ${p.outcome === 'CORRECT' ? 'pred-correct' : p.outcome === 'WRONG' ? 'pred-wrong' : ''}">
        <div class="pred-header">
          <span>
            ${outcomeTag(p.outcome)}
            <span class="${dirClass(p.direction)}">${p.direction}</span>
            @ $${fmt(p.entry_price)}
          </span>
          <span class="${signClass(pnl)}" style="font-family:var(--mono)">${pnlStr}</span>
        </div>
        <div class="pred-meta">
          ${p.timeframe} | conf ${pct(p.confidence)} | exit $${fmt(p.exit_price)} | ${p.bars_held || '?'} bars held
        </div>
        <div class="pred-reasons">
          ${reasons.map(r => `<div class="reason-line">${r}</div>`).join('')}
        </div>
        ${p.outcome_reason ? `<div class="pred-outcome-reason">${p.outcome_reason}</div>` : ''}
      </div>`;
    }).join('');
}

function renderTrades(d) {
  const trades = (d.trades || []).slice(0, 40);
  $('trade-count').textContent = trades.length + ' trades';

  $('trades-list').innerHTML = trades.map(t => {
    const time = (t.timestamp || '').slice(5, 16);
    const reasons = Array.isArray(t.reasons) ? t.reasons : [];
    const ctx = t.market_context || {};
    const confPct = typeof t.confidence === 'number' ? (t.confidence * 100).toFixed(0) : t.confidence;

    // Context pills
    let contextHtml = '';
    if (ctx.regime) contextHtml += `<span class="ctx-pill">regime: ${ctx.regime}</span>`;
    if (ctx.gold_pressure) contextHtml += `<span class="ctx-pill ${signClass(ctx.gold_pressure)}">pressure: ${fmt(ctx.gold_pressure, 1)}</span>`;
    if (ctx.risk_sentiment) contextHtml += `<span class="ctx-pill">risk: ${fmt(ctx.risk_sentiment, 2)}</span>`;
    if (t.session) contextHtml += `<span class="ctx-pill">${t.session}</span>`;

    return `
    <div class="trade-card">
      <div class="trade-header">
        <span>
          <span class="trade-time">${time}</span>
          <span class="${dirClass(t.direction)}">${t.direction}</span>
          <span style="font-family:var(--sans);font-weight:600">${t.strategy}</span>
        </span>
        <span>
          <span style="font-family:var(--mono)">$${fmt(t.entry_price)}</span>
          <span class="conf-badge">conf ${confPct}%</span>
        </span>
      </div>
      ${reasons.length > 0 ? `<div class="trade-reasons">${reasons.map(r => `<div class="reason-line">${r}</div>`).join('')}</div>` : ''}
      ${contextHtml ? `<div class="trade-context">${contextHtml}</div>` : ''}
      ${t.outcome ? `<div class="trade-outcome">${outcomeTag(t.outcome)} ${t.exit_price ? 'exit $' + fmt(t.exit_price) : ''} ${t.pnl ? (t.pnl >= 0 ? '+' : '') + fmt(t.pnl, 1) + ' pts' : ''}</div>` : ''}
    </div>`;
  }).join('');
}

function renderMarket(d) {
  // RMT badge
  const rmt = d.rmt_diagnostics || {};
  if (rmt.n_signal !== undefined) {
    $('rmt-badge').textContent = `RMT: ${rmt.n_signal}/${rmt.N} signal eigenvalues`;
  }

  // Correlation bars
  const gc = d.gold_correlations || {};
  const entries = Object.entries(gc).slice(0, 12);
  $('corr-bars').innerHTML = entries.map(([mkt, val]) => {
    const w = Math.min(Math.abs(val) * 50, 50);
    const cls = val >= 0 ? 'pos' : 'neg';
    const style = val >= 0
      ? `left:50%;width:${w}%`
      : `right:50%;width:${w}%`;
    return `<div class="corr-row">
      <span class="corr-label">${mkt}</span>
      <div class="corr-bar-wrap"><div class="corr-bar ${cls}" style="${style}"></div></div>
      <span class="corr-val ${signClass(val)}">${val >= 0 ? '+' : ''}${val.toFixed(3)}</span>
    </div>`;
  }).join('');

  // Lead-lag
  const ll = d.lead_lag || {};
  $('lead-lag-list').innerHTML = Object.entries(ll).slice(0, 6).map(([mkt, info]) => {
    const lag = info.best_lag || 0;
    const label = lag > 0 ? `leads +${lag}d` : lag < 0 ? `lags ${lag}d` : 'sync';
    return kv(mkt, `${label} (${fmt(info.correlation, 3)})`);
  }).join('');

  // Regime
  const regime = d.latest_regime || {};
  if (regime.regime) {
    const scores = regime.scores || {};
    $('regime-info').innerHTML = `
      <div style="font-size:18px;font-weight:700;margin:4px 0">${regime.regime.replace(/_/g, ' ').toUpperCase()}</div>
      ${Object.entries(scores).map(([k, v]) => kv(k.replace(/_/g, ' '), fmt(v, 1))).join('')}
    `;
  }
}

function renderIntel(d, knowledge) {
  // COT
  const cot = d.cot || {};
  if (cot.report_date) {
    const specNet = (cot.non_commercial_long || 0) - (cot.non_commercial_short || 0);
    $('cot-info').innerHTML = [
      kv('Report', cot.report_date),
      kv('Spec Net', specNet.toLocaleString(), signClass(specNet)),
      kv('Comm Net', (cot.commercial_net || 0).toLocaleString(), signClass(cot.commercial_net)),
      kv('Open Interest', (cot.open_interest_total || 0).toLocaleString()),
    ].join('');
  } else {
    $('cot-info').innerHTML = '<p class="sub">No COT data</p>';
  }

  // Calendar (from lifecycle / static)
  const lc = d.lifecycle || {};
  $('calendar-info').innerHTML = [
    kv('Daily trades', lc.daily_trade_count || 0),
    kv('Live trades', lc.daily_live_count || 0),
    kv('Positions', lc.current_positions || 0),
    kv('Daily PnL', fmt(lc.daily_pnl_pct, 2) + '%', signClass(lc.daily_pnl_pct)),
  ].join('');

  // Levels & sessions
  const sr = (knowledge || {}).support_resistance || [];
  const ss = (knowledge || {}).session_stats || {};
  let levelsHtml = sr.map(l => kv(l.type, '$' + fmt(l.level))).join('');
  levelsHtml += Object.entries(ss).map(([s, data]) =>
    kv(s + ' range', '$' + fmt(data.avg_range) + ` (${data.observations} obs)`)
  ).join('');
  $('levels-info').innerHTML = levelsHtml;

  // Lifecycle details
  const ll = d.indicator_lead_lag || {};
  $('lifecycle-info').innerHTML = [
    kv('Best predictor (5bar)', ll.strongest_predictor_5bar || '--'),
    kv('Best predictor (10bar)', ll.strongest_predictor_10bar || '--'),
    kv('Pre-move patterns', d.pre_move_pattern_count || 0),
    kv('Round # bounce rate', d.round_number?.bounce_rate ? pct(d.round_number.bounce_rate) : '--'),
    kv('Journal entries', d.journal_entries || 0),
  ].join('');
}

function renderLogs(d) {
  const el = $('log-output');
  const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
  el.textContent = (d.lines || []).join('\n');
  if (wasAtBottom) el.scrollTop = el.scrollHeight;
}

// ─── Bot Toggle ───
async function toggleBot() {
  const btn = $('bot-toggle');
  btn.disabled = true;
  btn.textContent = '...';
  try {
    const endpoint = botRunning ? '/api/bot/stop' : '/api/bot/start';
    await fetch(API + endpoint, { method: 'POST' });
    setTimeout(fetchAll, 2000);
  } catch (e) {
    console.error(e);
  }
  btn.disabled = false;
}

// ─── Init ───
fetchAll();
setInterval(fetchAll, 5000);
