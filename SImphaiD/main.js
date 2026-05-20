// main.js — SimphAI · FaceMesh + WebGazer + Attention Engine
// ─────────────────────────────────────────────────────────────

/* ======= Config ======= */
const PROCESS_EVERY_N      = 1;
const SMOOTH_WINDOW        = 5;
let   EAR_THRESH           = 0.42;
const ATTENTION_TRIGGER    = 0.5;
const ATTENTION_TRIGGER_MS = 4000;

/* ======= State ======= */
let streaming              = false;
let rafId                  = null;
let smoothBuffer           = [];
let blinkTotal             = 0;
let latestGaze             = null;
let attentionEMA           = 1.0;
let lastAttentionTimeBelow = 0;
let clickPoints            = [];
let clickLog               = [];
let calibrationReminderShown = false;
let sessionStartTime       = null;   // set on startCamera()
let attentionRecords       = [];     // filled per frame for summary

/* ======= Telemetry / Firebase ======= */
let telemetryEnabled = false;
let firestore        = null;
const firebaseConfig = null; // paste your Firebase config here to enable

/* ======= Blink state machine ======= */
let earBuffer       = [];
const EAR_SMOOTH_FRAMES = 4;
let eyeClosed       = false;
let eyeClosedSince  = 0;

/* ======= Landmark indices ======= */
const LEFT_EYE_IDX  = [33, 160, 158, 133, 144, 153];
const RIGHT_EYE_IDX = [263, 387, 385, 362, 373, 380];

/* ======= DOM refs ======= */
const videoEl          = document.getElementById('camera');
const canvas           = document.getElementById('overlay');
const ctx              = canvas.getContext('2d');
const startBtn         = document.getElementById('startBtn');
const stopBtn          = document.getElementById('stopBtn');
const adaptiveToggle   = document.getElementById('adaptiveToggle');
const calibrateBtn     = document.getElementById('calibrateBtn');
const downloadLogBtn   = document.getElementById('downloadLog');
const facePresentSpan  = document.getElementById('facePresent');
const attnValSpan      = document.getElementById('attnVal');
const blinkValSpan     = document.getElementById('blinkVal');
const gazeValSpan      = document.getElementById('gazeVal');
const headValSpan      = document.getElementById('headVal');
const earValSpan       = document.getElementById('earVal');
const calmToggle       = document.getElementById('calmModeToggle');
const calmBadge        = document.getElementById('calmBadge');
const attnBar          = document.getElementById('attnBar');

/* ======= Utility ======= */
function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function computeEAR(landmarks, idx) {
  try {
    const [p1, p2, p3, p4, p5, p6] = idx.map(i => landmarks[i]);
    const A = dist(p2, p6), B = dist(p3, p5), C = dist(p1, p4);
    return C === 0 ? 1.0 : (A + B) / (2.0 * C);
  } catch { return 1.0; }
}

function headOrientationScore(landmarks) {
  if (!landmarks) return 0;
  const left = landmarks[33], right = landmarks[263], nose = landmarks[1];
  const deviation = Math.abs(nose.x - (left.x + right.x) / 2);
  return Math.max(0, 1 - deviation / 0.12);
}

function pushSmooth(landmarks) {
  if (!landmarks) return null;
  const pts = landmarks.map(l => ({ x: l.x, y: l.y, z: l.z || 0 }));
  smoothBuffer.push(pts);
  if (smoothBuffer.length > SMOOTH_WINDOW) smoothBuffer.shift();
  const len = smoothBuffer[0].length;
  return Array.from({ length: len }, (_, i) => ({
    x: smoothBuffer.reduce((s, b) => s + b[i].x, 0) / smoothBuffer.length,
    y: smoothBuffer.reduce((s, b) => s + b[i].y, 0) / smoothBuffer.length,
    z: smoothBuffer.reduce((s, b) => s + b[i].z, 0) / smoothBuffer.length,
  }));
}

/* ======= Fallback drawing utils ======= */
if (typeof drawConnectors === 'undefined') {
  window.drawConnectors = (ctxIn, landmarks, connections, style = {}) => {
    ctxIn.save();
    ctxIn.strokeStyle = style.color || '#00FF00';
    ctxIn.lineWidth   = style.lineWidth || 1;
    const w = canvas.width, h = canvas.height;
    for (const [a, b] of connections) {
      if (!landmarks[a] || !landmarks[b]) continue;
      ctxIn.beginPath();
      ctxIn.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      ctxIn.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      ctxIn.stroke();
    }
    ctxIn.restore();
  };
}
if (typeof drawLandmarks === 'undefined') {
  window.drawLandmarks = (ctxIn, landmarks, style = {}) => {
    ctxIn.save();
    ctxIn.fillStyle = style.color || '#FFCC00';
    const r = style.radius || 1.2, w = canvas.width, h = canvas.height;
    for (const p of landmarks) {
      if (!p) continue;
      ctxIn.beginPath();
      ctxIn.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
      ctxIn.fill();
    }
    ctxIn.restore();
  };
}

/* ======= FaceMesh onResults ======= */
let frameCount = 0;

function onResults(results) {
  frameCount++;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  try { ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height); } catch {}

  const has = results?.multiFaceLandmarks?.length > 0;
  facePresentSpan.textContent = has ? 'Yes ✓' : 'No';

  if (!has) {
    attentionEMA = attentionEMA * 0.85;
    updateAttentionUI(attentionEMA);
    attentionRecords.push(attentionEMA);
    return;
  }

  const raw      = results.multiFaceLandmarks[0];
  const smoothed = pushSmooth(raw);

  if (smoothed) {
    if (typeof FACEMESH_TESSELATION !== 'undefined')
      try { drawConnectors(ctx, smoothed, FACEMESH_TESSELATION, { color: '#1e3a4a', lineWidth: 1 }); } catch {}
    if (typeof FACEMESH_LEFT_EYE  !== 'undefined') drawConnectors(ctx, smoothed, FACEMESH_LEFT_EYE,  { color: '#00FF88', lineWidth: 1 });
    if (typeof FACEMESH_RIGHT_EYE !== 'undefined') drawConnectors(ctx, smoothed, FACEMESH_RIGHT_EYE, { color: '#00FF88', lineWidth: 1 });
    if (typeof FACEMESH_LIPS      !== 'undefined') drawConnectors(ctx, smoothed, FACEMESH_LIPS,      { color: '#FF6666', lineWidth: 1 });
    drawLandmarks(ctx, smoothed, { color: '#38bdf8', radius: 1.0 });
  }

  /* --- EAR + blink detection --- */
  let ear = 1.0;
  try {
    const eL = computeEAR(raw, LEFT_EYE_IDX);
    const eR = computeEAR(raw, RIGHT_EYE_IDX);
    earBuffer.push((eL + eR) / 2);
    if (earBuffer.length > EAR_SMOOTH_FRAMES) earBuffer.shift();
    ear = earBuffer.reduce((s, v) => s + v, 0) / earBuffer.length;
    if (earValSpan) earValSpan.textContent = ear.toFixed(3);

    const now = Date.now();
    const MIN_CLOSED_MS = 80, MAX_CLOSED_MS = 1200;

    if (!eyeClosed && ear < EAR_THRESH) {
      eyeClosed = true;
      eyeClosedSince = now;
    } else if (eyeClosed && ear >= EAR_THRESH) {
      const dur = now - eyeClosedSince;
      if (dur >= MIN_CLOSED_MS && dur <= MAX_CLOSED_MS) {
        blinkTotal++;
        blinkValSpan.textContent = blinkTotal;
        console.log(`👁 Blink #${blinkTotal} — ${dur}ms`);
      }
      eyeClosed = false;
      eyeClosedSince = 0;
    }
  } catch (e) { console.warn('EAR error', e); }

  /* --- Head orientation --- */
  const headScore = headOrientationScore(raw);
  const headFacing = headScore > 0.7;
  headValSpan.textContent = headFacing
    ? `Forward (${headScore.toFixed(2)})`
    : `Turned (${headScore.toFixed(2)})`;

  /* --- Gaze score --- */
  let gazeScore = 0.3;
  if (latestGaze && typeof latestGaze.x === 'number') {
    const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
    const dx = (latestGaze.x - cx) / cx, dy = (latestGaze.y - cy) / cy;
    gazeScore = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy));
    gazeValSpan.textContent = `On page (d=${Math.sqrt(dx*dx+dy*dy).toFixed(2)})`;
  } else {
    gazeValSpan.textContent = 'Not calibrated';
  }

  /* --- Attention score --- */
  const OPEN = 0.28, CLOSED = 0.12;
  let E = (ear - CLOSED) / (OPEN - CLOSED);
  E = Math.max(0, Math.min(1, E));
  const rawAtt = 0.50 * gazeScore + 0.30 * headScore + 0.20 * E;

  // Asymmetric EMA: faster drop, slower rise
  attentionEMA = attentionEMA > rawAtt
    ? 0.70 * attentionEMA + 0.30 * rawAtt
    : 0.85 * attentionEMA + 0.15 * rawAtt;

  attentionRecords.push(attentionEMA);
  updateAttentionUI(attentionEMA);

  if (frameCount % 60 === 0)
    console.log(`📊 Attention: ${attentionEMA.toFixed(2)} | Gaze: ${gazeScore.toFixed(2)}, Head: ${headScore.toFixed(2)}, Eyes: ${E.toFixed(2)}`);

  /* --- Adaptive trigger --- */
  if (attentionEMA < ATTENTION_TRIGGER) {
    if (!lastAttentionTimeBelow) {
      lastAttentionTimeBelow = Date.now();
    } else {
      const elapsed = Date.now() - lastAttentionTimeBelow;

      // Auto calm mode on critical drop
      if (attentionEMA < 0.45 && !calmToggle.checked) {
        calmToggle.checked = true;
        applyCalmMode(true);
        console.log('🌙 Auto Calm Mode activated');
      }

      if (elapsed > ATTENTION_TRIGGER_MS && adaptiveToggle.checked) {
        showAdaptiveHint();
        lastAttentionTimeBelow = Date.now();
      }
    }
  } else {
    lastAttentionTimeBelow = 0;
  }

  drawClickPoints();
}

/* ======= Attention UI updater ======= */
function updateAttentionUI(score) {
  const pct = Math.round(score * 100);
  attnValSpan.textContent = score.toFixed(2);
  attnValSpan.style.color = score > 0.6 ? '#34d399' : score > 0.4 ? '#fbbf24' : '#f87171';
  if (attnBar) attnBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

/* ======= FaceMesh setup ======= */
const faceMesh = new FaceMesh({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`
});
faceMesh.setOptions({
  maxNumFaces:            1,
  refineLandmarks:        true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence:  0.6
});
faceMesh.onResults(onResults);

/* ======= rAF loop ======= */
async function loopSendFrame() {
  if (!streaming) return;
  try { await faceMesh.send({ image: videoEl }); } catch (e) { console.error('faceMesh.send', e); }
  rafId = requestAnimationFrame(loopSendFrame);
}

/* ======= Camera start/stop ======= */
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    videoEl.srcObject = stream;
    await videoEl.play();
    streaming       = true;
    sessionStartTime = Date.now();
    attentionRecords = [];
    blinkTotal       = 0;
    blinkValSpan.textContent = '0';
    smoothBuffer     = [];
    canvas.width     = videoEl.videoWidth  || 640;
    canvas.height    = videoEl.videoHeight || 480;

    adaptiveToggle.checked = true;
    rafId = requestAnimationFrame(loopSendFrame);
    console.log('✅ Camera started. Face detection running.');

    setTimeout(() => {
      if (!calibrationReminderShown && streaming) {
        calibrationReminderShown = true;
        showCalibrationReminder();
      }
    }, 5000);
  } catch (e) {
    console.error('❌ Camera error:', e);
    alert(`Camera error: ${e.message}\n\nPlease allow camera access and refresh.`);
  }
}

function stopCamera() {
  if (videoEl.srcObject) videoEl.srcObject.getTracks().forEach(t => t.stop());
  streaming = false;
  if (rafId) cancelAnimationFrame(rafId);
  showSummaryModal();
  console.log('Camera stopped.');
}

/* ======= WebGazer ======= */
async function initWebGazer() {
  try {
    await webgazer
      .setRegression('ridge')
      .setGazeListener((data) => {
        latestGaze = (data?.x != null) ? { x: data.x, y: data.y, t: Date.now() } : null;
      })
      .begin();

    webgazer.showVideo(false);
    webgazer.showPredictionPoints(false);
    webgazer.showFaceOverlay(false);

    // Ensure gaze dot exists
    if (!document.getElementById('gazeDot')) {
      const d = document.createElement('div');
      d.id = 'gazeDot';
      document.body.appendChild(d);
    }

    setInterval(() => {
      const el = document.getElementById('gazeDot');
      if (!el) return;
      if (latestGaze) {
        el.style.display = 'block';
        el.style.left    = `${Math.max(0, Math.min(window.innerWidth,  latestGaze.x))}px`;
        el.style.top     = `${Math.max(0, Math.min(window.innerHeight, latestGaze.y))}px`;
      } else {
        el.style.display = 'none';
      }
    }, 40);

    console.log('✅ WebGazer ready. Click Calibrate Gaze after starting for accuracy.');
  } catch (e) {
    console.warn('⚠️ WebGazer failed:', e);
  }
}

function calibrateGaze() {
  const points = [
    { x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 }, { x: 0.5, y: 0.5 },
    { x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 }
  ];
  let idx = 0;

  function showPoint() {
    if (idx >= points.length) {
      showToast('✅ Calibration complete!', 'ok');
      return;
    }
    const p   = points[idx];
    const dot = document.createElement('div');
    Object.assign(dot.style, {
      position: 'fixed',
      left: (p.x * 100) + '%',
      top:  (p.y * 100) + '%',
      transform: 'translate(-50%,-50%)',
      width: '22px', height: '22px',
      background: '#38bdf8',
      borderRadius: '50%',
      zIndex: '99999',
      boxShadow: '0 0 0 4px rgba(56,189,248,0.3)',
      cursor: 'pointer',
      transition: 'transform 0.1s'
    });
    document.body.appendChild(dot);
    dot.addEventListener('click', () => {
      dot.remove();
      idx++;
      setTimeout(showPoint, 150);
    });
  }

  showToast('Click each blue dot to calibrate gaze (5 points)', 'info');
  setTimeout(showPoint, 800);
}

/* ======= Adaptive hint ======= */
function showAdaptiveHint() {
  console.log('🚨 Adaptive intervention triggered!');
  const el = document.createElement('div');
  Object.assign(el.style, {
    position:     'fixed',
    left:         '50%',
    top:          '18%',
    transform:    'translateX(-50%)',
    padding:      '32px 52px',
    background:   '#dc2626',
    color:        '#fff',
    borderRadius: '18px',
    zIndex:       '999999',
    fontSize:     '28px',
    fontWeight:   '800',
    boxShadow:    '0 0 60px rgba(220,38,38,0.8)',
    border:       '4px solid rgba(255,255,255,0.3)',
    textAlign:    'center',
    animation:    'pulseAlert 0.6s ease-in-out infinite',
    letterSpacing: '1px'
  });
  el.innerHTML = '🚨 ATTENTION DROPPED<br><span style="font-size:16px;font-weight:500;display:block;margin-top:10px;opacity:0.9">Take a break or refocus!</span>';
  document.body.appendChild(el);
  document.body.style.boxShadow = 'inset 0 0 80px rgba(220,38,38,0.5)';
  setTimeout(() => {
    el.style.opacity    = '0';
    el.style.transition = 'opacity 0.4s';
    document.body.style.boxShadow = '';
    setTimeout(() => el.remove(), 400);
  }, 5000);
}

/* ======= Calibration reminder ======= */
function showCalibrationReminder() {
  showToast('💡 Click "Calibrate Gaze" for better attention accuracy', 'info');
}

/* ======= Toast helper ======= */
function showToast(msg, type = 'info') {
  const colors = { info: '#38bdf8', ok: '#34d399', warn: '#fbbf24' };
  const el = document.createElement('div');
  Object.assign(el.style, {
    position:     'fixed',
    left:         '50%',
    bottom:       '10%',
    transform:    'translateX(-50%)',
    padding:      '12px 20px',
    background:   'rgba(13,21,32,0.95)',
    border:       `1px solid ${colors[type] || colors.info}`,
    color:        '#e2eaf4',
    borderRadius: '10px',
    zIndex:       '99999',
    fontSize:     '13px',
    boxShadow:    '0 4px 20px rgba(0,0,0,0.5)',
    backdropFilter: 'blur(8px)',
    transition:   'opacity 0.4s',
    whiteSpace:   'nowrap'
  });
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 400);
  }, 4000);
}

/* ======= Click heatmap ======= */
function addClickPoint(x, y) {
  clickPoints.push({ x, y, t: Date.now() });
  if (clickPoints.length > 400) clickPoints.shift();
}
function drawClickPoints() {
  const now = Date.now();
  clickPoints = clickPoints.filter(p => now - p.t < 12000);
  for (const p of clickPoints) {
    const age    = (now - p.t) / 1000;
    const alpha  = Math.max(0, 1 - age / 8);
    const radius = 8 + Math.max(0, 18 * (1 - alpha));
    ctx.beginPath();
    ctx.fillStyle = `rgba(255,90,90,${alpha * 0.9})`;
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}
document.addEventListener('click', (e) => {
  const rect = document.querySelector('.video-wrap').getBoundingClientRect();
  if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
    addClickPoint(Math.round(e.clientX - rect.left), Math.round(e.clientY - rect.top));
  }
  clickLog.push({ t: Date.now(), x: e.clientX, y: e.clientY, tag: e.target.tagName });
});

/* ======= Session summary ======= */
function showSummaryModal() {
  const mins         = ((Date.now() - sessionStartTime) / 60000).toFixed(2);
  const avg          = attentionRecords.length
    ? (attentionRecords.reduce((a, b) => a + b, 0) / attentionRecords.length).toFixed(2)
    : '—';
  const stability    = attentionRecords.length
    ? Math.max(0, 1 - (Math.max(...attentionRecords) - Math.min(...attentionRecords))).toFixed(2)
    : '—';

  document.getElementById('sumDuration').textContent = mins + ' min';
  document.getElementById('sumAvgAttn').textContent  = avg;
  document.getElementById('sumBlinks').textContent   = blinkTotal;
  document.getElementById('sumFocus').textContent    = stability;
  document.getElementById('summaryModal').style.display = 'flex';
}

/* ======= Calm mode ======= */
function applyCalmMode(on) {
  document.body.classList.toggle('calm', on);
  document.querySelector('.video-wrap')?.classList.toggle('calm', on);
  if (calmBadge) calmBadge.style.display = on ? 'block' : 'none';
}

/* ======= Firebase (optional) ======= */
function initFirebaseIfConfigured() {
  if (!firebaseConfig) {
    console.log('Firebase not configured — telemetry stays local.');
    return;
  }
  try {
    firebase.initializeApp(firebaseConfig);
    firestore = firebase.firestore();
    console.log('Firebase initialized.');
  } catch (e) {
    console.warn('Firebase init failed:', e);
  }
}

/* ======= Event listeners ======= */
startBtn.addEventListener('click', () => startCamera());
stopBtn.addEventListener('click',  () => stopCamera());
calibrateBtn.addEventListener('click', calibrateGaze);
calmToggle.addEventListener('change', () => applyCalmMode(calmToggle.checked));

downloadLogBtn.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({ clicks: clickLog }, null, 2)], { type: 'application/json' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'simphai_session_log.json';
  a.click();
});

document.getElementById('closeSummary').addEventListener('click', () => {
  document.getElementById('summaryModal').style.display = 'none';
});

/* ======= Boot ======= */
document.addEventListener('DOMContentLoaded', () => {
  initFirebaseIfConfigured();
  document.getElementById('consentModal').style.display = 'flex';

  document.getElementById('consentAccept').addEventListener('click', () => {
    telemetryEnabled = document.getElementById('consentTelemetry').checked;
    document.getElementById('consentModal').style.display = 'none';
    console.log('Consent accepted. Telemetry:', telemetryEnabled);
  });

  document.getElementById('consentDecline').addEventListener('click', () => {
    telemetryEnabled = false;
    document.getElementById('consentModal').style.display = 'none';
    console.log('Consent declined.');
  });
});

window.addEventListener('load', () => {
  console.log('=== SimphAI Loading ===');
  initWebGazer();
  [earValSpan, facePresentSpan, attnValSpan, gazeValSpan, headValSpan]
    .forEach(el => { if (el) el.textContent = '—'; });
  if (blinkValSpan) blinkValSpan.textContent = '0';
  console.log('Ready. Accept consent → click Start.');
});
