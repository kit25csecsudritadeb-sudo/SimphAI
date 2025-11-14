// main.js — integrated FaceMesh + WebGazer + blink + attention + click heatmap fallback

/* ======= Config ======= */
const PROCESS_EVERY_N = 1;
const SMOOTH_WINDOW = 5;
let EAR_THRESH = 0.42; // you can tune this later
const ATTENTION_TRIGGER = 0.5;  // Raised from 0.4 to trigger easier
const ATTENTION_TRIGGER_MS = 4000;  // Reduced from 8s to 4s for faster testing

/* ======= DOM ======= */
const videoEl = document.getElementById('camera');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const adaptiveToggle = document.getElementById('adaptiveToggle');
const calibrateBtn = document.getElementById('calibrateBtn');
const downloadLogBtn = document.getElementById('downloadLog');

const facePresentSpan = document.getElementById('facePresent');
const attnValSpan = document.getElementById('attnVal');
const blinkValSpan = document.getElementById('blinkVal');
const gazeValSpan = document.getElementById('gazeVal');
const headValSpan = document.getElementById('headVal');
const earValSpan = document.getElementById('earVal');

/* ======= State ======= */
let streaming = false;
let rafId = null;
let smoothBuffer = [];
let blinkTotal = 0, closedFrames = 0;
let latestGaze = null;
let attentionEMA = 1.0;
let lastAttentionTimeBelow = 0;
let clickPoints = []; // for heatmap fallback
let clickLog = [];
let calibrationReminderShown = false;
/* ======= Telemetry + Firebase (optional) ======= */
let telemetryBuffer = []; // holds events to flush
const TELEMETRY_FLUSH_MS = 3000;
let telemetryEnabled = false; // user consent

// Placeholder: Firebase will be initialized below if user pasted config
let firestore = null; // will hold firebase.firestore() if initialized

// blink smoothing + state-machine
let earBuffer = [];                 // recent EAR values for smoothing
const EAR_SMOOTH_FRAMES = 4;        // average across this many frames
let eyeClosed = false;              // current closed state
let eyeClosedSince = 0;             // timestamp when closed started


/* ======= Utility helpers ======= */
function dist(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.sqrt(dx*dx+dy*dy); }
const LEFT_EYE_IDX = [33,160,158,133,144,153];
const RIGHT_EYE_IDX = [263,387,385,362,373,380];
function computeEAR(landmarks, idxArr){
  try {
    const p1 = landmarks[idxArr[0]], p2 = landmarks[idxArr[1]], p3 = landmarks[idxArr[2]];
    const p4 = landmarks[idxArr[3]], p5 = landmarks[idxArr[4]], p6 = landmarks[idxArr[5]];
    const A = dist(p2,p6), B = dist(p3,p5), C = dist(p1,p4);
    return C === 0 ? 1.0 : (A + B) / (2.0 * C);
  } catch(e){ return 1.0; }
}
function headFacingCenter(landmarks){
  if(!landmarks) return false;
  const left = landmarks[33], right = landmarks[263], nose = landmarks[1];
  const eyeCenterX = (left.x + right.x)/2;
  return Math.abs(nose.x - eyeCenterX) < 0.06;
}

// New function: returns gradient score 0..1 for head orientation
function headOrientationScore(landmarks){
  if(!landmarks) return 0;
  const left = landmarks[33], right = landmarks[263], nose = landmarks[1];
  const eyeCenterX = (left.x + right.x)/2;
  const deviation = Math.abs(nose.x - eyeCenterX);

  // deviation: 0 = perfect center, >0.15 = turned away significantly
  // Map to score: 0 deviation = 1.0, 0.15+ deviation = 0
  const score = Math.max(0, 1 - (deviation / 0.12));
  return score;
}

/* smoothing */
function pushSmooth(landmarks){
  if (!landmarks) return null;
  const pts = landmarks.map(l => ({x:l.x,y:l.y,z:l.z||0}));
  smoothBuffer.push(pts);
  if (smoothBuffer.length > SMOOTH_WINDOW) smoothBuffer.shift();
  const len = smoothBuffer[0].length;
  const avg = [];
  for (let i=0;i<len;i++){
    let sx=0, sy=0, sz=0;
    for (let b=0;b<smoothBuffer.length;b++){ sx+=smoothBuffer[b][i].x; sy+=smoothBuffer[b][i].y; sz+=smoothBuffer[b][i].z||0; }
    avg.push({x: sx/smoothBuffer.length, y: sy/smoothBuffer.length, z: sz/smoothBuffer.length});
  }
  return avg;
}

/* ======= Fallback drawing utils if not loaded by MediaPipe ======= */
if (typeof drawConnectors === 'undefined' || typeof drawLandmarks === 'undefined') {
  console.warn('Drawing utils missing - using fallback');
  window.drawConnectors = function(ctxIn, landmarks, connections, style={}) {
    ctxIn.save(); ctxIn.strokeStyle = style.color || '#00FF00'; ctxIn.lineWidth = style.lineWidth || 1;
    const w = canvas.width, h = canvas.height;
    for (let i=0;i<connections.length;i++){
      const [a,b] = connections[i];
      if (!landmarks[a] || !landmarks[b]) continue;
      ctxIn.beginPath();
      ctxIn.moveTo(landmarks[a].x*w, landmarks[a].y*h);
      ctxIn.lineTo(landmarks[b].x*w, landmarks[b].y*h);
      ctxIn.stroke();
    }
    ctxIn.restore();
  };
  window.drawLandmarks = function(ctxIn, landmarks, style={}) {
    ctxIn.save(); ctxIn.fillStyle = style.color || '#FFCC00'; const r = style.radius || 1.2;
    const w = canvas.width, h = canvas.height;
    for (let i=0;i<landmarks.length;i++){
      const p=landmarks[i]; if(!p) continue;
      ctxIn.beginPath(); ctxIn.arc(p.x*w, p.y*h, r, 0, Math.PI*2); ctxIn.fill();
    }
    ctxIn.restore();
  };
}

/* ======= FaceMesh onResults ======= */
let frameCount = 0;
function onResults(results){
  frameCount++;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  try { ctx.drawImage(videoEl,0,0,canvas.width,canvas.height); } catch(e){}

  const has = results && results.multiFaceLandmarks && results.multiFaceLandmarks.length>0;
  facePresentSpan.textContent = has ? 'Yes' : 'No';

  // Log every 60 frames (roughly once per 2 seconds at 30fps)
  if (frameCount % 60 === 0) {
    console.log('Face detected:', has, '| Frame:', frameCount);
  }

  if (!has) {
    attentionEMA = attentionEMA * 0.85;
    attnValSpan.textContent = attentionEMA.toFixed(2);
    return;
  }

  const raw = results.multiFaceLandmarks[0];
  const smoothed = pushSmooth(raw);

  if (smoothed) {
    if (typeof FACEMESH_TESSELATION !== 'undefined') {
      try { drawConnectors(ctx, smoothed, FACEMESH_TESSELATION, {color:'#888', lineWidth:1}); } catch(e){}
    }
    if (typeof FACEMESH_LEFT_EYE !== 'undefined') drawConnectors(ctx, smoothed, FACEMESH_LEFT_EYE, {color:'#00FF88', lineWidth:1});
    if (typeof FACEMESH_RIGHT_EYE !== 'undefined') drawConnectors(ctx, smoothed, FACEMESH_RIGHT_EYE, {color:'#00FF88', lineWidth:1});
    if (typeof FACEMESH_LIPS !== 'undefined') drawConnectors(ctx, smoothed, FACEMESH_LIPS, {color:'#FF6666', lineWidth:1});
    drawLandmarks(ctx, smoothed, {color:'#FFFF00', radius:1.2});
  }
    // --- EAR + blink detection (smoothed + state machine)
  let ear = 1.0;
  try {
    const eL = computeEAR(raw, LEFT_EYE_IDX);
    const eR = computeEAR(raw, RIGHT_EYE_IDX);
    const instantEAR = (eL + eR) / 2;
    // push into buffer & compute smoothed EAR
    earBuffer.push(instantEAR);
    if (earBuffer.length > EAR_SMOOTH_FRAMES) earBuffer.shift();
    // average
    const sum = earBuffer.reduce((s,v)=>s+v,0);
    ear = sum / earBuffer.length;
    // Log EAR every 60 frames to avoid spam
    if (frameCount % 60 === 0) {
      console.log('EAR:', ear.toFixed(3), '| Blinks:', blinkTotal);
    }
    if (earValSpan) earValSpan.textContent = ear.toFixed(3);


    // debug: show a few EAR values (uncomment to spam less)
    // console.log('EAR raw', instantEAR.toFixed(3), 'smoothed', ear.toFixed(3));

    const now = Date.now();
    const CLOSED_THRESH = EAR_THRESH; // use your EAR_THRESH (tune later)
    const MIN_CLOSED_MS = 80;   // ignore extremely short closures
    const MAX_CLOSED_MS = 1200;  // treat longer than this as not a blink (maybe attention drop)

    if (!eyeClosed && ear < CLOSED_THRESH) {
      // eyes just went closed
      eyeClosed = true;
      eyeClosedSince = now;
    } else if (eyeClosed) {
      // currently closed — check for re-open
      if (ear >= CLOSED_THRESH) {
        const closedFor = now - eyeClosedSince;
        // count as blink only if closed duration reasonable
        if (closedFor >= MIN_CLOSED_MS && closedFor <= MAX_CLOSED_MS) {
          blinkTotal++;
          blinkValSpan.textContent = blinkTotal;
          console.log('👁️ Blink detected! Duration:', closedFor, 'ms | Total blinks:', blinkTotal);
        } else {
          // too short or too long; only log if debugging
          // console.log('No-blink (dur ms)=', closedFor, 'ear=', ear.toFixed(3));
        }
        eyeClosed = false;
        eyeClosedSince = 0;
      } else {
        // still closed; nothing to do
      }
    }
  } catch(e){ console.warn('EAR error', e); }


  // --- head facing (use gradient score for more sensitivity)
  const headScore = headOrientationScore(raw);
  const headFacing = headScore > 0.7; // binary for display purposes
  headValSpan.textContent = headFacing ? 'Forward (' + headScore.toFixed(2) + ')' : 'Turned (' + headScore.toFixed(2) + ')';

  // --- gaze: compute continuous score from latestGaze distance to center
  let gazeScore = 0; // 0..1 (1 = looking at center)
  let gazeOnPage = false;
  if (latestGaze && typeof latestGaze.x === 'number') {
    const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
    const dx = (latestGaze.x - cx) / cx;
    const dy = (latestGaze.y - cy) / cy;
    const distNorm = Math.sqrt(dx*dx + dy*dy);
    gazeScore = Math.max(0, 1 - distNorm);
    gazeOnPage = true;
    gazeValSpan.textContent = 'On page (d=' + distNorm.toFixed(2) + ')';
  } else {
    // When gaze not available, assume not looking (penalize attention)
    gazeScore = 0.3; // small baseline instead of 0 to avoid too harsh penalty
    gazeOnPage = false;
    gazeValSpan.textContent = 'Not calibrated (assuming distracted)';
  }

  // --- attention components
  // Remove face presence as separate factor since we already know face exists here
  // Focus on: gaze direction, eye openness, and head orientation
  const G = gazeScore;
  const OPEN = 0.28, CLOSED = 0.12;
  let E = (ear - CLOSED) / (OPEN - CLOSED); E = Math.max(0, Math.min(1, E));
  const H = headScore; // Use gradient score instead of binary

  // Rebalanced weights (total = 1.0)
  // Gaze is most important, then head orientation, then eye openness
  const wG=0.50, wH=0.30, wE=0.20;
  const rawAtt = wG*G + wH*H + wE*E;

  // Faster decay when attention drops (more responsive)
  attentionEMA = attentionEMA > rawAtt ?
    0.70*attentionEMA + 0.30*rawAtt :  // Faster drop
    0.85*attentionEMA + 0.15*rawAtt;   // Slower rise

  attnValSpan.textContent = attentionEMA.toFixed(2);
  attnValSpan.style.color = attentionEMA > 0.6 ? '#4ade80' : attentionEMA > 0.4 ? '#fbbf24' : '#fb7185';

  // Log attention details every 60 frames
  if (frameCount % 60 === 0) {
    console.log('📊 Attention:', attentionEMA.toFixed(2), '| Raw:', rawAtt.toFixed(2), '| Gaze='+G.toFixed(2), 'Head='+H.toFixed(2), 'Eyes='+E.toFixed(2));
  }

  // --- adaptive trigger
  if (attentionEMA < ATTENTION_TRIGGER) {
    if (!lastAttentionTimeBelow) {
      lastAttentionTimeBelow = Date.now();
      console.log('⚠️ Attention below threshold! Timer started. Need to stay below', ATTENTION_TRIGGER, 'for', ATTENTION_TRIGGER_MS/1000, 'seconds');
      console.log('   Adaptive toggle is:', adaptiveToggle.checked ? 'ON ✅' : 'OFF ❌ (turn it ON to see alerts!)');
    } else {
      const timeBelow = Date.now() - lastAttentionTimeBelow;
      // Log every 60 frames while below threshold
      if (frameCount % 60 === 0) {
        console.log('   ⏱️ Low attention for', (timeBelow/1000).toFixed(1), 's /', (ATTENTION_TRIGGER_MS/1000), 's needed');
      }

      if (timeBelow > ATTENTION_TRIGGER_MS) {
        if (adaptiveToggle.checked) {
          showAdaptiveHint();
          lastAttentionTimeBelow = Date.now(); // reset timer
        } else {
          console.log('   ⚠️ Trigger condition met but Adaptive toggle is OFF! Turn it ON to see the alert.');
        }
      }
    }
  } else {
    if (lastAttentionTimeBelow) {
      console.log('✅ Attention recovered! Timer reset.');
    }
    lastAttentionTimeBelow = 0;
  }

  // draw click points last
  drawClickPoints();
}

/* mediaPipe FaceMesh */
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`
});
faceMesh.setOptions({maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.5, minTrackingConfidence:0.6});
faceMesh.onResults(onResults);

/* requestAnimationFrame loop */
async function loopSendFrame(){
  if (!streaming) return;
  try { await faceMesh.send({image: videoEl}); } catch(e){ console.error('faceMesh.send error', e); }
  rafId = requestAnimationFrame(loopSendFrame);
}

/* ======= Camera start/stop ======= */
async function startCamera(){
  try {
    console.log('Requesting camera access...');
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    videoEl.srcObject = stream;
    await videoEl.play();
    streaming = true;
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    smoothBuffer = [];
    rafId = requestAnimationFrame(loopSendFrame);

    // Auto-enable adaptive mode
    adaptiveToggle.checked = true;
    console.log('✅ Camera started successfully. Face detection running...');
    console.log('✅ Adaptive mode auto-enabled - alerts will show when attention drops!');
    console.log('Canvas size:', canvas.width, 'x', canvas.height);

    // Show calibration reminder after 5 seconds
    setTimeout(() => {
      if (!calibrationReminderShown && streaming) {
        calibrationReminderShown = true;
        showCalibrationReminder();
      }
    }, 5000);
  } catch(e){
    console.error('❌ Camera error:', e);
    alert('Camera error: '+e.message+'\n\nPlease allow camera access and refresh the page.');
  }
}
function stopCamera(){
  if (videoEl.srcObject) { videoEl.srcObject.getTracks().forEach(t=>t.stop()); videoEl.srcObject = null; }
  streaming = false; if (rafId){ cancelAnimationFrame(rafId); rafId = null; }
  smoothBuffer = []; ctx.clearRect(0,0,canvas.width,canvas.height);
  console.log('Camera stopped');
}

/* ======= WebGazer init and calibration ======= */
async function initWebGazer(){
  try {
    console.log('Starting WebGazer (gaze tracking)...');
    await webgazer.setRegression('ridge')
      .setGazeListener((data, t) => {
        // data can be null until model warms up / calibrated
        if (data && data.x != null && data.y != null) {
          latestGaze = { x: data.x, y: data.y, t: Date.now() };
        } else {
          latestGaze = null;
        }
      })
      .begin();
    webgazer.showVideo(false); webgazer.showPredictionPoints(false); webgazer.showFaceOverlay(false);

    // create or ensure gaze dot exists
    if (!document.getElementById('gazeDot')) {
      const d = document.createElement('div'); d.id = 'gazeDot'; document.body.appendChild(d);
      // style is in CSS but ensure basic fallback if missing:
      Object.assign(d.style, {position:'fixed', width:'12px', height:'12px', borderRadius:'50%', background:'#ff3b3b', pointerEvents:'none', transform:'translate(-50%,-50%)', zIndex:9999, display:'none'});
    }

    // move dot regularly if we have a prediction; otherwise hide it
    setInterval(()=> {
      const el = document.getElementById('gazeDot');
      if (latestGaze) {
        el.style.display = 'block';
        // clamp to viewport to avoid it jumping offscreen
        const x = Math.max(0, Math.min(window.innerWidth, latestGaze.x));
        const y = Math.max(0, Math.min(window.innerHeight, latestGaze.y));
        el.style.left = `${x}px`;
        el.style.top = `${y}px`;
      } else {
        el.style.display = 'none';
      }
    }, 40);

    console.log('✅ WebGazer initialized. Click "Calibrate Gaze" after starting camera for better accuracy.');
  } catch(e) {
    console.warn('⚠️ WebGazer init failed (gaze tracking will be limited):', e);
  }
}

function calibrateGaze(){
  // very simple 5-point click calibration
  alert('Calibration: Click the dots that appear (5 points).');
  const points = [{x:0.1,y:0.1},{x:0.9,y:0.1},{x:0.5,y:0.5},{x:0.1,y:0.9},{x:0.9,y:0.9}];
  let idx = 0;
  function showPoint(){
    if (idx >= points.length) { alert('Calibration done'); return; }
    const p = points[idx];
    const dot = document.createElement('div');
    Object.assign(dot.style, {position:'fixed',left:(p.x*100)+'%',top:(p.y*100)+'%',transform:'translate(-50%,-50%)',width:'18px',height:'18px',background:'#38bdf8',borderRadius:'9px',zIndex:9999});
    document.body.appendChild(dot);
    dot.addEventListener('click', ()=>{ dot.remove(); idx++; setTimeout(showPoint, 150); });
  }
  showPoint();
}

/* ======= Click heatmap fallback ======= */
function addClickPoint(relX, relY){
  clickPoints.push({x:relX, y:relY, t:Date.now()});
  if (clickPoints.length > 400) clickPoints.shift();
}
function drawClickPoints(){
  if (!clickPoints.length) return;
  for (let i=0;i<clickPoints.length;i++){
    const p = clickPoints[i];
    const age = (Date.now() - p.t)/1000;
    const alpha = Math.max(0, 1 - age/8);
    const radius = 8 + Math.max(0, 18*(1-alpha));
    ctx.beginPath(); ctx.fillStyle = `rgba(255,90,90,${alpha*0.9})`;
    ctx.arc(p.x, p.y, radius, 0, Math.PI*2); ctx.fill();
  }
  // remove very old
  clickPoints = clickPoints.filter(p => (Date.now() - p.t) < 12000);
}

/* click listener */
document.addEventListener('click', (e) => {
  const rect = document.querySelector('.video-wrap').getBoundingClientRect();
  const inVideo = e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;
  if (inVideo) {
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    addClickPoint(x, y);
  }
  clickLog.push({t:Date.now(), x:e.clientX, y:e.clientY, tag:e.target.tagName});
});

/* download log */
downloadLogBtn.addEventListener('click', ()=>{
  const blob = new Blob([JSON.stringify({clicks:clickLog},null,2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'session_log.json'; a.click();
});

/* adaptive hint */
function showAdaptiveHint(){
  console.log('🔔🔔🔔 ADAPTIVE INTERVENTION TRIGGERED! Attention has been low for ' + (ATTENTION_TRIGGER_MS/1000) + 's 🔔🔔🔔');
  const el = document.createElement('div');
  Object.assign(el.style, {
    position:'fixed',
    left:'50%',
    top:'20%',
    transform:'translateX(-50%)',
    padding:'40px 60px',
    background:'#DC2626',  // Very bright red
    color:'#FFFFFF',
    borderRadius:'20px',
    zIndex:999999,
    fontSize:'32px',
    fontWeight:'900',
    boxShadow:'0 0 60px rgba(220, 38, 38, 0.9), 0 0 100px rgba(220, 38, 38, 0.6)',
    border:'6px solid #FFFFFF',
    animation: 'pulse 0.5s ease-in-out infinite',
    textAlign:'center',
    letterSpacing:'2px',
    textShadow:'2px 2px 4px rgba(0,0,0,0.5)'
  });
  el.innerHTML = '🚨 ATTENTION DROPPED! 🚨<br><span style="font-size:20px; font-weight:600; display:block; margin-top:12px;">Take a break or refocus!</span>';
  document.body.appendChild(el);

  // Also flash the screen border
  document.body.style.boxShadow = 'inset 0 0 100px rgba(220, 38, 38, 0.8)';

  setTimeout(()=>{
    el.style.opacity = '0';
    el.style.transition = 'opacity 0.5s';
    document.body.style.boxShadow = '';
    setTimeout(()=>el.remove(), 500);
  }, 5000);
}

/* calibration reminder */
function showCalibrationReminder(){
  console.log('💡 Showing calibration reminder');
  const el = document.createElement('div');
  Object.assign(el.style, {
    position:'fixed',
    left:'50%',
    bottom:'10%',
    transform:'translateX(-50%)',
    padding:'14px 20px',
    background:'rgba(56, 189, 248, 0.95)',
    color:'#fff',
    borderRadius:'12px',
    zIndex:99999,
    fontSize:'14px',
    boxShadow:'0 4px 20px rgba(0,0,0,0.5)',
    border:'2px solid rgba(255,255,255,0.3)',
    animation: 'fadeIn 0.3s ease-in',
    textAlign: 'center'
  });
  el.innerHTML = '💡 <strong>Tip:</strong> Click "Calibrate Gaze" for more accurate attention tracking!<br><small style="opacity:0.8">(Head orientation tracking is already active)</small>';
  document.body.appendChild(el);
  setTimeout(()=>{
    el.style.opacity = '0';
    el.style.transition = 'opacity 0.5s';
    setTimeout(()=>el.remove(), 500);
  }, 6000);
}

/* ======= UI hooks ======= */
startBtn.addEventListener('click', async ()=>{ await startCamera(); });
stopBtn.addEventListener('click', ()=>{ stopCamera(); });
calibrateBtn.addEventListener('click', calibrateGaze);
downloadLogBtn.addEventListener('click', ()=>{ /* already attached above */ });

/* ======= 3B: Firebase init + Consent modal wiring ======= */
const firebaseConfig = null; 

function initFirebaseIfConfigured() {
  try {
    if (firebaseConfig) {
      // initialize Firebase app and Firestore (compat libs included in index.html)
      firebase.initializeApp(firebaseConfig);
      firestore = firebase.firestore();
      console.log('Firebase initialized.');
    } else {
      console.log('Firebase not configured (telemetry will remain local).');
    }
  } catch (err) {
    console.warn('Firebase init failed:', err);
  }
}

// show/hide consent modal helpers (the modal HTML is in index.html)
function showConsentModal() {
  const m = document.getElementById('consentModal');
  if (m) m.style.display = 'flex';
}
function hideConsentModal() {
  const m = document.getElementById('consentModal');
  if (m) m.style.display = 'none';
}

// wire the modal buttons after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // initialize Firebase (if user supplied config in the variable above)
  initFirebaseIfConfigured();

  // show modal automatically when the page loads
  showConsentModal();

  // find modal elements
  const acceptBtn = document.getElementById('consentAccept');
  const declineBtn = document.getElementById('consentDecline');
  const telemetryCheckbox = document.getElementById('consentTelemetry');

  // Accept: set telemetryEnabled based on the checkbox, then hide modal
  if (acceptBtn) {
    acceptBtn.addEventListener('click', () => {
      telemetryEnabled = !!(telemetryCheckbox && telemetryCheckbox.checked);
      console.log('Consent accepted. telemetryEnabled=', telemetryEnabled);
      hideConsentModal();
    });
  }

  // Decline: explicitly disable telemetry and hide modal
  if (declineBtn) {
    declineBtn.addEventListener('click', () => {
      telemetryEnabled = false;
      console.log('Consent declined. telemetryEnabled=false');
      hideConsentModal();
    });
  }
});

/* init on load */
window.addEventListener('load', ()=>{
  console.log('=== SimphAI Loading ===');
  console.log('Initializing WebGazer...');
  initWebGazer();

  // Set initial display values
  if (earValSpan) earValSpan.textContent = '—';
  if (facePresentSpan) facePresentSpan.textContent = '—';
  if (attnValSpan) attnValSpan.textContent = '—';
  if (blinkValSpan) blinkValSpan.textContent = '0';
  if (gazeValSpan) gazeValSpan.textContent = '—';
  if (headValSpan) headValSpan.textContent = '—';

  console.log('UI elements initialized. Click "Accept" on consent modal, then "Start" to begin.');
});
