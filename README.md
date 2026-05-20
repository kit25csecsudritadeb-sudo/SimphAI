SimphAI — The AI That Learns How You Learn

> A NeuroAdaptive Learning Platform that personalizes education in real time based on each learner's attention, gaze, and cognitive state — running entirely in the browser.


The Problem

Over **15% of students in India are neurodivergent** (Autism, ADHD, Dyslexia), yet most digital learning platforms follow a rigid one-size-fits-all approach.

- E-learning environments cause cognitive overload and frustration
- No sensory-friendly or adaptive-pacing tools exist at scale
- India's NEP 2020 calls for inclusive learning — but no accessible AI-driven solution exists yet


The Solution

SimphAI is a browser-based adaptive learning prototype that **senses and responds** to each learner's attention state in real time — no installs, no uploads, no servers.


Features

- **Live attention scoring** — weighted from gaze direction, eye openness (EAR), and head orientation
- **Blink detection** — EAR state-machine with 4-frame smoothing buffer
- **Gaze tracking** — WebGazer ridge regression with 5-point click calibration
- **Adaptive intervention** — alert fires when attention drops below threshold for 4+ seconds
- **Calm Mode** — auto-activates on critical attention drop, reduces visual stimulation
- **Session summary** — duration, average attention, blink count, focus stability score
- **Privacy-first** — all processing in-browser, no raw video uploaded or stored



Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | MediaPipe FaceMesh |
| Gaze Tracking | WebGazer.js |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Optional Telemetry | Firebase (user-consented) |



Run Locally

```bash
git clone https://github.com/kit25csecsudritadeb-sudo/SimphAI.git
cd SimphAI
```

Open `index.html` in any modern browser — no build step needed.

> Allow camera access when prompted. Click **Calibrate Gaze** after starting for best accuracy.

---

 SDG Alignment

| Goal | Relevance |
|---|---|
| 🎯 SDG 4 — Quality Education | Adaptive, inclusive learning for all |
| 🎯 SDG 10 — Reduced Inequalities | Accessible without specialised hardware |
| 🎯 SDG 3 — Good Health & Well-being | Reduces cognitive overload and learning stress |

---

Future Roadmap

- Emotion sensing via wearables
- Voice-based interface for low-literacy users
- Cloud dashboards for parents and institutions
- On-device ML for offline rural use



Author

**Sudrita Deb** — Team Leader, TechVerseTeam  
CSE, Kalaignarkarunanidhi Institute of Technology, Coimbatore  
 sudr2008.deb@gmail.com · GitHub: [@kit25csecsudritadeb-sudo](https://github.com/kit25csecsudritadeb-sudo)

*"SimphAI listens, senses, and evolves with every mind."*
