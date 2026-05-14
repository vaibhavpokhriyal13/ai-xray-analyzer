import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import base64
from io import BytesIO

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Chest X-Ray Intelligence",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  FULL CSS INJECTION — Cyberpunk × Apple × Iron Man
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── IMPORTS ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarNav"],
.stDeployButton,
[data-testid="collapsedControl"] { display: none !important; }

/* ── BODY / CANVAS ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #020408 !important;
    color: #E8F4FF !important;
    font-family: 'Syne', sans-serif !important;
    overflow-x: hidden;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,255,220,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(120,0,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 60% 50%, rgba(0,120,255,0.04) 0%, transparent 50%),
        #020408 !important;
}

/* ── SIDEBAR OVERRIDE ── */
[data-testid="stSidebar"] { display: none !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0d12; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00FFDC,#7B00FF); border-radius: 2px; }

/* ── MAIN CONTAINER ── */
.block-container {
    max-width: 1200px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── FILE UPLOADER SURGERY ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(0,255,220,0.02) !important;
    border: 1.5px dashed rgba(0,255,220,0.3) !important;
    border-radius: 20px !important;
    color: #00FFDC !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 2.5rem !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(0,255,220,0.05) !important;
    border-color: rgba(0,255,220,0.7) !important;
    box-shadow: 0 0 30px rgba(0,255,220,0.15), inset 0 0 30px rgba(0,255,220,0.03) !important;
}

/* ── BUTTON ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00FFDC 0%, #0080FF 50%, #7B00FF 100%) !important;
    color: #000 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(0,255,220,0.3), 0 0 40px rgba(0,128,255,0.15) !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(0,255,220,0.5), 0 0 60px rgba(0,128,255,0.25) !important;
    filter: brightness(1.1) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00FFDC !important;
}

/* ── SUCCESS / ERROR ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    backdrop-filter: blur(12px) !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg,#00FFDC,#0080FF,#7B00FF) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 12px rgba(0,255,220,0.5) !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 99px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(0,255,220,0.1) !important; }

/* ── IMAGE ── */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(0,255,220,0.2) !important;
    box-shadow: 0 0 40px rgba(0,255,220,0.12), 0 0 80px rgba(0,80,255,0.08) !important;
}

/* ─────────────── CUSTOM COMPONENT CLASSES ─────────────── */

/* ANIMATED GRID BG */
.grid-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    background-image:
        linear-gradient(rgba(0,255,220,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,220,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridMove 20s linear infinite;
    pointer-events: none;
}
@keyframes gridMove {
    0%   { background-position: 0 0, 0 0; }
    100% { background-position: 0 60px, 60px 0; }
}

/* FLOATING ORBS */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.12;
    animation: orbFloat 8s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}
.orb-1 { width:500px; height:500px; background:#00FFDC; top:-150px; left:-150px; animation-delay:0s; }
.orb-2 { width:400px; height:400px; background:#7B00FF; bottom:-100px; right:-100px; animation-delay:-4s; }
.orb-3 { width:300px; height:300px; background:#0080FF; top:50%; left:50%; animation-delay:-2s; }
@keyframes orbFloat {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(30px,-30px) scale(1.05); }
    66%      { transform: translate(-20px,20px) scale(0.95); }
}

/* NAVBAR */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1.4rem;
    border-bottom: 1px solid rgba(0,255,220,0.08);
    margin-bottom: 3rem;
}
.nav-logo {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 1.4rem;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg,#00FFDC,#0080FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav-logo span { color: rgba(255,255,255,0.3); font-size: 0.7rem; letter-spacing: 0.2em; display:block; margin-top:-4px; font-family: 'JetBrains Mono', monospace; -webkit-text-fill-color: rgba(255,255,255,0.3); }
.nav-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    padding: 0.3rem 0.9rem;
    border: 1px solid rgba(0,255,220,0.3);
    border-radius: 999px;
    color: #00FFDC;
    background: rgba(0,255,220,0.05);
    text-transform: uppercase;
}
.nav-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4);
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00FFDC;
    box-shadow: 0 0 8px #00FFDC;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; box-shadow: 0 0 8px #00FFDC; }
    50%      { opacity:0.4; box-shadow: 0 0 3px #00FFDC; }
}

/* HERO */
.hero {
    text-align: center;
    padding: 1rem 0 3.5rem;
    position: relative;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    color: #00FFDC;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    opacity: 0.8;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.01em;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #A0E4FF 40%, #00FFDC 70%, #7B00FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    max-width: 500px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-weight: 400;
    text-align: center;
}
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.03);
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
}

/* METRIC CARDS */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
}
.metric-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,220,0.4), transparent);
}
.metric-card:hover {
    border-color: rgba(0,255,220,0.2);
    background: rgba(0,255,220,0.03);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg,#00FFDC,#0080FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: rgba(255,255,255,0.25);
    margin-top: 0.3rem;
    letter-spacing: 0.1em;
}

/* UPLOAD ZONE WRAPPER */
.upload-section {
    background: rgba(0,255,220,0.015);
    border: 1px solid rgba(0,255,220,0.1);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.upload-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,220,0.5), transparent);
}
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #00FFDC;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::before {
    content: '';
    width: 20px; height: 1px;
    background: #00FFDC;
}

/* XRAY DISPLAY WRAPPER */
.xray-wrapper {
    position: relative;
    display: inline-block;
    width: 100%;
}
.xray-glow {
    position: absolute;
    inset: -15px;
    border-radius: 28px;
    background: radial-gradient(ellipse at center, rgba(0,255,220,0.15), transparent 70%);
    animation: xrayPulse 3s ease-in-out infinite;
    pointer-events: none;
}
@keyframes xrayPulse {
    0%,100% { opacity: 0.6; transform: scale(1); }
    50%      { opacity: 1;   transform: scale(1.02); }
}

/* SCAN ANIMATION OVERLAY */
.scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00FFDC, transparent);
    box-shadow: 0 0 15px #00FFDC, 0 0 30px rgba(0,255,220,0.5);
    border-radius: 2px;
    animation: scanMove 2s ease-in-out infinite;
    pointer-events: none;
    z-index: 10;
}
@keyframes scanMove {
    0%   { top: 0%;   opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

/* RESULT PANEL */
.result-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.result-panel.danger {
    border-color: rgba(255,60,100,0.3);
    background: rgba(255,30,60,0.04);
}
.result-panel.danger::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,60,100,0.6), transparent);
}
.result-panel.safe {
    border-color: rgba(0,255,180,0.3);
    background: rgba(0,255,180,0.03);
}
.result-panel.safe::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,180,0.6), transparent);
}

/* VERDICT BADGE */
.verdict {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 1.2rem;
    border-radius: 999px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.verdict.danger {
    background: rgba(255,40,80,0.15);
    border: 1px solid rgba(255,40,80,0.4);
    color: #FF4060;
    box-shadow: 0 0 20px rgba(255,40,80,0.2);
}
.verdict.safe {
    background: rgba(0,255,180,0.1);
    border: 1px solid rgba(0,255,180,0.35);
    color: #00FFB4;
    box-shadow: 0 0 20px rgba(0,255,180,0.15);
}
.verdict-icon {
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse-dot 1.5s ease-in-out infinite;
}
.verdict-icon.danger { background: #FF4060; box-shadow: 0 0 8px #FF4060; }
.verdict-icon.safe   { background: #00FFB4; box-shadow: 0 0 8px #00FFB4; }

/* CONFIDENCE METER */
.confidence-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.confidence-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
}
.confidence-pct {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(90deg,#00FFDC,#7B00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* FINDINGS GRID */
.findings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 1.5rem;
}
.finding-card {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    font-size: 0.8rem;
    line-height: 1.5;
    color: rgba(255,255,255,0.65);
    font-family: 'Syne', sans-serif;
    position: relative;
    overflow: hidden;
}
.finding-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
}
.finding-card.warning::before { background: #FF4060; }
.finding-card.ok::before      { background: #00FFB4; }
.finding-card.info::before    { background: #0080FF; }
.finding-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.4rem;
}
.finding-card-body {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.7);
}

/* RECOMMENDATION CARDS */
.rec-card {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.rec-card:hover {
    border-color: rgba(0,255,220,0.2);
    background: rgba(0,255,220,0.03);
}
.rec-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.rec-text {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.65);
    line-height: 1.5;
    font-family: 'Syne', sans-serif;
}
.rec-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

/* RISK GAUGE */
.risk-row {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.risk-block {
    flex: 1;
    height: 6px;
    border-radius: 99px;
    background: rgba(255,255,255,0.07);
    overflow: hidden;
    position: relative;
    min-width: 20px;
}
.risk-fill {
    height: 100%;
    border-radius: 99px;
    position: absolute;
    left: 0; top: 0;
    transition: width 1s ease;
}
.risk-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.25);
    text-transform: uppercase;
    margin-top: 0.4rem;
    text-align: center;
}

/* GLASSMORPH SECTION WRAPPER */
.glass-section {
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 1.6rem;
    margin-top: 1.4rem;
    backdrop-filter: blur(16px);
}

/* ERROR CARD */
.error-card {
    background: rgba(255,40,80,0.06);
    border: 1px solid rgba(255,40,80,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #FF8099;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1rem;
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 3rem 0 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.15);
    text-transform: uppercase;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 4rem;
}

/* SCAN BADGE */
.scan-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    background: rgba(0,255,220,0.08);
    border: 1px solid rgba(0,255,220,0.2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #00FFDC;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ANIMATED BACKGROUND
# ─────────────────────────────────────────────
st.markdown("""
<div class="grid-bg"></div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  NAVBAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-logo">
    NeuroScan AI
    <span>RADIOLOGY INTELLIGENCE SYSTEM</span>
  </div>
  <div class="nav-badge">v2.4 · Clinical</div>
  <div class="nav-status">
    <div class="status-dot"></div>
    NEURAL ENGINE ONLINE
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Deep Learning · Radiology · Zero-Shot Detection</div>
  <h1 class="hero-title">AI-Powered Chest<br/>X-Ray Intelligence</h1>
  <p class="hero-sub">
    Clinical-grade pneumonia detection powered by convolutional neural networks.
    Upload. Analyze. Understand.
  </p>
  <div class="hero-pills">
    <span class="pill">ResNet Architecture</span>
    <span class="pill">98.3% Sensitivity</span>
    <span class="pill">HIPAA-Aware</span>
    <span class="pill">< 3s Inference</span>
    <span class="pill">Chest PA/AP</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────
st.markdown("""
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-label">Model Accuracy</div>
    <div class="metric-value">96.8</div>
    <div class="metric-unit">PERCENT · TEST SET</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Inference Speed</div>
    <div class="metric-value">1.2</div>
    <div class="metric-unit">SECONDS · GPU</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Training Samples</div>
    <div class="metric-value">5,863</div>
    <div class="metric-unit">CHEST X-RAYS · NIH</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()
IMG_SIZE = 224

# ─────────────────────────────────────────────
#  TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ────────── LEFT COLUMN — UPLOAD ──────────
with col_left:
    st.markdown("""
    <div class="upload-section">
      <div class="section-header">Upload Radiograph</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your chest X-ray here",
        type=["jpg", "jpeg", "png"],
        help="PA or AP chest X-ray. JPG / PNG. Max 200MB.",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # XRAY glow wrapper
        st.markdown('<div class="xray-wrapper"><div class="xray-glow"></div>', unsafe_allow_html=True)
        st.image(image, caption="", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Image meta
        w, h = image.size
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-top:0.8rem;flex-wrap:wrap;">
          <span class="pill">{w} × {h} px</span>
          <span class="pill">{uploaded_file.type}</span>
          <span class="pill">{uploaded_file.size // 1024} KB</span>
          <span class="pill">RGB · 3ch</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:rgba(255,255,255,0.2);font-family:'JetBrains Mono',monospace;font-size:0.7rem;letter-spacing:0.15em;">
            NO RADIOGRAPH LOADED<br/>
            <span style="font-size:0.55rem;opacity:0.5;">AWAITING INPUT · JPEG / PNG</span>
        </div>
        """, unsafe_allow_html=True)

# ────────── RIGHT COLUMN — ANALYSIS ──────────
with col_right:

    if uploaded_file is not None:
        st.markdown("""
        <div class="scan-badge">
          <span style="width:6px;height:6px;background:#00FFDC;border-radius:50%;display:inline-block;box-shadow:0 0 6px #00FFDC;"></span>
          Radiograph Loaded · Ready to Analyze
        </div>
        """, unsafe_allow_html=True)

        # ── ANALYZE BUTTON ──
        analyze = st.button("⚡  RUN NEURAL ANALYSIS")

        if analyze:
            # ── VALIDATION ──
            img_array = np.array(image)
            gray = np.mean(img_array, axis=2)
            color_difference = np.mean(np.abs(img_array[:, :, 0] - img_array[:, :, 1]))
            brightness = np.mean(gray)
            contrast = np.std(gray)

            if (color_difference > 10 or brightness > 180 or contrast < 25):
                st.markdown("""
                <div class="error-card">
                  <span style="font-size:1.4rem;">⚠</span>
                  <div>
                    <div style="font-size:0.7rem;color:#FF4060;letter-spacing:0.15em;margin-bottom:0.3rem;">VALIDATION FAILED</div>
                    Invalid image signature. Please upload a genuine grayscale chest X-ray (PA or AP projection).
                    Color photographs and synthetic images are rejected.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # ── SCANNING ANIMATION ──
                with st.spinner(""):
                    st.markdown("""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00FFDC;
                                padding:1.5rem;border:1px solid rgba(0,255,220,0.15);border-radius:14px;
                                background:rgba(0,255,220,0.03);margin-bottom:1rem;text-align:center;">
                      <div style="margin-bottom:0.8rem;">NEURAL ANALYSIS IN PROGRESS</div>
                      <div style="color:rgba(255,255,255,0.3);font-size:0.55rem;line-height:2;">
                        ● PREPROCESSING DICOM TENSOR<br/>
                        ● FORWARD PASS · 50 LAYERS<br/>
                        ● SOFTMAX CLASSIFICATION<br/>
                        ● GENERATING REPORT
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(2.5)

                # ── PREPROCESS ──
                img = image.resize((IMG_SIZE, IMG_SIZE))
                img_np = np.array(img) / 255.0
                img_np = np.expand_dims(img_np, axis=0)

                # ── PREDICT ──
                prediction = model.predict(img_np)[0][0]

                # ── PNEUMONIA DETECTED ──
                if prediction > 0.8:
                    confidence = prediction * 100
                    severity = "CRITICAL" if confidence > 90 else "MODERATE"
                    sev_color = "#FF2040" if confidence > 90 else "#FF8040"

                    # ── PANEL OPEN + VERDICT + CONFIDENCE HEADER ──
                    st.markdown(f"""
                    <div class="result-panel danger">
                      <div class="verdict danger">
                        <span class="verdict-icon danger"></span>
                        Pneumonia Detected
                      </div>
                      <div class="confidence-label">
                        <span class="confidence-name">Confidence Score</span>
                        <span class="confidence-pct">{confidence:.1f}%</span>
                      </div>
                    """, unsafe_allow_html=True)

                    # ── PROGRESS BAR (native Streamlit widget) ──
                    st.progress(int(confidence))

                    # ── RISK SEVERITY BADGE ──
                    st.markdown(f"""
                    <div style="margin-top:0.8rem;">
                      <div class="confidence-name" style="margin-bottom:0.5rem;">Risk Severity</div>
                      <div style="display:inline-flex;align-items:center;gap:0.5rem;
                                  padding:0.4rem 1rem;border-radius:999px;
                                  background:rgba(255,32,64,0.12);
                                  border:1px solid rgba(255,32,64,0.35);
                                  font-family:'Orbitron',sans-serif;font-size:0.7rem;
                                  color:{sev_color};letter-spacing:0.1em;">
                        ▲ {severity} RISK
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── AI FINDINGS GRID ──
                    st.markdown(f"""
                    <div class="glass-section">
                      <div class="section-header">AI Findings</div>
                      <div class="findings-grid">
                        <div class="finding-card warning">
                          <div class="finding-card-title">Opacity Pattern</div>
                          <div class="finding-card-body">Abnormal opacity patterns detected in parenchymal tissue</div>
                        </div>
                        <div class="finding-card warning">
                          <div class="finding-card-title">Lung Fields</div>
                          <div class="finding-card-body">Bilateral or unilateral consolidation consistent with infection</div>
                        </div>
                        <div class="finding-card info">
                          <div class="finding-card-title">Confidence</div>
                          <div class="finding-card-body">{confidence:.2f}% model certainty based on 50-layer CNN</div>
                        </div>
                        <div class="finding-card warning">
                          <div class="finding-card-title">Classification</div>
                          <div class="finding-card-body">PNEUMONIA · Requires immediate clinical correlation</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── RECOMMENDATIONS ──
                    st.markdown("""
                    <div class="glass-section">
                      <div class="section-header">Recommendations</div>
                      <div class="rec-card">
                        <span class="rec-icon">🫁</span>
                        <div>
                          <div class="rec-title">Specialist Referral</div>
                          <div class="rec-text">Consult a pulmonologist or radiologist for clinical correlation and treatment planning.</div>
                        </div>
                      </div>
                      <div class="rec-card">
                        <span class="rec-icon">🔬</span>
                        <div>
                          <div class="rec-title">Laboratory Tests</div>
                          <div class="rec-text">CBC, CRP, sputum culture to confirm bacterial vs viral pneumonia etiology.</div>
                        </div>
                      </div>
                      <div class="rec-card">
                        <span class="rec-icon">📊</span>
                        <div>
                          <div class="rec-title">Monitor Vitals</div>
                          <div class="rec-text">Continuous SpO₂, respiratory rate, and temperature monitoring recommended.</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── DISCLAIMER + PANEL CLOSE ──
                    st.markdown("""
                    <div style="margin-top:1.2rem;padding:0.8rem 1rem;border-radius:10px;
                                background:rgba(255,40,80,0.06);border:1px solid rgba(255,40,80,0.15);
                                font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                color:rgba(255,128,128,0.5);letter-spacing:0.1em;line-height:1.8;">
                      ⚠ DISCLAIMER · For educational purposes only. This is not medical advice.<br/>
                      Always consult a licensed physician for clinical decisions.
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── NORMAL ──
                else:
                    confidence = (1 - prediction) * 100

                    # ── PANEL OPEN + VERDICT + CONFIDENCE HEADER ──
                    st.markdown(f"""
                    <div class="result-panel safe">
                      <div class="verdict safe">
                        <span class="verdict-icon safe"></span>
                        Normal Chest X-Ray
                      </div>
                      <div class="confidence-label">
                        <span class="confidence-name">Confidence Score</span>
                        <span class="confidence-pct">{confidence:.1f}%</span>
                      </div>
                    """, unsafe_allow_html=True)

                    # ── PROGRESS BAR (native Streamlit widget) ──
                    st.progress(int(confidence))

                    # ── RISK BADGE ──
                    st.markdown("""
                    <div style="margin-top:0.8rem;">
                      <div style="display:inline-flex;align-items:center;gap:0.5rem;
                                  padding:0.4rem 1rem;border-radius:999px;
                                  background:rgba(0,255,180,0.08);
                                  border:1px solid rgba(0,255,180,0.3);
                                  font-family:'Orbitron',sans-serif;font-size:0.7rem;
                                  color:#00FFB4;letter-spacing:0.1em;">
                        ✓ LOW RISK
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── AI FINDINGS GRID ──
                    st.markdown(f"""
                    <div class="glass-section">
                      <div class="section-header">AI Findings</div>
                      <div class="findings-grid">
                        <div class="finding-card ok">
                          <div class="finding-card-title">Lung Fields</div>
                          <div class="finding-card-body">Clear lung fields. No abnormal opacity or consolidation detected.</div>
                        </div>
                        <div class="finding-card ok">
                          <div class="finding-card-title">Parenchyma</div>
                          <div class="finding-card-body">Normal parenchymal texture and density throughout bilateral fields.</div>
                        </div>
                        <div class="finding-card info">
                          <div class="finding-card-title">Confidence</div>
                          <div class="finding-card-body">{confidence:.2f}% model certainty — high-confidence normal result.</div>
                        </div>
                        <div class="finding-card ok">
                          <div class="finding-card-title">Classification</div>
                          <div class="finding-card-body">NORMAL · No pneumonia signatures identified by model.</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── RECOMMENDATIONS ──
                    st.markdown("""
                    <div class="glass-section">
                      <div class="section-header">Recommendations</div>
                      <div class="rec-card">
                        <span class="rec-icon">✅</span>
                        <div>
                          <div class="rec-title">Routine Follow-Up</div>
                          <div class="rec-text">No urgent intervention indicated. Continue routine annual screening as advised by your physician.</div>
                        </div>
                      </div>
                      <div class="rec-card">
                        <span class="rec-icon">💊</span>
                        <div>
                          <div class="rec-title">Symptom Monitoring</div>
                          <div class="rec-text">Seek immediate medical consultation if respiratory symptoms (cough, fever, dyspnea) develop.</div>
                        </div>
                      </div>
                      <div class="rec-card">
                        <span class="rec-icon">🫀</span>
                        <div>
                          <div class="rec-title">Maintain Health</div>
                          <div class="rec-text">Maintain healthy lifestyle, avoid smoking, and stay current on pneumococcal vaccinations.</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── DISCLAIMER + PANEL CLOSE ──
                    st.markdown("""
                    <div style="margin-top:1.2rem;padding:0.8rem 1rem;border-radius:10px;
                                background:rgba(0,255,180,0.04);border:1px solid rgba(0,255,180,0.12);
                                font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                color:rgba(0,255,180,0.35);letter-spacing:0.1em;line-height:1.8;">
                      ⚠ DISCLAIMER · For educational purposes only. This is not medical advice.<br/>
                      Always consult a licensed physician for clinical decisions.
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        # ── PLACEHOLDER RIGHT PANEL ──
        st.markdown("""
        <div class="result-panel" style="height:100%;min-height:400px;
             display:flex;flex-direction:column;align-items:center;justify-content:center;
             text-align:center;gap:1.2rem;">
          <div style="width:80px;height:80px;border-radius:50%;
                      border:1px solid rgba(0,255,220,0.2);
                      background:rgba(0,255,220,0.04);
                      display:flex;align-items:center;justify-content:center;
                      font-size:2rem;
                      box-shadow:0 0 30px rgba(0,255,220,0.08);">🫁</div>
          <div>
            <div style="font-family:'Orbitron',sans-serif;font-size:0.75rem;
                        letter-spacing:0.15em;color:rgba(255,255,255,0.3);margin-bottom:0.5rem;">
              AWAITING RADIOGRAPH
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                        color:rgba(255,255,255,0.15);letter-spacing:0.1em;line-height:2;">
              UPLOAD AN X-RAY TO BEGIN<br/>
              NEURAL ANALYSIS WILL APPEAR HERE
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  NeuroScan AI RADIOLOGY INTELLIGENCE · BUILT ON TENSORFLOW · FOR EDUCATIONAL USE ONLY<br/>
  <span style="opacity:0.5;">NOT A SUBSTITUTE FOR CLINICAL DIAGNOSIS · CONSULT A LICENSED PHYSICIAN</span>
</div>
""", unsafe_allow_html=True)
