"""
MAPLE Local Agent Workbench & Web UI (Zero-Electron, 100% Offline)
==================================================================
Lightweight FastAPI + single-page HTML/CSS/JS workbench for your local 4-tier
model stack (DeepSeek-Coder-V2-Lite 54-Exp, Gemma 4 26B, Bonsai 27B, Qwen 1.5B).

Features:
  - 100% offline, zero external CDN dependencies, zero Electron bloat.
  - Model Chooser pills (Auto, DeepSeek, Gemma, Bonsai, Fast) + Chat / Agent Task modes.
  - Live streaming Markdown & code rendering.
  - Right-hand "Live Work & File Renderer" pane: shows files being written in real time,
    colorized unified diffs on edits, terminal execution cards, and live HTML/Markdown preview.
  - Live RTX 4050 6GB VRAM telemetry bar.

Author: Aman Sachan (amansachan92905@gmail.com)
"""

import sys
import os
import re
import json
import time
import difflib
import asyncio
import threading
import webbrowser
from typing import Dict, Any, Optional

import requests
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_agent_harness import (
    MODEL_ROSTER,
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    DEFAULT_OLLAMA_API,
    DEFAULT_BONSAI_API,
    Tools,
    auto_choose_model,
    ensure_vram_for_model,
)

WORKBENCH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MAPLE Local Agent Workbench</title>
<style>
  :root {
    --bg: #0b0f14;
    --panel: #111722;
    --panel-2: #161e2c;
    --border: #232f42;
    --text: #e6edf3;
    --muted: #8b9bb4;
    --accent: #38bdf8;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #a78bfa;
    --mono: "Cascadia Code", "JetBrains Mono", Consolas, monospace;
    --sans: "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  /* Top Header */
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.4px;
  }
  .brand-badge {
    background: rgba(56, 189, 248, 0.14);
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-family: var(--mono);
  }
  .model-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .pill {
    background: var(--panel-2);
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    font-weight: 500;
  }
  .pill:hover { border-color: var(--accent); color: var(--text); }
  .pill.active {
    background: rgba(56, 189, 248, 0.16);
    color: var(--accent);
    border-color: var(--accent);
    font-weight: 600;
  }
  .pill[data-model="bonsai"].active {
    background: rgba(167, 139, 250, 0.18);
    color: var(--purple);
    border-color: var(--purple);
  }
  .pill[data-model="deepseek"].active {
    background: rgba(16, 185, 129, 0.16);
    color: var(--green);
    border-color: var(--green);
  }
  .pill[data-model="fast"].active {
    background: rgba(245, 158, 11, 0.16);
    color: var(--amber);
    border-color: var(--amber);
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .vram-box {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 8px;
    font-size: 11px;
    font-family: var(--mono);
  }
  .vram-track {
    width: 72px;
    height: 7px;
    background: #090d12;
    border-radius: 999px;
    overflow: hidden;
  }
  .vram-fill {
    height: 100%;
    width: 0%;
    background: var(--green);
    transition: width 0.4s ease, background 0.3s ease;
  }
  /* Main Split Layout */
  .workspace {
    flex: 1;
    display: grid;
    grid-template-columns: 1.05fr 0.95fr;
    min-height: 0;
  }
  .pane-left {
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    min-height: 0;
  }
  .pane-right {
    display: flex;
    flex-direction: column;
    background: #0d121a;
    min-height: 0;
  }
  /* Chat Stream */
  .chat-stream {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    background: var(--panel);
    line-height: 1.55;
    font-size: 13.5px;
  }
  .msg.user {
    background: #141d2b;
    border-color: #293850;
    align-self: flex-end;
    max-width: 88%;
  }
  .msg.assistant {
    background: var(--panel);
    width: 100%;
  }
  .msg-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 8px;
    font-family: var(--mono);
  }
  .role-tag {
    padding: 2px 7px;
    border-radius: 5px;
    background: rgba(56, 189, 248, 0.12);
    color: var(--accent);
    font-weight: 600;
  }
  .msg-body pre {
    background: #090d12;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    overflow-x: auto;
    margin: 8px 0;
    font-family: var(--mono);
    font-size: 12.5px;
    line-height: 1.45;
  }
  .msg-body code {
    font-family: var(--mono);
    background: rgba(255,255,255,0.06);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 12.5px;
  }
  .msg-body pre code { background: none; padding: 0; }
  .msg-body h1, .msg-body h2, .msg-body h3 { margin: 10px 0 6px; color: #fff; }
  .msg-body ul, .msg-body ol { margin-left: 20px; margin-bottom: 8px; }
  /* Tool Execution Cards */
  .tool-card {
    background: #0d131d;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 9px 12px;
    margin: 8px 0;
    font-family: var(--mono);
    font-size: 12px;
  }
  .tool-card.write { border-left-color: var(--green); }
  .tool-card.edit { border-left-color: var(--amber); }
  .tool-card.err { border-left-color: var(--red); }
  .tool-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--text);
    font-weight: 600;
    cursor: pointer;
  }
  .tool-out {
    margin-top: 6px;
    color: var(--muted);
    white-space: pre-wrap;
    max-height: 160px;
    overflow-y: auto;
    font-size: 11.5px;
  }
  /* Composer */
  .composer {
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .composer-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 12px;
    color: var(--muted);
  }
  .mode-switch {
    display: inline-flex;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 7px;
    overflow: hidden;
  }
  .mode-btn {
    background: none;
    border: none;
    color: var(--muted);
    padding: 4px 10px;
    font-size: 11.5px;
    cursor: pointer;
  }
  .mode-btn.active {
    background: var(--accent);
    color: #081018;
    font-weight: 700;
  }
  .workdir-input {
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 11.5px;
    width: 260px;
  }
  .input-row {
    display: flex;
    gap: 8px;
  }
  textarea {
    flex: 1;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    padding: 10px 12px;
    font-family: var(--sans);
    font-size: 13.5px;
    resize: none;
    min-height: 46px;
    max-height: 140px;
    outline: none;
  }
  textarea:focus { border-color: var(--accent); }
  .send-btn {
    background: var(--accent);
    color: #071019;
    border: none;
    border-radius: 8px;
    padding: 0 18px;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
  }
  .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  /* Right Pane: Live File & Renderer */
  .right-tabs {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  .tab-group { display: flex; gap: 6px; }
  .tab-btn {
    background: var(--panel-2);
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 5px 11px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
  }
  .tab-btn.active {
    background: rgba(16, 185, 129, 0.16);
    color: var(--green);
    border-color: var(--green);
    font-weight: 600;
  }
  .active-file-label {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 300px;
  }
  .right-content {
    flex: 1;
    overflow: auto;
    padding: 14px;
    min-height: 0;
  }
  .code-viewer {
    background: #090d12;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 12.5px;
    line-height: 1.5;
    overflow-x: auto;
  }
  .code-line {
    display: flex;
    padding: 0 10px;
  }
  .code-line:hover { background: rgba(255,255,255,0.03); }
  .line-num {
    width: 42px;
    color: #4b5a70;
    user-select: none;
    text-align: right;
    padding-right: 12px;
    flex-shrink: 0;
  }
  .line-text { white-space: pre; }
  .diff-add { background: rgba(16, 185, 129, 0.14); color: #6ee7b7; }
  .diff-del { background: rgba(239, 68, 68, 0.14); color: #fca5a5; }
  .diff-hunk { background: rgba(56, 189, 248, 0.12); color: var(--accent); }
  .render-frame {
    width: 100%;
    height: 100%;
    min-height: 400px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #ffffff;
  }
  .file-list-item {
    display: flex;
    justify-content: space-between;
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 12px;
    cursor: pointer;
  }
  .file-list-item:hover { background: var(--panel-2); color: var(--accent); }
  .empty-state {
    color: var(--muted);
    text-align: center;
    margin-top: 60px;
    font-size: 13px;
    line-height: 1.7;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>MAPLE Local Workbench</span>
    <span class="brand-badge">4-Tier Router</span>
  </div>
  <div class="model-bar" id="modelBar">
    <button class="pill active" data-model="auto" title="Auto-select best model per prompt">Auto Chooser</button>
    <button class="pill" data-model="deepseek" title="DeepSeek-Coder-V2-Lite 54-Expert MoE">DeepSeek (Systems)</button>
    <button class="pill" data-model="gemma" title="Gemma 4 26B Sub-5GB">Gemma 4 (General)</button>
    <button class="pill" data-model="bonsai" title="Ternary Bonsai 2 27B PTQ1_0">Bonsai 27B (Reasoning)</button>
    <button class="pill" data-model="fast" title="Qwen 2.5 Coder 1.5B">Fast 1.5B (Triage)</button>
  </div>
  <div class="header-right">
    <div class="vram-box" title="NVIDIA RTX 4050 6GB VRAM">
      <span>VRAM</span>
      <div class="vram-track"><div class="vram-fill" id="vramFill"></div></div>
      <span id="vramText">0 / 6141 MiB</span>
    </div>
  </div>
</header>

<div class="workspace">
  <div class="pane-left">
    <div class="chat-stream" id="chatStream">
      <div class="msg assistant">
        <div class="msg-meta">
          <span class="role-tag">WORKBENCH READY</span>
          <span>100% Local Inference</span>
        </div>
        <div class="msg-body">
          Select <strong>Chat</strong> for direct coding/reasoning or <strong>Agent Task</strong> to let the harness read, write, edit, and execute files with live visual rendering on the right pane.
        </div>
      </div>
    </div>
    <div class="composer">
      <div class="composer-controls">
        <div class="mode-switch">
          <button class="mode-btn" id="modeChat" onclick="setMode('chat')">Chat</button>
          <button class="mode-btn active" id="modeTask" onclick="setMode('task')">Agent Task (Tools)</button>
        </div>
        <div>
          <span>Workdir: </span>
          <input class="workdir-input" id="workdirInput" value="E:\OllamaModels" />
        </div>
      </div>
      <div class="input-row">
        <textarea id="promptInput" placeholder="Ask a coding/architecture question or give an autonomous file task... (Enter to send, Shift+Enter for newline)"></textarea>
        <button class="send-btn" id="sendBtn" onclick="sendPrompt()">Run</button>
      </div>
    </div>
  </div>

  <div class="pane-right">
    <div class="right-tabs">
      <div class="tab-group">
        <button class="tab-btn active" id="tabCode" onclick="setRightTab('code')">Live File / Diff</button>
        <button class="tab-btn" id="tabRender" onclick="setRightTab('render')">Rendered View</button>
        <button class="tab-btn" id="tabFiles" onclick="setRightTab('files')">Workspace Files</button>
      </div>
      <span class="active-file-label" id="activeFileLabel">No active file</span>
    </div>
    <div class="right-content" id="rightContent">
      <div class="empty-state">
        Files written, edited, or read by the agent appear here live.<br/>
        Click <strong>Workspace Files</strong> to inspect any file in your working directory.
      </div>
    </div>
  </div>
</div>

<script>
let selectedModel = "auto";
let selectedMode = "task";
let rightTab = "code";
let activeFilePath = "";
let activeFileContent = "";
let activeDiffText = "";
let isRunning = false;

document.querySelectorAll(".pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    selectedModel = btn.dataset.model;
  });
});

function setMode(m) {
  selectedMode = m;
  document.getElementById("modeChat").classList.toggle("active", m === "chat");
  document.getElementById("modeTask").classList.toggle("active", m === "task");
}

function setRightTab(t) {
  rightTab = t;
  document.getElementById("tabCode").classList.toggle("active", t === "code");
  document.getElementById("tabRender").classList.toggle("active", t === "render");
  document.getElementById("tabFiles").classList.toggle("active", t === "files");
  if (t === "files") {
    loadWorkspaceFiles();
  } else {
    renderRightPane();
  }
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Lightweight zero-dependency Markdown renderer
function renderMarkdown(md) {
  let html = escapeHtml(md);
  // Fenced code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code>${code}</code></pre>`;
  });
  // Inline code
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  // Headings
  html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  // Line breaks outside pre
  const parts = html.split(/(<pre>[\s\S]*?<\/pre>)/g);
  return parts.map(p => p.startsWith("<pre>") ? p : p.replace(/\n/g, "<br/>")).join("");
}

function renderCodeLines(text, isDiff = false) {
  const lines = String(text || "").split("\n");
  return `<div class="code-viewer">` + lines.map((line, i) => {
    let cls = "code-line";
    if (isDiff) {
      if (line.startsWith("+") && !line.startsWith("+++")) cls += " diff-add";
      else if (line.startsWith("-") && !line.startsWith("---")) cls += " diff-del";
      else if (line.startsWith("@@")) cls += " diff-hunk";
    }
    return `<div class="${cls}"><span class="line-num">${isDiff ? "" : i + 1}</span><span class="line-text">${escapeHtml(line)}</span></div>`;
  }).join("") + `</div>`;
}

function renderRightPane() {
  const container = document.getElementById("rightContent");
  document.getElementById("activeFileLabel").textContent = activeFilePath || "No active file";

  if (rightTab === "code") {
    if (activeDiffText) {
      container.innerHTML = renderCodeLines(activeDiffText, true);
    } else if (activeFileContent !== "") {
      container.innerHTML = renderCodeLines(activeFileContent, false);
    } else {
      container.innerHTML = `<div class="empty-state">Waiting for file write, edit diff, or file read...</div>`;
    }
  } else if (rightTab === "render") {
    if (!activeFilePath) {
      container.innerHTML = `<div class="empty-state">No file loaded to render.</div>`;
      return;
    }
    const lower = activeFilePath.toLowerCase();
    if (lower.endsWith(".html") || lower.endsWith(".htm") || lower.endsWith(".svg")) {
      const iframe = document.createElement("iframe");
      iframe.className = "render-frame";
      iframe.setAttribute("sandbox", "allow-scripts");
      iframe.srcdoc = activeFileContent;
      container.innerHTML = "";
      container.appendChild(iframe);
    } else if (lower.endsWith(".md")) {
      container.innerHTML = `<div class="msg assistant"><div class="msg-body">${renderMarkdown(activeFileContent)}</div></div>`;
    } else {
      container.innerHTML = renderCodeLines(activeFileContent, false);
    }
  }
}

async function loadWorkspaceFiles() {
  const workdir = document.getElementById("workdirInput").value.trim() || ".";
  const container = document.getElementById("rightContent");
  container.innerHTML = `<div class="empty-state">Loading files...</div>`;
  try {
    const r = await fetch(`/api/files?path=${encodeURIComponent(workdir)}`);
    const data = await r.json();
    if (data.error) {
      container.innerHTML = `<div class="empty-state">${escapeHtml(data.error)}</div>`;
      return;
    }
    const entries = data.entries || [];
    container.innerHTML = entries.map(e => {
      const tag = e.is_dir ? "[DIR]" : `${e.size} B`;
      const fullPath = workdir.replace(/[\\/]+$/, "") + "\\" + e.name;
      if (e.is_dir) {
        return `<div class="file-list-item" onclick="document.getElementById('workdirInput').value='${escapeHtml(fullPath).replace(/\\/g, "\\\\")}'; loadWorkspaceFiles();"><span>${escapeHtml(e.name)}/</span><span>${tag}</span></div>`;
      }
      return `<div class="file-list-item" onclick="openWorkspaceFile('${escapeHtml(fullPath).replace(/\\/g, "\\\\")}')"><span>${escapeHtml(e.name)}</span><span>${tag}</span></div>`;
    }).join("");
  } catch (err) {
    container.innerHTML = `<div class="empty-state">Failed to list files: ${escapeHtml(err)}</div>`;
  }
}

async function openWorkspaceFile(path) {
  const r = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
  const data = await r.json();
  if (data.content !== undefined) {
    activeFilePath = path;
    activeFileContent = data.content;
    activeDiffText = "";
    setRightTab("code");
  }
}

async function pollVram() {
  try {
    const r = await fetch("/api/status");
    const data = await r.json();
    const v = data.vram || {};
    const used = parseInt(v.used_mib || "0", 10);
    const total = parseInt(v.total_mib || "6141", 10);
    const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
    const fill = document.getElementById("vramFill");
    fill.style.width = pct + "%";
    fill.style.background = pct > 85 ? "var(--amber)" : "var(--green)";
    document.getElementById("vramText").textContent = `${used} / ${total} MiB`;
  } catch (_) {}
}
setInterval(pollVram, 3000);
pollVram();

function appendUserMsg(text) {
  const stream = document.getElementById("chatStream");
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="msg-body">${escapeHtml(text).replace(/\n/g, "<br/>")}</div>`;
  stream.appendChild(div);
  stream.scrollTop = stream.scrollHeight;
}

async function sendPrompt() {
  if (isRunning) return;
  const input = document.getElementById("promptInput");
  const prompt = input.value.trim();
  if (!prompt) return;

  input.value = "";
  appendUserMsg(prompt);
  isRunning = true;
  document.getElementById("sendBtn").disabled = true;

  const stream = document.getElementById("chatStream");
  const card = document.createElement("div");
  card.className = "msg assistant";
  card.innerHTML = `
    <div class="msg-meta">
      <span class="role-tag" id="tag-${Date.now()}">ROUTING...</span>
      <span class="telemetry-span"></span>
    </div>
    <div class="tools-box"></div>
    <div class="msg-body">Thinking...</div>
  `;
  stream.appendChild(card);
  stream.scrollTop = stream.scrollHeight;

  const roleTag = card.querySelector(".role-tag");
  const telemSpan = card.querySelector(".telemetry-span");
  const toolsBox = card.querySelector(".tools-box");
  const bodyDiv = card.querySelector(".msg-body");

  let accText = "";
  const workdir = document.getElementById("workdirInput").value.trim();

  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, model: selectedModel, mode: selectedMode, workdir })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === "[DONE]") continue;
        let ev;
        try { ev = JSON.parse(raw); } catch { continue; }

        if (ev.type === "model_selected") {
          roleTag.textContent = `${ev.key.toUpperCase()} - ${ev.name}`;
        } else if (ev.type === "step") {
          accText += (accText ? "\n\n---\n" : "") + `**Step ${ev.step}/${ev.max_steps}**\n\n`;
          bodyDiv.innerHTML = renderMarkdown(accText);
        } else if (ev.type === "token") {
          accText += ev.text;
          bodyDiv.innerHTML = renderMarkdown(accText);
        } else if (ev.type === "file_write") {
          activeFilePath = ev.path;
          activeFileContent = ev.content;
          activeDiffText = "";
          if (rightTab === "files") setRightTab("code");
          else renderRightPane();
          const tc = document.createElement("div");
          tc.className = "tool-card write";
          tc.innerHTML = `<div class="tool-head"><span>WRITE FILE: ${escapeHtml(ev.path)}</span><span>${ev.content.length} chars</span></div>`;
          toolsBox.appendChild(tc);
        } else if (ev.type === "file_edit") {
          activeFilePath = ev.path;
          activeFileContent = ev.new_content;
          activeDiffText = ev.diff;
          if (rightTab === "files") setRightTab("code");
          else renderRightPane();
          const tc = document.createElement("div");
          tc.className = "tool-card edit";
          tc.innerHTML = `<div class="tool-head"><span>EDIT DIFF: ${escapeHtml(ev.path)}</span><span>updated</span></div>`;
          toolsBox.appendChild(tc);
        } else if (ev.type === "file_read") {
          activeFilePath = ev.path;
          activeFileContent = ev.content;
          activeDiffText = "";
          if (rightTab === "files") setRightTab("code");
          else renderRightPane();
          const tc = document.createElement("div");
          tc.className = "tool-card";
          tc.innerHTML = `<div class="tool-head"><span>READ FILE: ${escapeHtml(ev.path)}</span><span>${ev.total_lines} lines</span></div>`;
          toolsBox.appendChild(tc);
        } else if (ev.type === "cmd_result") {
          const tc = document.createElement("div");
          tc.className = "tool-card" + (ev.exit_code !== 0 ? " err" : "");
          const outPreview = (ev.stdout || "") + (ev.stderr ? "\n[STDERR]\n" + ev.stderr : "");
          tc.innerHTML = `<div class="tool-head"><span>${escapeHtml(ev.title)}: ${escapeHtml(ev.cmd)}</span><span>EXIT ${ev.exit_code}</span></div><div class="tool-out">${escapeHtml(outPreview.slice(0, 800))}</div>`;
          toolsBox.appendChild(tc);
        } else if (ev.type === "telemetry") {
          telemSpan.textContent = `~${ev.tokens} tok | ${ev.elapsed}s | ${ev.tps} TPS`;
        } else if (ev.type === "error") {
          accText += `\n\n[Error: ${ev.message}]`;
          bodyDiv.innerHTML = renderMarkdown(accText);
        }
        stream.scrollTop = stream.scrollHeight;
      }
    }
  } catch (err) {
    bodyDiv.innerHTML = `<span style="color:var(--red)">Request failed: ${escapeHtml(err)}</span>`;
  } finally {
    isRunning = false;
    document.getElementById("sendBtn").disabled = false;
    pollVram();
  }
}

document.getElementById("promptInput").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});
</script>
</body>
</html>
"""

app = FastAPI(title="MAPLE Local Agent Workbench")

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(WORKBENCH_HTML)

@app.get("/api/status")
async def api_status():
    return {
        "roster": MODEL_ROSTER,
        "vram": Tools.vram_status(),
    }

@app.get("/api/files")
async def api_files(path: str = Query(".")):
    return Tools.list_dir(path, recursive=False)

@app.get("/api/file")
async def api_file(path: str = Query(...)):
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": f"File not found: {path}"})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"path": path, "content": content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def _resolve_path(path: str, workdir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(workdir or os.getcwd(), path))

def _stream_model_tokens(model_key: str, prompt: str, system: Optional[str] = None):
    """Yields (token_text, stats_dict_or_none) synchronously for SSE streaming."""
    cfg = MODEL_ROSTER.get(model_key, MODEL_ROSTER["gemma"])
    ensure_vram_for_model(model_key)

    endpoint = f"{DEFAULT_OLLAMA_API}/api/generate"
    if cfg["backend"] == "bonsai_or_router":
        try:
            r = requests.get(f"{DEFAULT_OLLAMA_API}/api/tags", timeout=1.0)
            has_bonsai = any("bonsai" in m.get("name", "") for m in r.json().get("models", []))
            if not has_bonsai:
                endpoint = f"{DEFAULT_BONSAI_API}/completion"
        except Exception:
            endpoint = f"{DEFAULT_BONSAI_API}/completion"

    start_t = time.time()
    collected = []

    if endpoint.endswith("/completion"):
        prompt_fmt = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            if system
            else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        llama_payload = {
            "prompt": prompt_fmt,
            "stream": True,
            "n_predict": cfg["max_tokens"],
            "temperature": cfg["temperature"],
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        resp = requests.post(endpoint, json=llama_payload, stream=True, timeout=180)
        for line in resp.iter_lines():
            if not line:
                continue
            ls = line.decode("utf-8", errors="replace")
            if ls.startswith("data: "):
                ds = ls[6:].strip()
                if ds == "[DONE]":
                    break
                try:
                    tok = json.loads(ds).get("content", "")
                    if tok:
                        collected.append(tok)
                        yield tok, None
                except Exception:
                    pass
    else:
        payload = {
            "model": cfg["id"],
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": cfg["temperature"], "num_predict": cfg["max_tokens"]},
        }
        if system:
            payload["system"] = system
        resp = requests.post(endpoint, json=payload, stream=True, timeout=180)
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8", errors="replace"))
                tok = chunk.get("response", "")
                if tok:
                    collected.append(tok)
                    yield tok, None
                if chunk.get("done", False):
                    break
            except Exception:
                pass

    elapsed = round(time.time() - start_t, 2)
    full_text = "".join(collected)
    tokens_est = max(1, len(full_text) // 4)
    tps = round(tokens_est / elapsed, 1) if elapsed > 0 else 0.0
    yield "", {"tokens": tokens_est, "elapsed": elapsed, "tps": tps}

@app.post("/api/run")
async def api_run(req: Request):
    body = await req.json()
    prompt = (body.get("prompt") or "").strip()
    requested_model = (body.get("model") or "auto").strip().lower()
    mode = (body.get("mode") or "task").strip().lower()
    workdir = (body.get("workdir") or os.getcwd()).strip()

    chosen = auto_choose_model(prompt) if requested_model == "auto" else requested_model
    if chosen not in MODEL_ROSTER:
        chosen = "gemma"
    cfg = MODEL_ROSTER[chosen]

    def sse(obj: Dict[str, Any]) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def event_stream():
        yield sse({
            "type": "model_selected",
            "key": chosen,
            "name": cfg["name"],
            "role": cfg["role"],
        })

        try:
            if mode == "chat":
                for tok, stats in _stream_model_tokens(chosen, prompt):
                    if tok:
                        yield sse({"type": "token", "text": tok})
                    if stats:
                        yield sse({"type": "telemetry", **stats})
                yield "data: [DONE]\n\n"
                return

            # Autonomous Task Mode with live file/diff/command events
            system_prompt = (
                f"You are an autonomous engineering agent operating in working directory: {workdir}\n"
                f"Role: {cfg['role']}\n"
                f"{TOOL_SCHEMAS}\n"
                f"Rules:\n"
                f"1. To execute work, write TOOL: <name> followed by ARGS: <json>.\n"
                f"2. After tools execute, inspect their outputs and decide the next step.\n"
                f"3. Only when all tool results have been observed and the task is accomplished, output 'FINAL ANSWER:' followed by your concise report.\n"
                f"4. Ban placeholders. Write complete, working solutions."
            )
            history = f"TASK GOAL: {prompt}\n"
            max_steps = 6

            for step in range(1, max_steps + 1):
                yield sse({"type": "step", "step": step, "max_steps": max_steps})
                step_text_parts = []
                for tok, stats in _stream_model_tokens(chosen, history + "\nWhat is your next action?", system=system_prompt):
                    if tok:
                        step_text_parts.append(tok)
                        yield sse({"type": "token", "text": tok})
                    if stats:
                        yield sse({"type": "telemetry", **stats})

                response_text = "".join(step_text_parts)
                tool_matches = list(re.finditer(r"TOOL:\s*(\w+)\s*\nARGS:\s*({.*?})", response_text, re.DOTALL))
                executed_any = False

                for m in tool_matches:
                    tool_name = m.group(1).strip()
                    try:
                        args = json.loads(m.group(2).strip())
                    except Exception:
                        args = {}

                    if tool_name not in TOOL_REGISTRY:
                        history += f"\nERROR: Unknown tool '{tool_name}'\n"
                        continue

                    executed_any = True
                    # Execute cleanly without Rich CLI output, emitting structured UI events
                    if tool_name == "write_file":
                        fpath = _resolve_path(args.get("path", "output.txt"), workdir)
                        content = args.get("content", "")
                        os.makedirs(os.path.dirname(fpath), exist_ok=True)
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(content)
                        result = {"success": True, "path": fpath, "bytes_written": len(content)}
                        yield sse({"type": "file_write", "path": fpath, "content": content})

                    elif tool_name == "edit_file":
                        fpath = _resolve_path(args.get("path", ""), workdir)
                        target = args.get("target", "")
                        replacement = args.get("replacement", "")
                        if not os.path.exists(fpath):
                            result = {"error": f"File not found: {fpath}"}
                        else:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                old_data = f.read()
                            if target not in old_data:
                                result = {"error": "Target string not found in file."}
                            else:
                                new_data = old_data.replace(target, replacement)
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(new_data)
                                diff_str = "\n".join(difflib.unified_diff(
                                    old_data.splitlines(),
                                    new_data.splitlines(),
                                    fromfile=f"a/{os.path.basename(fpath)}",
                                    tofile=f"b/{os.path.basename(fpath)}",
                                    lineterm=""
                                ))
                                result = {"success": True, "path": fpath}
                                yield sse({"type": "file_edit", "path": fpath, "diff": diff_str, "new_content": new_data})

                    elif tool_name == "read_file":
                        fpath = _resolve_path(args.get("path", ""), workdir)
                        if not os.path.exists(fpath):
                            result = {"error": f"File not found: {fpath}"}
                        else:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                            raw_txt = "".join(lines)
                            result = {"path": fpath, "total_lines": len(lines), "content": raw_txt[:8000]}
                            yield sse({"type": "file_read", "path": fpath, "content": raw_txt, "total_lines": len(lines)})

                    elif tool_name == "run_command":
                        cmd = args.get("cmd", "")
                        result = Tools.run_command(f'cd /d "{workdir}" && {cmd}' if os.name == "nt" else cmd, silent=True)
                        yield sse({"type": "cmd_result", "title": "SHELL", "cmd": cmd, **result})

                    elif tool_name == "python_eval":
                        code = args.get("code", "")
                        result = Tools.python_eval(code)
                        yield sse({"type": "cmd_result", "title": "PYTHON", "cmd": code.splitlines()[0][:80] if code else "", **result})

                    else:
                        result = TOOL_REGISTRY[tool_name](**args)

                    history += f"\nACTION TAKEN: {tool_name} with {args}\nRESULT: {json.dumps(result)}\n"

                if not executed_any and ("FINAL ANSWER:" in response_text or "DONE" in response_text):
                    break

        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("MAPLE_UI_PORT", "7860"))
    print("=" * 70)
    print(f"  MAPLE Local Agent Workbench UI: http://127.0.0.1:{port}")
    print("  Zero-Electron | 100% Offline | Live File & Diff Renderer")
    print("=" * 70)
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
