const statusEl = document.getElementById("conv-state") || document.createElement("span");
const logEl = document.getElementById("log") || document.createElement("pre");

let client = null;
let localAudioTrack = null;
let session = null;
let remoteAnalyserCtx = null;
let remoteAudioSource = null;
let remoteProcessor = null;
let remoteMuteGain = null;

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
  console.log(line);
}

function setStatus(text) {
  statusEl.innerHTML = `<span class="state-badge state-idle">${text}</span>`;
  log(text);
}

function matchKeywords(label, keywords) {
  const lc = (label || "").toLowerCase();
  return keywords.some(k => lc.includes(k.toLowerCase()));
}

async function fetchSession() {
  const res = await fetch("/api/agora/session");
  if (!res.ok) throw new Error(`Session fetch failed: ${res.status}`);
  return res.json();
}

async function postStartAgent() {
  const res = await fetch("/api/agora/agent/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(`Agent start failed: ${res.status}`);
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Agent start failed");
  return data;
}

async function pickDevice(devices, keywords, kind) {
  const match = devices.find(d => matchKeywords(d.label, keywords));
  if (match) {
    log(`Found ${kind}: ${match.label}`);
    return match;
  }
  if (devices.length > 0) {
    log(`${kind} not matched, using: ${devices[0].label}`);
    return devices[0];
  }
  throw new Error(`No ${kind} device found`);
}

// --- Audio analysis for robot head wobble ---

function stopRemoteAudio() {
  if (remoteProcessor) { try { remoteProcessor.disconnect(); } catch(_){} remoteProcessor = null; }
  if (remoteAudioSource) { try { remoteAudioSource.disconnect(); } catch(_){} remoteAudioSource = null; }
  if (remoteMuteGain) { try { remoteMuteGain.disconnect(); } catch(_){} remoteMuteGain = null; }
  if (remoteAnalyserCtx) { remoteAnalyserCtx.close().catch(()=>{}); remoteAnalyserCtx = null; }
}

function startRemoteAudioAnalysis(audioTrack) {
  stopRemoteAudio();
  if (!audioTrack?.getMediaStreamTrack) return;

  try {
    const msTrack = audioTrack.getMediaStreamTrack();
    const stream = new MediaStream([msTrack]);
    remoteAnalyserCtx = new (window.AudioContext || window.webkitAudioContext)();
    remoteAudioSource = remoteAnalyserCtx.createMediaStreamSource(stream);
    remoteProcessor = remoteAnalyserCtx.createScriptProcessor(512, 1, 1);
    remoteMuteGain = remoteAnalyserCtx.createGain();
    remoteMuteGain.gain.value = 0.0; // Don't double-play, just analyze

    remoteAudioSource.connect(remoteProcessor);
    remoteProcessor.connect(remoteMuteGain);
    remoteMuteGain.connect(remoteAnalyserCtx.destination);

    if (remoteAnalyserCtx.state === "suspended") {
      remoteAnalyserCtx.resume().catch(()=>{});
    }

    remoteProcessor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      if (!input || input.length === 0) return;

      // Compute RMS energy level
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      const level = Math.min(Math.max(rms * 4.0, 0), 1);

      // Send to backend for robot head wobble
      if (level > 0.01) {
        fetch("/api/motion/audio-chunk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pcm_b64: "", level, sample_rate: 16000 }),
          keepalive: true,
        }).catch(()=>{});
      }
    };
    log("Remote audio analysis started (for robot wobble)");
  } catch (err) {
    log(`Audio analysis init failed: ${err.message}`);
  }
}

// --- Main flow ---

async function join() {
  if (client) return;

  session = await fetchSession();
  setStatus("Connecting...");

  client = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });

  // Handle datastream messages from agent (tool calls, state, etc.)
  client.on("stream-message", async (...args) => {
    let data;
    if (args.length >= 3) data = args[2];
    else if (args.length === 2) data = args[1];
    else if (args.length === 1 && args[0].data) data = args[0].data;
    else return;

    let text;
    try {
      if (typeof data === "string") text = data;
      else if (data instanceof Uint8Array) text = new TextDecoder().decode(data);
      else if (data instanceof ArrayBuffer) text = new TextDecoder().decode(new Uint8Array(data));
      else text = String(data);
    } catch (e) { return; }

    // Forward to backend for robot actions + logging
    try {
      await fetch("/api/datastream/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, ts: Date.now() }),
        keepalive: true,
      });
    } catch (_) {}
  });

  // Subscribe to remote audio (agent's TTS)
  client.on("user-published", async (user, mediaType) => {
    await client.subscribe(user, mediaType);
    log(`Subscribed to agent ${mediaType} (uid ${user.uid})`);
    if (mediaType === "audio") {
      if (user.audioTrack?.setPlaybackDevice && session.speakerDeviceId) {
        try {
          await user.audioTrack.setPlaybackDevice(session.speakerDeviceId);
        } catch (e) {
          log(`Speaker routing failed: ${e.message}`);
        }
      }
      // Start audio analysis for robot head wobble
      startRemoteAudioAnalysis(user.audioTrack);
      user.audioTrack.play();
      log("Agent audio playing");
    }
  });

  client.on("user-unpublished", (user, mediaType) => {
    log(`Agent ${mediaType} unpublished (uid ${user.uid})`);
    if (mediaType === "audio") {
      stopRemoteAudio();
    }
  });

  // Get mic permission
  await navigator.mediaDevices.getUserMedia({ audio: true });

  const mics = await AgoraRTC.getMicrophones();
  const speakers = await AgoraRTC.getPlaybackDevices();
  log(`Mics: ${mics.map(m => m.label).join(", ")}`);
  log(`Speakers: ${speakers.map(s => s.label).join(", ")}`);

  const keywords = session.deviceKeywords || ["Reachy", "Hollyland", "Wireless", "USB"];
  const mic = await pickDevice(mics, keywords, "microphone");

  const spkKeywords = session.speakerKeywords || ["Reachy", "Bluetooth", "bluez", "Pollen"];
  let speaker = null;
  try {
    speaker = await pickDevice(speakers, spkKeywords, "speaker");
    session.speakerDeviceId = speaker.deviceId;
  } catch (_) {
    log("No matched speaker, using default output");
  }

  localAudioTrack = await AgoraRTC.createMicrophoneAudioTrack({
    microphoneId: mic.deviceId,
  });

  await client.join(session.appId, session.channel, session.token || null, session.uid || null);
  log("Joined channel, starting agent...");

  // Notify backend session is active
  fetch("/api/motion/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active: true }),
  }).catch(()=>{});

  await postStartAgent();
  await client.publish([localAudioTrack]);
  setStatus(`Connected to ${session.channel} as uid ${session.uid}`);
}

async function leave() {
  fetch("/api/motion/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active: false }),
    keepalive: true,
  }).catch(()=>{});

  stopRemoteAudio();
  if (localAudioTrack) {
    localAudioTrack.stop();
    localAudioTrack.close();
    localAudioTrack = null;
  }
  if (client) {
    await client.leave();
    client.removeAllListeners();
    client = null;
  }
  setStatus("Disconnected");
}

window.addEventListener("load", async () => {
  try {
    await join();
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    // Show error on dashboard too
    const el = document.getElementById("agent-text");
    if (el) el.textContent = `ERROR: ${err.message}`;
    console.error(err);
  }
});

window.addEventListener("beforeunload", () => {
  leave();
});
