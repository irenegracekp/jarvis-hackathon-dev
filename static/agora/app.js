const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let client = null;
let localAudioTrack = null;
let session = null;

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
  console.log(line);
}

function setStatus(text) {
  statusEl.textContent = text;
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
  // Try to match by keywords
  const match = devices.find(d => matchKeywords(d.label, keywords));
  if (match) {
    log(`Found ${kind}: ${match.label}`);
    return match;
  }
  // Fallback to first available
  if (devices.length > 0) {
    log(`${kind} not matched by keywords, using: ${devices[0].label}`);
    return devices[0];
  }
  throw new Error(`No ${kind} device found`);
}

async function join() {
  if (client) return;

  session = await fetchSession();
  setStatus("Connecting...");

  client = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });

  // Handle datastream messages from agent (tool calls, etc.)
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

    // Forward to backend for processing
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
      // Route to selected speaker
      if (user.audioTrack?.setPlaybackDevice && session.speakerDeviceId) {
        try {
          await user.audioTrack.setPlaybackDevice(session.speakerDeviceId);
        } catch (e) {
          log(`Speaker routing failed: ${e.message}`);
        }
      }
      user.audioTrack.play();
      log("Agent audio playing");
    }
  });

  client.on("user-unpublished", (user, mediaType) => {
    log(`Agent ${mediaType} unpublished (uid ${user.uid})`);
  });

  // Get mic permission first
  await navigator.mediaDevices.getUserMedia({ audio: true });

  const mics = await AgoraRTC.getMicrophones();
  const speakers = await AgoraRTC.getPlaybackDevices();
  log(`Mics: ${mics.map(m => m.label).join(", ")}`);
  log(`Speakers: ${speakers.map(s => s.label).join(", ")}`);

  const keywords = session.deviceKeywords || ["Hollyland", "Wireless", "Shenzhen", "USB"];
  const mic = await pickDevice(mics, keywords, "microphone");

  // Find Bluetooth or USB speaker
  const spkKeywords = session.speakerKeywords || ["Bluetooth", "bluez", "A2DP"];
  let speaker = null;
  try {
    speaker = await pickDevice(speakers, spkKeywords, "speaker");
    session.speakerDeviceId = speaker.deviceId;
  } catch (_) {
    log("No Bluetooth speaker found, using default output");
  }

  localAudioTrack = await AgoraRTC.createMicrophoneAudioTrack({
    microphoneId: mic.deviceId,
  });

  await client.join(session.appId, session.channel, session.token || null, session.uid || null);
  log("Joined channel, starting agent...");

  await postStartAgent();
  await client.publish([localAudioTrack]);
  setStatus(`Connected to ${session.channel} as uid ${session.uid}`);
}

async function leave() {
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

// Auto-join on load
window.addEventListener("load", async () => {
  try {
    await join();
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    console.error(err);
  }
});

window.addEventListener("beforeunload", () => {
  leave();
});
