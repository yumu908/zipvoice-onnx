// App state and UI handlers
let selectedLanguage = 'zh';
let selectedMode = 'non-stream';
let voicesList = [];

// Audio variables
let audioCtx = null;
let nextPlayTime = 0;
let visualizerAnalyser = null;
let activeSources = [];
let audioPlayerElement = document.getElementById("audio-player");
let refAudioPlayer = document.getElementById("ref-audio-player");
let isAudioPlaying = false;
let isRefAudioPlaying = false;
let currentAudioUrl = null;

// Preset texts
const presets = {
    zh: [
        {
            name: "诗词意境 (短)",
            text: "大漠飞雪，是一场跨越时空的凄美邂逅。金黄浩瀚的沙海，本是烈日与焦土的领地，却在寒流骤起之时，也是荒野的叹息。"
        },
        {
            name: "散文风光 (长)",
            text: "狂风卷着雪花，在大地上奔走呼号，千万朵洁白在沙丘间碰撞、破碎、又旋转而起。沙子是暖色的苍凉，雪花是冷色的孤傲，二者在交织中褪去了原本的颜色，幻化成一种混沌而深邃的苍茫。行人踏过，脚印瞬间被风雪填平，留不住过往，更看不清前路。"
        }
    ],
    en: [
        {
            name: "Descriptive",
            text: "There is a sublime, terrifying silence beneath the gale; the shifting sands are quickly buried under a pristine, shifting mantle of frost. The endless ridges of the desert, usually defined by their sweeping curves, are now softened and blurred by the blinding curtain of snow."
        },
        {
            name: "Technical",
            text: "ZipVoice leverages a distilled flow matching decoder coupled with a feed forward text encoder to deliver fast and robust speech synthesis directly inside your web application."
        }
    ]
};

// Console logs helper
function log(message, type = 'info') {
    const logsContainer = document.getElementById("console-logs");
    const line = document.createElement("div");
    line.className = `log-line ${type}-msg`;
    
    const time = new Date().toLocaleTimeString();
    line.innerText = `[${time}] ${message}`;
    
    logsContainer.appendChild(line);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Format time
function formatTime(seconds) {
    if (isNaN(seconds) || seconds === Infinity) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

// Update UI displays for sliders
function initSliders() {
    const speedSlider = document.getElementById("speed-slider");
    const speedVal = document.getElementById("speed-val");
    speedSlider.addEventListener("input", () => speedVal.innerText = parseFloat(speedSlider.value).toFixed(2));

    const stepsSlider = document.getElementById("steps-slider");
    const stepsVal = document.getElementById("steps-val");
    stepsSlider.addEventListener("input", () => stepsVal.innerText = stepsSlider.value);

    const guidanceSlider = document.getElementById("guidance-slider");
    const guidanceVal = document.getElementById("guidance-val");
    guidanceSlider.addEventListener("input", () => guidanceVal.innerText = parseFloat(guidanceSlider.value).toFixed(1));

    const tshiftSlider = document.getElementById("tshift-slider");
    const tshiftVal = document.getElementById("tshift-val");
    tshiftSlider.addEventListener("input", () => tshiftVal.innerText = parseFloat(tshiftSlider.value).toFixed(2));
}

// Load and render presets
function loadPresets(lang) {
    const container = document.getElementById("presets-container");
    container.innerHTML = "";
    presets[lang].forEach(preset => {
        const btn = document.createElement("button");
        btn.className = "preset-btn";
        btn.innerText = preset.name;
        btn.addEventListener("click", () => {
            document.getElementById("target-text").value = preset.text;
            updateCharCount();
            log(`Loaded preset: "${preset.name}"`);
        });
        container.appendChild(btn);
    });
}

function updateCharCount() {
    const text = document.getElementById("target-text").value;
    document.getElementById("char-count-val").innerText = text.length;
}

// Fetch voice lists from server
async function fetchVoices() {
    try {
        const res = await fetch("/api/voices");
        if (!res.ok) throw new Error("Failed to load voices list");
        voicesList = await res.json();
        
        const select = document.getElementById("voice-select");
        select.innerHTML = "";
        
        voicesList.forEach(v => {
            const opt = document.createElement("option");
            opt.value = v.path;
            const prefix = v.type === 'system' ? '💻 System' : '📂 Custom';
            opt.innerText = `${prefix} - ${v.name}`;
            select.appendChild(opt);
        });
        
        // Select first item by default
        if (voicesList.length > 0) {
            select.value = voicesList[0].path;
            document.getElementById("ref-play-btn").disabled = false;
            document.getElementById("ref-recognize-btn").disabled = false;
            log(`Loaded ${voicesList.length} voices successfully.`);
            document.getElementById("engine-status").innerText = "Engine Idle";
        } else {
            document.getElementById("ref-play-btn").disabled = true;
            document.getElementById("ref-recognize-btn").disabled = true;
            document.getElementById("engine-status").innerText = "No voices found";
        }
    } catch (e) {
        log(`Failed to retrieve voices: ${e.message}`, 'error');
        document.getElementById("engine-status").innerText = "Error loading engine";
        const dot = document.querySelector(".pulse-dot");
        dot.className = "pulse-dot error";
    }
}

// Handle Drag & Drop Voice Upload
function initUpload() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });
}

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith(".wav")) {
        log("Error: Only .wav files are supported for voice cloning", "error");
        return;
    }
    
    log(`Uploading reference audio "${file.name}"...`);
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("Upload failed");
        
        const data = await res.json();
        log(`Successfully uploaded: ${data.name}. Cloned voice is active.`, 'success');
        
        // Refresh voices and select the new one
        await fetchVoices();
        document.getElementById("voice-select").value = data.path;
    } catch (e) {
        log(`Upload failed: ${e.message}`, 'error');
    }
}

// Set up UI components & interactions
document.addEventListener("DOMContentLoaded", () => {
    initSliders();
    initUpload();
    
    // Language tabs
    document.querySelectorAll(".lang-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".lang-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            selectedLanguage = tab.dataset.lang;
            loadPresets(selectedLanguage);
            log(`Language switched to: ${selectedLanguage === 'zh' ? 'Chinese' : 'English'}`);
        });
    });
    
    // Mode selectors
    document.querySelectorAll(".mode-option").forEach(opt => {
        opt.addEventListener("click", () => {
            document.querySelectorAll(".mode-option").forEach(o => o.classList.remove("active"));
            opt.classList.add("active");
            selectedMode = opt.dataset.mode;
            log(`Generation mode set to: ${opt.querySelector("h4").innerText}`);
            
            // Adjust player layout based on streaming vs standard
            const downloadBtn = document.getElementById("download-audio-btn");
            const playBtn = document.getElementById("play-pause-btn");
            if (selectedMode !== 'non-stream') {
                downloadBtn.style.display = 'none';
                playBtn.disabled = true; // Streaming manages its own audio output context
            } else {
                if (currentAudioUrl) {
                    downloadBtn.style.display = 'flex';
                    playBtn.disabled = false;
                }
            }
        });
    });

    // Text area listener
    document.getElementById("target-text").addEventListener("input", updateCharCount);

    // Initial load
    selectedLanguage = 'zh';
    loadPresets('zh');
    fetchVoices();
    
    // Action button
    document.getElementById("generate-btn").addEventListener("click", executeSynthesis);
    
    // Audio Player Elements
    const playPauseBtn = document.getElementById("play-pause-btn");
    const progressTrack = document.getElementById("progress-track");
    const progressFill = document.getElementById("progress-fill");
    
    playPauseBtn.addEventListener("click", toggleStandardPlayback);
    audioPlayerElement.addEventListener("timeupdate", () => {
        const current = audioPlayerElement.currentTime;
        const total = audioPlayerElement.duration || 0;
        document.getElementById("current-time").innerText = formatTime(current);
        document.getElementById("total-time").innerText = formatTime(total);
        if (total > 0) {
            progressFill.style.width = `${(current / total) * 100}%`;
        }
    });

    audioPlayerElement.addEventListener("ended", () => {
        playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        isAudioPlaying = false;
        document.getElementById("visualizer-idle-msg").style.opacity = '1';
    });

    progressTrack.addEventListener("click", (e) => {
        if (!currentAudioUrl || selectedMode !== 'non-stream') return;
        const rect = progressTrack.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        audioPlayerElement.currentTime = percent * audioPlayerElement.duration;
    });

    document.getElementById("clear-console-btn").addEventListener("click", () => {
        document.getElementById("console-logs").innerHTML = "";
        log("Console cleared.");
    });

    // Reference Audio Preview Button
    const refPlayBtn = document.getElementById("ref-play-btn");
    refPlayBtn.addEventListener("click", () => {
        const selectedVoicePath = document.getElementById("voice-select").value;
        if (!selectedVoicePath) return;

        if (isRefAudioPlaying) {
            refAudioPlayer.pause();
            refPlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
            isRefAudioPlaying = false;
        } else {
            // Stop main generated audio playback only
            audioPlayerElement.pause();
            audioPlayerElement.currentTime = 0;
            document.getElementById("play-pause-btn").innerHTML = '<i class="fa-solid fa-play"></i>';
            isAudioPlaying = false;

            activeSources.forEach(s => {
                try { s.stop(); } catch(e) {}
            });
            activeSources = [];
            nextPlayTime = 0;
            document.getElementById("visualizer-idle-msg").style.opacity = '1';

            const previewUrl = `/api/voices/audio?path=${encodeURIComponent(selectedVoicePath)}`;
            if (!refAudioPlayer.src || !refAudioPlayer.src.includes(encodeURIComponent(selectedVoicePath))) {
                refAudioPlayer.src = previewUrl;
            }
            
            refAudioPlayer.play().then(() => {
                refPlayBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
                isRefAudioPlaying = true;
            }).catch(err => {
                log(`Failed to play preview audio: ${err.message}`, 'error');
            });
        }
    });

    refAudioPlayer.addEventListener("ended", () => {
        refPlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        isRefAudioPlaying = false;
    });

    document.getElementById("voice-select").addEventListener("change", () => {
        // Stop current preview
        refAudioPlayer.pause();
        refAudioPlayer.src = "";
        refPlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        isRefAudioPlaying = false;
    });

    // Reference Text Auto Recognition (Whisper)
    const refRecognizeBtn = document.getElementById("ref-recognize-btn");
    refRecognizeBtn.addEventListener("click", async () => {
        const selectedVoicePath = document.getElementById("voice-select").value;
        if (!selectedVoicePath) {
            log("Error: Please select a reference voice first", "error");
            return;
        }

        // Disable inputs & buttons
        refRecognizeBtn.disabled = true;
        refRecognizeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>Auto</span>';
        const refTextInput = document.getElementById("ref-text-input");
        refTextInput.disabled = true;

        log(`Automatically transcribing reference audio: ${selectedVoicePath}...`);

        try {
            const res = await fetch("/api/voices/transcribe", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ path: selectedVoicePath })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Transcription failed");
            }

            const data = await res.json();
            refTextInput.value = data.text;
            log(`Transcription complete: "${data.text}"`, 'success');
        } catch (e) {
            log(`Transcription failed: ${e.message}`, 'error');
        } finally {
            refRecognizeBtn.disabled = false;
            refRecognizeBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i><span>Auto</span>';
            refTextInput.disabled = false;
        }
    });
    
    // Setup Visualizer Animation Loop
    setupVisualizerCanvas();
    drawVisualizer();
});

// Setup custom audio visualizer Web Audio API nodes
function initAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        visualizerAnalyser = audioCtx.createAnalyser();
        visualizerAnalyser.fftSize = 256;
        visualizerAnalyser.connect(audioCtx.destination);
        
        // Connect HTML Audio tag to visualizer context
        const source = audioCtx.createMediaElementSource(audioPlayerElement);
        source.connect(visualizerAnalyser);
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function stopAllPlayback() {
    // Stop standard player
    audioPlayerElement.pause();
    audioPlayerElement.currentTime = 0;
    document.getElementById("play-pause-btn").innerHTML = '<i class="fa-solid fa-play"></i>';
    isAudioPlaying = false;
    
    // Stop reference audio preview player
    if (refAudioPlayer) {
        refAudioPlayer.pause();
        refAudioPlayer.currentTime = 0;
        document.getElementById("ref-play-btn").innerHTML = '<i class="fa-solid fa-play"></i>';
        isRefAudioPlaying = false;
    }
    
    // Stop active scheduled Web Audio sources (for streaming/ws)
    activeSources.forEach(s => {
        try { s.stop(); } catch(e) {}
    });
    activeSources = [];
    nextPlayTime = 0;
    
    document.getElementById("visualizer-idle-msg").style.opacity = '1';
}

function playPCMChunk(base64Audio, sampleRate) {
    initAudioContext();
    
    // Decode base64 to binary array
    const binaryString = atob(base64Audio);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Convert 16-bit signed PCM bytes to Float32 array
    const int16Array = new Int16Array(bytes.buffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }
    
    // Create AudioBuffer
    const audioBuffer = audioCtx.createBuffer(1, float32Array.length, sampleRate);
    audioBuffer.copyToChannel(float32Array, 0);
    
    // Create Source Node
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    
    // Connect to visualizer analyser
    source.connect(visualizerAnalyser);
    
    // Schedule playback
    const now = audioCtx.currentTime;
    if (nextPlayTime < now) {
        nextPlayTime = now;
    }
    source.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;
    
    activeSources.push(source);
    source.onended = () => {
        activeSources = activeSources.filter(s => s !== source);
        if (activeSources.length === 0 && nextPlayTime <= audioCtx.currentTime) {
            document.getElementById("visualizer-idle-msg").style.opacity = '1';
        }
    };
    
    return audioBuffer.duration;
}

function toggleStandardPlayback() {
    if (!currentAudioUrl) return;
    initAudioContext();
    
    if (isAudioPlaying) {
        audioPlayerElement.pause();
        document.getElementById("play-pause-btn").innerHTML = '<i class="fa-solid fa-play"></i>';
        isAudioPlaying = false;
        document.getElementById("visualizer-idle-msg").style.opacity = '1';
    } else {
        audioPlayerElement.play();
        document.getElementById("play-pause-btn").innerHTML = '<i class="fa-solid fa-pause"></i>';
        isAudioPlaying = true;
        document.getElementById("visualizer-idle-msg").style.opacity = '0';
    }
}

// Dynamic visualizer drawing loop
function setupVisualizerCanvas() {
    const canvas = document.getElementById("visualizer");
    // Handle retina display
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.parentElement.clientWidth * dpr;
    canvas.height = canvas.parentElement.clientHeight * dpr;
    
    window.addEventListener('resize', () => {
        canvas.width = canvas.parentElement.clientWidth * dpr;
        canvas.height = canvas.parentElement.clientHeight * dpr;
    });
}

function drawVisualizer() {
    requestAnimationFrame(drawVisualizer);
    const canvas = document.getElementById("visualizer");
    const canvasCtx = canvas.getContext("2d");
    
    if (!visualizerAnalyser) {
        canvasCtx.fillStyle = '#090d16';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        return;
    }
    
    const bufferLength = visualizerAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Waveform visualization
    visualizerAnalyser.getByteTimeDomainData(dataArray);
    
    canvasCtx.fillStyle = 'rgba(9, 13, 22, 0.4)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    canvasCtx.lineWidth = 3;
    const gradient = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, '#06b6d4');
    gradient.addColorStop(0.5, '#a855f7');
    gradient.addColorStop(1, '#06b6d4');
    canvasCtx.strokeStyle = gradient;
    
    canvasCtx.beginPath();
    
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    let isSilent = true;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        
        if (Math.abs(dataArray[i] - 128) > 2) {
            isSilent = false;
        }
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    if (isSilent) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, canvas.height / 2);
        canvasCtx.lineTo(canvas.width, canvas.height / 2);
    } else {
        canvasCtx.lineTo(canvas.width, canvas.height / 2);
    }
    canvasCtx.stroke();
}

// Core Synthesis Function
async function executeSynthesis() {
    const text = document.getElementById("target-text").value.strip ? document.getElementById("target-text").value.strip() : document.getElementById("target-text").value.trim();
    if (!text) {
        log("Error: Target text cannot be empty", "error");
        return;
    }
    
    const refWav = document.getElementById("voice-select").value;
    if (!refWav) {
        log("Error: Please select a reference voice first", "error");
        return;
    }
    
    // Collect parameters
    const refText = document.getElementById("ref-text-input").value.trim() || null;
    const speed = parseFloat(document.getElementById("speed-slider").value);
    const numSteps = parseInt(document.getElementById("steps-slider").value);
    const guidanceScale = parseFloat(document.getElementById("guidance-slider").value);
    const tShift = parseFloat(document.getElementById("tshift-slider").value);
    
    const requestData = {
        text: text,
        language: selectedLanguage,
        ref_wav: refWav,
        ref_text: refText,
        speed: speed,
        num_steps: numSteps,
        guidance_scale: guidanceScale,
        t_shift: tShift
    };
    
    // Reset/Stop previous audio playback
    stopAllPlayback();
    
    // Update button status
    const generateBtn = document.getElementById("generate-btn");
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="btn-text">Generating...</span><i class="fa-solid fa-spinner fa-spin"></i>';
    document.getElementById("engine-status").innerText = "Generating Speech...";
    const dot = document.querySelector(".pulse-dot");
    dot.className = "pulse-dot loading";
    
    log(`Starting synthesis in [${selectedMode}] mode...`);
    
    try {
        if (selectedMode === 'non-stream') {
            await runNonStreaming(requestData);
        } else if (selectedMode === 'stream') {
            await runStreamingHTTP(requestData);
        } else if (selectedMode === 'websocket') {
            await runStreamingWebSocket(requestData);
        }
    } catch (e) {
        log(`Synthesis failed: ${e.message}`, 'error');
        dot.className = "pulse-dot error";
        document.getElementById("engine-status").innerText = "Inference Failed";
    } finally {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<span class="btn-text">Generate Speech</span><i class="fa-solid fa-play"></i>';
        if (dot.className.includes("loading")) {
            dot.className = "pulse-dot";
            document.getElementById("engine-status").innerText = "Engine Idle";
        }
    }
}

// 1. HTTP Non-streaming
async function runNonStreaming(payload) {
    log("Requesting full audio generation...");
    const start = performance.now();
    
    const res = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "HTTP non-streaming synthesis failed");
    }
    
    const blob = await res.blob();
    const duration = (performance.now() - start) / 1000;
    log(`Audio fully generated in ${duration.toFixed(2)} seconds. Loading to player.`, 'success');
    
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
    }
    currentAudioUrl = URL.createObjectURL(blob);
    
    audioPlayerElement.src = currentAudioUrl;
    
    // Enable controls
    document.getElementById("play-pause-btn").disabled = false;
    const dlBtn = document.getElementById("download-audio-btn");
    dlBtn.href = currentAudioUrl;
    dlBtn.style.display = 'flex';
    
    // Auto play
    toggleStandardPlayback();
}

// 2. HTTP Streaming (ndjson)
async function runStreamingHTTP(payload) {
    log("Opening sentence-by-sentence HTTP stream...");
    
    const res = await fetch("/api/generate_stream?format=json", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "HTTP streaming synthesis failed");
    }
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    
    // Hide idle msg
    document.getElementById("visualizer-idle-msg").style.opacity = '0';
    
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last partial line in buffer
        buffer = lines.pop();
        
        for (const line of lines) {
            if (!line.trim()) continue;
            const data = JSON.parse(line);
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (data.audio) {
                log(`Synthesized chunk ${data.index + 1}: "${data.text.substring(0, 15)}..."`, 'info');
                playPCMChunk(data.audio, data.sample_rate);
            }
        }
    }
    
    log("HTTP streaming complete.", 'success');
}

// 3. WebSocket Streaming
async function runStreamingWebSocket(payload) {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/synthesize`;
        
        log(`Connecting to WebSocket at ${wsUrl}...`);
        const socket = new WebSocket(wsUrl);
        
        // Hide idle msg
        document.getElementById("visualizer-idle-msg").style.opacity = '0';
        
        socket.onopen = () => {
            log("WebSocket connected. Sending synthesis parameters...", 'ws');
            socket.send(JSON.stringify(payload));
        };
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.event === 'audio') {
                log(`WS Audio Received (chunk ${data.index + 1}): "${data.text.substring(0, 15)}..."`, 'ws');
                playPCMChunk(data.audio, data.sample_rate);
            } else if (data.event === 'done') {
                log("WebSocket generation finished.", 'success');
                socket.close();
                resolve();
            } else if (data.event === 'error') {
                log(`WebSocket Engine Error: ${data.message}`, 'error');
                socket.close();
                reject(new Error(data.message));
            }
        };
        
        socket.onerror = (err) => {
            log("WebSocket error occurred.", 'error');
            reject(err);
        };
        
        socket.onclose = () => {
            log("WebSocket connection closed.", 'ws');
        };
    });
}
