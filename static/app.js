/**
 * Audio Transcriber & AI Summarizer Client-side Application
 * Event-Driven Architecture with Background Job Streaming
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const audioFileInput = document.getElementById("audio-file-input");
  const selectedFileCard = document.getElementById("selected-file-card");
  const fileNameDisplay = document.getElementById("file-name");
  const fileSizeDisplay = document.getElementById("file-size");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const audioPreview = document.getElementById("audio-preview");

  const tabUploadBtn = document.getElementById("tab-upload-btn");
  const tabRecordBtn = document.getElementById("tab-record-btn");
  const dropzonePanel = document.getElementById("dropzone-panel");
  const recorderPanel = document.getElementById("recorder-panel");
  const recordToggleBtn = document.getElementById("record-toggle-btn");
  const recordTimer = document.getElementById("record-timer");
  const recordStatus = document.getElementById("record-status");
  const recordPreview = document.getElementById("record-preview");

  const whisperEngineSelect = document.getElementById("whisper-engine-select");
  const whisperModelSelect = document.getElementById("whisper-model-select");
  const languageSelect = document.getElementById("language-select");
  const vadCheckbox = document.getElementById("vad-checkbox");
  const llmProviderSelect = document.getElementById("llm-provider-select");
  const llmModelSelect = document.getElementById("llm-model-select");
  const refreshModelsBtn = document.getElementById("refresh-models-btn");
  const notifyCheckbox = document.getElementById("notify-checkbox");
  const startProcessBtn = document.getElementById("start-process-btn");

  const summaryOptionsBox = document.getElementById("summary-options-box");
  const progressContainer = document.getElementById("progress-container");
  const currentStepLabel = document.getElementById("current-step-label");
  const stepCounter = document.getElementById("step-counter");
  const progressBar = document.getElementById("progress-bar");

  const resultsContainer = document.getElementById("results-container");
  const resTabTranscript = document.getElementById("res-tab-transcript");
  const resTabPolish = document.getElementById("res-tab-polish");
  const resTabSummary = document.getElementById("res-tab-summary");
  const viewportTranscript = document.getElementById("viewport-transcript");
  const viewportPolish = document.getElementById("viewport-polish");
  const viewportSummary = document.getElementById("viewport-summary");
  const transcriptSegmentsList = document.getElementById("transcript-segments-list");
  const transcriptPlainText = document.getElementById("transcript-plain-text");
  const toggleTimestamps = document.getElementById("toggle-timestamps");
  const transcriptionStats = document.getElementById("transcription-stats");
  const polishContent = document.getElementById("polish-content");
  const summaryContent = document.getElementById("summary-content");

  const copyBtn = document.getElementById("copy-btn");
  const downloadTxtBtn = document.getElementById("download-txt-btn");
  const downloadSrtBtn = document.getElementById("download-srt-btn");
  const downloadVttBtn = document.getElementById("download-vtt-btn");
  const downloadAssBtn = document.getElementById("download-ass-btn");
  const downloadJsonBtn = document.getElementById("download-json-btn");

  const tabBadgePolish = document.getElementById("tab-badge-polish");
  const tabBadgeSummary = document.getElementById("tab-badge-summary");
  const polishToolbar = document.getElementById("polish-toolbar");
  const repolishBtn = document.getElementById("repolish-btn");
  const summaryToolbar = document.getElementById("summary-toolbar");
  const resummarizeBtn = document.getElementById("resummarize-btn");
  const resummarizeLevelBtns = document.querySelectorAll(".resummarize-level-btn");
  let resummarizeActiveLevel = "bullets";
  let ondemandSummaryActiveLevel = "bullets";

  const diarizationCheckbox = document.getElementById("diarization-checkbox");
  const diarizationBadge = document.getElementById("diarization-badge");
  const numSpeakersSelect = document.getElementById("num-speakers-select");

  const openSettingsBtn = document.getElementById("open-settings-btn");
  const closeSettingsBtn = document.getElementById("close-settings-btn");
  const settingsModal = document.getElementById("settings-modal");
  const testTelegramBtn = document.getElementById("test-telegram-btn");
  const telegramTokenInput = document.getElementById("telegram-token-input");
  const telegramChatIdInput = document.getElementById("telegram-chat-id-input");
  const telegramTestResult = document.getElementById("telegram-test-result");
  const pullModelBtn = document.getElementById("pull-model-btn");
  const pullModelNameInput = document.getElementById("pull-model-name-input");
  const pullModelStatus = document.getElementById("pull-model-status");
  const ollamaStatusBadge = document.getElementById("ollama-status-badge");

  const testGroqBtn = document.getElementById("test-groq-btn");
  const groqApiKeyInput = document.getElementById("groq-api-key-input");
  const groqTestResult = document.getElementById("groq-test-result");
  const testOpenrouterBtn = document.getElementById("test-openrouter-btn");
  const openrouterApiKeyInput = document.getElementById("openrouter-api-key-input");
  const openrouterTestResult = document.getElementById("openrouter-test-result");
  const testOpenaiBtn = document.getElementById("test-openai-btn");
  const openaiBaseUrlInput = document.getElementById("openai-base-url-input");
  const openaiApiKeyInput = document.getElementById("openai-api-key-input");
  const openaiTestResult = document.getElementById("openai-test-result");
  const saveSettingsBtn = document.getElementById("save-settings-btn");
  const settingsSaveFeedback = document.getElementById("settings-save-feedback");

  const liveStreamCheckbox = document.getElementById("live-stream-checkbox");
  const liveStreamBox = document.getElementById("live-stream-box");
  const liveTranscriptFeed = document.getElementById("live-transcript-feed");
  const liveFinalText = document.getElementById("live-final-text");
  const liveInterimText = document.getElementById("live-interim-text");
  const liveCursor = document.getElementById("live-cursor");

  // State
  let selectedFile = null;
  let recordedBlob = null;
  let isRecording = false;
  let mediaRecorder = null;
  let recordChunks = [];
  let recordTimerInterval = null;
  let recordSeconds = 0;

  let liveWs = null;
  let liveAudioCtx = null;
  let liveScriptNode = null;
  let liveMicStream = null;

  let activeAiAction = "summary"; // 'raw', 'polish', 'summary'
  let activeSummaryLevel = "bullets"; // 'tldr', 'bullets', 'detailed', 'action_items', 'custom'

  // LocalStorage UI Preferences Persistence
  const UI_STORAGE_KEY = "transcriber_ui_preferences_v1";

  function getStoredPreferences() {
    try {
      const raw = localStorage.getItem(UI_STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) {
      return {};
    }
  }

  function savePreference(key, value) {
    try {
      const prefs = getStoredPreferences();
      prefs[key] = value;
      localStorage.setItem(UI_STORAGE_KEY, JSON.stringify(prefs));
    } catch (_) {}
  }

  function applyStoredPreferences() {
    const prefs = getStoredPreferences();

    // 1. Whisper Engine
    if (prefs.whisper_engine && whisperEngineSelect) {
      const exists = Array.from(whisperEngineSelect.options).some((o) => o.value === prefs.whisper_engine);
      if (exists) whisperEngineSelect.value = prefs.whisper_engine;
    }

    // 2. Language
    if (prefs.language && languageSelect) {
      const exists = Array.from(languageSelect.options).some((o) => o.value === prefs.language);
      if (exists) languageSelect.value = prefs.language;
    }

    // 3. AI Action (raw, polish, summary)
    if (prefs.ai_action) {
      activeAiAction = prefs.ai_action;
      const aiBtns = document.querySelectorAll(".ai-action-btn");
      aiBtns.forEach((btn) => {
        if (btn.getAttribute("data-action") === activeAiAction) {
          btn.classList.add("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
          btn.classList.remove("border-slate-700", "bg-slate-950", "text-slate-300");
        } else {
          btn.classList.remove("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
          btn.classList.add("border-slate-700", "bg-slate-950", "text-slate-300");
        }
      });
      if (summaryOptionsBox) {
        if (activeAiAction === "summary") {
          summaryOptionsBox.classList.remove("hidden");
        } else {
          summaryOptionsBox.classList.add("hidden");
        }
      }
    }

    // 4. Summary Detail Level
    if (prefs.summary_level) {
      activeSummaryLevel = prefs.summary_level;
      const lvlBtns = document.querySelectorAll(".summary-level-btn");
      lvlBtns.forEach((btn) => {
        if (btn.getAttribute("data-level") === activeSummaryLevel) {
          btn.classList.add("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
          btn.classList.remove("border-slate-700", "bg-slate-950", "text-slate-300");
        } else {
          btn.classList.remove("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
          btn.classList.add("border-slate-700", "bg-slate-950", "text-slate-300");
        }
      });
    }

    // 5. LLM Provider
    if (prefs.llm_provider && llmProviderSelect) {
      const exists = Array.from(llmProviderSelect.options).some((o) => o.value === prefs.llm_provider);
      if (exists) llmProviderSelect.value = prefs.llm_provider;
    }

    // 6. Diarization & Checkboxes
    if (prefs.diarization_enabled !== undefined && diarizationCheckbox) {
      diarizationCheckbox.checked = Boolean(prefs.diarization_enabled);
    }
    if (prefs.num_speakers !== undefined && numSpeakersSelect) {
      numSpeakersSelect.value = prefs.num_speakers;
    }
    if (prefs.vad_enabled !== undefined && vadCheckbox) {
      vadCheckbox.checked = Boolean(prefs.vad_enabled);
    }
    if (prefs.notify_enabled !== undefined && notifyCheckbox) {
      notifyCheckbox.checked = Boolean(prefs.notify_enabled);
    }
  }

  let currentResult = {
    text: "",
    srt: "",
    vtt: "",
    ass: "",
    word_vtt: "",
    segments: [],
    speaker_turns: [],
    num_speakers: 0,
    polished: "",
    summary: "",
    duration: 0,
    processing_time: 0,
    filename: "",
    task_id: "",
  };

  const speakerPalette = [
    { badge: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40 hover:bg-indigo-500/30" },
    { badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/30" },
    { badge: "bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30" },
    { badge: "bg-rose-500/20 text-rose-300 border-rose-500/40 hover:bg-rose-500/30" },
    { badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 hover:bg-cyan-500/30" },
    { badge: "bg-purple-500/20 text-purple-300 border-purple-500/40 hover:bg-purple-500/30" },
  ];

  function getSpeakerStyle(speakerName) {
    if (!speakerName) return speakerPalette[0];
    let hash = 0;
    for (let i = 0; i < speakerName.length; i++) {
      hash = (hash << 5) - hash + speakerName.charCodeAt(i);
      hash |= 0;
    }
    const idx = Math.abs(hash) % speakerPalette.length;
    return speakerPalette[idx];
  }

  // Check Ollama and System status on load
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      
      if (data.ollama_connection && data.ollama_connection.online) {
        ollamaStatusBadge.innerHTML = `
          <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span>Ollama Online (${data.ollama_connection.model_count} models)</span>
        `;
      } else {
        ollamaStatusBadge.innerHTML = `
          <span class="w-2 h-2 rounded-full bg-amber-400"></span>
          <span>Ollama Offline / Remote</span>
        `;
      }

      if (data.diarization_engines) {
        const availableEngine = data.diarization_engines.find((e) => e.available);
        if (availableEngine && diarizationBadge) {
          diarizationBadge.className = "text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium";
          diarizationBadge.textContent = "ONNX Ready";
        } else if (diarizationBadge) {
          diarizationBadge.className = "text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium";
          diarizationBadge.textContent = "Models Missing";
          if (diarizationCheckbox) diarizationCheckbox.checked = false;
        }
      }

      await loadLlmModels();
    } catch (e) {
      console.error("Status check failed", e);
    }
  }

  async function loadLlmModels() {
    const provider = llmProviderSelect.value;
    const prefs = getStoredPreferences();
    const savedModel = (prefs.llm_models && prefs.llm_models[provider]) || (prefs.llm_provider === provider ? prefs.llm_model : null);

    try {
      const res = await fetch(`/api/llm/models?provider=${provider}`);
      if (res.ok) {
        const data = await res.json();
        llmModelSelect.innerHTML = "";
        if (data.models && data.models.length > 0) {
          let matched = false;
          data.models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            if (savedModel && m === savedModel) {
              opt.selected = true;
              matched = true;
            }
            llmModelSelect.appendChild(opt);
          });
          if (matched && savedModel) {
            llmModelSelect.value = savedModel;
          } else {
            const models = prefs.llm_models || {};
            models[provider] = llmModelSelect.value;
            savePreference("llm_models", models);
            savePreference("llm_model", llmModelSelect.value);
          }
        } else {
          const opt = document.createElement("option");
          opt.value = "llama3.2";
          opt.textContent = "llama3.2 (default)";
          llmModelSelect.appendChild(opt);
        }
      }
    } catch (e) {
      console.error("Failed to load LLM models", e);
    }
  }

  async function updateWhisperModels() {
    const engine = whisperEngineSelect.value;
    const prefs = getStoredPreferences();
    const savedModel = (prefs.whisper_models && prefs.whisper_models[engine]) || (prefs.whisper_engine === engine ? prefs.whisper_model : null);

    try {
      const res = await fetch(`/api/whisper/models?engine=${engine}`);
      if (res.ok) {
        const data = await res.json();
        if (data.models && data.models.length > 0) {
          whisperModelSelect.innerHTML = "";
          let matched = false;
          data.models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m;
            let label = m;
            if (m === "base") label = "Base (Recommended)";
            else if (m === "tiny") label = "Tiny (Fastest)";
            else if (m === "small") label = "Small (More Accurate)";
            else if (m === "medium") label = "Medium (High Precision)";
            else if (m === "whisper-large-v3-turbo") label = "Whisper Large v3 Turbo (Recommended)";
            else if (m === "whisper-large-v3") label = "Whisper Large v3";
            else if (m === "distil-whisper-large-v3-en") label = "Distil-Whisper Large v3 (English)";
            else if (m === "openai/whisper-large-v3") label = "OpenAI Whisper Large v3 (Default)";
            else if (m === "openai/whisper-large-v3-turbo") label = "OpenAI Whisper Large v3 Turbo";
            opt.textContent = label;
            if (savedModel && m === savedModel) {
              opt.selected = true;
              matched = true;
            }
            whisperModelSelect.appendChild(opt);
          });
          if (matched && savedModel) {
            whisperModelSelect.value = savedModel;
          } else {
            const models = prefs.whisper_models || {};
            models[engine] = whisperModelSelect.value;
            savePreference("whisper_models", models);
            savePreference("whisper_model", whisperModelSelect.value);
          }
        }
      }
    } catch (e) {
      console.warn("Failed to fetch whisper models:", e);
    }
  }

  whisperEngineSelect.addEventListener("change", () => {
    savePreference("whisper_engine", whisperEngineSelect.value);
    updateWhisperModels();
  });

  whisperModelSelect.addEventListener("change", () => {
    const engine = whisperEngineSelect.value;
    const model = whisperModelSelect.value;
    savePreference("whisper_model", model);
    const prefs = getStoredPreferences();
    const models = prefs.whisper_models || {};
    models[engine] = model;
    savePreference("whisper_models", models);
  });

  languageSelect.addEventListener("change", () => {
    savePreference("language", languageSelect.value);
  });

  llmProviderSelect.addEventListener("change", () => {
    savePreference("llm_provider", llmProviderSelect.value);
    loadLlmModels();
  });

  llmModelSelect.addEventListener("change", () => {
    const provider = llmProviderSelect.value;
    const model = llmModelSelect.value;
    savePreference("llm_model", model);
    const prefs = getStoredPreferences();
    const models = prefs.llm_models || {};
    models[provider] = model;
    savePreference("llm_models", models);
  });

  refreshModelsBtn.addEventListener("click", loadLlmModels);

  if (diarizationCheckbox) {
    diarizationCheckbox.addEventListener("change", () => {
      savePreference("diarization_enabled", diarizationCheckbox.checked);
    });
  }

  if (numSpeakersSelect) {
    numSpeakersSelect.addEventListener("change", () => {
      savePreference("num_speakers", numSpeakersSelect.value);
    });
  }

  if (vadCheckbox) {
    vadCheckbox.addEventListener("change", () => {
      savePreference("vad_enabled", vadCheckbox.checked);
    });
  }

  if (notifyCheckbox) {
    notifyCheckbox.addEventListener("change", () => {
      savePreference("notify_enabled", notifyCheckbox.checked);
    });
  }

  // Tabs: Upload vs Record
  tabUploadBtn.addEventListener("click", () => {
    tabUploadBtn.classList.add("border-indigo-500", "text-indigo-400");
    tabUploadBtn.classList.remove("border-transparent", "text-slate-400");
    tabRecordBtn.classList.remove("border-indigo-500", "text-indigo-400");
    tabRecordBtn.classList.add("border-transparent", "text-slate-400");
    dropzonePanel.classList.remove("hidden");
    recorderPanel.classList.add("hidden");
  });

  tabRecordBtn.addEventListener("click", () => {
    tabRecordBtn.classList.add("border-indigo-500", "text-indigo-400");
    tabRecordBtn.classList.remove("border-transparent", "text-slate-400");
    tabUploadBtn.classList.remove("border-indigo-500", "text-indigo-400");
    tabUploadBtn.classList.add("border-transparent", "text-slate-400");
    recorderPanel.classList.remove("hidden");
    dropzonePanel.classList.add("hidden");
  });

  // Dropzone handling
  dropzone.addEventListener("click", () => audioFileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-indigo-500", "bg-indigo-950/20");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-indigo-500", "bg-indigo-950/20");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-indigo-500", "bg-indigo-950/20");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  audioFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    recordedBlob = null;
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    selectedFileCard.classList.remove("hidden");
    dropzone.classList.add("hidden");

    const url = URL.createObjectURL(file);
    audioPreview.src = url;
    audioPreview.classList.remove("hidden");
    lucide.createIcons();
  }

  removeFileBtn.addEventListener("click", () => {
    selectedFile = null;
    audioFileInput.value = "";
    selectedFileCard.classList.add("hidden");
    dropzone.classList.remove("hidden");
    audioPreview.src = "";
    audioPreview.classList.add("hidden");
  });

  // Helper to downsample Float32 audio buffer to 16kHz 16-bit PCM little-endian
  function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
    if (inputSampleRate === outputSampleRate) {
      const pcm16 = new Int16Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) {
        const s = Math.max(-1, Math.min(1, buffer[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      return pcm16.buffer;
    }
    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Int16Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      const val = count > 0 ? accum / count : 0;
      const s = Math.max(-1, Math.min(1, val));
      result[offsetResult] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result.buffer;
  }

  // Trigger AI Polish or Summary (live streaming or on-demand)
  async function triggerStreamingAiProcessing(text, action, detailLevel) {
    if (!text || !text.trim()) return;
    const provider = llmProviderSelect.value;
    const model = llmModelSelect.value;
    const chosenLevel = detailLevel || activeSummaryLevel || "bullets";

    if (action === "polish") {
      switchResultTab("polish");
      if (polishToolbar) polishToolbar.classList.add("hidden");
      polishContent.innerHTML = `
        <div class="space-y-3">
          <div class="flex items-center gap-2 text-xs font-medium text-indigo-400 animate-pulse">
            <svg class="w-4 h-4 animate-spin text-indigo-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            <span>Polishing transcript with ${provider} (${model})...</span>
          </div>
          <div id="polish-streaming-target" class="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap"></div>
        </div>
      `;
    } else if (action === "summary") {
      switchResultTab("summary");
      if (summaryToolbar) summaryToolbar.classList.add("hidden");
      summaryContent.innerHTML = `
        <div class="space-y-3">
          <div class="flex items-center gap-2 text-xs font-medium text-indigo-400 animate-pulse">
            <svg class="w-4 h-4 animate-spin text-indigo-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            <span>Generating ${chosenLevel} summary with ${provider} (${model})...</span>
          </div>
          <div id="summary-streaming-target" class="prose prose-invert max-w-none text-sm leading-relaxed"></div>
        </div>
      `;
    }

    try {
      const resp = await fetch("/api/process-llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          action: action,
          detail_level: chosenLevel,
          provider: provider,
          model: model,
        }),
      });

      if (!resp.ok) {
        let errMsg = `LLM processing request failed (${resp.status})`;
        try {
          const errData = await resp.json();
          if (errData.detail) errMsg = errData.detail;
        } catch (_) {}
        throw new Error(errMsg);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6);
            if (dataStr === "[DONE]") break;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.error) {
                throw new Error(parsed.error);
              }
              if (parsed.token) {
                accumulated += parsed.token;
                if (action === "polish") {
                  const target = document.getElementById("polish-streaming-target");
                  if (target) target.textContent = accumulated;
                } else if (action === "summary") {
                  const target = document.getElementById("summary-streaming-target");
                  if (target) {
                    target.innerHTML = typeof marked !== "undefined" ? marked.parse(accumulated) : accumulated;
                  }
                }
              }
            } catch (err) {
              if (err.message && (err.message.includes("error") || err.message.includes("failed") || err.message.includes("503") || err.message.includes("404"))) {
                throw err;
              }
            }
          }
        }
      }

      if (action === "polish") {
        currentResult.polished = accumulated;
        polishContent.textContent = accumulated;
        if (polishToolbar) polishToolbar.classList.remove("hidden");
      } else if (action === "summary") {
        currentResult.summary = accumulated;
        summaryContent.innerHTML = typeof marked !== "undefined" ? marked.parse(accumulated) : accumulated;
        if (summaryToolbar) summaryToolbar.classList.remove("hidden");
      }

      updateResultTabStates();
      if (typeof lucide !== "undefined") lucide.createIcons();

    } catch (e) {
      console.warn("AI processing error:", e);
      if (action === "polish") {
        polishContent.innerHTML = renderAiUnavailableNotice("Polish", e.message);
        if (polishToolbar) polishToolbar.classList.add("hidden");
      } else if (action === "summary") {
        summaryContent.innerHTML = renderAiUnavailableNotice("Summary", e.message);
        if (summaryToolbar) summaryToolbar.classList.add("hidden");
      }
      updateResultTabStates();
    }
  }

  // Microphone recording & Live WebSocket Streaming
  recordToggleBtn.addEventListener("click", async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        liveMicStream = stream;
        mediaRecorder = new MediaRecorder(stream);
        recordChunks = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) recordChunks.push(e.data);
        };

        const isLive = liveStreamCheckbox && liveStreamCheckbox.checked;

        mediaRecorder.onstop = () => {
          recordedBlob = new Blob(recordChunks, { type: "audio/webm" });
          selectedFile = new File([recordedBlob], "mic_recording.webm", { type: "audio/webm" });
          const audioUrl = URL.createObjectURL(recordedBlob);
          recordPreview.src = audioUrl;
          recordPreview.classList.remove("hidden");
          audioPreview.src = audioUrl;
          recordStatus.textContent = isLive ? "Live dictation complete! Results loaded below." : "Recording saved! Ready to transcribe.";
        };

        mediaRecorder.start(100);

        if (isLive) {
          if (liveFinalText) liveFinalText.textContent = "";
          if (liveInterimText) liveInterimText.textContent = "";
          if (liveStreamBox) liveStreamBox.classList.remove("hidden");
          recordStatus.textContent = "Live streaming dictation active... Speak into your mic.";

          const proto = location.protocol === "https:" ? "wss:" : "ws:";
          const wsUrl = `${proto}//${location.host}/api/ws/transcribe`;
          liveWs = new WebSocket(wsUrl);
          liveWs.binaryType = "arraybuffer";

          liveWs.onopen = () => {
            liveWs.send(JSON.stringify({
              action: "start",
              whisper_engine: whisperEngineSelect.value,
              whisper_model: whisperModelSelect.value,
              language: languageSelect.value,
              vad_filter: vadCheckbox.checked,
            }));
          };

          liveWs.onmessage = (e) => {
            try {
              const data = JSON.parse(e.data);
              if (data.event === "partial") {
                if (liveInterimText) liveInterimText.textContent = " " + data.text;
                if (liveTranscriptFeed) liveTranscriptFeed.scrollTop = liveTranscriptFeed.scrollHeight;
              } else if (data.event === "segment") {
                if (liveFinalText) {
                  liveFinalText.textContent += (liveFinalText.textContent ? " " : "") + data.text;
                }
                if (liveInterimText) liveInterimText.textContent = "";
                if (liveTranscriptFeed) liveTranscriptFeed.scrollTop = liveTranscriptFeed.scrollHeight;
              } else if (data.event === "completed") {
                if (data.result) {
                  currentResult = Object.assign(currentResult, data.result);
                  resultsContainer.classList.remove("hidden");
                  renderSpeakerDialogue(currentResult);
                  const spkCountStr = currentResult.num_speakers ? ` • ${currentResult.num_speakers} Speakers` : "";
                  transcriptionStats.textContent = `Duration: ${currentResult.duration.toFixed(1)}s • Live Streaming • Language: ${(currentResult.language || "auto").toUpperCase()}${spkCountStr}`;
                  transcriptPlainText.textContent = currentResult.text;

                  // Trigger AI post-processing if requested
                  if (activeAiAction === "polish" || activeAiAction === "summary") {
                    triggerStreamingAiProcessing(currentResult.text, activeAiAction, activeSummaryLevel);
                  }
                }
              }
            } catch (err) {
              console.warn("WebSocket parse error:", err);
            }
          };

          liveWs.onerror = (err) => {
            console.error("Live WebSocket error:", err);
            recordStatus.textContent = "Live streaming connection error.";
          };

          // Setup AudioContext for 16kHz PCM streaming
          liveAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const source = liveAudioCtx.createMediaStreamSource(stream);
          liveScriptNode = liveAudioCtx.createScriptProcessor(4096, 1, 1);
          liveScriptNode.onaudioprocess = (e) => {
            if (liveWs && liveWs.readyState === WebSocket.OPEN) {
              const channelData = e.inputBuffer.getChannelData(0);
              const pcmData = downsampleBuffer(channelData, liveAudioCtx.sampleRate, 16000);
              liveWs.send(pcmData);
            }
          };
          source.connect(liveScriptNode);
          liveScriptNode.connect(liveAudioCtx.destination);
        } else {
          if (liveStreamBox) liveStreamBox.classList.add("hidden");
          recordStatus.textContent = "Recording in progress... Click to stop.";
        }

        isRecording = true;
        recordToggleBtn.classList.add("recording-active");
        recordSeconds = 0;
        recordTimer.textContent = "00:00";
        recordTimerInterval = setInterval(() => {
          recordSeconds++;
          const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
          const secs = String(recordSeconds % 60).padStart(2, "0");
          recordTimer.textContent = `${mins}:${secs}`;
        }, 1000);

      } catch (err) {
        alert("Microphone access denied or not available: " + err.message);
      }
    } else {
      isRecording = false;
      recordToggleBtn.classList.remove("recording-active");
      clearInterval(recordTimerInterval);

      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }

      if (liveScriptNode) {
        try { liveScriptNode.disconnect(); } catch (e) {}
        liveScriptNode = null;
      }
      if (liveAudioCtx) {
        try { liveAudioCtx.close(); } catch (e) {}
        liveAudioCtx = null;
      }
      if (liveMicStream) {
        liveMicStream.getTracks().forEach((track) => track.stop());
        liveMicStream = null;
      }

      if (liveWs) {
        if (liveWs.readyState === WebSocket.OPEN) {
          liveWs.send(JSON.stringify({ action: "stop" }));
        }
      }
    }
  });

  // Action Buttons
  const aiActionBtns = document.querySelectorAll(".ai-action-btn");
  aiActionBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      aiActionBtns.forEach((b) => {
        b.classList.remove("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
        b.classList.add("border-slate-700", "bg-slate-950", "text-slate-300");
      });
      btn.classList.add("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
      btn.classList.remove("border-slate-700", "bg-slate-950", "text-slate-300");

      activeAiAction = btn.getAttribute("data-action");
      savePreference("ai_action", activeAiAction);
      if (activeAiAction === "summary") {
        summaryOptionsBox.classList.remove("hidden");
      } else {
        summaryOptionsBox.classList.add("hidden");
      }
    });
  });

  // Summary Level Buttons
  const summaryLevelBtns = document.querySelectorAll(".summary-level-btn");
  summaryLevelBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      summaryLevelBtns.forEach((b) => {
        b.classList.remove("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
        b.classList.add("border-slate-700", "bg-slate-950", "text-slate-300");
      });
      btn.classList.add("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400");
      btn.classList.remove("border-slate-700", "bg-slate-950", "text-slate-300");
      activeSummaryLevel = btn.getAttribute("data-level");
      savePreference("summary_level", activeSummaryLevel);
    });
  });

  function updateProgressBar(percent, label, stepInfo) {
    progressContainer.classList.remove("hidden");
    currentStepLabel.textContent = label;
    stepCounter.textContent = stepInfo || `${percent}%`;
    progressBar.style.width = `${percent}%`;
  }

  // Primary Execution via Async Background Job Queue & SSE Stream
  startProcessBtn.addEventListener("click", async () => {
    const fileToUpload = selectedFile || (recordedBlob ? new File([recordedBlob], "mic.webm", { type: "audio/webm" }) : null);
    if (!fileToUpload) {
      alert("Please upload an audio/video file or record with your microphone first.");
      return;
    }

    startProcessBtn.disabled = true;
    startProcessBtn.classList.add("opacity-50", "cursor-not-allowed");
    resultsContainer.classList.remove("hidden");

    // Reset results views
    transcriptSegmentsList.innerHTML = "";
    transcriptPlainText.textContent = "";
    polishContent.textContent = "";
    summaryContent.innerHTML = "";
    if (polishToolbar) polishToolbar.classList.add("hidden");
    if (summaryToolbar) summaryToolbar.classList.add("hidden");
    if (tabBadgePolish) tabBadgePolish.classList.add("hidden");
    if (tabBadgeSummary) tabBadgeSummary.classList.add("hidden");
    currentResult.segments = [];
    currentResult.text = "";
    currentResult.polished = "";
    currentResult.summary = "";

    switchResultTab("transcript");
    updateProgressBar(10, "Uploading media and queuing job...", "Queued");

    try {
      const formData = new FormData();
      formData.append("file", fileToUpload);
      formData.append("whisper_engine", whisperEngineSelect.value);
      formData.append("whisper_model", whisperModelSelect.value);
      formData.append("language", languageSelect.value);
      formData.append("vad_filter", vadCheckbox ? vadCheckbox.checked : true);
      formData.append("enable_diarization", diarizationCheckbox ? diarizationCheckbox.checked : false);
      formData.append("num_speakers", numSpeakersSelect ? numSpeakersSelect.value : -1);
      formData.append("ai_action", activeAiAction);
      formData.append("summary_level", activeSummaryLevel);
      formData.append("llm_provider", llmProviderSelect.value);
      formData.append("llm_model", llmModelSelect.value);
      formData.append("notify", notifyCheckbox.checked);

      const jobRes = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });

      if (!jobRes.ok) {
        const err = await jobRes.json();
        throw new Error(err.detail || "Failed to submit job.");
      }

      const { job_id } = await jobRes.json();
      currentResult.task_id = job_id;
      currentResult.filename = fileToUpload.name;

      // Connect to Server-Sent Events stream for real-time progress
      const eventSource = new EventSource(`/api/jobs/${job_id}/stream`);

      eventSource.addEventListener("status", (e) => {
        const data = JSON.parse(e.data);
        updateProgressBar(data.progress, data.message, `${data.progress}%`);
      });

      eventSource.addEventListener("diarization", (e) => {
        const data = JSON.parse(e.data);
        updateProgressBar(65, `Diarization complete (${data.num_speakers} speakers detected)`, "65%");
      });

      eventSource.addEventListener("speaker_renamed", (e) => {
        const data = JSON.parse(e.data);
        if (data.speaker_turns) {
          currentResult.speaker_turns = data.speaker_turns;
          renderSpeakerDialogue(currentResult);
        }
      });

      eventSource.addEventListener("segment", (e) => {
        const seg = JSON.parse(e.data);
        currentResult.segments.push(seg);
        appendLiveSegment(seg);
      });

      eventSource.addEventListener("ai_token", (e) => {
        const data = JSON.parse(e.data);
        if (data.action === "polish") {
          currentResult.polished += data.token;
          polishContent.textContent = currentResult.polished;
          switchResultTab("polish");
        } else if (data.action === "summary") {
          currentResult.summary += data.token;
          summaryContent.innerHTML = marked.parse(currentResult.summary);
          switchResultTab("summary");
        }
      });

      eventSource.addEventListener("completed", (e) => {
        const finalResult = JSON.parse(e.data);
        eventSource.close();
        progressContainer.classList.add("hidden");
        startProcessBtn.disabled = false;
        startProcessBtn.classList.remove("opacity-50", "cursor-not-allowed");

        currentResult = Object.assign(currentResult, finalResult);

        if (finalResult.segments && finalResult.segments.length > 0) {
          renderSpeakerDialogue(currentResult);
        }

        if (finalResult.summary && !finalResult.summary.startsWith("[AI summary skipped")) {
          currentResult.summary = finalResult.summary;
          summaryContent.innerHTML = typeof marked !== "undefined" ? marked.parse(finalResult.summary) : finalResult.summary;
          if (summaryToolbar) summaryToolbar.classList.remove("hidden");
        } else if (activeAiAction === "summary") {
          summaryContent.innerHTML = renderAiUnavailableNotice("Summary", finalResult.ai_warning || (finalResult.summary ? finalResult.summary.replace(/^\[|\]$/g, "") : null));
          if (summaryToolbar) summaryToolbar.classList.add("hidden");
        }

        if (finalResult.polished && !finalResult.polished.startsWith("[AI polish skipped")) {
          currentResult.polished = finalResult.polished;
          polishContent.textContent = finalResult.polished;
          if (polishToolbar) polishToolbar.classList.remove("hidden");
        } else if (activeAiAction === "polish") {
          polishContent.innerHTML = renderAiUnavailableNotice("Polish", finalResult.ai_warning || (finalResult.polished ? finalResult.polished.replace(/^\[|\]$/g, "") : null));
          if (polishToolbar) polishToolbar.classList.add("hidden");
        }

        const speakerCountStr = finalResult.num_speakers ? ` • ${finalResult.num_speakers} Speaker${finalResult.num_speakers > 1 ? "s" : ""} Identified` : "";
        transcriptionStats.textContent = `Duration: ${finalResult.duration.toFixed(1)}s • Processed in: ${finalResult.processing_time}s • Language: ${(finalResult.language || "auto").toUpperCase()}${speakerCountStr}`;
        transcriptPlainText.textContent = finalResult.text;

        updateResultTabStates();
        if (activeAiAction === "summary" && currentResult.summary) {
          switchResultTab("summary");
        } else if (activeAiAction === "polish" && currentResult.polished) {
          switchResultTab("polish");
        } else {
          switchResultTab("transcript");
        }
      });

      eventSource.addEventListener("failed", (e) => {
        const data = JSON.parse(e.data);
        eventSource.close();
        progressContainer.classList.add("hidden");
        startProcessBtn.disabled = false;
        startProcessBtn.classList.remove("opacity-50", "cursor-not-allowed");
        alert("Processing failed: " + data.error);
      });

      eventSource.onerror = (err) => {
        eventSource.close();
        progressContainer.classList.add("hidden");
        startProcessBtn.disabled = false;
        startProcessBtn.classList.remove("opacity-50", "cursor-not-allowed");
      };

    } catch (err) {
      alert("Error: " + err.message);
      progressContainer.classList.add("hidden");
      startProcessBtn.disabled = false;
      startProcessBtn.classList.remove("opacity-50", "cursor-not-allowed");
    }
  });

  function appendLiveSegment(seg) {
    const div = document.createElement("div");
    div.className = "segment-item flex items-start gap-3 p-2.5 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs hover:border-slate-700/80 transition-colors";
    
    const timeBtn = document.createElement("button");
    timeBtn.className = "px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-mono text-[11px] flex-shrink-0 hover:bg-indigo-500/20";
    timeBtn.textContent = formatTimestamp(seg.start);
    timeBtn.title = "Click to seek audio";
    timeBtn.addEventListener("click", () => {
      const activeAudio = audioPreview.src ? audioPreview : recordPreview;
      if (activeAudio) {
        activeAudio.currentTime = seg.start;
        activeAudio.play();
      }
    });

    const textSpan = document.createElement("span");
    textSpan.className = "text-slate-200 leading-relaxed flex-1 flex flex-wrap gap-x-1 gap-y-0.5";

    if (seg.words && seg.words.length > 0) {
      seg.words.forEach((w) => {
        const wSpan = document.createElement("span");
        wSpan.className = "word-token px-1 py-0.5 rounded cursor-pointer transition-colors duration-150 hover:bg-indigo-500/30 hover:text-indigo-200";
        wSpan.textContent = w.word;
        wSpan.dataset.start = w.start;
        wSpan.dataset.end = w.end;
        const confStr = w.probability !== null && w.probability !== undefined ? ` (${Math.round(w.probability * 100)}%)` : "";
        wSpan.title = `${w.start.toFixed(2)}s - ${w.end.toFixed(2)}s${confStr}`;
        wSpan.addEventListener("click", (e) => {
          e.stopPropagation();
          const activeAudio = audioPreview.src ? audioPreview : recordPreview;
          if (activeAudio) {
            activeAudio.currentTime = w.start;
            activeAudio.play();
          }
        });
        textSpan.appendChild(wSpan);
      });
    } else {
      textSpan.textContent = seg.text;
    }

    div.appendChild(timeBtn);
    div.appendChild(textSpan);
    transcriptSegmentsList.appendChild(div);
    transcriptSegmentsList.scrollTop = transcriptSegmentsList.scrollHeight;
  }

  function formatTimestamp(secs) {
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${String(mins).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function renderSpeakerDialogue(result) {
    if (!result || !result.segments) return;
    transcriptSegmentsList.innerHTML = "";

    const hasSpeakers = result.segments.some((s) => s.speaker);

    if (hasSpeakers) {
      // Group consecutive segments from the same speaker into dialogue turns
      const turns = [];
      result.segments.forEach((seg) => {
        const spk = seg.speaker || "Speaker 0";
        if (turns.length > 0 && turns[turns.length - 1].speaker === spk) {
          turns[turns.length - 1].end = seg.end;
          turns[turns.length - 1].segments.push(seg);
        } else {
          turns.push({
            speaker: spk,
            start: seg.start,
            end: seg.end,
            segments: [seg],
          });
        }
      });

      turns.forEach((turn) => {
        const card = document.createElement("div");
        card.className = "speaker-turn-card p-3.5 rounded-xl bg-slate-950/50 border border-slate-800/80 space-y-2 mb-3.5 hover:border-slate-700/80 transition-colors";

        const header = document.createElement("div");
        header.className = "flex items-center justify-between";

        const style = getSpeakerStyle(turn.speaker);
        const speakerBtn = document.createElement("button");
        speakerBtn.className = `speaker-badge-btn px-2.5 py-1 rounded-md text-xs font-semibold flex items-center gap-1.5 border ${style.badge}`;
        speakerBtn.innerHTML = `
          <i data-lucide="user" class="w-3.5 h-3.5 inline"></i>
          <span>${turn.speaker}</span>
          <i data-lucide="pencil" class="w-2.5 h-2.5 inline opacity-60 ml-0.5"></i>
        `;
        speakerBtn.title = "Click to rename this speaker across all turns";
        speakerBtn.addEventListener("click", () => handleSpeakerRename(turn.speaker));

        const timeBtn = document.createElement("button");
        timeBtn.className = "px-2 py-0.5 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 font-mono text-[11px] transition-colors";
        timeBtn.textContent = `${formatTimestamp(turn.start)} - ${formatTimestamp(turn.end)}`;
        timeBtn.title = "Seek audio to turn start";
        timeBtn.addEventListener("click", () => {
          const activeAudio = audioPreview.src ? audioPreview : recordPreview;
          if (activeAudio) {
            activeAudio.currentTime = turn.start;
            activeAudio.play();
          }
        });

        header.appendChild(speakerBtn);
        header.appendChild(timeBtn);
        card.appendChild(header);

        // Turn words & segments
        const textContainer = document.createElement("div");
        textContainer.className = "text-sm text-slate-200 leading-relaxed flex flex-wrap gap-x-1 gap-y-0.5 pt-1";

        turn.segments.forEach((seg) => {
          if (seg.words && seg.words.length > 0) {
            seg.words.forEach((w) => {
              const wSpan = document.createElement("span");
              wSpan.className = "word-token px-1 py-0.5 rounded cursor-pointer transition-colors duration-150 hover:bg-indigo-500/30 hover:text-indigo-200";
              wSpan.textContent = w.word;
              wSpan.dataset.start = w.start;
              wSpan.dataset.end = w.end;
              const confStr = w.probability !== null && w.probability !== undefined ? ` (${Math.round(w.probability * 100)}%)` : "";
              wSpan.title = `${w.start.toFixed(2)}s - ${w.end.toFixed(2)}s${confStr}`;
              wSpan.addEventListener("click", (e) => {
                e.stopPropagation();
                const activeAudio = audioPreview.src ? audioPreview : recordPreview;
                if (activeAudio) {
                  activeAudio.currentTime = w.start;
                  activeAudio.play();
                }
              });
              textContainer.appendChild(wSpan);
            });
          } else {
            const segSpan = document.createElement("span");
            segSpan.textContent = seg.text + " ";
            textContainer.appendChild(segSpan);
          }
        });

        card.appendChild(textContainer);
        transcriptSegmentsList.appendChild(card);
      });
    } else {
      // Standard segment list without speakers
      result.segments.forEach((seg) => appendLiveSegment(seg));
    }

    if (window.lucide) {
      lucide.createIcons();
    }
  }

  async function handleSpeakerRename(oldSpeaker) {
    const newName = prompt(`Enter new name for "${oldSpeaker}":`, oldSpeaker);
    if (!newName || !newName.trim() || newName.trim() === oldSpeaker) return;
    const cleanName = newName.trim();

    if (currentResult.task_id) {
      try {
        const resp = await fetch(`/api/jobs/${currentResult.task_id}/rename-speaker`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ old_name: oldSpeaker, new_name: cleanName }),
        });
        if (resp.ok) {
          const data = await resp.json();
          currentResult = Object.assign(currentResult, data.result);
          renderSpeakerDialogue(currentResult);
          transcriptPlainText.textContent = currentResult.text;
          return;
        }
      } catch (e) {
        console.warn("Backend rename failed, using local update:", e);
      }
    }

    // Local client-side rename fallback
    if (currentResult.segments) {
      currentResult.segments.forEach((s) => {
        if (s.speaker === oldSpeaker) s.speaker = cleanName;
        if (s.words) s.words.forEach((w) => { if (w.speaker === oldSpeaker) w.speaker = cleanName; });
      });
    }
    if (currentResult.srt) currentResult.srt = currentResult.srt.replaceAll(`${oldSpeaker}:`, `${cleanName}:`);
    if (currentResult.vtt) currentResult.vtt = currentResult.vtt.replaceAll(`<v ${oldSpeaker}>`, `<v ${cleanName}>`);
    if (currentResult.text) currentResult.text = currentResult.text.replaceAll(`[${oldSpeaker}]:`, `[${cleanName}]:`);
    if (currentResult.ass) currentResult.ass = currentResult.ass.replaceAll(`,${oldSpeaker},`, `,${cleanName},`);
    renderSpeakerDialogue(currentResult);
    transcriptPlainText.textContent = currentResult.text;
  }

  toggleTimestamps.addEventListener("change", () => {
    if (toggleTimestamps.checked) {
      transcriptSegmentsList.classList.remove("hidden");
      transcriptPlainText.classList.add("hidden");
    } else {
      transcriptSegmentsList.classList.add("hidden");
      transcriptPlainText.classList.remove("hidden");
    }
  });

  function renderAiUnavailableNotice(actionName, warning) {
    const detail = warning || "Ollama or OpenAI-compatible AI processor was not running or configured.";
    return `
      <div class="p-5 rounded-xl border border-amber-500/30 bg-amber-500/10 text-slate-200 space-y-3">
        <div class="flex items-center gap-2 font-medium text-sm text-amber-400">
          <svg class="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
          </svg>
          <span>AI ${actionName} Error / Not Available</span>
        </div>
        <p class="text-xs text-slate-300 leading-relaxed">${detail}</p>
        <div class="pt-2 border-t border-amber-500/20 text-xs text-slate-400 space-y-1">
          <p class="font-medium text-slate-300">To resolve:</p>
          <ul class="list-disc list-inside space-y-0.5 text-slate-300">
            <li>Verify your API Key / host in <span class="text-indigo-400 font-medium cursor-pointer hover:underline" onclick="document.getElementById('open-settings-btn').click()">Settings</span></li>
            <li>If using local Ollama, ensure it is running with <code class="px-1.5 py-0.5 bg-slate-900 rounded font-mono text-amber-300 text-[11px]">ollama serve</code></li>
          </ul>
        </div>
      </div>
    `;
  }

  function renderOnDemandPolishCard() {
    return `
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-1">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center flex-shrink-0">
            <i data-lucide="sparkles" class="w-4 h-4"></i>
          </div>
          <div>
            <h4 class="text-xs font-semibold text-slate-200">Transcript Polishing Not Run</h4>
            <p class="text-[11px] text-slate-400 mt-0.5">Clean up grammar, fix punctuation, and remove filler words using your selected AI model.</p>
          </div>
        </div>
        <button id="ondemand-polish-btn" class="flex-shrink-0 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-sm flex items-center justify-center gap-1.5 transition-all">
          <i data-lucide="sparkles" class="w-3.5 h-3.5"></i> Polish Now
        </button>
      </div>
    `;
  }

  function renderOnDemandSummaryCard() {
    return `
      <div class="space-y-3 p-1">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center flex-shrink-0">
              <i data-lucide="list-collapse" class="w-4 h-4"></i>
            </div>
            <div>
              <h4 class="text-xs font-semibold text-slate-200">AI Summary Not Run</h4>
              <p class="text-[11px] text-slate-400 mt-0.5">Choose a detail format and generate an AI summary from this transcript.</p>
            </div>
          </div>
          <button id="ondemand-summary-btn" class="flex-shrink-0 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-sm flex items-center justify-center gap-1.5 transition-all">
            <i data-lucide="list-collapse" class="w-3.5 h-3.5"></i> Generate Summary
          </button>
        </div>
        <div class="flex items-center gap-1.5 pt-2 border-t border-slate-800/80 text-xs">
          <span class="text-[11px] text-slate-500 mr-1">Format:</span>
          <button type="button" data-ondemand-level="tldr" class="ondemand-level-chip py-1 px-2.5 rounded-md border border-slate-800 bg-slate-950 text-slate-400 text-[11px] hover:border-slate-700 transition-all">TL;DR</button>
          <button type="button" data-ondemand-level="bullets" class="ondemand-level-chip active py-1 px-2.5 rounded-md border border-indigo-500/60 bg-indigo-500/15 text-indigo-300 text-[11px] font-medium transition-all">Key Points</button>
          <button type="button" data-ondemand-level="detailed" class="ondemand-level-chip py-1 px-2.5 rounded-md border border-slate-800 bg-slate-950 text-slate-400 text-[11px] hover:border-slate-700 transition-all">Detailed</button>
          <button type="button" data-ondemand-level="action_items" class="ondemand-level-chip py-1 px-2.5 rounded-md border border-slate-800 bg-slate-950 text-slate-400 text-[11px] hover:border-slate-700 transition-all">Action Items</button>
        </div>
      </div>
    `;
  }

  function attachOnDemandPolishListeners() {
    const btn = document.getElementById("ondemand-polish-btn");
    if (btn) {
      btn.addEventListener("click", () => {
        if (!currentResult.text || !currentResult.text.trim()) {
          alert("No transcript content available to polish.");
          return;
        }
        triggerStreamingAiProcessing(currentResult.text, "polish");
      });
    }
    if (typeof lucide !== "undefined") lucide.createIcons();
  }

  function attachOnDemandSummaryListeners() {
    const chips = summaryContent.querySelectorAll(".ondemand-level-chip");
    chips.forEach((c) => {
      c.addEventListener("click", () => {
        chips.forEach((x) => {
          x.classList.remove("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400", "font-medium");
          x.classList.add("border-slate-700", "bg-slate-900", "text-slate-300");
        });
        c.classList.add("active", "border-indigo-500", "bg-indigo-500/10", "text-indigo-400", "font-medium");
        c.classList.remove("border-slate-700", "bg-slate-900", "text-slate-300");
        ondemandSummaryActiveLevel = c.getAttribute("data-ondemand-level");
      });
    });

    const btn = document.getElementById("ondemand-summary-btn");
    if (btn) {
      btn.addEventListener("click", () => {
        if (!currentResult.text || !currentResult.text.trim()) {
          alert("No transcript content available to summarize.");
          return;
        }
        triggerStreamingAiProcessing(currentResult.text, "summary", ondemandSummaryActiveLevel);
      });
    }
    if (typeof lucide !== "undefined") lucide.createIcons();
  }

  function updateResultTabStates() {
    if (tabBadgePolish) {
      if (currentResult.polished && currentResult.polished.trim()) {
        tabBadgePolish.textContent = "Ready";
        tabBadgePolish.className = "text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
        tabBadgePolish.classList.remove("hidden");
        if (polishToolbar) polishToolbar.classList.remove("hidden");
      } else {
        tabBadgePolish.textContent = "+ Polish";
        tabBadgePolish.className = "text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-slate-800 text-slate-400 border border-slate-700/60";
        tabBadgePolish.classList.remove("hidden");
        if (polishToolbar) polishToolbar.classList.add("hidden");
      }
    }

    if (tabBadgeSummary) {
      if (currentResult.summary && currentResult.summary.trim()) {
        tabBadgeSummary.textContent = "Ready";
        tabBadgeSummary.className = "text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
        tabBadgeSummary.classList.remove("hidden");
        if (summaryToolbar) summaryToolbar.classList.remove("hidden");
      } else {
        tabBadgeSummary.textContent = "+ Summary";
        tabBadgeSummary.className = "text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-slate-800 text-slate-400 border border-slate-700/60";
        tabBadgeSummary.classList.remove("hidden");
        if (summaryToolbar) summaryToolbar.classList.add("hidden");
      }
    }
  }

  function switchResultTab(tabName) {
    [resTabTranscript, resTabPolish, resTabSummary].forEach((b) => {
      b.classList.remove("active", "text-indigo-400", "bg-indigo-500/10");
      b.classList.add("text-slate-400");
    });
    [viewportTranscript, viewportPolish, viewportSummary].forEach((v) => v.classList.add("hidden"));

    if (tabName === "transcript") {
      resTabTranscript.classList.add("active", "text-indigo-400", "bg-indigo-500/10");
      resTabTranscript.classList.remove("text-slate-400");
      viewportTranscript.classList.remove("hidden");
    } else if (tabName === "polish") {
      resTabPolish.classList.add("active", "text-indigo-400", "bg-indigo-500/10");
      resTabPolish.classList.remove("text-slate-400");
      viewportPolish.classList.remove("hidden");
      if (currentResult.polished && currentResult.polished.trim()) {
        polishContent.textContent = currentResult.polished;
        if (polishToolbar) polishToolbar.classList.remove("hidden");
      } else if (!polishContent.textContent.trim()) {
        polishContent.innerHTML = renderOnDemandPolishCard();
        attachOnDemandPolishListeners();
      }
    } else if (tabName === "summary") {
      resTabSummary.classList.add("active", "text-indigo-400", "bg-indigo-500/10");
      resTabSummary.classList.remove("text-slate-400");
      viewportSummary.classList.remove("hidden");
      if (currentResult.summary && currentResult.summary.trim()) {
        summaryContent.innerHTML = typeof marked !== "undefined" ? marked.parse(currentResult.summary) : currentResult.summary;
        if (summaryToolbar) summaryToolbar.classList.remove("hidden");
      } else if (!summaryContent.textContent.trim()) {
        summaryContent.innerHTML = renderOnDemandSummaryCard();
        attachOnDemandSummaryListeners();
      }
    }
    if (typeof lucide !== "undefined") lucide.createIcons();
  }

  resTabTranscript.addEventListener("click", () => switchResultTab("transcript"));
  resTabPolish.addEventListener("click", () => switchResultTab("polish"));
  resTabSummary.addEventListener("click", () => switchResultTab("summary"));

  // Re-run buttons in toolbars
  if (repolishBtn) {
    repolishBtn.addEventListener("click", () => {
      if (!currentResult.text || !currentResult.text.trim()) return;
      triggerStreamingAiProcessing(currentResult.text, "polish");
    });
  }

  if (resummarizeBtn) {
    resummarizeBtn.addEventListener("click", () => {
      if (!currentResult.text || !currentResult.text.trim()) return;
      triggerStreamingAiProcessing(currentResult.text, "summary", resummarizeActiveLevel);
    });
  }

  if (resummarizeLevelBtns) {
    resummarizeLevelBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        resummarizeLevelBtns.forEach((b) => {
          b.classList.remove("active", "text-indigo-400", "bg-indigo-500/10", "font-medium");
          b.classList.add("text-slate-400");
        });
        btn.classList.add("active", "text-indigo-400", "bg-indigo-500/10", "font-medium");
        btn.classList.remove("text-slate-400");
        resummarizeActiveLevel = btn.getAttribute("data-resummarize-level");
        if (currentResult.text && currentResult.text.trim()) {
          triggerStreamingAiProcessing(currentResult.text, "summary", resummarizeActiveLevel);
        }
      });
    });
  }

  // Copy & Download
  copyBtn.addEventListener("click", () => {
    let contentToCopy = currentResult.text;
    if (!viewportPolish.classList.contains("hidden") && currentResult.polished) {
      contentToCopy = currentResult.polished;
    } else if (!viewportSummary.classList.contains("hidden") && currentResult.summary) {
      contentToCopy = currentResult.summary;
    }
    navigator.clipboard.writeText(contentToCopy);
    copyBtn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i> Copied!`;
    setTimeout(() => {
      copyBtn.innerHTML = `<i data-lucide="copy" class="w-3.5 h-3.5"></i> Copy`;
      lucide.createIcons();
    }, 2000);
  });

  function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  downloadTxtBtn.addEventListener("click", () => {
    let content = currentResult.text;
    let filename = "transcript.txt";
    if (!viewportPolish.classList.contains("hidden") && currentResult.polished) {
      content = currentResult.polished;
      filename = "transcript_polished.txt";
    } else if (!viewportSummary.classList.contains("hidden") && currentResult.summary) {
      content = currentResult.summary;
      filename = "transcript_summary.txt";
    }
    downloadFile(filename, content, "text/plain");
  });
  downloadSrtBtn.addEventListener("click", () => downloadFile("subtitles.srt", currentResult.srt, "text/plain"));
  downloadVttBtn.addEventListener("click", () => downloadFile("subtitles.vtt", currentResult.vtt, "text/vtt"));
  if (downloadAssBtn) {
    downloadAssBtn.addEventListener("click", () => downloadFile("subtitles.ass", currentResult.ass || "", "text/plain"));
  }
  if (downloadJsonBtn) {
    downloadJsonBtn.addEventListener("click", () => {
      downloadFile("transcription.json", JSON.stringify(currentResult, null, 2), "application/json");
    });
  }

  // Real-time audio playback word synchronization (Karaoke highlight)
  function setupAudioWordSync(audioEl) {
    if (!audioEl) return;
    audioEl.addEventListener("timeupdate", () => {
      const cur = audioEl.currentTime;
      const allWordTokens = transcriptSegmentsList.querySelectorAll(".word-token");
      allWordTokens.forEach((token) => {
        const st = parseFloat(token.dataset.start);
        const en = parseFloat(token.dataset.end);
        if (cur >= st && cur <= en) {
          token.classList.add("bg-indigo-600", "text-white", "font-semibold", "shadow-sm");
        } else {
          token.classList.remove("bg-indigo-600", "text-white", "font-semibold", "shadow-sm");
        }
      });
    });
  }
  setupAudioWordSync(audioPreview);
  setupAudioWordSync(recordPreview);

  // Settings Modal Handlers
  async function loadSettings() {
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        const data = await res.json();
        if (groqApiKeyInput && data.groq?.api_key) groqApiKeyInput.value = data.groq.api_key;
        if (openrouterApiKeyInput && data.openrouter?.api_key) openrouterApiKeyInput.value = data.openrouter.api_key;
        if (openaiBaseUrlInput && data.openai_compatible?.base_url) openaiBaseUrlInput.value = data.openai_compatible.base_url;
        if (openaiApiKeyInput && data.openai_compatible?.api_key) openaiApiKeyInput.value = data.openai_compatible.api_key;
        if (telegramTokenInput && data.notifications?.telegram?.bot_token) telegramTokenInput.value = data.notifications.telegram.bot_token;
        if (telegramChatIdInput && data.notifications?.telegram?.chat_id) telegramChatIdInput.value = data.notifications.telegram.chat_id;
      }
    } catch (e) {
      console.warn("Failed to load settings:", e);
    }
  }

  openSettingsBtn.addEventListener("click", () => {
    loadSettings();
    settingsModal.classList.remove("hidden");
  });
  closeSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));

  async function testProviderConnection(provider, apiKey, baseUrl, resultEl) {
    if (!resultEl) return;
    resultEl.innerHTML = `<span class="text-indigo-400">Testing connection...</span>`;
    try {
      const res = await fetch("/api/settings/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: provider,
          api_key: apiKey,
          base_url: baseUrl,
        }),
      });
      const data = await res.json();
      if (data.online) {
        const lat = data.latency_ms ? ` (${data.latency_ms} ms)` : "";
        resultEl.innerHTML = `<span class="text-emerald-400">✅ Connected successfully!${lat}</span>`;
      } else {
        resultEl.innerHTML = `<span class="text-red-400">❌ Connection failed: ${data.error || "Unknown error"}</span>`;
      }
    } catch (e) {
      resultEl.innerHTML = `<span class="text-red-400">❌ Request failed: ${e.message}</span>`;
    }
  }

  if (testGroqBtn) {
    testGroqBtn.addEventListener("click", () => {
      testProviderConnection("groq", groqApiKeyInput.value.trim(), null, groqTestResult);
    });
  }

  if (testOpenrouterBtn) {
    testOpenrouterBtn.addEventListener("click", () => {
      testProviderConnection("openrouter", openrouterApiKeyInput.value.trim(), null, openrouterTestResult);
    });
  }

  if (testOpenaiBtn) {
    testOpenaiBtn.addEventListener("click", () => {
      testProviderConnection("openai_compat", openaiApiKeyInput.value.trim(), openaiBaseUrlInput.value.trim(), openaiTestResult);
    });
  }

  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener("click", async () => {
      saveSettingsBtn.disabled = true;
      if (settingsSaveFeedback) {
        settingsSaveFeedback.innerHTML = `<span class="text-indigo-400">Saving configuration...</span>`;
      }
      const payload = {
        groq: { api_key: groqApiKeyInput.value.trim() },
        openrouter: { api_key: openrouterApiKeyInput.value.trim() },
        openai_compatible: {
          base_url: openaiBaseUrlInput.value.trim(),
          api_key: openaiApiKeyInput.value.trim(),
        },
        notifications: {
          telegram: {
            bot_token: telegramTokenInput.value.trim(),
            chat_id: telegramChatIdInput.value.trim(),
          },
        },
      };
      try {
        const res = await fetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          if (settingsSaveFeedback) {
            settingsSaveFeedback.innerHTML = `<span class="text-emerald-400 font-medium">✅ Settings saved!</span>`;
          }
          await checkSystemStatus();
          await loadLlmModels();
          await updateWhisperModels();
          setTimeout(() => {
            if (settingsSaveFeedback) settingsSaveFeedback.textContent = "";
          }, 3500);
        } else {
          const err = await res.json();
          if (settingsSaveFeedback) {
            settingsSaveFeedback.innerHTML = `<span class="text-red-400">❌ Failed: ${err.detail || "Error"}</span>`;
          }
        }
      } catch (e) {
        if (settingsSaveFeedback) {
          settingsSaveFeedback.innerHTML = `<span class="text-red-400">❌ Error: ${e.message}</span>`;
        }
      } finally {
        saveSettingsBtn.disabled = false;
      }
    });
  }

  testTelegramBtn.addEventListener("click", async () => {
    telegramTestResult.textContent = "Testing connection...";
    try {
      const res = await fetch("/api/notifications/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "telegram",
          telegram_bot_token: telegramTokenInput.value.trim(),
          telegram_chat_id: telegramChatIdInput.value.trim(),
        }),
      });
      const data = await res.json();
      if (data.success) {
        telegramTestResult.innerHTML = `<span class="text-emerald-400">✅ ${data.message}</span>`;
      } else {
        telegramTestResult.innerHTML = `<span class="text-red-400">❌ ${data.message} ${data.error_details || ""}</span>`;
      }
    } catch (e) {
      telegramTestResult.innerHTML = `<span class="text-red-400">❌ Error: ${e.message}</span>`;
    }
  });

  pullModelBtn.addEventListener("click", async () => {
    const modelName = pullModelNameInput.value.trim();
    if (!modelName) return;

    pullModelBtn.disabled = true;
    pullModelStatus.textContent = `Pulling model '${modelName}'...`;

    try {
      const formData = new FormData();
      formData.append("model_name", modelName);
      const res = await fetch("/api/llm/pull", { method: "POST", body: formData });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        pullModelStatus.textContent = chunk.slice(-80);
      }
      pullModelStatus.innerHTML = `<span class="text-emerald-400">✅ Model '${modelName}' pulled successfully!</span>`;
      await loadLlmModels();
    } catch (e) {
      pullModelStatus.innerHTML = `<span class="text-red-400">❌ Pull failed: ${e.message}</span>`;
    } finally {
      pullModelBtn.disabled = false;
    }
  });

  function loadDemoData() {
    selectedFileCard.classList.remove("hidden");
    dropzone.classList.add("hidden");
    fileNameDisplay.textContent = "tech_talk_keynote.mp3";
    fileSizeDisplay.textContent = "12.4 MB (00:42 duration)";
    audioPreview.classList.remove("hidden");

    const demoResult = {
      task_id: "demo-showcase",
      filename: "tech_talk_keynote.mp3",
      duration: 42.8,
      processing_time: 1.4,
      language: "en",
      num_speakers: 2,
      text: "[Alex (Host)]: Welcome back to Tech Horizons. Today we're exploring local AI architectures and offline speech recognition.\n[Dr. Aris (AI Researcher)]: Thanks Alex. Running neural models directly on CPU using INT8 quantization and ONNX has unlocked incredible speed and privacy for end users.\n[Alex (Host)]: The ability to isolate audio, align speaker diarization, and generate karaoke subtitles without cloud dependency is a major leap forward.",
      summary: "### Executive Briefing\n- **Edge AI Acceleration**: Advances in INT8 quantization and ONNX runtime enable full Whisper transcription and neural speaker diarization directly on standard CPUs without GPU requirements.\n- **Zero-Latency Privacy**: Self-hosted speech-to-text ensures complete data isolation and confidential media processing.\n\n### Key Takeaways\n1. **High Precision Diarization**: Multi-speaker attribution accurately separates overlapping turns.\n2. **Subtitle Generation**: Complete support for SRT, WebVTT voice cues, and karaoke-timed ASS format.",
      polished: "Welcome back to Tech Horizons. Today we are exploring local AI architectures and offline speech recognition.\n\nThanks Alex. Running neural models directly on CPU using INT8 quantization and ONNX has unlocked incredible speed and privacy for end users.\n\nThe ability to isolate audio, align speaker diarization, and generate karaoke subtitles without cloud dependency is a major leap forward.",
      segments: [
        {
          start: 0.0,
          end: 6.8,
          speaker: "Alex (Host)",
          text: "Welcome back to Tech Horizons. Today we're exploring local AI architectures and offline speech recognition.",
          words: [
            { word: "Welcome", start: 0.0, end: 0.5, probability: 0.98, speaker: "Alex (Host)" },
            { word: "back", start: 0.55, end: 0.85, probability: 0.99, speaker: "Alex (Host)" },
            { word: "to", start: 0.9, end: 1.05, probability: 0.99, speaker: "Alex (Host)" },
            { word: "Tech", start: 1.1, end: 1.45, probability: 0.97, speaker: "Alex (Host)" },
            { word: "Horizons.", start: 1.5, end: 2.1, probability: 0.99, speaker: "Alex (Host)" },
            { word: "Today", start: 2.3, end: 2.7, probability: 0.96, speaker: "Alex (Host)" },
            { word: "we're", start: 2.75, end: 3.0, probability: 0.98, speaker: "Alex (Host)" },
            { word: "exploring", start: 3.05, end: 3.65, probability: 0.98, speaker: "Alex (Host)" },
            { word: "local", start: 3.7, end: 4.1, probability: 0.97, speaker: "Alex (Host)" },
            { word: "AI", start: 4.15, end: 4.5, probability: 0.99, speaker: "Alex (Host)" },
            { word: "architectures", start: 4.55, end: 5.4, probability: 0.99, speaker: "Alex (Host)" },
            { word: "and", start: 5.45, end: 5.65, probability: 0.98, speaker: "Alex (Host)" },
            { word: "offline", start: 5.7, end: 6.15, probability: 0.99, speaker: "Alex (Host)" },
            { word: "speech", start: 6.2, end: 6.5, probability: 0.99, speaker: "Alex (Host)" },
            { word: "recognition.", start: 6.55, end: 6.8, probability: 0.98, speaker: "Alex (Host)" }
          ]
        },
        {
          start: 7.2,
          end: 16.5,
          speaker: "Dr. Aris (AI Researcher)",
          text: "Thanks Alex. Running neural models directly on CPU using INT8 quantization and ONNX has unlocked incredible speed and privacy for end users.",
          words: [
            { word: "Thanks", start: 7.2, end: 7.6, probability: 0.98, speaker: "Dr. Aris (AI Researcher)" },
            { word: "Alex.", start: 7.65, end: 8.05, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "Running", start: 8.2, end: 8.65, probability: 0.97, speaker: "Dr. Aris (AI Researcher)" },
            { word: "neural", start: 8.7, end: 9.1, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "models", start: 9.15, end: 9.6, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "directly", start: 9.65, end: 10.15, probability: 0.98, speaker: "Dr. Aris (AI Researcher)" },
            { word: "on", start: 10.2, end: 10.35, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "CPU", start: 10.4, end: 10.85, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "using", start: 10.9, end: 11.25, probability: 0.97, speaker: "Dr. Aris (AI Researcher)" },
            { word: "INT8", start: 11.3, end: 11.8, probability: 0.98, speaker: "Dr. Aris (AI Researcher)" },
            { word: "quantization", start: 11.85, end: 12.7, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "and", start: 12.75, end: 12.95, probability: 0.98, speaker: "Dr. Aris (AI Researcher)" },
            { word: "ONNX", start: 13.0, end: 13.6, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "has", start: 13.65, end: 13.85, probability: 0.98, speaker: "Dr. Aris (AI Researcher)" },
            { word: "unlocked", start: 13.9, end: 14.45, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "incredible", start: 14.5, end: 15.1, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "speed", start: 15.15, end: 15.65, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" },
            { word: "and", start: 15.7, end: 15.85, probability: 0.97, speaker: "Dr. Aris (AI Researcher)" },
            { word: "privacy.", start: 15.9, end: 16.5, probability: 0.99, speaker: "Dr. Aris (AI Researcher)" }
          ]
        },
        {
          start: 17.0,
          end: 26.2,
          speaker: "Alex (Host)",
          text: "The ability to isolate audio, align speaker diarization, and generate karaoke subtitles without cloud dependency is a major leap forward.",
          words: [
            { word: "The", start: 17.0, end: 17.2, probability: 0.99, speaker: "Alex (Host)" },
            { word: "ability", start: 17.25, end: 17.75, probability: 0.99, speaker: "Alex (Host)" },
            { word: "to", start: 17.8, end: 17.95, probability: 0.98, speaker: "Alex (Host)" },
            { word: "isolate", start: 18.0, end: 18.55, probability: 0.99, speaker: "Alex (Host)" },
            { word: "audio,", start: 18.6, end: 19.1, probability: 0.98, speaker: "Alex (Host)" },
            { word: "align", start: 19.2, end: 19.65, probability: 0.99, speaker: "Alex (Host)" },
            { word: "speaker", start: 19.7, end: 20.15, probability: 0.99, speaker: "Alex (Host)" },
            { word: "diarization,", start: 20.2, end: 21.05, probability: 0.99, speaker: "Alex (Host)" },
            { word: "and", start: 21.1, end: 21.25, probability: 0.98, speaker: "Alex (Host)" },
            { word: "generate", start: 21.3, end: 21.85, probability: 0.99, speaker: "Alex (Host)" },
            { word: "karaoke", start: 21.9, end: 22.45, probability: 0.98, speaker: "Alex (Host)" },
            { word: "subtitles", start: 22.5, end: 23.2, probability: 0.99, speaker: "Alex (Host)" },
            { word: "without", start: 23.25, end: 23.7, probability: 0.99, speaker: "Alex (Host)" },
            { word: "cloud", start: 23.75, end: 24.15, probability: 0.99, speaker: "Alex (Host)" },
            { word: "dependency", start: 24.2, end: 24.95, probability: 0.99, speaker: "Alex (Host)" },
            { word: "is", start: 25.0, end: 25.15, probability: 0.98, speaker: "Alex (Host)" },
            { word: "a", start: 25.2, end: 25.3, probability: 0.99, speaker: "Alex (Host)" },
            { word: "major", start: 25.35, end: 25.75, probability: 0.99, speaker: "Alex (Host)" },
            { word: "leap", start: 25.8, end: 26.05, probability: 0.99, speaker: "Alex (Host)" },
            { word: "forward.", start: 26.1, end: 26.2, probability: 0.99, speaker: "Alex (Host)" }
          ]
        }
      ]
    };

    currentResult = Object.assign(currentResult, demoResult);
    resultsContainer.classList.remove("hidden");
    renderSpeakerDialogue(currentResult);
    transcriptionStats.textContent = "Duration: 42.8s • Processed in: 1.4s • Language: EN • 2 Speakers Identified";
    transcriptPlainText.textContent = demoResult.text;
    if (typeof marked !== "undefined") {
      summaryContent.innerHTML = marked.parse(demoResult.summary);
    }
    polishContent.textContent = demoResult.polished;

    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("tab") === "summary") {
      switchResultTab("summary");
    } else if (urlParams.get("tab") === "polish") {
      switchResultTab("polish");
    }

    if (window.lucide) {
      lucide.createIcons();
    }
  }

  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("demo") === "1" || urlParams.get("demo") === "true") {
    loadDemoData();
  }

  applyStoredPreferences();
  checkSystemStatus();
  updateWhisperModels();
});
