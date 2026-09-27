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

  // State
  let selectedFile = null;
  let recordedBlob = null;
  let isRecording = false;
  let mediaRecorder = null;
  let recordChunks = [];
  let recordTimerInterval = null;
  let recordSeconds = 0;

  let activeAiAction = "summary"; // 'raw', 'polish', 'summary'
  let activeSummaryLevel = "bullets"; // 'tldr', 'bullets', 'detailed', 'action_items', 'custom'

  let currentResult = {
    text: "",
    srt: "",
    vtt: "",
    ass: "",
    word_vtt: "",
    segments: [],
    polished: "",
    summary: "",
    duration: 0,
    processing_time: 0,
    filename: "",
    task_id: "",
  };

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

      await loadLlmModels();
    } catch (e) {
      console.error("Status check failed", e);
    }
  }

  async function loadLlmModels() {
    const provider = llmProviderSelect.value;
    try {
      const res = await fetch(`/api/llm/models?provider=${provider}`);
      if (res.ok) {
        const data = await res.json();
        llmModelSelect.innerHTML = "";
        if (data.models && data.models.length > 0) {
          data.models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            llmModelSelect.appendChild(opt);
          });
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

  llmProviderSelect.addEventListener("change", loadLlmModels);
  refreshModelsBtn.addEventListener("click", loadLlmModels);

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

  // Microphone recording
  recordToggleBtn.addEventListener("click", async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        recordChunks = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) recordChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
          recordedBlob = new Blob(recordChunks, { type: "audio/webm" });
          selectedFile = new File([recordedBlob], "mic_recording.webm", { type: "audio/webm" });
          recordPreview.src = URL.createObjectURL(recordedBlob);
          recordPreview.classList.remove("hidden");
          recordStatus.textContent = "Recording saved! Ready to transcribe.";
        };

        mediaRecorder.start();
        isRecording = true;
        recordToggleBtn.classList.add("recording-active");
        recordStatus.textContent = "Recording in progress... Click to stop.";
        
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
      mediaRecorder.stop();
      isRecording = false;
      recordToggleBtn.classList.remove("recording-active");
      clearInterval(recordTimerInterval);
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

        currentResult.text = finalResult.text;
        currentResult.srt = finalResult.srt;
        currentResult.vtt = finalResult.vtt;
        currentResult.ass = finalResult.ass || "";
        currentResult.word_vtt = finalResult.word_vtt || "";
        currentResult.duration = finalResult.duration;
        currentResult.processing_time = finalResult.processing_time;
        currentResult.segments = finalResult.segments || [];

        if (finalResult.segments && finalResult.segments.length > 0) {
          transcriptSegmentsList.innerHTML = "";
          finalResult.segments.forEach((seg) => appendLiveSegment(seg));
        }

        if (finalResult.summary && !finalResult.summary.startsWith("[AI summary skipped")) {
          currentResult.summary = finalResult.summary;
          summaryContent.innerHTML = typeof marked !== "undefined" ? marked.parse(finalResult.summary) : finalResult.summary;
        } else {
          summaryContent.innerHTML = renderAiUnavailableNotice("Summary", finalResult.ai_warning || (finalResult.summary ? finalResult.summary.replace(/^\[|\]$/g, "") : null));
        }

        if (finalResult.polished && !finalResult.polished.startsWith("[AI polish skipped")) {
          currentResult.polished = finalResult.polished;
          polishContent.textContent = finalResult.polished;
        } else {
          polishContent.innerHTML = renderAiUnavailableNotice("Polish", finalResult.ai_warning || (finalResult.polished ? finalResult.polished.replace(/^\[|\]$/g, "") : null));
        }

        transcriptionStats.textContent = `Duration: ${finalResult.duration.toFixed(1)}s • Processed in: ${finalResult.processing_time}s • Language: ${finalResult.language.toUpperCase()}`;
        transcriptPlainText.textContent = finalResult.text;
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
          <span>AI ${actionName} Not Available</span>
        </div>
        <p class="text-xs text-slate-300 leading-relaxed">${detail}</p>
        <div class="pt-2 border-t border-amber-500/20 text-xs text-slate-400 space-y-1">
          <p class="font-medium text-slate-300">To enable AI ${actionName.toLowerCase()}:</p>
          <ul class="list-disc list-inside space-y-0.5 text-slate-300">
            <li>Start local Ollama with <code class="px-1.5 py-0.5 bg-slate-900 rounded font-mono text-amber-300 text-[11px]">ollama serve</code></li>
            <li>Or configure a remote Ollama host / cloud API key in <span class="text-indigo-400 font-medium cursor-pointer hover:underline" onclick="document.getElementById('settings-btn').click()">Settings</span></li>
          </ul>
        </div>
      </div>
    `;
  }

  function switchResultTab(tabName) {
    [resTabTranscript, resTabPolish, resTabSummary].forEach((b) => {
      b.classList.remove("active", "text-indigo-400", "bg-indigo-500/10");
      b.classList.add("text-slate-400");
    });
    [viewportTranscript, viewportPolish, viewportSummary].forEach((v) => v.classList.add("hidden"));

    if (tabName === "transcript") {
      resTabTranscript.classList.add("active", "text-indigo-400", "bg-indigo-500/10");
      viewportTranscript.classList.remove("hidden");
    } else if (tabName === "polish") {
      resTabPolish.classList.add("active", "text-indigo-400", "bg-indigo-500/10");
      viewportPolish.classList.remove("hidden");
      if (!currentResult.polished && !polishContent.textContent.trim()) {
        polishContent.innerHTML = renderAiUnavailableNotice("Polish");
      }
    } else if (tabName === "summary") {
      resTabSummary.classList.add("active", "text-indigo-400", "bg-indigo-500/10");
      viewportSummary.classList.remove("hidden");
      if (!currentResult.summary && !summaryContent.textContent.trim()) {
        summaryContent.innerHTML = renderAiUnavailableNotice("Summary");
      }
    }
  }

  resTabTranscript.addEventListener("click", () => switchResultTab("transcript"));
  resTabPolish.addEventListener("click", () => switchResultTab("polish"));
  resTabSummary.addEventListener("click", () => switchResultTab("summary"));

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

  downloadTxtBtn.addEventListener("click", () => downloadFile("transcript.txt", currentResult.text, "text/plain"));
  downloadSrtBtn.addEventListener("click", () => downloadFile("subtitles.srt", currentResult.srt, "text/plain"));
  downloadVttBtn.addEventListener("click", () => downloadFile("subtitles.vtt", currentResult.vtt, "text/vtt"));
  if (downloadAssBtn) {
    downloadAssBtn.addEventListener("click", () => downloadFile("subtitles.ass", currentResult.ass || "", "text/plain"));
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
  openSettingsBtn.addEventListener("click", () => settingsModal.classList.remove("hidden"));
  closeSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));

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

  checkSystemStatus();
});
