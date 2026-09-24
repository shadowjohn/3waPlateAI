/**
 * 3waPlateAI Web Studio Client Logic
 */
$(function () {
    // State
    let activeTaskId = null;
    let currentPollingTaskId = null;
    let lastLogCount = 0;

    // 3wa Mascot Dialogue Script
    const mascotLines = [
        "上車囉！老司機準備出發～🚗💨",
        "點擊上方按鈕，老司機隨時為您帶路！🏁",
        "貼上截圖 (Ctrl+V) 試試看，老司機一眼就看清！👀",
        "台灣車牌有 34 種合法字元，數字 4 也妥妥支援！🔢",
        "建置與訓練在本機完成，模型權重不隨原始碼發行。✨",
        "1788 API 目前是 source-only 雛形，尚未接上真實推論。🚀"
    ];

    function setMascotLine(line) {
        $("#mascot-bubble-text").text(line);
    }

    $("#sidebar-mascot").on("click", function () {
        const rand = mascotLines[Math.floor(Math.random() * mascotLines.length)];
        setMascotLine(rand);
    });

    // Navigation Switcher
    $(".nav-item").on("click", function (e) {
        e.preventDefault();
        const tab = $(this).data("tab");
        $(".nav-item").removeClass("active");
        $(this).addClass("active");

        $(".page-section").hide();
        $("#section-" + tab).fadeIn(150);

        if (tab === "benchmark") {
            setTimeout(function () { initBenchmarkChart(benchmarkRecords); }, 200);
        } else if (tab === "train") {
            loadTrainDatasets();
            checkActiveTraining();
            setTimeout(function () {
                if (trainChart) trainChart.resize();
                else initTrainChart();
                if (gramChart) gramChart.resize();
                else initGramChart();
            }, 200);
        }
    });

    $(window).on("resize", function () {
        if (trainChart) trainChart.resize();
        if (gramChart) gramChart.resize();
        if (benchmarkChart) benchmarkChart.resize();
    });

    // Toggle Terminal Detail
    $(document).on("click", ".btn-toggle-terminal", function () {
        const target = $(this).data("target");
        $(target).slideToggle(200);
    });

    // Refresh System Status
    function refreshStatus() {
        $.getJSON("/api/status", function (data) {
            $("#stat-gpu").text(data.system.gpu);
            $("#stat-ezcon").text(data.datasets.ezcon_ready ? "已下載 (就緒)" : "未下載");
            $("#stat-tlpd").text(data.datasets.tlpd_ready ? "已下載 (就緒)" : "未下載");
            $("#stat-synth").text(data.datasets.synthetic_count + " 張");
            $("#stat-model").text(data.model.ready ? "ONNX 已就緒" : "未訓練/未匯出");
            $("#stat-model").attr("title", data.model.error || data.model.path || "尚無可用模型");
        });
    }
    refreshStatus();
    loadTrainDatasets();
    checkActiveTraining();
    setInterval(pollGpuMemory, 3000);
    pollGpuMemory();

    // Global Toast Notification Helper
    function showToast(title, message, isSuccess = true) {
        const toastEl = document.getElementById("app-toast");
        if (!toastEl) return;
        const $t = $(toastEl);
        if (isSuccess) {
            $t.removeClass("bg-danger").addClass("bg-success");
            $("#toast-icon").text("✅");
        } else {
            $t.removeClass("bg-success").addClass("bg-danger");
            $("#toast-icon").text("⚠️");
        }
        $("#toast-title").text(title);
        $("#toast-message").text(message);
        const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
        toast.show();
    }

    // Universal OK Modal Helper
    function showOkModal(title, heading, msg) {
        $("#modal-ok-title").text(title || "任務執行完成");
        $("#modal-ok-heading").text(heading || "執行完成 OK！");
        $("#modal-ok-msg").text(msg || "作業已順利處理完畢。");
        const modalEl = document.getElementById("modal-task-ok");
        if (modalEl) {
            const modal = new bootstrap.Modal(modalEl);
            modal.show();
        }
    }

    // Generic Task Runner & Log Poller
    function runTask(url, payload, terminalId, progressId, statusBoxId, buttonId, onComplete) {
        const $term = $(terminalId);
        const $body = $term.find(".terminal-body");
        const $prog = $(progressId);
        const $btn = buttonId ? $(buttonId) : null;

        if ($btn) {
            if (!$btn.data("idle-label")) $btn.data("idle-label", $btn.text());
            $btn.text($btn.data("idle-label")).prop("disabled", true)
                .removeClass("btn-success btn-danger").addClass("disabled btn-primary");
        }

        $term.show();
        $body.html("<div class='text-muted'>正在啟動背景任務...</div>");
        $prog.removeClass("bg-success bg-danger").addClass("bg-primary progress-bar-animated").css("width", "5%").attr("aria-valuenow", 5).text("5%");
        
        if (statusBoxId) {
            $(statusBoxId).show().html(`
                <div class="alert alert-info py-2 px-3 mb-3 d-flex align-items-center gap-2">
                    <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
                    <strong>任務正在進行中，請稍候...</strong>
                </div>
            `);
        }
        lastLogCount = 0;

        $.ajax({
            url: url,
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify(payload || {}),
            success: function (res) {
                activeTaskId = res.task_id;
                pollTask(activeTaskId, terminalId, progressId, statusBoxId, buttonId, onComplete);
            },
            error: function (err) {
                if ($btn) {
                    $btn.prop("disabled", false).removeClass("disabled");
                }
                if (err.status === 409) {
                    const data = err.responseJSON || {};
                    showToast("已有訓練任務進行中", data.message || "已有模型正在訓練中，不可同時重複啟動！", false);
                    setMascotLine("老司機報告：目前已經有訓練任務正在狂飆中囉！已為您自動同步進度～🏎️💨");
                    checkActiveTraining();
                    return;
                }
                $body.append("<div class='text-danger'>啟動失敗: " + err.responseText + "</div>");
                if (statusBoxId) {
                    $(statusBoxId).show().html(`
                        <div class="alert alert-danger py-2 px-3 mb-3">
                            <strong>啟動失敗:</strong> ${err.responseText || "無法連接後端"}
                        </div>
                    `);
                }
            }
        });
    }

    function pollTask(taskId, terminalId, progressId, statusBoxId, buttonId, onComplete) {
        let pollInterval = null;
        currentPollingTaskId = taskId;
        const $body = $(terminalId).find(".terminal-body");
        const $prog = $(progressId);
        const $btn = buttonId ? $(buttonId) : null;

        pollInterval = setInterval(function () {
            $.getJSON("/api/tasks/" + taskId, function (task) {
                // Update Progress
                const pct = task.progress;
                $prog.css("width", pct + "%").attr("aria-valuenow", pct).text(pct + "%");

                // Update Logs
                if (task.logs && task.logs.length > 0) {
                    $body.empty();
                    task.logs.forEach(function (line) {
                        $body.append($("<div>").text(line));
                    });
                    // Auto-scroll to bottom
                    $body.scrollTop($body[0].scrollHeight);
                }

                // Legacy task payloads may also carry training metrics.
                if (task.result && task.result.history && task.result.history.length > 0) {
                    updateTrainChartAndStats(task.result.history, task.result.current_metrics);
                }

                // Check Completion
                if (task.status === "completed") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    currentPollingTaskId = null;
                    
                    // Progress Bar OK
                    $prog.removeClass("progress-bar-animated bg-primary bg-info bg-warning").addClass("bg-success").css("width", "100%").text("100% OK");
                    $body.append("<div class='text-success fw-bold mt-2'>[OK] 全部步驟已順利完成！</div>");
                    $body.scrollTop($body[0].scrollHeight);

                    // Update Button State
                    if ($btn) {
                        $btn.prop("disabled", false).removeClass("disabled btn-secondary btn-warning btn-danger").addClass("btn-success");
                        const origText = ($btn.data("idle-label") || $btn.text()).replace(/^🚀 |^📥 |^🖨️ |^🏎️ |^✅ /, "");
                        $btn.html(`✅ ${origText} (OK)`);
                    }

                    // Render Prominent Success Banner
                    if (statusBoxId) {
                        $(statusBoxId).show().html(`
                            <div class="alert alert-success d-flex align-items-center gap-3 py-3 px-4 shadow-sm mb-3 border-success">
                                <span class="fs-1">🎉</span>
                                <div>
                                    <h5 class="alert-heading mb-1 fw-bold text-success">【${task.name}】執行完成 OK！</h5>
                                    <div class="small text-success-emphasis">${task.message || '全部步驟已執行完畢，狀態就緒。'}</div>
                                </div>
                            </div>
                        `);
                    }

                    // Popup Universal OK Modal
                    showOkModal(task.name, `【${task.name}】執行完成 OK！`, task.message || "作業已全數處理完畢，請按 OK 確認。");
                    showToast("執行完成 OK", `【${task.name}】已全部處理完畢！`, true);
                    setMascotLine(`太棒了！老司機確認【${task.name}】已順利完成 OK！✨👍`);
                    
                    refreshStatus();
                    if (onComplete) onComplete(task.result);
                } else if (task.status === "cancelled") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    currentPollingTaskId = null;
                    $prog.removeClass("progress-bar-animated bg-primary bg-info bg-danger").addClass("bg-warning text-dark").text("已手動中斷 (Cancelled)");
                    if ($btn) {
                        $btn.prop("disabled", false).removeClass("disabled btn-secondary btn-success").addClass("btn-danger").html("🏎️ 重新開始訓練");
                    }
                    $("#btn-stop-train").prop("disabled", false);

                    if (statusBoxId) {
                        $(statusBoxId).show().html(`
                            <div class="alert alert-warning d-flex align-items-center gap-3 py-3 px-4 shadow-sm mb-3 border-warning">
                                <span class="fs-2">🛑</span>
                                <div>
                                    <h5 class="alert-heading mb-1 fw-bold text-warning-emphasis">【${task.name}】已手動中斷</h5>
                                    <div class="small text-secondary">訓練引擎已安全停止，狀態已歸位。您可以隨時點擊「重新開始訓練」。</div>
                                </div>
                            </div>
                        `);
                    }

                    $("#train-chart-status").removeClass("bg-success-subtle text-success bg-warning-subtle text-warning").addClass("bg-secondary-subtle text-secondary").text("已手動中斷 (可隨時再啟動)");
                    showToast("訓練中斷", `【${task.name}】已安全中斷`, false);
                    setMascotLine("老司機已緊急煞車！訓練已中斷，隨時可以再啟動～🛑");
                } else if (task.status === "failed") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    currentPollingTaskId = null;
                    $prog.removeClass("progress-bar-animated").addClass("bg-danger");
                    if ($btn) {
                        $btn.prop("disabled", false).removeClass("disabled");
                    }
                    
                    if (statusBoxId) {
                        $(statusBoxId).show().html(`
                            <div class="alert alert-danger d-flex align-items-center gap-3 py-3 px-4 shadow-sm mb-3 border-danger">
                                <span class="fs-2">⚠️</span>
                                <div>
                                    <h5 class="alert-heading mb-1 fw-bold text-danger">【${task.name}】執行失敗！</h5>
                                    <div class="small text-danger-emphasis">${task.error || task.message}</div>
                                </div>
                            </div>
                        `);
                    }

                    showToast("執行失敗", `【${task.name}】遇到問題，請查看 Detail 紀錄`, false);
                    setMascotLine("哎呀！遇到障礙了，快看終端機 Log 排除一下～⚠️");
                }
            }).fail(function () {
                $(statusBoxId).empty().show().append($("<div>").addClass("alert alert-warning").text("暫時無法取得任務狀態，正在重新連線；尚未確認完成。"));
            });
        }, 800);
    }

    // 1. Environment Build
    $("#btn-build-env").on("click", function () {
        setMascotLine("正在檢查依賴與引擎環境，黑底綠字咻咻咻中！⚡");
        runTask("/api/env/build", {}, "#term-env", "#prog-env", "#status-box-env", "#btn-build-env");
    });

    // 2. Fetch Datasets
    $("#btn-fetch-ezcon").on("click", function () {
        setMascotLine("開始抓取 EZCon 真實車牌資料集，請稍候喔！📦");
        if (!$("#ezcon-license-ack").prop("checked")) {
            showToast("請確認資料用途", "勾選 EZCon 本機實驗授權說明後才能下載", false); return;
        }
        runTask("/api/dataset/fetch_ezcon", { acknowledge_unreviewed_license: true }, "#term-dataset", "#prog-dataset", "#status-box-dataset", "#btn-fetch-ezcon");
    });
    $("#btn-fetch-tlpd").on("click", function () {
        setMascotLine("開始同步 TLPD 台灣車輛 3,032 張檢測集！🚗");
        runTask("/api/dataset/fetch_tlpd", {}, "#term-dataset", "#prog-dataset", "#status-box-dataset", "#btn-fetch-tlpd");
    });

    // 3. Generate 10000 Plates
    $("#btn-generate-plates").on("click", function () {
        const count = parseInt($("#gen-count").val()) || 10000;
        const font = $("#gen-font").val() || "taiwan_plate";
        setMascotLine(`準備狂印 ${count} 張車牌！引擎全開！🖨️💨`);
        runTask("/api/dataset/generate", { count: count, font: font }, "#term-gen", "#prog-gen", "#status-box-gen", "#btn-generate-plates");
    });

    // Training Dataset Selector
    function loadTrainDatasets() {
        $.getJSON("/api/train/datasets", function (res) {
            const $sel = $("#train-dataset-select");
            const previous = $sel.val();
            $sel.empty();
            if (!res.datasets || res.datasets.length === 0) {
                $sel.append('<option value="">自動產生預設訓練集 (2,000 張)</option>');
                return;
            }
            let hasSelected = false;
            res.datasets.forEach(function (d) {
                const recText = ` (${d.count.toLocaleString()} 張)`;
                const opt = $("<option>").val(d.path).text(d.name + recText + (d.eligible ? "" : ` — 不可用：${d.reason}`));
                opt.prop("disabled", !d.eligible).attr("title", d.reason || "資料設定契約符合；開始訓練前會完整驗證樣本");
                if (d.eligible && (!hasSelected || d.path === previous)) {
                    opt.prop("selected", true);
                    hasSelected = true;
                }
                $sel.append(opt);
            });
            if (!hasSelected) $sel.prepend($("<option>").val("").text("自動產生預設訓練集 (2,000 張)").prop("selected", true));
        });
    }

    // PyTorch CTC Training Metrics Chart
    let trainChart = null;
    function initTrainChart() {
        const dom = document.getElementById("chart-train-metrics");
        if (!dom) return;
        if (!trainChart) {
            trainChart = echarts.init(dom);
        }
        const option = {
            title: {
                text: "PyTorch CTC 指標收斂曲線 (Loss vs. Accuracy)",
                subtext: "即時追蹤 PyTorch CTC 訓練損失與驗證精準度",
                left: "center",
                textStyle: { fontSize: 13, fontWeight: "bold", color: "#1e293b" },
                subtextStyle: { fontSize: 11, color: "#64748b" }
            },
            tooltip: {
                trigger: "axis",
                axisPointer: { type: "cross" }
            },
            legend: {
                data: ["訓練損失 (Train Loss)", "驗證損失 (Val Loss)", "驗證準確率 (Val Acc %)"],
                bottom: 2,
                textStyle: { fontSize: 12 }
            },
            grid: {
                top: 55,
                left: "4%",
                right: "5%",
                bottom: 40,
                containLabel: true
            },
            xAxis: {
                type: "category",
                boundaryGap: false,
                name: "Epoch",
                data: []
            },
            yAxis: [
                {
                    type: "value",
                    name: "Loss",
                    position: "left",
                    min: 0,
                    splitLine: { lineStyle: { type: "dashed", color: "#e2e8f0" } }
                },
                {
                    type: "value",
                    name: "Accuracy (%)",
                    position: "right",
                    min: 0,
                    max: 100,
                    splitLine: { show: false },
                    axisLabel: { formatter: "{value}%" }
                }
            ],
            series: [
                {
                    name: "訓練損失 (Train Loss)",
                    type: "line",
                    smooth: true,
                    yAxisIndex: 0,
                    itemStyle: { color: "#ef4444" },
                    lineStyle: { width: 3 },
                    symbol: "circle",
                    symbolSize: 6,
                    data: []
                },
                {
                    name: "驗證損失 (Val Loss)",
                    type: "line",
                    smooth: true,
                    yAxisIndex: 0,
                    itemStyle: { color: "#8b5cf6" },
                    lineStyle: { width: 3, type: "dashed" },
                    symbol: "diamond",
                    symbolSize: 6,
                    data: []
                },
                {
                    name: "驗證準確率 (Val Acc %)",
                    type: "line",
                    smooth: true,
                    yAxisIndex: 1,
                    itemStyle: { color: "#06b6d4" },
                    lineStyle: { width: 3 },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: "rgba(6,182,212,0.25)" },
                            { offset: 1, color: "rgba(6,182,212,0.01)" }
                        ])
                    },
                    symbol: "rect",
                    symbolSize: 6,
                    data: []
                }
            ]
        };
        trainChart.setOption(option);
    }

    function updateTrainChartAndStats(history, currentMetrics) {
        if (!trainChart) {
            initTrainChart();
        }
        const completedHistory = Array.isArray(history) ? history : [];
        const last = currentMetrics || completedHistory[completedHistory.length - 1];
        if (!last) {
            $("#train-stat-loss, #train-stat-vloss, #train-stat-acc").text("尚未驗證");
        } else {
            const tot = last.total_epochs || $("#train-epochs").val() || 5;
            $("#train-stat-epoch").text(`${last.epoch} / ${tot}`);
            $("#train-stat-loss").text(last.train_loss == null ? "尚未驗證" : Number(last.train_loss).toFixed(4));
            $("#train-stat-vloss").text(last.val_loss == null ? "尚未驗證" : Number(last.val_loss).toFixed(4));
            $("#train-stat-acc").text(last.val_acc == null ? "尚未驗證" : `${Number(last.val_acc).toFixed(1)}%`);
            $("#train-chart-status")
                .removeClass("bg-secondary-subtle text-secondary bg-warning-subtle text-warning")
                .addClass("bg-success-subtle text-success")
                .text(`即時收斂中 (回合 ${last.epoch}/${tot} | 準確率 ${last.val_acc}%)`);
        }
        if (completedHistory.length === 0) return;

        trainChart.setOption({
            xAxis: { data: completedHistory.map(h => `Epoch ${h.epoch}`) },
            series: [
                { name: "訓練損失 (Train Loss)", data: completedHistory.map(h => h.train_loss) },
                { name: "驗證損失 (Val Loss)", data: completedHistory.map(h => h.val_loss) },
                { name: "驗證準確率 (Val Acc %)", data: completedHistory.map(h => h.val_acc) }
            ]
        });
    }

    // GPU utilization real-time monitor, with VRAM retained as supporting context.
    let gramChart = null;
    let gramHistory = [];
    const MAX_GRAM_POINTS = 30;

    function initGramChart() {
        const dom = document.getElementById("chart-train-gram");
        if (!dom) return;
        if (!gramChart) {
            gramChart = echarts.init(dom);
        }
        const option = {
            title: {
                text: "GPU 使用率 即時負載",
                subtext: "每 3 秒透過 NVIDIA 驅動讀取 GPU 使用率",
                left: "center",
                textStyle: { fontSize: 13, fontWeight: "bold", color: "#1e293b" },
                subtextStyle: { fontSize: 11, color: "#64748b" }
            },
            tooltip: {
                trigger: "axis",
                axisPointer: { type: "cross" },
                formatter: function (params) {
                    if (!params || !params.length) return "";
                    const time = params[0].name;
                    let html = `<strong>${time}</strong><br/>`;
                    params.forEach(p => {
                        html += `${p.marker} ${p.seriesName}: <strong>${p.value}</strong>${p.seriesIndex === 0 ? "%" : " GB"}<br/>`;
                    });
                    return html;
                }
            },
            legend: {
                data: ["GPU 運算 (%)", "顯存用量 (GB)"],
                bottom: 2,
                textStyle: { fontSize: 12 }
            },
            grid: {
                top: 55,
                left: "4%",
                right: "5%",
                bottom: 40,
                containLabel: true
            },
            xAxis: {
                type: "category",
                boundaryGap: false,
                data: []
            },
            yAxis: [
                {
                    type: "value",
                    name: "GPU 使用率 (%)",
                    position: "left",
                    min: 0,
                    max: 100,
                    axisLabel: { formatter: "{value}%" },
                    splitLine: { lineStyle: { type: "dashed", color: "#e2e8f0" } }
                },
                {
                    type: "value",
                    name: "VRAM (GB)",
                    position: "right",
                    min: 0,
                    splitLine: { show: false },
                    axisLabel: { formatter: "{value} GB" }
                }
            ],
            series: [
                {
                    name: "GPU 運算 (%)",
                    type: "line",
                    smooth: true,
                    yAxisIndex: 0,
                    itemStyle: { color: "#06b6d4" },
                    lineStyle: { width: 3 },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: "rgba(6, 182, 212, 0.35)" },
                            { offset: 1, color: "rgba(6, 182, 212, 0.02)" }
                        ])
                    },
                    data: []
                },
                {
                    name: "顯存用量 (GB)",
                    type: "line",
                    smooth: true,
                    yAxisIndex: 1,
                    itemStyle: { color: "#6366f1" },
                    lineStyle: { width: 2, type: "dashed" },
                    data: []
                }
            ]
        };
        gramChart.setOption(option);
    }

    function pollGpuMemory() {
        $.getJSON("/api/system/gpu_memory", function (info) {
            if (!info) return;

            if (info.device_name) {
                $("#gram-device-badge").text(info.device_name);
            }
            if (info.available) {
                const gpuUsage = info.gpu_utilization_percent;
                const usageText = gpuUsage == null ? "GPU 使用率無法取得" : `${gpuUsage}% GPU`;
                $("#gram-usage-text").text(`${usageText} · ${info.used_gb} GB / ${info.total_gb} GB VRAM`);
            } else {
                $("#gram-usage-text").text(`${info.used_mb} MB (${info.percent}%)`);
            }

            const nowTime = info.timestamp || new Date().toTimeString().split(" ")[0];
            gramHistory.push({
                time: nowTime,
                used_gb: info.used_gb,
                gpu_utilization_percent: info.gpu_utilization_percent,
                total_gb: info.total_gb
            });
            if (gramHistory.length > MAX_GRAM_POINTS) {
                gramHistory.shift();
            }

            if (!gramChart) {
                initGramChart();
            }
            if (gramChart) {
                gramChart.setOption({
                    xAxis: {
                        data: gramHistory.map(g => g.time)
                    },
                    yAxis: [
                        { max: 100 },
                        { max: info.total_gb ? Math.ceil(info.total_gb) : undefined }
                    ],
                    series: [
                        { name: "GPU 運算 (%)", data: gramHistory.map(g => g.gpu_utilization_percent) },
                        { name: "顯存用量 (GB)", data: gramHistory.map(g => g.used_gb) }
                    ]
                });
            }
        });
    }

    // Training uses its own timer so other task panels cannot stop its recovery loop.
    let trainTaskId = null;
    let trainPollTimer = null;
    let trainPollInFlight = false;
    let trainRetryDelay = 1000;
    let trainStartedHere = false;

    function stopTrainPolling() {
        if (trainPollTimer) clearTimeout(trainPollTimer);
        trainPollTimer = null;
        trainPollInFlight = false;
    }

    function scheduleTrainPoll(delay) {
        if (trainPollTimer) clearTimeout(trainPollTimer);
        trainPollTimer = setTimeout(function () { pollTrainingTask(trainTaskId); }, delay);
    }

    function setTrainingStatus(task) {
        const terminal = ["completed", "failed", "cancelled"].includes(task.status);
        const $start = $("#btn-start-train");
        const $stop = $("#btn-stop-train");
        $start.prop("disabled", !terminal).toggleClass("disabled", !terminal);
        $stop.prop("disabled", terminal || task.cancel_requested);
        if (!terminal) $start.text("🏎️ 訓練進行中...");
        else if (task.status === "completed") $start.text("🏎️ 開始模型訓練");
        else $start.text("🏎️ 重新開始訓練");
    }

    function renderTrainingTask(task, notifyCompletion) {
        const terminal = ["completed", "failed", "cancelled"].includes(task.status);
        const $progress = $("#prog-train");
        const $body = $("#term-train .terminal-body");
        const $status = $("#status-box-train");
        const result = task.result || {};
        const metrics = result.current_metrics || null;
        const history = Array.isArray(result.history) ? result.history : [];

        setTrainingStatus(task);
        $progress.css("width", `${task.progress || 0}%`).attr("aria-valuenow", task.progress || 0).text(`${task.progress || 0}%`);
        if (Array.isArray(task.logs)) {
            $body.empty();
            task.logs.forEach(function (line) { $body.append($("<div>").text(line)); });
            if ($body[0]) $body.scrollTop($body[0].scrollHeight);
        }
        const totalEpochs = result.total_epochs || (task.request && task.request.epochs) || (history.length && history[history.length - 1].epoch);
        updateTrainChartAndStats(history.map(row => ({ ...row, total_epochs: totalEpochs })), metrics ? { ...metrics, total_epochs: totalEpochs } : null);

        if (!terminal) {
            const batchText = result.current_batch && result.total_batches
                ? `batch ${result.current_batch}/${result.total_batches}`
                : (task.last_progress_at ? "進度已持續回報" : "等待 worker 接手");
            $("#train-chart-status").removeClass("bg-secondary-subtle text-secondary bg-success text-white").addClass("bg-success-subtle text-success").text(`${task.message || "訓練進行中"}（${batchText}）`);
            return;
        }

        $status.empty().show();
        if (task.status === "completed") {
            $progress.removeClass("progress-bar-animated bg-danger bg-warning").addClass("bg-success").text("100% OK");
            $("#train-chart-status").removeClass("bg-warning-subtle text-warning bg-success-subtle text-success").addClass("bg-success text-white").text("已匯出，尚未啟用");
            $status.append($("<div>").addClass("alert alert-success py-2 px-3 mb-0").text("訓練與 ONNX 匯出完成；模型尚未自動啟用。"));
            if (result.bundle_dir) $status.append($("<div>").addClass("small text-success-emphasis mt-1").text(`Bundle: ${result.bundle_dir}`));

            // Extract bundle directory name
            const bundleName = result.bundle_dir ? result.bundle_dir.split(/[\\\/]/).pop() : (`train-${task.id}`);

            // Render Mascot Pointing Activation Card
            const $activateCard = $(`
                <div class="mascot-activate-card mt-3">
                    <div class="d-flex align-items-center justify-content-between flex-wrap gap-3">
                        <div class="d-flex align-items-center gap-3">
                            <div class="mascot-pointing-box">
                                <img src="/assets/mascot_pointing.png" class="mascot-pointing-img" alt="3wa 看板娘用力指著啟用按鈕" title="點擊啟用最新模型！">
                                <span class="pointing-arrow ms-1">👉</span>
                            </div>
                            <div>
                                <h5 class="fw-bold text-success mb-1">
                                    🎉 新模型已訓練就緒！
                                </h5>
                                <div class="text-secondary small mb-1">
                                    <strong>3wa 老司機看板娘用力指著：</strong>「新大腦訓練出爐啦！用力按旁邊這顆【一鍵啟用】，換裝現役模型飛馳吧！🚗💨」
                                </div>
                                <div class="small text-muted font-monospace">
                                    產出 Bundle：<span class="badge bg-light text-dark border">${bundleName}</span>
                                </div>
                            </div>
                        </div>
                        <div>
                            <button class="btn btn-activate-model d-flex align-items-center gap-2" id="btn-activate-now" data-bundle="${bundleName}">
                                <span>⚡</span>
                                <span>一鍵啟用為現役模型 (active-v1)</span>
                                <span>🚀</span>
                            </button>
                        </div>
                    </div>
                </div>
            `);
            $status.append($activateCard);

            if (notifyCompletion) showToast("訓練完成", "已匯出，尚未啟用", true);
        } else if (task.status === "cancelled") {
            $progress.removeClass("progress-bar-animated bg-primary bg-danger").addClass("bg-warning text-dark").text("已停止");
            $("#train-chart-status").removeClass("bg-success-subtle text-success").addClass("bg-secondary-subtle text-secondary").text("已停止；已發布產物保留");
            $status.append($("<div>").addClass("alert alert-warning py-2 px-3 mb-0").text(task.message || "已停止；已發布產物保留。"));
        } else {
            $progress.removeClass("progress-bar-animated").addClass("bg-danger");
            $("#train-chart-status").removeClass("bg-success-subtle text-success").addClass("bg-warning-subtle text-warning").text("訓練失敗，請查看 Detail");
            $status.append($("<div>").addClass("alert alert-danger py-2 px-3 mb-0").text(task.error || task.message || "訓練失敗，請查看 Detail。"));
        }
    }

    function recoverMissingTrainingTask() {
        $.getJSON("/api/train/active", function (res) {
            if (res.active && res.task) {
                trainTaskId = res.task.id;
                renderTrainingTask(res.task, false);
                scheduleTrainPoll(0);
            } else {
                $("#train-chart-status").text("找不到先前任務，請確認目前訓練狀態。");
                $("#btn-start-train").prop("disabled", false).removeClass("disabled");
            }
        });
    }

    function pollTrainingTask(taskId) {
        if (!taskId || taskId !== trainTaskId || trainPollInFlight) return;
        trainPollInFlight = true;
        $.ajax({ url: `/api/tasks/${encodeURIComponent(taskId)}`, dataType: "json" })
            .done(function (task) {
                if (taskId !== trainTaskId) return;
                trainRetryDelay = 1000;
                const terminal = ["completed", "failed", "cancelled"].includes(task.status);
                renderTrainingTask(task, terminal && trainStartedHere);
                if (terminal) {
                    stopTrainPolling();
                    trainStartedHere = false;
                } else {
                    scheduleTrainPoll(1000);
                }
            })
            .fail(function (xhr) {
                if (taskId !== trainTaskId) return;
                if (xhr.status === 404) {
                    stopTrainPolling();
                    recoverMissingTrainingTask();
                    return;
                }
                $("#train-chart-status")
                    .removeClass("bg-success-subtle text-success bg-success text-white")
                    .addClass("bg-warning-subtle text-warning")
                    .text(`連線中斷，${Math.ceil(trainRetryDelay / 1000)} 秒後重試...`);
                scheduleTrainPoll(trainRetryDelay);
                trainRetryDelay = Math.min(trainRetryDelay * 2, 10000);
            })
            .always(function () { trainPollInFlight = false; });
    }

    function startTrainingPolling(taskId, startedHere) {
        if (trainTaskId !== taskId) stopTrainPolling();
        trainTaskId = taskId;
        trainStartedHere = Boolean(startedHere);
        scheduleTrainPoll(0);
    }

    // Check if an independently running training task exists after a page reload.
    function checkActiveTraining() {
        $.getJSON("/api/train/active", function (res) {
            if (!res.task) {
                $("#btn-start-train").prop("disabled", false).removeClass("disabled");
                $("#btn-stop-train").prop("disabled", true);
                return;
            }
            renderTrainingTask(res.task, false);
            if (res.active) startTrainingPolling(res.task.id, false);
        });
    }

    // 4. Train Model
    $("#btn-start-train").on("click", function () {
        const epochs = parseInt($("#train-epochs").val(), 10) || 5;
        const trainDataset = $("#train-dataset-select").val() || null;
        $("#btn-start-train").prop("disabled", true).addClass("disabled");
        $("#train-stat-epoch").text(`0 / ${epochs}`);
        $("#train-stat-loss, #train-stat-vloss, #train-stat-acc").text("尚未驗證");
        $("#train-chart-status").removeClass("bg-success text-white").addClass("bg-warning-subtle text-warning").text("正在建立背景訓練任務...");
        initTrainChart();
        initGramChart();

        $.ajax({
            url: "/api/train/start",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ epochs: epochs, train_dataset: trainDataset })
        }).done(function (res) {
            startTrainingPolling(res.task_id, true);
        }).fail(function (xhr) {
            $("#btn-start-train").prop("disabled", false).removeClass("disabled");
            const message = (xhr.responseJSON && xhr.responseJSON.detail) || "無法建立背景訓練任務";
            $("#status-box-train").empty().show().append($("<div>").addClass("alert alert-danger py-2 px-3 mb-0").text(message));
            if (xhr.status === 409) checkActiveTraining();
        });
    });

    // Stop only asks the persisted worker to stop; polling continues until it confirms a terminal state.
    $("#btn-stop-train").on("click", function () {
        $("#btn-stop-train").prop("disabled", true);
        $.post("/api/train/stop").done(function (res) {
            $("#train-chart-status").text(res.message || "停止中，等待訓練在安全邊界結束...");
            if (res.task_id) startTrainingPolling(res.task_id, false);
        }).fail(function () {
            $("#btn-stop-train").prop("disabled", false);
            $("#status-box-train").empty().show().append($("<div>").addClass("alert alert-danger py-2 px-3 mb-0").text("停止要求未確認，訓練可能仍在執行。"));
        });
    });

    // One-Click Model Activation (active-v1)
    $(document).on("click", "#btn-activate-now", function () {
        const bundle = $(this).data("bundle");
        const $btn = $(this);
        $btn.prop("disabled", true).html('<span class="spinner-border spinner-border-sm me-2"></span>老司機正在全速換裝現役模型...');

        $.ajax({
            url: "/api/model/activate",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ bundle_name: bundle }),
            success: function (res) {
                $btn.removeClass("btn-activate-model").addClass("btn-secondary disabled").html('✅ 已成功啟用為現役模型！');
                $("#train-chart-status").removeClass("bg-success text-white").addClass("bg-primary text-white").text("✅ 已成功啟用為現役模型 (active-v1)");
                setMascotLine("太讚了！老司機已成功換上最新 AI 大腦！快去『即時驗證』上傳圖片試試看！🎯✨");

                $("#modal-ok-title").text("模型啟用成功");
                $("#modal-ok-heading").text("🎉 現役模型已成功換裝！");
                $("#modal-ok-msg").html(`已成功將 <strong>${res.bundle}</strong> 部署為現役模型 (active-v1)。<br>推論引擎已自動熱重載完成，隨時可以進行即時辨識！`);
                const modal = new bootstrap.Modal(document.getElementById("modal-task-ok"));
                modal.show();
            },
            error: function (err) {
                $btn.prop("disabled", false).html('<span>⚡</span><span>一鍵啟用為現役模型 (active-v1)</span><span>🚀</span>');
                alert("啟用失敗: " + (err.responseJSON ? err.responseJSON.detail : err.statusText));
            }
        });
    });

    // 5. Benchmark
    let benchmarkChart = null;
    let benchmarkRecords = [];
    function initBenchmarkChart(records) {
        const dom = document.getElementById("chart-benchmark");
        if (!dom || !$(dom).is(":visible") || !dom.clientWidth) return;
        if (!benchmarkChart) {
            benchmarkChart = echarts.init(dom);
        }
        const cases = (records || []).map(r => r.case);
        const latencies = (records || []).map(r => r.avg_ms);

        const option = {
            title: { text: "各測試案例平均延遲 (ms)", left: "center", textStyle: { fontSize: 14 } },
            tooltip: { trigger: "axis" },
            xAxis: { type: "category", data: cases, axisLabel: { interval: 0, rotate: 15 } },
            yAxis: { type: "value", name: "ms" },
            series: [{
                data: latencies,
                type: "bar",
                itemStyle: { color: "#2e6da4" },
                label: { show: true, position: "top" }
            }]
        };
        benchmarkChart.setOption(option);
        benchmarkChart.resize();
    }

    $("#btn-run-benchmark").on("click", function () {
        setMascotLine("Benchmark 測速與成功率評測中，大家坐穩囉！⏱️");
        const payload = { rounds: Number($("#bench-rounds").val()), warmup: Number($("#bench-warmup").val()),
            sample_limit: Number($("#bench-samples").val()), model_kind: $("#bench-model").val(),
            dataset: $("#bench-dataset").val(), mode: $("#bench-mode").val() };
        if (!Number.isInteger(payload.rounds) || payload.rounds < 1 || payload.rounds > 20 ||
            !Number.isInteger(payload.warmup) || payload.warmup < 0 || payload.warmup > 20 ||
            !Number.isInteger(payload.sample_limit) || payload.sample_limit < 1 || payload.sample_limit > 100) {
            showToast("參數無效", "回合 1–20、暖機 0–20、樣本 1–100", false); return;
        }
        $("#benchmark-result-panel").hide();
        benchmarkRecords = [];
        runTask("/api/benchmark/run", payload, "#term-benchmark", "#prog-benchmark", "#status-box-benchmark", "#btn-run-benchmark", function (res) {
            if (res && res.results) {
                renderBenchmarkResults(res.results, res.markdown);
                benchmarkRecords = res.results;
                $("#benchmark-result-panel").stop(true, true).show();
                initBenchmarkChart(benchmarkRecords);
            }
        });
    });

    function renderBenchmarkResults(results, markdown) {
        const $tbody = $("#table-benchmark tbody");
        $tbody.empty();
        results.forEach(function (r) {
            const tr = `<tr>
                <td>${r.case}</td>
                <td><code>${r.api}</code></td>
                <td>${r.query}</td>
                <td><span class="badge bg-success">${r.success}</span></td>
                <td><strong>${r.accuracy}</strong></td>
                <td>${r.avg_ms}</td>
                <td>${r.p50_ms}</td>
                <td>${r.p95_ms}</td>
                <td>${r.max_ms}</td>
                <td><span class="badge bg-primary">${r.qps}</span></td>
            </tr>`;
            $tbody.append(tr);
        });
        $("#benchmark-markdown").text(markdown);
        $("#benchmark-result-panel").fadeIn();
    }

    $("#btn-copy-markdown").on("click", function () {
        const text = $("#benchmark-markdown").text();
        navigator.clipboard.writeText(text).then(function () {
            alert("Markdown 評測表格已成功複製到剪貼簿！可直接貼至 README.md！");
        });
    });

    // 6. One-Click Release
    $("#btn-release").on("click", function () {
        setMascotLine("正在輸出 Port 1788 source-only API 雛形，不附模型。🚀");
        runTask("/api/release/build", {}, "#term-release", "#prog-release", "#status-box-release", "#btn-release", function (res) {
            $("#release-success-card").fadeIn();
        });
    });

    // 7. Live Inference (Drag & Drop + Ctrl+V Paste)
    const canvas = document.getElementById("canvas-visual");
    const ctx = canvas ? canvas.getContext("2d") : null;
    let currentImageObj = null;

    $("#toggle-candidate-preview").on("change", function () {
        $("#candidate-preview-kind").prop("disabled", !this.checked);
    });

    function handleImageInference(formData) {
        const previewCandidate = $("#toggle-candidate-preview").prop("checked");
        if (previewCandidate) {
            const selected = $("#candidate-preview-kind").val();
            formData.set("candidate_kind", selected === "fpga-lpr-mit" ? "fpga-lpr-mit" : "native-preview");
        }
        setMascotLine("抓到了！老司機眼睛一亮，正在全速辨識中...🔍");
        $("#infer-loading").show();

        $.ajax({
            url: previewCandidate ? "/api/predict/compare" : "/api/predict",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function (res) {
                $("#infer-loading").hide();
                if (previewCandidate) {
                    renderInferenceResult(res.active, res.candidate, res.preview);
                } else {
                    renderInferenceResult(res);
                }
            },
            error: function (err) {
                $("#infer-loading").hide();
                alert("辨識失敗: " + (err.responseJSON ? err.responseJSON.detail : err.statusText));
            }
        });
    }

    function overlayDiagnosticCandidates(records, color, dashed, label) {
        if (!ctx) return;
        records.forEach(function (det) {
            if (!det.polygon || det.polygon.length < 4) return;
            ctx.save();
            ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
            ctx.strokeStyle = color;
            ctx.setLineDash(dashed ? [8, 5] : []);
            ctx.beginPath();
            ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
            for (let i = 1; i < det.polygon.length; i++) ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
            ctx.closePath();
            ctx.stroke();
            const tagX = det.box ? det.box[0] : det.polygon[0][0];
            const tagY = Math.max(24, (det.box ? det.box[1] : det.polygon[0][1]) - 5);
            ctx.font = `bold ${Math.max(15, Math.round(canvas.width / 44))}px Consolas, sans-serif`;
            ctx.fillStyle = color;
            ctx.fillText(label, tagX, tagY);
            ctx.restore();
        });
    }

    function renderCandidatePreview(candidate, preview) {
        const $summary = $("#infer-ab-summary");
        const $candidateList = $("#infer-candidate-list").empty().hide();
        $summary.empty();
        if (!candidate) {
            $summary.hide();
            return;
        }
        const candidateDiag = candidate.diagnostics || {};
        const accepted = candidate.detections || [];
        const rejected = candidate.rejections || [];
        const label = candidateDiag.bundle_name || (preview && preview.candidate_bundle) || "候選模型";
        const isExternalOcr = candidateDiag.recognizer_type === "fpga-lpr-mit";
        const source = isExternalOcr ? "MIT CPM + LPRNet" : "原生候選模型";
        $candidateList.append($("<strong>").text(`候選結果 · ${source}`),
            $("<div>").addClass("small text-muted mb-2").text(
                `${label} · ${accepted.length} ${isExternalOcr ? "筆 OCR 輸出" : "通過"}／${rejected.length} 拒絕 · 未替換現役模型`));
        if (isExternalOcr) $candidateList.append($("<p>").addClass("small text-muted").text("OCR 輸出可能為碎片或誤報；分數未校準，不代表正確率。"));
        if (candidateDiag.error) {
            $candidateList.append($("<div>").addClass("alert alert-danger small").text(`候選推論失敗：${candidateDiag.error}`));
        } else if (!accepted.length) {
            $candidateList.append($("<p>").addClass("text-muted small").text("候選模型沒有可顯示的辨識輸出。"));
        }
        accepted.forEach(function (plate, index) {
            const $entry = $("<div>").addClass("border rounded p-2 mb-2");
            const score = Number.isFinite(plate.confidence) ? `${plate.confidence}%` : "未提供";
            $entry.append($("<div>").addClass("fw-bold mb-1").text(`${index + 1}. ${plate.plate_text || "空白"}`));
            if (plate.crop_base64) $entry.append($("<img>").attr({ src: plate.crop_base64, alt: `${source} 實際裁切` })
                .addClass("crop-inspect-btn rounded border mb-1").css({ maxWidth: "100%", height: "55px", cursor: "zoom-in" })
                .data({ crop: plate.crop_base64, plate: plate.plate_text || "空白", raw: plate.raw_greedy_text || "空白", rule: plate.rule_id || "無", conf: score }));
            $entry.append($("<div>").addClass("small").text(`原文：${plate.raw_greedy_text || "空白"} · OCR 分數：${score}（未校準）`));
            const timing = plate.timings || {};
            const inferenceMs = isExternalOcr ? timing.cpm + timing.lprnet : timing.onnx_ms;
            const decodeMs = isExternalOcr ? timing.decode : timing.ctc_ms;
            $entry.append($("<div>").addClass("small text-muted").text(
                `推論：${Number.isFinite(inferenceMs) ? inferenceMs.toFixed(1) + " ms" : "未提供"} · 解碼：${Number.isFinite(decodeMs) ? decodeMs.toFixed(1) + " ms" : "未提供"}`));
            $candidateList.append($entry);
        });
        if (rejected.length) {
            const $details = $("<details>").addClass("small").append($("<summary>").text(`候選模型拒絕診斷（${rejected.length} 筆）`));
            rejected.forEach(function (plate) {
                const $entry = $("<div>").addClass("border-bottom py-2");
                if (plate.crop_base64) $entry.append($("<img>").attr({ src: plate.crop_base64, alt: "候選拒絕裁切" }).css({ width: "95px", height: "40px", objectFit: "contain" }));
                $entry.append($("<span>").text(` ${plate.reason || "未通過辨識"} · 原文：${plate.raw_greedy_text || "空白"}`));
                $details.append($entry);
            });
            $candidateList.append($details);
        }
        $candidateList.show();
        const $title = $("<div>").addClass("fw-bold mb-1").text(
            isExternalOcr ? "本機 A/B：作者 CPM + LPRNet OCR（未啟用、不影響現役模型）" :
                "本機 A/B 定位對照（未啟用、不影響現役模型）"
        );
        const $counts = $("<div>").text(
            isExternalOcr
                ? `現役：${(preview && preview.active_bundle) || "active-v1"}；候選：${label}；OCR 輸出 ${accepted.length} 筆／拒絕 ${rejected.length} 筆（尚未驗證為真車牌）`
                : `現役：${(preview && preview.active_bundle) || "active-v1"}；候選：${label}；候選 ${accepted.length} 通過／${rejected.length} 拒絕`
        );
        const $notice = $("<div>").addClass("mt-1 text-danger").text(
            isExternalOcr
                ? "紫色實線＝OCR 有輸出，仍可能是碎片或誤報；紫色虛線＝候選定位到但 OCR 未輸出。"
                : "紫色實線＝候選通過辨識；紫色虛線＝候選定位到但未通過 OCR，僅供定位診斷。"
        );
        $summary.append($title, $counts, $notice);
        if (isExternalOcr) {
            $summary.append($("<div>").addClass("mt-1").text(
                `辨識器：${candidateDiag.model_id || "fpga-lpr-mit-v1"}；定位：${candidateDiag.detector_bundle || "未載入"}；OCR 分數未校準，不能當正確率。`
            ));
            $summary.append($("<a>").attr({
                href: "https://github.com/evan6007/FPGA-LPR", target: "_blank", rel: "noopener noreferrer"
            }).addClass("d-inline-block mt-1").text("作者原始碼與來源資訊"));
            accepted.forEach(function (plate, index) {
                $summary.append($("<div>").addClass("mt-1").text(
                    `候選 ${index + 1}：${plate.plate_text || "空白"}（原文：${plate.raw_greedy_text || "空白"}）`
                ));
            });
            rejected.slice(0, 10).forEach(function (plate, index) {
                $summary.append($("<div>").addClass("mt-1 text-danger").text(
                    `拒絕 ${index + 1}：${plate.reason || "未知原因"}`
                ));
            });
        }
        const warnings = (candidateDiag.warnings || []).slice();
        if (candidateDiag.error) warnings.unshift(`候選推論錯誤：${candidateDiag.error}`);
        warnings.forEach(function (warning) {
            $summary.append($("<div>").addClass("mt-1").text(warning));
        });
        const crops = accepted.concat(rejected).filter(function (item) { return item.crop_base64; }).slice(0, 3);
        if (crops.length) {
            const $crops = $("<div>").addClass("d-flex align-items-center gap-2 mt-2");
            $crops.append($("<span>").text("候選實際裁切："));
            crops.forEach(function (item) {
                $crops.append($("<img>").attr("src", item.crop_base64).attr("alt", "候選定位裁切")
                    .addClass("rounded border").css({ width: "114px", height: "48px", objectFit: "contain" }));
            });
            $summary.append($crops);
        }
        $summary.show();
        overlayDiagnosticCandidates(accepted, "#7c3aed", false, "候選");
        overlayDiagnosticCandidates(rejected, "#7c3aed", true, "候選診斷");
    }

    function renderInferenceResult(res, candidate, preview) {
        if (!ctx || !currentImageObj) return;

        // Set canvas dimensions matching original image
        canvas.width = currentImageObj.width;
        canvas.height = currentImageObj.height;
        ctx.drawImage(currentImageObj, 0, 0);

        // Render Diagnostics Bar
        const diag = res.diagnostics || {};
        const tb = diag.timing_breakdown || {};

        $("#diag-bundle").text(`Bundle: ${diag.bundle_name || "active-v1"}`);
        $("#infer-active-label").text(`現役模型 · ${diag.bundle_name || "active-v1"}`);
        $("#diag-model-id").text(`Model: ${diag.model_id || "-"}`);

        if (diag.pipeline_mode === "unavailable" || res.status === "inference_error") {
            $("#diag-pipeline-mode").removeClass("bg-success bg-warning text-dark").addClass("bg-danger text-white").text("管線不可用：請查看錯誤原因");
            $("#diag-locator").text(diag.locator_type || "定位器未載入");
        } else if (diag.pipeline_mode === "neural_full_pipeline") {
            $("#diag-pipeline-mode").removeClass("bg-danger");
            $("#diag-pipeline-mode").removeClass("bg-warning text-dark").addClass("bg-success text-white").text("管線: 神經網路端到端 (PlateReader)");
            $("#diag-locator").removeClass("bg-warning text-dark").addClass("bg-primary text-white").text(`定位: ${diag.locator_type}`);
        } else {
            $("#diag-pipeline-mode").removeClass("bg-success bg-danger text-white").addClass("bg-warning text-dark").text("管線: 混合啟發式 (OpenCV輪廓+ONNX)");
            $("#diag-locator").removeClass("bg-primary text-white").addClass("bg-secondary text-white").text("定位: OpenCV 輪廓啟發式 (無神經網絡 Detector)");
        }

        $("#time-locator").text(tb.locator_ms !== undefined ? tb.locator_ms : 0);
        $("#time-rect").text(tb.rectifier_ms !== undefined ? tb.rectifier_ms : 0);
        $("#time-onnx").text(tb.onnx_inference_ms !== undefined ? tb.onnx_inference_ms : 0);
        $("#time-ctc").text(tb.ctc_decoding_ms !== undefined ? tb.ctc_decoding_ms : 0);
        $("#time-total").text(res.latency_ms || tb.total_ms || 0);
        $("#infer-diagnostic-bar").fadeIn();
        $("#infer-ab-summary").hide().empty();

        let $warnings = $("#infer-model-warnings");
        if (!$warnings.length) {
            $warnings = $("<div>").attr("id", "infer-model-warnings").addClass("alert alert-warning small mt-2")
                .insertAfter("#infer-diagnostic-bar");
        }
        const warnings = (diag.warnings || []).slice();
        if (diag.error) warnings.unshift(`推論錯誤：${diag.error}`);
        $warnings.text(warnings.join("\n")).css("white-space", "pre-line").toggle(warnings.length > 0);

        // Rejected candidates remain inspectable, but are never drawn as plates.
        let $rejected = $("#infer-rejected-candidates");
        if (!$rejected.length) {
            $rejected = $("<details>").attr("id", "infer-rejected-candidates").addClass("mt-3 small")
                .insertAfter($("#table-infer-list").closest(".table-responsive"));
        }
        $rejected.empty();
        const rejected = res.rejections || [];
        const reasons = {
            low_recognition_score: "OCR 分數不足", decoder_disagreement: "Greedy 與規則解碼不一致",
            missing_greedy_evidence: "缺少原始解碼證據", ambiguous: "四角方向不明確",
            low_area: "四角面積過小", duplicate: "四角重複", non_convex: "四角非凸形"
        };
        $rejected.append($("<summary>").text(`已拒絕 ${rejected.length} 個候選（展開診斷，不視為車牌）`));
        rejected.forEach(function (item) {
            const $entry = $("<div>").addClass("d-flex gap-2 align-items-center border-bottom py-2");
            if (item.crop_base64) $entry.append($("<img>").attr("src", item.crop_base64).css({ width: "95px", height: "40px", objectFit: "contain" }));
            const score = Number.isFinite(item.recognition_score) ? `；OCR 分數 ${(item.recognition_score * 100).toFixed(1)}%` : "";
            const raw = item.raw_greedy_text == null ? "無" : (item.raw_greedy_text || "空白");
            $entry.append($("<span>").text(`${reasons[item.reason] || item.reason || "候選不合格"}${score}；Greedy：${raw}；規則候選：${item.plate_text || "無"}`));
            $rejected.append($entry);
        });
        $rejected.toggle(rejected.length > 0);

        const $list = $("#table-infer-list tbody");
        $list.empty();
        $("#badge-det-count").text(`${(res.detections || []).length} 通過／${rejected.length} 拒絕`);

        if (!res.detections || res.detections.length === 0) {
            const message = diag.error ? "模型或推論失敗，請查看診斷資訊。" :
                (rejected.length ? "沒有通過辨識條件的車牌；候選與拒絕原因可在下方展開。" : "目前定位器未找到車牌。");
            $list.append($("<tr>").append($("<td>").attr("colspan", 6).addClass("text-center text-muted py-4").text(message)));
            setMascotLine(message);
        } else {
            res.detections.forEach(function (det, idx) {
            // Draw polygon / bounding box
            ctx.lineWidth = Math.max(3, Math.round(canvas.width / 250));
            ctx.strokeStyle = "#22c55e"; // bright green
            ctx.fillStyle = "rgba(34, 197, 94, 0.15)";

            if (det.polygon && det.polygon.length >= 4) {
                ctx.beginPath();
                ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
                for (let i = 1; i < det.polygon.length; i++) {
                    ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
                }
                ctx.closePath();
                ctx.stroke();
                ctx.fill();
            } else if (det.box) {
                const [x1, y1, x2, y2] = det.box;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            }

            // Draw text tag
            const tagX = det.box ? det.box[0] : (det.polygon ? det.polygon[0][0] : 20);
            const tagY = Math.max(30, (det.box ? det.box[1] : (det.polygon ? det.polygon[0][1] : 30)) - 10);
            
            ctx.font = `bold ${Math.max(18, Math.round(canvas.width / 35))}px Consolas, sans-serif`;
            const textWidth = ctx.measureText(det.plate_text).width;

            ctx.fillStyle = "rgba(17, 36, 58, 0.85)";
            ctx.fillRect(tagX, tagY - 24, textWidth + 16, 30);

            ctx.fillStyle = "#ffffff";
            ctx.fillText(det.plate_text, tagX + 8, tagY - 3);

            // Crop thumbnail
            const cropHtml = det.crop_base64
                ? `<img src="${det.crop_base64}" class="crop-inspect-btn rounded border shadow-sm" style="max-height: 38px; max-width: 90px; cursor: zoom-in;" title="點擊檢視送審裁切圖" data-crop="${det.crop_base64}" data-plate="${det.plate_text}" data-raw="${det.raw_greedy_text || ''}" data-rule="${det.rule_id || ''}" data-conf="${det.confidence}%">`
                : `<span class="text-muted small">無裁切</span>`;

            // Stage timing tags
            const timings = det.timings || {};
            const inferenceText = timings.onnx_scope === "shared_batch" ? "共用批次（見上方）" : `${Number(timings.onnx_ms || 0).toFixed(1)}ms`;
            const timingText = `<div class="small text-muted font-monospace" style="font-size: 11px;">推論: ${inferenceText}<br>解碼: ${Number(timings.ctc_ms || 0).toFixed(1)}ms</div>`;

            // Add row to table
            const row = `<tr>
                <td>${idx + 1}</td>
                <td>${cropHtml}</td>
                <td>
                    <span class="badge-plate">${det.plate_text}</span>
                    <div class="text-muted" style="font-size: 11px;">${det.plate_type || "一般號牌"}</div>
                </td>
                <td><code class="text-dark bg-light px-1 border rounded">${det.raw_greedy_text == null ? "無" : (det.raw_greedy_text || "空白")}</code></td>
                <td>
                    <span class="badge bg-secondary" title="未經實拍校準的 OCR 分數，不是正確率">${det.confidence}%</span>
                    <div class="text-muted" style="font-size: 10px;">${det.rule_id || ""}</div>
                </td>
                <td>${timingText}</td>
            </tr>`;
                $list.append(row);
            });

            setMascotLine(`辨識結果【${res.detections[0].plate_text}】，${rejected.length} 個候選被拒絕；總耗時 ${res.latency_ms}ms。`);
        }
        renderCandidatePreview(candidate, preview);
        if (candidate) {
            const outputs = candidate.detections || [];
            const source = (candidate.diagnostics || {}).recognizer_type === "fpga-lpr-mit" ? "MIT 候選" : "原生候選";
            setMascotLine(`${source}：${(candidate.diagnostics || {}).error ? "推論失敗" : outputs.length ? outputs.map(p => p.plate_text || "空白").join("、") : "無辨識輸出"}；現役：${(res.detections || []).length} 通過／${rejected.length} 拒絕。兩套結果分開顯示。`);
        }
    }

    $(document).on("click", ".crop-inspect-btn", function () {
        const crop = $(this).data("crop");
        const plate = $(this).data("plate");
        const raw = $(this).data("raw");
        const rule = $(this).data("rule");
        const conf = $(this).data("conf");

        $("#modal-crop-img").attr("src", crop);
        $("#modal-crop-details").html(`
            <div><strong>車牌預測：</strong> <span class="badge-plate">${plate}</span></div>
            <div><strong>Greedy 原文：</strong> <code>${raw || "無"}</code></div>
            <div><strong>約束規則：</strong> <code>${rule || "無"}</code></div>
            <div><strong>OCR 分數：</strong> ${conf}（未校準，非正確率）</div>
            <div class="mt-2 text-muted" style="font-size: 11px;">說明：上方圖像是推論引擎實際送入 ONNX Recognizer 前進行擺正或裁切之真實影像。</div>
        `);
        const modal = new bootstrap.Modal(document.getElementById("modal-crop-inspect"));
        modal.show();
    });

    // Dropzone Event
    const $drop = $("#drop-zone");
    $drop.on("dragover", function (e) {
        e.preventDefault();
        $(this).addClass("dragover");
    });
    $drop.on("dragleave", function () {
        $(this).removeClass("dragover");
    });
    $drop.on("drop", function (e) {
        e.preventDefault();
        $(this).removeClass("dragover");
        const files = e.originalEvent.dataTransfer.files;
        if (files.length > 0) {
            loadFile(files[0]);
        }
    });

    $("#file-input").on("change", function () {
        if (this.files && this.files[0]) {
            loadFile(this.files[0]);
        }
    });

    // Global Paste Event (Ctrl + V)
    document.addEventListener("paste", function (e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let index in items) {
            const item = items[index];
            if (item.kind === "file" && item.type.indexOf("image") !== -1) {
                const blob = item.getAsFile();
                loadFile(blob);
                setMascotLine("偵測到剪貼簿截圖！老司機立即為您解析～📋");
                break;
            }
        }
    });

    function loadFile(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.onload = function () {
                currentImageObj = img;
                const fd = new FormData();
                fd.append("file", file);
                handleImageInference(fd);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});
