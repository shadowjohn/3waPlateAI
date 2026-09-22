/**
 * 3waPlateAI Web Studio Client Logic
 */
$(function () {
    // State
    let activeTaskId = null;
    let pollInterval = null;
    let lastLogCount = 0;

    // 3wa Mascot Dialogue Script
    const mascotLines = [
        "上車囉！老司機準備出發～🚗💨",
        "點擊上方按鈕，老司機隨時為您帶路！🏁",
        "貼上截圖 (Ctrl+V) 試試看，老司機一眼就看清！👀",
        "台灣車牌有 34 種合法字元，數字 4 也妥妥支援！🔢",
        "建置、訓練、發行一鍵搞定，超穩～✨",
        "發行後搬去其他專案跑 1788 一起發發！🚀"
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
            setTimeout(initBenchmarkChart, 200);
        }
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
        });
    }
    refreshStatus();

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
            $btn.prop("disabled", true).addClass("disabled");
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
        if (pollInterval) clearInterval(pollInterval);
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

                // Check Completion
                if (task.status === "completed") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    
                    // Progress Bar OK
                    $prog.removeClass("progress-bar-animated bg-primary bg-info bg-warning").addClass("bg-success").css("width", "100%").text("100% OK");
                    $body.append("<div class='text-success fw-bold mt-2'>[OK] 全部步驟已順利完成！</div>");
                    $body.scrollTop($body[0].scrollHeight);

                    // Update Button State
                    if ($btn) {
                        $btn.prop("disabled", false).removeClass("disabled btn-primary btn-secondary btn-warning btn-danger").addClass("btn-success");
                        const origText = $btn.text().replace(/^🚀 |^📥 |^🖨️ |^🏎️ |^✅ /, "");
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
                } else if (task.status === "failed") {
                    clearInterval(pollInterval);
                    pollInterval = null;
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
        runTask("/api/dataset/fetch_ezcon", {}, "#term-dataset", "#prog-dataset", "#status-box-dataset", "#btn-fetch-ezcon");
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

    // 4. Train Model
    $("#btn-start-train").on("click", function () {
        const epochs = parseInt($("#train-epochs").val()) || 5;
        setMascotLine(`模型訓練開始！${epochs} 個 Epoch 衝刺中，老司機緊握方向盤！🏎️`);
        runTask("/api/train/start", { epochs: epochs }, "#term-train", "#prog-train", "#status-box-train", "#btn-start-train");
    });

    // 5. Benchmark
    let benchmarkChart = null;
    function initBenchmarkChart(records) {
        const dom = document.getElementById("chart-benchmark");
        if (!dom) return;
        if (!benchmarkChart) {
            benchmarkChart = echarts.init(dom);
        }
        const cases = (records || []).map(r => r.case);
        const latencies = (records || []).map(r => r.avg_ms);

        const option = {
            title: { text: "各測試案例平均延遲 (ms)", left: "center", textStyle: { fontSize: 14 } },
            tooltip: { trigger: "axis" },
            xAxis: { type: "category", data: cases.length ? cases : ["新式客車", "姿態偵測", "CTC解碼", "機車", "批次"], axisLabel: { interval: 0, rotate: 15 } },
            yAxis: { type: "value", name: "ms" },
            series: [{
                data: latencies.length ? latencies : [12.4, 18.2, 5.1, 11.8, 28.5],
                type: "bar",
                itemStyle: { color: "#2e6da4" },
                label: { show: true, position: "top" }
            }]
        };
        benchmarkChart.setOption(option);
    }

    $("#btn-run-benchmark").on("click", function () {
        setMascotLine("Benchmark 測速與成功率評測中，大家坐穩囉！⏱️");
        runTask("/api/benchmark/run", { rounds: 10 }, "#term-benchmark", "#prog-benchmark", "#status-box-benchmark", "#btn-run-benchmark", function (res) {
            if (res && res.results) {
                renderBenchmarkResults(res.results, res.markdown);
                initBenchmarkChart(res.results);
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
        setMascotLine("一鍵發行中！即將輸出 Port 1788 獨立服務包！🚀");
        runTask("/api/release/build", {}, "#term-release", "#prog-release", "#status-box-release", "#btn-release", function (res) {
            $("#release-success-card").fadeIn();
        });
    });

    // 7. Live Inference (Drag & Drop + Ctrl+V Paste)
    const canvas = document.getElementById("canvas-visual");
    const ctx = canvas ? canvas.getContext("2d") : null;
    let currentImageObj = null;

    function handleImageInference(formData) {
        setMascotLine("抓到了！老司機眼睛一亮，正在全速辨識中...🔍");
        $("#infer-loading").show();

        $.ajax({
            url: "/api/predict",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function (res) {
                $("#infer-loading").hide();
                renderInferenceResult(res);
            },
            error: function (err) {
                $("#infer-loading").hide();
                alert("辨識失敗: " + (err.responseJSON ? err.responseJSON.detail : err.statusText));
            }
        });
    }

    function renderInferenceResult(res) {
        if (!ctx || !currentImageObj) return;

        // Set canvas dimensions matching original image
        canvas.width = currentImageObj.width;
        canvas.height = currentImageObj.height;
        ctx.drawImage(currentImageObj, 0, 0);

        const $list = $("#table-infer-list tbody");
        $list.empty();

        if (!res.detections || res.detections.length === 0) {
            $list.append("<tr><td colspan='5' class='text-center text-muted'>未偵測到車牌</td></tr>");
            return;
        }

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

            // Add row to table
            const row = `<tr>
                <td>${idx + 1}</td>
                <td><span class="badge-plate">${det.plate_text}</span></td>
                <td>${det.plate_type || "一般號牌"}</td>
                <td><span class="badge bg-success">${det.confidence}%</span></td>
                <td>${res.latency_ms} ms</td>
            </tr>`;
            $list.append(row);
        });

        setMascotLine(`辨識成功！發現車牌【${res.detections[0].plate_text}】，耗時僅 ${res.latency_ms}ms！🎯`);
    }

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
