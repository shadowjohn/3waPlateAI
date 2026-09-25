(function () {
    "use strict";
    let ready = false, busy = false, checking = false, poseTask = null;
    const terminal = task => ["completed", "failed", "cancelled"].includes(task.status);
    const element = id => document.getElementById(id);
    function controls() {
        element("pose-start").disabled = !ready || busy || checking || !element("pose-ack").checked;
        element("pose-check").disabled = checking || busy;
        element("pose-device").disabled = busy;
        element("pose-stop").disabled = !poseTask || terminal(poseTask) || poseTask.cancel_requested;
    }
    async function request(url, options) {
        const response = await fetch(url, {...options, signal: AbortSignal.timeout(45000)});
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`);
        return data;
    }
    function render(task) {
        busy = !terminal(task);
        if (task.request?.kind !== "pose") { poseTask = null; controls(); return; }
        poseTask = task;
        const result = task.result || {};
        element("pose-status").textContent = task.error || task.message || task.status;
        const bar = element("pose-progress"), progress = Math.max(0, Math.min(100, task.progress || 0));
        bar.style.width = `${progress}%`;
        bar.textContent = `${progress}%`;
        bar.setAttribute("aria-valuenow", progress);
        bar.classList.toggle("bg-danger", task.status === "failed");
        const logs = element("pose-logs");
        logs.textContent = (task.logs || []).join("\n");
        const output = element("pose-result");
        output.replaceChildren();
        const metrics = result.current_metrics;
        if (metrics) {
            const text = document.createElement("div");
            text.textContent = `Epoch ${metrics.epoch} / 10 · Train loss ${metrics.train_loss == null ? "—" : Number(metrics.train_loss).toFixed(4)} · holdout 尚未用於選模`;
            output.append(text);
        }
        if (terminal(task) && result.package_sha256 && /^[0-9a-f]{12}$/.test(task.id)) {
            const link = document.createElement("a");
            link.href = `/api/train/pose/artifacts/${task.id}/package`;
            link.className = "btn btn-success btn-sm my-2";
            link.textContent = "下載 Pose .pt ＋ hash ＋ 設定／評估 ZIP";
            output.append(link);
            const text = document.createElement("div");
            text.textContent = `Detector SHA-256: ${result.checkpoint_sha256}。未啟用；現役 API 不變。`;
            output.append(text);
            if (result.metrics) {
                const details = document.createElement("details"), summary = document.createElement("summary"), pre = document.createElement("pre");
                summary.textContent = `${result.holdout_count ?? "—"} 張內部 holdout 幾何指標（不是全新盲測）`;
                pre.textContent = JSON.stringify(result.metrics, null, 2);
                details.append(summary, pre); output.append(details);
            }
        }
        controls();
    }
    window.PoseTraining = {
        render,
        setBusy(value) { busy = value; controls(); },
        connectionError(message) { if (poseTask && !terminal(poseTask)) element("pose-status").textContent = message; }
    };
    document.addEventListener("DOMContentLoaded", () => {
        element("pose-ack").addEventListener("change", controls);
        element("pose-check").addEventListener("click", async () => {
            checking = true; ready = false; controls();
            const checks = element("pose-checks"); checks.textContent = "檢查中…";
            try {
                const result = await request("/api/train/pose/preflight");
                ready = result.ready; checks.replaceChildren();
                for (const check of result.checks) {
                    const row = document.createElement("div");
                    row.className = `p-2 mb-1 rounded ${check.ok ? "bg-success-subtle" : "bg-warning-subtle"}`;
                    row.style.overflowWrap = "anywhere";
                    row.textContent = `${check.ok ? "✓" : "✕"} ${check.id}: ${typeof check.detail === "string" ? check.detail : JSON.stringify(check.detail)}`;
                    if (!check.ok) { const remedy = document.createElement("div"); remedy.textContent = check.remedy; row.append(remedy); }
                    checks.append(row);
                }
            } catch (error) { checks.textContent = `預檢失敗：${error.message}`; }
            finally { checking = false; controls(); }
        });
        element("pose-start").addEventListener("click", async () => {
            busy = true; poseTask = null; controls();
            window.dispatchEvent(new Event("pose-training-pending"));
            element("pose-result").replaceChildren();
            element("pose-status").textContent = "正在建立 Pose 背景任務…";
            try {
                const result = await request("/api/train/start", {method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({kind: "pose", epochs: 10, device: element("pose-device").value, acknowledge_unreviewed_license: element("pose-ack").checked})});
                window.dispatchEvent(new CustomEvent("pose-training-started", {detail: result.task_id}));
            } catch (error) {
                busy = false; controls(); element("pose-status").textContent = `啟動未確認：${error.message}`;
                window.dispatchEvent(new Event("pose-training-refresh"));
            }
        });
        element("pose-stop").addEventListener("click", async () => {
            element("pose-stop").disabled = true;
            try {
                const result = await request("/api/train/stop", {method: "POST"});
                element("pose-status").textContent = result.message || "停止中…";
                if (result.task_id) window.dispatchEvent(new CustomEvent("pose-training-started", {detail: result.task_id}));
            } catch (error) { controls(); element("pose-status").textContent = `中止未確認：${error.message}`; }
        });
        controls();
    });
})();
