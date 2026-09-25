/* Paged report rows; image bytes never travel through task polling JSON. */
function benchmarkBboxText(row) {
    const isGt = row.bbox_source === "gt_crop";
    const boxes = isGt ? [row.gt_box] : row.predicted_boxes;
    if (!boxes.length) return "未定位";
    return boxes.map((box, index) => {
        const coordinates = box.map(value => Math.round(value * 10) / 10).join(",");
        return `${isGt ? "GT" : `P${index + 1}`} [${coordinates}]`;
    }).join("\n");
}

$(function () {
    let reportId = null;
    let offset = 0;
    let request = null;
    let revision = 0;
    const pageSize = 100;
    const $panel = $("#benchmark-details");
    const $rows = $("#table-benchmark-details tbody");

    function imageCell(url, title, width, height) {
        const $cell = $("<td>");
        if (!url) return $cell.text("無裁切").addClass("text-muted small");
        const $button = $("<button>").attr({ type: "button", title, "aria-label": title })
            .addClass("btn p-0 border-0 benchmark-image-open").data({ url, title });
        $button.append($("<img>").attr({ src: url, alt: title, loading: "lazy", decoding: "async", width, height })
            .addClass("rounded border").css({ objectFit: "contain", background: "#f3f4f6" }));
        return $cell.append($button);
    }

    function showRows(rows) {
        $rows.empty();
        rows.forEach(function (row) {
            const $tr = $("<tr>");
            $tr.append($("<td>").text(row.index + 1).attr("title", row.filename));
            $tr.append(imageCell(row.image_url, `${row.filename} · 綠 GT／紫預測`, 180, 115));
            $tr.append(imageCell(row.crop_url, `實際 OCR 裁切 · GT ${row.expected}`, 130, 64));
            $tr.append($("<td>").addClass("font-monospace").text(row.expected));
            // String correctness and spatial correctness are independent.
            $tr.append($("<td>").addClass("font-monospace fw-bold")
                .toggleClass("benchmark-text-mismatch text-danger", !row.text_match)
                .text(row.predicted || "未辨識"));
            $tr.append($("<td>").addClass("font-monospace small").css("white-space", "pre")
                .text(benchmarkBboxText(row)).attr("title", "原圖像素座標 [x1,y1,x2,y2]；顯示至小數 1 位"));
            const reasons = { no_detection: "未定位到車牌", no_gt_match: "定位未匹配 GT（IoU < 0.5）" };
            const status = row.correct ? "✓ 正確" : row.reason ? (reasons[row.reason] || row.reason) : "文字不符";
            const $status = $("<td>").addClass("small").append($("<div>").toggleClass("text-success", row.correct).text(status));
            $status.append($("<div>").addClass("text-muted").text(`${row.latency_ms} ms · 第 1 回合`));
            $status.append($("<div>").addClass("text-muted").text(row.bbox_source === "gt_crop" ? "GT Crop（非模型定位）" : `預測框 ${row.predicted_boxes.length} 個`));
            if (row.scene_outputs && row.scene_outputs.length > 1) {
                $status.append($("<div>").addClass("text-muted").text(`場景全部輸出：${row.scene_outputs.join("、")}`));
            }
            $tr.append($status);
            $rows.append($tr);
        });
        if (!rows.length) $rows.append($("<tr>").append($("<td>").attr("colspan", 7).addClass("text-muted text-center py-3").text("此條件下沒有樣本")));
    }

    function loadPage() {
        const current = ++revision;
        if (request) request.abort();
        if (!reportId) return;
        $("#bench-detail-prev, #bench-detail-next").prop("disabled", true);
        $("#bench-detail-page").text("載入中…");
        $rows.empty();
        request = $.getJSON(`/api/benchmark/reports/${encodeURIComponent(reportId)}/rows`, {
            group: $("#bench-detail-group").val(), offset, limit: pageSize,
            errors_only: $("#bench-detail-errors").prop("checked")
        }).done(function (page) {
            if (current !== revision) return;
            showRows(page.rows);
            $("#bench-detail-page").text(`${page.total ? offset + 1 : 0}–${Math.min(offset + pageSize, page.total)}／${page.total} 筆`);
            $("#bench-detail-prev").prop("disabled", offset === 0);
            $("#bench-detail-next").prop("disabled", offset + pageSize >= page.total);
        }).fail(function (_, status) {
            if (status === "abort" || current !== revision) return;
            $("#bench-detail-page").text("明細載入失敗，請切換模型／篩選重試；總分不受影響。");
        });
    }

    window.renderBenchmarkDetails = function (id, results) {
        reportId = id;
        offset = 0;
        ++revision;
        if (request) request.abort();
        $rows.empty();
        const $groups = $("#bench-detail-group").empty();
        $("#bench-detail-errors").prop("checked", false);
        $panel.toggle(Boolean(id));
        if (!id) return;
        results.forEach(r => $groups.append($("<option>").val(r.group).text(r.case)));
        loadPage();
    };
    $("#bench-detail-group, #bench-detail-errors").on("change", function () { offset = 0; loadPage(); });
    $("#bench-detail-prev").on("click", function () { offset = Math.max(0, offset - pageSize); loadPage(); });
    $("#bench-detail-next").on("click", function () { offset += pageSize; loadPage(); });
    $(document).on("click", ".benchmark-image-open", function () {
        $("#benchmark-image-title").text($(this).data("title"));
        $("#benchmark-image-zoom").attr("src", $(this).data("url"));
        bootstrap.Modal.getOrCreateInstance(document.getElementById("benchmark-image-modal")).show();
    });
});
