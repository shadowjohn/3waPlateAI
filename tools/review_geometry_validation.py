"""Serve the local image-level geometry review editor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit
import webbrowser

from plateai_trainer.detection.dataset import DetectionDataError
from plateai_trainer.detection.real_dataset import canonical_quad


_TOOLS = Path(__file__).resolve().parent
_ASSETS = {
    "/": (_TOOLS / "geometry_review.html", "text/html; charset=utf-8"),
    "/app.js": (_TOOLS / "geometry_review.js", "text/javascript; charset=utf-8"),
    "/style.css": (_TOOLS / "geometry_review.css", "text/css; charset=utf-8"),
}
_STATUSES = {"pending", "accepted", "excluded"}


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_manifest(path: Path, rows) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


class ReviewStore:
    def __init__(self, workspace: Path):
        self.root = workspace.resolve()
        review_set = json.loads((self.root / "review_set.json").read_text(encoding="utf-8"))
        if (review_set.get("schema_version") != "geometry-validation-review-v1"
                or review_set.get("status") != "pending_manual_review"):
            raise DetectionDataError("workspace is not a pending geometry review set")
        self.manifest_path = self.root / "manifest.jsonl"
        self.lock = threading.Lock()
        self._reload()

    def _reload(self):
        self.rows = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines()]
        if len(self.rows) != 120 or len({row["validation_index"] for row in self.rows}) != len(self.rows):
            raise DetectionDataError("review manifest count/identity mismatch")
        self.by_index = {row["validation_index"]: row for row in self.rows}

    def _safe_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        path.relative_to(self.root)
        return path

    def state(self):
        with self.lock:
            self._reload()
            records = []
            for row in self.rows:
                label = json.loads(self._safe_path(row["label_path"]).read_text(encoding="utf-8"))
                records.append({
                    **row,
                    "image_size_wh": label["image_size_wh"],
                    "original_count": len(label["original_corners"]),
                    "reviewed_count": len(label["reviewed_corners"]),
                    "reviewed_corners": label["reviewed_corners"],
                    "original_corners": label["original_corners"],
                })
            return {"records": records}

    def image(self, index: int):
        with self.lock:
            row = self.by_index.get(index)
            if row is None:
                raise KeyError(index)
            payload = self._safe_path(row["image_path"]).read_bytes()
        content_type = "image/png" if payload.startswith(b"\x89PNG") else "image/webp" if payload.startswith(b"RIFF") else "image/jpeg"
        return payload, content_type

    def save(self, index: int, document: dict):
        if not isinstance(document, dict):
            raise ValueError("request body must be an object")
        status = document.get("review_status")
        reason = document.get("exclude_reason")
        note = document.get("reviewer_note", "")
        corners = document.get("reviewed_corners")
        if status not in _STATUSES:
            raise ValueError("invalid review_status")
        if not isinstance(note, str) or len(note) > 500:
            raise ValueError("reviewer_note must be a string of at most 500 characters")
        if status == "excluded":
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 100:
                raise ValueError("excluded records require an exclude_reason")
        elif reason is not None:
            raise ValueError("exclude_reason is only valid for excluded records")
        if not isinstance(corners, list) or (status == "accepted" and not corners):
            raise ValueError("accepted records require at least one plate")

        with self.lock:
            self._reload()
            row = self.by_index.get(index)
            if row is None:
                raise KeyError(index)
            label_path = self._safe_path(row["label_path"])
            label = json.loads(label_path.read_text(encoding="utf-8"))
            width, height = label["image_size_wh"]
            reviewed = []
            for plate in corners:
                try:
                    values = [[float(point[0]), float(point[1])] for point in plate]
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError("each plate must contain four numeric points") from error
                if len(values) != 4 or any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in values):
                    raise ValueError("each plate must contain four finite points")
                normalized = canonical_quad(values, width, height)
                if not all(abs(values[i][axis] - float(normalized[i, axis])) <= 1e-3
                           for i in range(4) for axis in range(2)):
                    raise ValueError("corners must be semantic LT, RT, RB, LB without self-intersection")
                reviewed.append(values)
            label["reviewed_corners"] = reviewed
            _atomic_json(label_path, label)
            row["review_status"] = status
            row["exclude_reason"] = reason.strip() if isinstance(reason, str) else None
            row["reviewer_note"] = note
            row["reviewed_at"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                if status in ("accepted", "excluded") else None
            )
            _atomic_manifest(self.manifest_path, self.rows)
            return {"ok": True, "reviewed_at": row["reviewed_at"]}


def _handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PlateAIGeometryReview/1"

        def log_message(self, format, *args):
            print(f"review: {self.address_string()} {format % args}")

        def _send(self, status, payload, content_type="application/json; charset=utf-8"):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            try:
                request_path = urlsplit(self.path).path
                if request_path in _ASSETS:
                    path, content_type = _ASSETS[request_path]
                    self._send(HTTPStatus.OK, path.read_bytes(), content_type)
                    return
                if request_path == "/favicon.ico":
                    self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                    return
                if request_path == "/api/state":
                    self._send(HTTPStatus.OK, store.state())
                    return
                if request_path.startswith("/api/image/"):
                    payload, content_type = store.image(int(request_path.rsplit("/", 1)[1]))
                    self._send(HTTPStatus.OK, payload, content_type)
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (KeyError, OSError, ValueError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def do_POST(self):
            try:
                if not self.path.startswith("/api/review/"):
                    self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                document = json.loads(self.rfile.read(length).decode("utf-8"))
                result = store.save(int(self.path.rsplit("/", 1)[1]), document)
                self._send(HTTPStatus.OK, result)
            except (DetectionDataError, KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    store = ReviewStore(args.workspace)
    server = ThreadingHTTPServer((args.host, args.port), _handler(store))
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(json.dumps({"url": url, "workspace": str(store.root), "records": len(store.rows)}, ensure_ascii=False), flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
