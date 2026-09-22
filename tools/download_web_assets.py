"""Download static web assets for offline local UI."""
import os
import urllib.request

ASSETS = {
    "web/css/reset.css": "https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.min.css",
    "web/css/bootstrap.min.css": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
    "web/js/bootstrap.bundle.min.js": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
    "web/js/jquery.min.js": "https://code.jquery.com/jquery-3.7.1.min.js",
    "web/js/echarts.min.js": "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js",
}

def main():
    for rel_path, url in ASSETS.items():
        os.makedirs(os.path.dirname(rel_path), exist_ok=True)
        print(f"Fetching {url} -> {rel_path}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            with open(rel_path, "wb") as f:
                f.write(content)
            print(f"Saved {rel_path} ({len(content)} bytes)")
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

if __name__ == "__main__":
    main()
