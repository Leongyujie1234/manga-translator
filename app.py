import os
import sys
import io
import json
import uuid
import time
import threading
import shutil
import cv2
from flask import Flask, request, jsonify, render_template, send_file

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fetcher import fetch_nhentai, fetch_generic
from pipeline import process_single, OR_MODEL

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

sessions = {}
sessions_lock = threading.Lock()


def get_api_key():
    for var in ["OPENROUTER_KEY", "OPENROUTER_API_KEY"]:
        val = os.environ.get(var, "").strip()
        if val:
            return val
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                for prefix in ["OPENROUTER_KEY=", "OPENROUTER_API_KEY="]:
                    if line.startswith(prefix):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def process_session(session_id, url, api_key):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    dl_dir = os.path.join(session_dir, "originals")
    out_dir = os.path.join(session_dir, "translated")
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with sessions_lock:
        sessions[session_id]["status"] = "fetching"

    if "nhentai.net" in url:
        images = fetch_nhentai(url, dl_dir)
    else:
        images = fetch_generic(url, dl_dir)

    if not images:
        with sessions_lock:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["error"] = "No images found"
        return

    total = len(images)
    with sessions_lock:
        sessions[session_id]["total"] = total
        sessions[session_id]["status"] = "translating"

    import concurrent.futures

    def process_one(idx_path):
        idx, img_path = idx_path
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        out_path = os.path.splitext(out_path)[0] + ".jpg"
        ok = process_single(img_path, out_path, api_key)
        return idx, ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(process_one, (i, p)): i for i, p in enumerate(images)}
        for future in concurrent.futures.as_completed(futures):
            idx, ok = future.result()
            if ok:
                with sessions_lock:
                    sessions[session_id]["done"] += 1

    with sessions_lock:
        sessions[session_id]["status"] = "done"


@app.route("/")
def index():
    api_key = get_api_key()
    return render_template("index.html", has_key=bool(api_key), model=OR_MODEL)


@app.route("/start", methods=["POST"])
def start():
    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "No OpenRouter API key set"}), 400

    data = request.get_json()
    url = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "URL required"}), 400

    if not url.startswith("http"):
        url = "https://" + url

    session_id = uuid.uuid4().hex[:12]
    with sessions_lock:
        sessions[session_id] = {
            "status": "starting",
            "total": 0,
            "done": 0,
            "url": url,
            "error": None,
        }

    t = threading.Thread(target=process_session, args=(session_id, url, api_key), daemon=True)
    t.start()

    return jsonify({"session_id": session_id})


@app.route("/status/<session_id>")
def status(session_id):
    with sessions_lock:
        s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status": s["status"],
        "total": s["total"],
        "done": s["done"],
        "error": s.get("error"),
    })


@app.route("/image/<session_id>/<filename>")
def serve_image(session_id, filename):
    path = os.path.join(SESSIONS_DIR, session_id, "translated", filename)
    if not os.path.exists(path):
        path = os.path.join(SESSIONS_DIR, session_id, "originals", filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/reader/<session_id>")
def reader(session_id):
    s = sessions.get(session_id)
    if not s:
        return "Session not found", 404

    out_dir = os.path.join(SESSIONS_DIR, session_id, "translated")
    images = sorted(
        [f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png", ".webp"))],
        key=lambda f: int("".join(c for c in f if c.isdigit()) or "0"),
    ) if os.path.exists(out_dir) else []

    return render_template("reader.html", session_id=session_id, images=images, url=s.get("url", ""))


@app.route("/pdf/<session_id>")
def download_pdf(session_id):
    out_dir = os.path.join(SESSIONS_DIR, session_id, "translated")
    if not os.path.exists(out_dir):
        return "Not found", 404

    images = sorted(
        [f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png", ".webp"))],
        key=lambda f: int("".join(c for c in f if c.isdigit()) or "0"),
    )
    if not images:
        return "No images", 404

    import img2pdf
    img_paths = [os.path.join(out_dir, im) for im in images]
    pdf_bytes = img2pdf.convert(img_paths)

    from io import BytesIO
    from flask import Response
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=manga_{session_id}.pdf"},
    )


@app.route("/zip/<session_id>")
def download_zip(session_id):
    out_dir = os.path.join(SESSIONS_DIR, session_id, "translated")
    if not os.path.exists(out_dir):
        return "Not found", 404

    zip_path = os.path.join(SESSIONS_DIR, session_id, "manga.zip")
    if not os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(out_dir)):
                if f.endswith((".jpg", ".png", ".webp")):
                    zf.write(os.path.join(out_dir, f), f)

    return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name=f"manga_{session_id}.zip")


if __name__ == "__main__":
    key = get_api_key()
    if not key:
        print("WARNING: No OpenRouter API key found!")
        print("  Set env: $env:OPENROUTER_KEY='sk-or-...'")
    else:
        print(f"Model: {OR_MODEL}")
    print("Starting at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
