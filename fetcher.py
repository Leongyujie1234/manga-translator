import re
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse


def fetch_nhentai(gallery_url, download_dir):
    gallery_id = None
    for m in re.finditer(r"/g/(\d+)", gallery_url):
        gallery_id = m.group(1)
        break

    if not gallery_id:
        return []

    api_url = f"https://nhentai.net/api/gallery/{gallery_id}"
    headers = {"User-Agent": "MangaTranslator/1.0"}
    resp = requests.get(api_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        return []

    data = resp.json()
    media_id = data.get("media_id", "")
    base_url = f"https://i.nhentai.net/galleries/{media_id}"
    num_pages = data.get("num_pages", 0)

    exts = ["jpg", "png", "webp"]
    urls = []
    for page in range(1, num_pages + 1):
        found = False
        images = data.get("images", {}).get("pages", [])
        if images and page - 1 < len(images):
            img_ext = images[page - 1].get("t", "j")
            if img_ext == "j":
                img_ext = "jpg"
            ext = img_ext if img_ext in exts else "jpg"
            urls.append((f"{base_url}/{page}.{ext}", page))
            found = True
        if not found:
            for ext in exts:
                urls.append((f"{base_url}/{page}.{ext}", page))
                break

    return _download_parallel(urls, download_dir)


def fetch_generic(url, download_dir):
    headers = {"User-Agent": "MangaTranslator/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return [str(e)]

    content_type = resp.headers.get("Content-Type", "")
    ext_map = {
        "image/jpeg": ".jpg", "image/png": ".png",
        "image/webp": ".webp", "image/bmp": ".bmp",
    }

    if any(ct in content_type for ct in ext_map):
        ext = ext_map.get(content_type.split(";")[0].strip(), ".jpg")
        path = os.path.join(download_dir, f"page_001{ext}")
        with open(path, "wb") as f:
            f.write(resp.content)
        return [path]

    img_urls = []
    for m in re.finditer(r'(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp))', resp.text, re.I):
        url_clean = m.group(1).split("?")[0]
        if url_clean not in img_urls:
            img_urls.append(url_clean)

    if not img_urls:
        return []

    urls = [(u, i + 1) for i, u in enumerate(img_urls)]
    return _download_parallel(urls, download_dir)


def _download_one(args):
    url, page, dl_dir = args
    headers = {"User-Agent": "MangaTranslator/1.0", "Referer": "https://nhentai.net/"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        ext = os.path.splitext(urlparse(url).path)[1]
        if not ext or len(ext) > 5:
            ct = resp.headers.get("Content-Type", "")
            ext = ".jpg" if "jpeg" in ct else ".png" if "png" in ct else ".webp" if "webp" in ct else ".jpg"
        path = os.path.join(dl_dir, f"page_{page:03d}{ext}")
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception as e:
        return f"ERR:{url}:{e}"


def _download_parallel(urls, dl_dir, workers=8):
    os.makedirs(dl_dir, exist_ok=True)
    tasks = [(u, p, dl_dir) for u, p in urls]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, t): t for t in tasks}
        for f in as_completed(futures):
            results.append(f.result())
    return sorted(
        [r for r in results if r and not r.startswith("ERR:")],
        key=lambda p: int(re.search(r"page_(\d+)", os.path.basename(p)).group(1)) if re.search(r"page_(\d+)", os.path.basename(p)) else 0,
    )
