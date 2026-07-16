import re
import os
import cloudscraper
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept": "application/json, text/html", "Accept-Language": "en-US,en;q=0.9"}
IMG_HEADERS = {"User-Agent": UA, "Referer": "https://nhentai.net/", "Accept": "image/webp,image/*"}  # noqa: S105 - public referer only


def _get_scraper():
    return cloudscraper.create_scraper(
        browser={"custom": UA},
        delay=10,
    )


def fetch_nhentai(gallery_url, download_dir):
    gallery_id = None
    for m in re.finditer(r"/g/(\d+)", gallery_url):
        gallery_id = m.group(1)
        break
    if not gallery_id:
        return []

    api_url = f"https://nhentai.net/api/v2/galleries/{gallery_id}"
    scraper = _get_scraper()
    try:
        resp = scraper.get(api_url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    media_id = data.get("media_id", "")
    pages_data = data.get("pages", [])
    if not media_id or not pages_data:
        return []

    urls = []
    for p in pages_data:
        num = p.get("number", 0)
        path = p.get("path", "")
        ext = os.path.splitext(path)[1] if path else ".webp"
        if not ext:
            ext = ".webp"
        urls.append((f"https://i.nhentai.net/{path}", num))

    return _download_parallel(urls, download_dir)


def fetch_generic(url, download_dir):
    scraper = _get_scraper()
    try:
        resp = scraper.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return []

    content_type = resp.headers.get("Content-Type", "")
    if "image/" in content_type:
        ext = ".jpg" if "jpeg" in content_type else ".png" if "png" in content_type else ".webp"
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
    scraper = _get_scraper()
    try:
        resp = scraper.get(url, headers=IMG_HEADERS, timeout=30)
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
        key=lambda p: int(re.search(r"page_(\d+)", os.path.basename(p)).group(1))
        if re.search(r"page_(\d+)", os.path.basename(p)) else 0,
    )
