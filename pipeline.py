import os
import io
import json
import sys
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import torch

torch.set_num_threads(6)
torch.set_num_interop_threads(4)
cv2.setNumThreads(2)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TRANSLATE_MODEL = "deepseek/deepseek-chat"

_eocr = None
_mocr = None
_font_path = None


def _get_reader():
    global _eocr
    if _eocr is None:
        _eocr = easyocr.Reader(["ja"], gpu=False, verbose=False)
    return _eocr


def _get_mocr():
    global _mocr
    if _mocr is None:
        import manga_ocr
        _mocr = manga_ocr.MangaOcr()
    return _mocr


def _get_font():
    global _font_path
    if _font_path is None:
        for p in ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\msgothic.ttc"]:
            if os.path.exists(p):
                _font_path = p
                break
    return _font_path


def _wrap_text(text, font, max_w, draw):
    lines = []
    for word in text.split(" "):
        if not lines:
            lines.append(word)
            continue
        test = lines[-1] + " " + word
        bb = draw.textbbox((0, 0), test, font=font)
        if (bb[2] - bb[0]) <= max_w:
            lines[-1] = test
        else:
            lines.append(word)
    return lines


# ── step 1: local detection (EasyOCR + Manga OCR) ──

def detect_local(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(w, h) > 1200:
        scale = 1200 / max(w, h)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    reader = _get_reader()
    detections = reader.readtext(small)

    if not detections:
        return []

    mocr = _get_mocr()
    inv_scale = 1.0 / scale
    boxes = []
    jp_texts = []

    for bbox, _, _ in detections:
        pts = np.array(bbox, dtype=np.float32)
        x = int(np.min(pts[:, 0]) * inv_scale)
        y = int(np.min(pts[:, 1]) * inv_scale)
        x2 = int(np.max(pts[:, 0]) * inv_scale)
        y2 = int(np.max(pts[:, 1]) * inv_scale)
        rw, rh = x2 - x, y2 - y
        if rw < 8 or rh < 6:
            continue

        pad = 4
        cx, cy = max(0, x - pad), max(0, y - pad)
        cw = min(rw + pad * 2, w - cx)
        ch = min(rh + pad * 2, h - cy)
        crop = img_bgr[cy: cy + ch, cx: cx + cw]
        if crop.size == 0:
            continue

        try:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            text = mocr(pil_crop).strip()
        except Exception:
            continue
        if not text:
            continue

        boxes.append((x, y, rw, rh))
        jp_texts.append(text)

    return boxes, jp_texts


# ── step 2: translate with DeepSeek ──

def translate_texts(api_key, jp_texts):
    if not jp_texts:
        return {}

    items = []
    for t in jp_texts:
        items.append({"id": len(items), "japanese": t.strip()})

    prompt = (
        "Translate these manga lines to natural English. Preserve tone. "
        "Mark sound effects (onomatopoeia like ぱん/ドキ/ふあ) as sfx with empty translation. "
        "Return ONLY JSON array: [{\"id\":0,\"type\":\"dialog\",\"translation\":\"...\"},...]\n\n"
    )
    for item in items:
        prompt += f"{item['id']}: {item['japanese']}\n"

    body = {
        "model": TRANSLATE_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 4096,
    }

    resp = requests.post(
        OR_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"  Translate HTTP {resp.status_code}")
        return {}

    content = resp.json()["choices"][0]["message"]["content"]
    parsed = _parse_json(content)

    result = {}
    if isinstance(parsed, list):
        for item in parsed:
            idx = item.get("id")
            if item.get("type") != "sfx" and item.get("translation", "").strip():
                result[idx] = item["translation"].strip()
    return result


def _parse_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            s = content.index("[")
            e = content.rindex("]") + 1
            return json.loads(content[s:e])
        except (ValueError, json.JSONDecodeError):
            return None


# ── step 3: overlay ──

def overlay(img_bgr, boxes, translations, jp_texts):
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_file = _get_font()
    count = 0

    for idx, (x, y, rw, rh) in enumerate(boxes):
        en = translations.get(idx, "")
        if not en:
            continue

        box_pad = max(14, int(rh * 0.4))
        bx1, by1 = max(0, x - box_pad), max(0, y - box_pad)
        bx2, by2 = min(w, x + rw + box_pad), min(h, y + rh + box_pad)
        bw, bh = bx2 - bx1, by2 - by1

        radius = max(5, int(min(bw, bh) * 0.1))
        draw.rounded_rectangle(
            [bx1, by1, bx2, by2], radius=radius, fill=(255, 255, 255), outline=(170, 170, 170)
        )

        fsize = max(22, min(48, int(bh * 0.65)))
        if font_file:
            try:
                font = ImageFont.truetype(font_file, fsize)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        lines = _wrap_text(en, font, bw - 12, draw)
        lh = int(fsize * 1.15)
        total_h = len(lines) * lh
        ty = by1 + max(5, (bh - total_h) // 2)

        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            lw = bb[2] - bb[0]
            draw.text((bx1 + (bw - lw) // 2, ty), line, fill=(0, 0, 0), font=font)
            ty += lh

        count += 1

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), count


def process_single(image_path, output_path, api_key):
    img = cv2.imread(image_path)
    if img is None:
        return False

    print("    OCR...", end="", flush=True)
    boxes, jp_texts = detect_local(img)
    print(f" {len(boxes)} regions")

    if not boxes:
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True

    print("    Translate (DeepSeek)...", end="", flush=True)
    translations = translate_texts(api_key, jp_texts)
    print(f" {len(translations)} translations")

    result, count = overlay(img, boxes, translations, jp_texts)
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return count > 0
