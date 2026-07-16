import os
import io
import json
import base64
import re
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
DETECT_MODEL = "google/gemini-2.5-flash"
TRANSLATE_MODEL = "deepseek/deepseek-chat"
_font_path = None


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


# ── step 1: Gemini detects text + boxes (cheap, good at spatial) ──

def detect_with_gemini(img_path, api_key):
    ext = os.path.splitext(img_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = (
        "Find ALL Japanese text on this manga page (dialog, narration, margin text). "
        "For each, give exact pixel box [x1,y1,x2,y2] (0,0=top-left). "
        "Return ONLY JSON array: [{\"japanese\":\"...\",\"box\":[x1,y1,x2,y2]},...]"
    )

    body = {
        "model": DETECT_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        }],
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }

    r = requests.post(
        OR_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=90,
    )
    if r.status_code != 200:
        print(f"  Detect HTTP {r.status_code}")
        return None

    content = r.json()["choices"][0]["message"]["content"]
    parsed = _parse_json(content)
    return parsed if isinstance(parsed, list) else parsed.get("texts", []) if isinstance(parsed, dict) else None


# ── step 2: Claude translates (accurate, preserves tone) ──

def translate_with_claude(entries, api_key):
    texts = [e.get("japanese", "").strip() for e in entries if e.get("japanese", "").strip()]
    if not texts:
        return {}

    prompt = (
        "Translate these manga lines to natural English. Preserve tone. "
        "Mark sound effects (onomatopoeia like ぱん/ドキ/ふあ) as sfx with empty translation. "
        "Do not fabricate text.\n"
        "Return ONLY JSON array: [{\"id\":0,\"type\":\"dialog\",\"translation\":\"...\"},...]\n\n"
    )
    for i, t in enumerate(texts):
        prompt += f"{i}: {t}\n"

    body = {
        "model": TRANSLATE_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }

    r = requests.post(
        OR_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if r.status_code != 200:
        print(f"  Translate HTTP {r.status_code}")
        return {}

    content = r.json()["choices"][0]["message"]["content"]
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


# ── step 3: render ──

def overlay(img_bgr, entries, translations):
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_file = _get_font()
    count = 0

    for idx, entry in enumerate(entries):
        en = translations.get(idx, "")
        if not en:
            continue

        box = entry.get("box", [])
        if len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except (ValueError, TypeError):
            continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 + 6 or y2 <= y1 + 6:
            continue

        box_pad = max(14, int((y2 - y1) * 0.4))
        bx1, by1 = max(0, x1 - box_pad), max(0, y1 - box_pad)
        bx2, by2 = min(w, x2 + box_pad), min(h, y2 + box_pad)
        bw, bh = bx2 - bx1, by2 - by1

        radius = max(5, int(min(bw, bh) * 0.1))
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=radius, fill=(255, 255, 255), outline=(170, 170, 170))

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
    oh, ow = img.shape[:2]

    resized = cv2.resize(img, (700, int(oh * 700 / ow)), interpolation=cv2.INTER_AREA)
    tmp = image_path + ".tmp.jpg"
    cv2.imwrite(tmp, resized, [cv2.IMWRITE_JPEG_QUALITY, 60])

    print("    Detecting (Gemini)...", end="", flush=True)
    entries = detect_with_gemini(tmp, api_key)
    os.remove(tmp)

    if entries is None:
        return False
    if not entries:
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True

    print(f" {len(entries)} regions")

    sx = ow / resized.shape[1]
    sy = oh / resized.shape[0]
    for e in entries:
        box = e.get("box", [])
        if len(box) == 4:
            e["box"] = [int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)]

    print("    Translating (Claude)...", end="", flush=True)
    translations = translate_with_claude(entries, api_key)
    print(f" {len(translations)} translations")

    result, count = overlay(img, entries, translations)
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return count > 0
