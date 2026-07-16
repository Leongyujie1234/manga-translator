import os
import io
import json
import base64
import re
import time
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OR_MODEL = "openai/gpt-4o-mini"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

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


def resize_for_api(img_bgr, max_width=900):
    h, w = img_bgr.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_w = max_width
        new_h = int(h * ratio)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr


def call_vision_api(img_path, api_key, prompt_override=None):
    ext = os.path.splitext(img_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = prompt_override or (
        "You are translating a manga page. Find ALL dialog and narration text (inside speech bubbles and boxes). "
        "IGNORE sound effects/onomatopoeia (single kana, repeated sounds like ぱん, どきどき, ふあっ). "
        "Return ONLY a JSON array of objects with:\n"
        '- "japanese": the exact Japanese text\n'
        '- "english": natural English translation (keep tone, intent)\n'
        '- "box": [x1, y1, x2, y2] pixel coordinates (0,0 top-left)\n'
        "Return: [{\"japanese\":\"...\",\"english\":\"...\",\"box\":[x1,y1,x2,y2]}, ...]"
    )

    body = {
        "model": OR_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        }],
        "max_tokens": 4096,
    }

    resp = requests.post(
        OR_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=90,
    )

    if resp.status_code != 200:
        print(f"[API] HTTP {resp.status_code}: {resp.text[:200]}")
        return None, None

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    cost = data.get("usage", {}).get("total_tokens", 0)

    entries = []
    try:
        parsed = json.loads(content)
        entries = parsed if isinstance(parsed, list) else parsed.get("texts", [])
    except json.JSONDecodeError:
        for match in re.finditer(
            r'\{(?:[^{}]|\{[^{}]*\})*\}', content
        ):
            try:
                entry = json.loads(match.group())
                if "english" in entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries, cost


def overlay_translations(img_bgr, entries):
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_file = _get_font()

    for entry in entries:
        en = entry.get("english", "").strip()
        box = entry.get("box", [])
        if not en or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except (ValueError, TypeError):
            continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            continue

        pad = max(10, int((y2 - y1) * 0.3))
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)
        bw, bh = bx2 - bx1, by2 - by1

        radius = max(4, int(min(bw, bh) * 0.12))
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=radius, fill=(255, 255, 255), outline=(200, 200, 200))

        fsize = max(14, min(32, int(bh * 0.45)))
        if font_file:
            try:
                font = ImageFont.truetype(font_file, fsize)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        lines = _wrap_text(en, font, bw - 10, draw)
        lh = int(fsize * 1.2)
        total_h = len(lines) * lh
        ty = by1 + max(5, (bh - total_h) // 2)

        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            lw = bb[2] - bb[0]
            draw.text((bx1 + (bw - lw) // 2, ty), line, fill=(0, 0, 0), font=font)
            ty += lh

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_single(image_path, output_path, api_key):
    img = cv2.imread(image_path)
    if img is None:
        return False

    resized = resize_for_api(img, 900)
    temp_path = image_path + ".resized.jpg"
    cv2.imwrite(temp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 75])

    entries, _ = call_vision_api(temp_path, api_key)
    os.remove(temp_path)

    if entries is None:
        return False

    if not entries:
        cv2.imwrite(output_path, img)
        return True

    scale_x = img.shape[1] / resized.shape[1]
    scale_y = img.shape[0] / resized.shape[0]

    for entry in entries:
        box = entry.get("box", [])
        if len(box) == 4:
            entry["box"] = [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y),
            ]

    result = overlay_translations(img, entries)
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True
