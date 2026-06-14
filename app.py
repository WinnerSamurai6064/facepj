import base64
import io
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageDraw

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

active_mask_rgba: Optional[np.ndarray] = None


def make_default_mask(size: int = 768) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((105, 55, size - 105, size - 55), fill=(255, 122, 26, 205))
    draw.ellipse((250, 300, 305, 338), fill=(0, 0, 0, 110))
    draw.ellipse((463, 300, 518, 338), fill=(0, 0, 0, 110))
    draw.ellipse((330, 470, 438, 510), fill=(0, 0, 0, 135))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


active_mask_rgba = make_default_mask()


def decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame")
    return frame


def encode_jpeg_data_url(frame_bgr: np.ndarray, quality: int = 74) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode frame")
    payload = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def read_upload_as_mask(file_storage) -> np.ndarray:
    raw = file_storage.read()
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    img.thumbnail((900, 900), Image.LANCZOS)

    canvas = Image.new("RGBA", (900, 900), (0, 0, 0, 0))
    x = (900 - img.width) // 2
    y = (900 - img.height) // 2
    canvas.alpha_composite(img, (x, y))

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGRA)


def lm_xy(landmarks, index: int, width: int, height: int) -> Tuple[int, int]:
    lm = landmarks[index]
    return int(lm.x * width), int(lm.y * height)


def overlay_bgra(base_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = overlay_bgra.shape[:2]
    base_h, base_w = base_bgr.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(base_w, x + w)
    y2 = min(base_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return base_bgr

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_crop = overlay_bgra[oy1:oy2, ox1:ox2]
    rgb = overlay_crop[:, :, :3].astype(np.float32)
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0

    region = base_bgr[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * rgb + (1.0 - alpha) * region
    base_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)
    return base_bgr


def rotate_and_resize_mask(mask_bgra: np.ndarray, width: int, height: int, angle_degrees: float) -> np.ndarray:
    resized = cv2.resize(mask_bgra, (max(2, width), max(2, height)), interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(rot[0, 0])
    sin = abs(rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot[0, 2] += (new_w / 2) - center[0]
    rot[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(resized, rot, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def process_frame(frame_bgr: np.ndarray, mask_scale: float = 1.35) -> Tuple[np.ndarray, bool]:
    global active_mask_rgba

    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return frame_bgr, False

    landmarks = result.multi_face_landmarks[0].landmark

    forehead = lm_xy(landmarks, 10, width, height)
    chin = lm_xy(landmarks, 152, width, height)
    left_side = lm_xy(landmarks, 234, width, height)
    right_side = lm_xy(landmarks, 454, width, height)
    nose = lm_xy(landmarks, 1, width, height)
    upper_lip = lm_xy(landmarks, 13, width, height)
    lower_lip = lm_xy(landmarks, 14, width, height)
    mouth_left = lm_xy(landmarks, 61, width, height)
    mouth_right = lm_xy(landmarks, 291, width, height)

    face_w = max(80, int(abs(right_side[0] - left_side[0]) * mask_scale))
    face_h = max(110, int(abs(chin[1] - forehead[1]) * mask_scale * 1.22))

    mouth_open = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)) / max(1, face_h)
    mouth_width = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right)) / max(1, face_w)

    stretch_x = 1.0 + min(0.08, mouth_width * 0.09)
    stretch_y = 1.0 + min(0.10, mouth_open * 0.85)
    face_w = int(face_w * stretch_x)
    face_h = int(face_h * stretch_y)

    angle = math.degrees(math.atan2(right_side[1] - left_side[1], right_side[0] - left_side[0]))
    warped_mask = rotate_and_resize_mask(active_mask_rgba, face_w, face_h, angle)

    mh, mw = warped_mask.shape[:2]
    x = int(nose[0] - mw / 2)
    y = int(nose[1] - mh / 2 + face_h * 0.03)

    out = overlay_bgra(frame_bgr.copy(), warped_mask, x, y)

    # Old-school mouth cue on top of static masks.
    mx = int((mouth_left[0] + mouth_right[0]) / 2)
    my = int((upper_lip[1] + lower_lip[1]) / 2)
    rx = max(5, int(face_w * 0.075))
    ry = max(3, int(face_h * (0.012 + min(0.055, mouth_open * 0.75))))
    cv2.ellipse(out, (mx, my), (rx, ry), angle, 0, 360, (8, 8, 8), -1, cv2.LINE_AA)

    return out, True


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "engine": "python-mediapipe-opencv"})


@app.route("/api/mask", methods=["POST"])
def upload_mask():
    global active_mask_rgba
    file = request.files.get("mask")
    if not file:
        return jsonify({"ok": False, "error": "missing mask file"}), 400
    active_mask_rgba = read_upload_as_mask(file)
    return jsonify({"ok": True})


@app.route("/api/process", methods=["POST"])
def process():
    payload = request.get_json(force=True)
    frame_data = payload.get("image")
    mask_scale = float(payload.get("maskScale", 1.35))

    if not frame_data:
        return jsonify({"ok": False, "error": "missing image"}), 400

    frame = decode_data_url(frame_data)
    processed, found = process_frame(frame, mask_scale=mask_scale)
    return jsonify({"ok": True, "found": found, "image": encode_jpeg_data_url(processed)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, threaded=True)
