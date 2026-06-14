import math
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except Exception:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)


def make_default_mask(size: int = 768) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((105, 55, size - 105, size - 55), fill=(255, 122, 26, 210))
    draw.ellipse((250, 300, 305, 338), fill=(0, 0, 0, 120))
    draw.ellipse((463, 300, 518, 338), fill=(0, 0, 0, 120))
    draw.ellipse((330, 470, 438, 510), fill=(0, 0, 0, 150))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def pil_to_bgra_mask(mask_image: Optional[Image.Image]) -> np.ndarray:
    if mask_image is None:
        return make_default_mask()

    img = mask_image.convert("RGBA")
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

    crop = overlay_bgra[oy1:oy2, ox1:ox2]
    rgb = crop[:, :, :3].astype(np.float32)
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    region = base_bgr[y1:y2, x1:x2].astype(np.float32)
    base_bgr[y1:y2, x1:x2] = (alpha * rgb + (1.0 - alpha) * region).astype(np.uint8)
    return base_bgr


def rotate_and_resize_mask(mask_bgra: np.ndarray, width: int, height: int, angle_degrees: float) -> np.ndarray:
    width = max(2, min(int(width), 1400))
    height = max(2, min(int(height), 1600))
    resized = cv2.resize(mask_bgra, (width, height), interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(rot[0, 0])
    sin = abs(rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot[0, 2] += (new_w / 2) - center[0]
    rot[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        resized,
        rot,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def process_frame_bgr(frame_bgr: np.ndarray, mask_bgra: np.ndarray, mask_scale: float) -> Tuple[np.ndarray, bool]:
    mask_scale = max(0.7, min(float(mask_scale), 2.0))
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return frame_bgr, False

    landmarks = result.multi_face_landmarks[0].landmark
    forehead = lm_xy(landmarks, 10, w, h)
    chin = lm_xy(landmarks, 152, w, h)
    left_side = lm_xy(landmarks, 234, w, h)
    right_side = lm_xy(landmarks, 454, w, h)
    nose = lm_xy(landmarks, 1, w, h)
    upper_lip = lm_xy(landmarks, 13, w, h)
    lower_lip = lm_xy(landmarks, 14, w, h)
    mouth_left = lm_xy(landmarks, 61, w, h)
    mouth_right = lm_xy(landmarks, 291, w, h)

    face_w = max(60, int(abs(right_side[0] - left_side[0]) * mask_scale))
    face_h = max(80, int(abs(chin[1] - forehead[1]) * mask_scale * 1.22))

    mouth_open = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)) / max(1, face_h)
    mouth_width = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right)) / max(1, face_w)
    face_w = int(face_w * (1.0 + min(0.08, mouth_width * 0.09)))
    face_h = int(face_h * (1.0 + min(0.10, mouth_open * 0.85)))

    angle = math.degrees(math.atan2(right_side[1] - left_side[1], right_side[0] - left_side[0]))
    warped_mask = rotate_and_resize_mask(mask_bgra, face_w, face_h, angle)

    mh, mw = warped_mask.shape[:2]
    x = int(nose[0] - mw / 2)
    y = int(nose[1] - mh / 2 + face_h * 0.03)

    out = overlay_bgra(frame_bgr.copy(), warped_mask, x, y)

    mx = int((mouth_left[0] + mouth_right[0]) / 2)
    my = int((upper_lip[1] + lower_lip[1]) / 2)
    rx = max(4, int(face_w * 0.075))
    ry = max(2, int(face_h * (0.012 + min(0.055, mouth_open * 0.75))))
    cv2.ellipse(out, (mx, my), (rx, ry), angle, 0, 360, (8, 8, 8), -1, cv2.LINE_AA)
    return out, True


def process_image(input_image: Image.Image, mask_image: Optional[Image.Image], mask_scale: float):
    if input_image is None:
        return None, "Upload or capture an image first."

    try:
        frame_rgb = np.array(input_image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        mask_bgra = pil_to_bgra_mask(mask_image)
        out_bgr, found = process_frame_bgr(frame_bgr, mask_bgra, mask_scale)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        status = "Face locked — mask applied." if found else "No face found — returned original image."
        return Image.fromarray(out_rgb), status
    except Exception as exc:
        traceback.print_exc()
        return None, f"Processing failed: {exc}"


def process_video(video_path: str, mask_image: Optional[Image.Image], mask_scale: float, max_seconds: int):
    if not video_path:
        return None, "Upload a video first."

    try:
        mask_bgra = pil_to_bgra_mask(mask_image)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video."

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        fps = min(max(fps, 1), 30)
        total_limit = int(fps * max_seconds)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = min(1.0, 640 / max(width, height))
        out_w = max(2, int(width * scale))
        out_h = max(2, int(height * scale))

        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )

        count = 0
        found_count = 0
        while count < total_limit:
            ok, frame = cap.read()
            if not ok:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            processed, found = process_frame_bgr(frame, mask_bgra, mask_scale)
            found_count += int(found)
            writer.write(processed)
            count += 1

        cap.release()
        writer.release()

        status = f"Processed {count} frames. Face detected in {found_count} frames."
        return output_path, status
    except Exception as exc:
        traceback.print_exc()
        return None, f"Video processing failed: {exc}"


CSS = """
body, .gradio-container { background: #050505 !important; color: #f7f7f7 !important; }
.gradio-container { max-width: 1100px !important; margin: auto !important; }
#title { text-align: center; }
"""

with gr.Blocks(css=CSS, title="FacePJ Gradio") as demo:
    gr.Markdown(
        """
        # FacePJ — Gradio Mask Lab
        Upload/capture a face image or upload a short video, then apply a character mask.
        This route is designed for Hugging Face/Kaggle testing instead of fragile live webcam streaming.
        """,
        elem_id="title",
    )

    with gr.Tab("Image / Camera"):
        with gr.Row():
            input_image = gr.Image(label="Face image / camera capture", sources=["upload", "webcam"], type="pil")
            mask_image = gr.Image(label="Mask / character image", sources=["upload"], type="pil")
        mask_scale = gr.Slider(0.8, 1.8, value=1.35, step=0.05, label="Mask size")
        image_button = gr.Button("Apply mask", variant="primary")
        output_image = gr.Image(label="Processed image", type="pil")
        image_status = gr.Textbox(label="Status", interactive=False)
        image_button.click(process_image, [input_image, mask_image, mask_scale], [output_image, image_status])

    with gr.Tab("Video"):
        gr.Markdown("Keep videos short on free CPU. Start with 5–10 seconds at 480p/720p.")
        input_video = gr.Video(label="Target video")
        video_mask = gr.Image(label="Mask / character image", sources=["upload"], type="pil")
        video_scale = gr.Slider(0.8, 1.8, value=1.35, step=0.05, label="Mask size")
        max_seconds = gr.Slider(2, 20, value=8, step=1, label="Max seconds to process")
        video_button = gr.Button("Process video", variant="primary")
        output_video = gr.Video(label="Processed MP4")
        video_status = gr.Textbox(label="Status", interactive=False)
        video_button.click(process_video, [input_video, video_mask, video_scale, max_seconds], [output_video, video_status])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860)
