# FacePJ

Old-fashioned Python-powered live face masking experiment.

FacePJ keeps the custom browser UI, but routes camera frames to a small Python Flask backend. Python uses MediaPipe face landmarks plus OpenCV to place an uploaded image over the face, follow basic head rotation, fake mouth movement, and return the processed frame to the UI. The browser records the processed canvas locally.

This is intended for consent-based avatar/filter experiments and private hackathon testing, not impersonation.

## What runs where

- **Browser UI:** camera access, mask upload button, processed preview, recording, local save.
- **Python backend:** frame processing, MediaPipe face landmarks, OpenCV mask placement.
- **Hugging Face Docker Space:** runs the Flask app on port `7860`.
- **No LLM:** no language model is used.

## Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open:

```txt
http://localhost:7860
```

Allow camera permission, click **Start camera**, then upload a PNG/JPG/WebP mask image.

## Hugging Face Space

Create a **Docker Space**, connect this repo, and Hugging Face will run the included `Dockerfile` on port `7860`.

## Optional mask prep helper

```bash
python tools/prepare_mask.py input.jpg public/mask.png --oval
```

The browser uploader works without this step. The helper is only for pre-cutting/resizing a cleaner PNG asset.

## Notes

- This version is heavier than browser-only because video frames travel to Python and back.
- It is more aligned with an old-fashioned Python/OpenCV hackathon workflow.
- Keep camera resolution modest for Hugging Face free tier: 480x360 or lower.
- The current version uses 2D image placement, not full 3D face texture warping.
