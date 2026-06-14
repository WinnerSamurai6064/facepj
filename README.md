# FacePJ

Old-fashioned browser-first live face masking experiment.

FacePJ uses MediaPipe face landmarks in the browser to place an uploaded image over the live camera feed. It can track basic head rotation, mouth opening, smile movement, and cheek movement cues, then record the combined canvas locally.

This is intended for consent-based avatar/filter experiments and private hackathon testing, not impersonation.

## What runs where

- **Browser:** camera, face tracking, uploaded mask overlay, recording, local save.
- **Python:** optional tiny helper to prepare a PNG mask image before using it in the browser.
- **Hugging Face:** static/Docker-hosted web demo only.

## Local web test

```bash
npm install
npm run dev
```

Open the local Vite URL, allow camera permission, click **Start camera**, then upload a PNG/JPG/WebP mask image.

## Prepare a mask image with Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/prepare_mask.py input.jpg public/mask.png --oval
```

The browser uploader works without this step. The Python script is only for pre-cutting/resizing a cleaner PNG asset.

## Hugging Face Space

Create a **Docker Space**, connect this repo, and Hugging Face will run the included `Dockerfile` on port `7860`.

## Notes

- No LLM is used.
- No server-side video processing is required for the MVP.
- Best performance comes from keeping tracking in the browser.
- The first version uses fast Canvas rendering. A later version can upgrade to Three.js/WebGL face-mesh texture warping.
