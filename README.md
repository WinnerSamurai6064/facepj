# FacePJ

Old-fashioned Python-powered face masking experiment.

FacePJ now has two routes:

1. **Gradio Mask Lab** — recommended for Hugging Face/Kaggle upload-process-download testing.
2. **Flask live-frame demo** — experimental browser camera route that streams frames to Python.

The Gradio route is more stable because it avoids fragile live frame streaming. It supports image/camera capture, character mask upload, short video upload, and MP4 output.

This is intended for consent-based avatar/filter experiments and private hackathon testing, not impersonation.

## Recommended: Gradio app

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app_gradio.py
```

Open:

```txt
http://localhost:7860
```

## Kaggle quickstart

See [`KAGGLE.md`](KAGGLE.md) for the full setup.

One-cell Kaggle version:

```bash
!git clone https://github.com/WinnerSamurai6064/facepj.git
%cd facepj
!pip install -r requirements-kaggle.txt
!python kaggle_run.py
```

Kaggle should print a `gradio.live` link. Open that link to use the app.

## Hugging Face Space

Create a **Docker Space**, connect this repo, and Hugging Face will run the included `Dockerfile` on port `7860`.

## Experimental Flask live demo

```bash
python app.py
```

Then open:

```txt
http://localhost:7860
```

## Optional mask prep helper

```bash
python tools/prepare_mask.py input.jpg public/mask.png --oval
```

The browser/Gradio uploader works without this step. The helper is only for pre-cutting/resizing a cleaner PNG asset.

## Notes

- No LLM is used.
- The current version uses 2D image placement, not full 3D face texture warping.
- Kaggle is best for short clips and experiments.
- Hugging Face is best for a public demo.
- A private GPU VM is best for the eventual heavier AI route.
