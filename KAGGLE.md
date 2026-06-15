# FacePJ on Kaggle

This is the recommended route for testing FacePJ as an upload/process/download tool instead of a fragile live webcam backend.

## 1. Create a Kaggle Notebook

Use a normal Kaggle notebook. GPU is optional for this version because the current pipeline uses MediaPipe + OpenCV, but enabling GPU will not hurt.

## 2. Clone the repo

Run this in a Kaggle code cell:

```bash
!git clone https://github.com/WinnerSamurai6064/facepj.git
%cd facepj
```

If the repo already exists:

```bash
%cd /kaggle/working/facepj
!git pull
```

## 3. Install Kaggle requirements

```bash
!pip install -r requirements-kaggle.txt
```

If Kaggle asks to restart the session after dependency changes, restart and run from step 2 again.

## 4. Start the Gradio app

```bash
!python kaggle_run.py
```

Kaggle should print a `gradio.live` link. Open that link to use the app.

## One-cell quickstart

```bash
!git clone https://github.com/WinnerSamurai6064/facepj.git
%cd facepj
!pip install -r requirements-kaggle.txt
!python kaggle_run.py
```

## What works best on Kaggle

- Image upload/camera snapshot processing
- Short video processing
- Testing masks and character images
- Exporting processed MP4 files

## Recommended video settings

Start small:

```txt
5–10 seconds
480p or 720p
single face
mask size around 1.25–1.45
```

The Gradio video tab automatically downscales large videos to keep processing reasonable.

## Files added for Kaggle

- `requirements-kaggle.txt` — pinned Kaggle runtime packages
- `kaggle_run.py` — starts Gradio with public sharing enabled
- `app_gradio.py` — main Gradio app

## Notes

This is not a permanent hosting setup. Kaggle is best used as a processing lab. For a public demo, use Hugging Face Spaces. For the final private version, use a private GPU VM or local machine.
