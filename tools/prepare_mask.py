#!/usr/bin/env python3
"""
Tiny mask-prep helper for FacePJ.

This is not used for live tracking. The browser does that.
Use this only when you want to turn a character image into a lightweight PNG mask.

Examples:
  python tools/prepare_mask.py input.jpg public/mask.png
  python tools/prepare_mask.py input.jpg public/mask.png --size 768 --oval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFilter


def fit_image(img: Image.Image, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.alpha_composite(img, (x, y))
    return canvas


def apply_soft_oval_alpha(img: Image.Image, feather: int) -> Image.Image:
    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    pad_x = int(width * 0.09)
    pad_y = int(height * 0.045)
    draw.ellipse((pad_x, pad_y, width - pad_x, height - pad_y), fill=255)

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    existing_alpha = out.getchannel("A")
    combined = ImageChops_multiply(existing_alpha, mask)
    out.putalpha(combined)
    return out


def ImageChops_multiply(a: Image.Image, b: Image.Image) -> Image.Image:
    # Avoid importing the whole ImageChops namespace in older tiny Python envs.
    from PIL import ImageChops

    return ImageChops.multiply(a, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a FacePJ PNG mask image.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--size", type=int, default=768, help="Square output size in pixels")
    parser.add_argument("--oval", action="store_true", help="Apply a soft oval alpha mask")
    parser.add_argument("--feather", type=int, default=18, help="Oval edge blur amount")

    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(args.input) as img:
      prepared = fit_image(img, args.size)
      if args.oval:
          prepared = apply_soft_oval_alpha(prepared, args.feather)
      prepared.save(args.output, "PNG", optimize=True)

    print(f"Saved mask: {args.output}")


if __name__ == "__main__":
    main()
