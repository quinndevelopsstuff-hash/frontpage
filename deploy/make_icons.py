"""Regenerate the PWA icons in app/static/icons/ (run once; output is committed).

    .venv/bin/python deploy/make_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "icons"
BG, INK = "#0d0f14", "#ECEFF4"
ACCENTS = ["#FF6B6B", "#A78BFA", "#60A5FA", "#FBBF24"]


def draw_icon(size: int, inset: float) -> Image.Image:
    """A tiny front page: a white masthead bar over four topic-coloured lines.

    inset is the margin as a fraction of the size (maskable icons need more).
    """
    scale = 4  # supersample for smooth edges
    s = size * scale
    img = Image.new("RGB", (s, s), BG)
    draw = ImageDraw.Draw(img)
    left, right = s * inset, s * (1 - inset)
    width = right - left
    top = s * inset
    unit = (s - 2 * top) / 11  # 5 bars + 4 gaps, masthead double height
    draw.rounded_rectangle([left, top, right, top + 2.4 * unit], radius=unit * 0.6, fill=INK)
    lengths = [1.0, 0.72, 0.88, 0.55]
    y = top + 2.4 * unit + unit
    for colour, length in zip(ACCENTS, lengths):
        draw.rounded_rectangle([left, y, left + width * length, y + 1.2 * unit], radius=unit * 0.6, fill=colour)
        y += 1.2 * unit + 0.75 * unit
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_icon(192, 0.20).save(OUT / "icon-192.png")
    draw_icon(512, 0.20).save(OUT / "icon-512.png")
    draw_icon(512, 0.28).save(OUT / "icon-maskable-512.png")
    draw_icon(180, 0.20).save(OUT / "apple-touch-icon.png")
    print(f"icons written to {OUT}")


if __name__ == "__main__":
    main()
