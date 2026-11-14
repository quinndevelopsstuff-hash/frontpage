"""Topic colours: the preset swatches, WCAG contrast maths and card styling.

Every topic has one accent colour (topics.accent_color). The rest of a card's
look is derived from it here, so a new topic needs no extra configuration:
    accent  topic label, match bar, pressed buttons, cover headline
    tint    the small caption on a typographic cover
    grad    the cover / placeholder gradient, darkened towards the page colour
"""
from __future__ import annotations

import re

PAGE_BG = "#0d0f14"
CARD_BG = "#161a22"

# The swatches offered in the UI. Each has at least 4.5:1 contrast (WCAG AA
# for normal text) against the card background; tests/test_topics.py checks it.
PRESET_ACCENTS: list[tuple[str, str]] = [
    ("Coral", "#FF6B6B"),
    ("Violet", "#A78BFA"),
    ("Sky", "#60A5FA"),
    ("Amber", "#FBBF24"),
    ("Teal", "#2DD4BF"),
    ("Pink", "#F472B6"),
    ("Green", "#4ADE80"),
    ("Orange", "#FB923C"),
]
PRESET_HEXES = [hex_ for _, hex_ in PRESET_ACCENTS]

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_hex(value: str) -> bool:
    return bool(_HEX_RE.match(value or ""))


def normalize_hex(value: str) -> str:
    return value.strip().upper()


def _rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(color: str, other: str, weight: float) -> str:
    """weight * color + (1 - weight) * other, as #rrggbb."""
    a, b = _rgb(color), _rgb(other)
    return "#%02x%02x%02x" % tuple(round(x * weight + y * (1 - weight)) for x, y in zip(a, b))


def relative_luminance(hex_: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in _rgb(hex_))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 2 contrast ratio between two colours (1..21)."""
    light, dark = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def cover_gradient(accent: str) -> str:
    """CSS gradient stops for a topic's cover. The first stop keeps the accent
    headline above 3:1 (AA for large text) for every preset."""
    return f"{mix(accent, PAGE_BG, 0.35)}, {mix(accent, PAGE_BG, 0.15)} 60%, {mix(accent, PAGE_BG, 0.05)}"


def topic_style(accent: str | None) -> dict[str, str]:
    """CSS custom-property values for a card, reader or row in this colour."""
    accent = accent if is_hex(accent or "") else "#9AA3B2"
    return {"accent": accent, "tint": mix(accent, "#ffffff", 0.35), "grad": cover_gradient(accent)}
