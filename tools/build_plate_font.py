"""Build a local Taiwan plate TrueType font from the checked-in reference images."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "assets" / "spec_images"
OUTPUT_FONT_PATH = (
    REPOSITORY_ROOT / "assets" / "local" / "fonts" / "TaiwanPlate-Regular.ttf"
)
VERIFICATION_PATH = REPOSITORY_ROOT / "out" / "user_preview" / "taiwan_plate_font_sheet.png"

GRID_LAYOUT = [
    (0, 0, ["A", "B", "C", "D", "E"]),
    (0, 1, ["F", "G", "H", "I", "J"]),
    (1, 0, ["K", "L", "M", "N", "O"]),
    (1, 1, ["P", "Q", "R", "S", "T"]),
    (2, 0, ["1", "2", "3", "4", "5"]),
    (2, 1, ["6", "7", "8", "9", "0"]),
    (3, 0, ["U", "V", "W", "X", "Y"]),
    (3, 1, ["Z"]),
]

COL_BOUNDS = [
    (0, 360),
    (360, 770),
    (770, 1180),
    (1180, 1590),
    (1590, 1950),
]


def clean_extract_cell(cell: np.ndarray, char_hint: str | None = None, k_size: int = 7) -> np.ndarray | None:
    """Extract a solid binary mask from the scanned outline image cell."""
    _, bin_c = cv2.threshold(cell, 210, 255, cv2.THRESH_BINARY_INV)

    # Character D in the official scan has an open scan gap at the top right of its inner hole
    if char_hint == "D":
        cv2.line(bin_c, (150, 76), (175, 85), 255, 4)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed_stroke = cv2.morphologyEx(bin_c, cv2.MORPH_CLOSE, kernel)

    if char_hint == "D":
        contours, _ = cv2.findContours(closed_stroke, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        large_c = sorted([c for c in contours if cv2.contourArea(c) > 5000], key=lambda c: cv2.contourArea(c), reverse=True)
        if len(large_c) >= 3:
            mask = np.zeros_like(cell)
            cv2.drawContours(mask, [large_c[0]], -1, 255, -1)
            cv2.drawContours(mask, [large_c[2]], -1, 0, -1)
            ys, xs = np.where(mask == 255)
            if len(ys) > 0:
                return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    inv = cv2.bitwise_not(closed_stroke)
    inv[0, :] = 255
    inv[-1, :] = 255
    inv[:, 0] = 255
    inv[:, -1] = 255
    filled = inv.copy()
    mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    res = np.zeros_like(cell)
    res[labels == largest_label] = 255
    ys, xs = np.where(res == 255)
    if len(ys) == 0:
        return None
    return res[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def polygon_area_oriented(pts: np.ndarray) -> float:
    """Shoelace formula: positive for CCW, negative for CW in standard Cartesian."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def glyph_mask_to_ttglyph(
    mask: np.ndarray,
    cap_height: float = 700.0,
    lsb: float = 40.0,
) -> tuple[object, int]:
    """Convert binary glyph mask to a TrueType glyph with proper winding direction."""
    pad = 4
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    h_img, w_img = mask.shape
    scale = cap_height / float(h_img)

    contours, hierarchy = cv2.findContours(
        padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    pen = TTGlyphPen(None)
    if not contours or hierarchy is None:
        return pen.glyph(), int(w_img * scale + 2 * lsb)

    hier = hierarchy[0]
    for i, c in enumerate(contours):
        approx = cv2.approxPolyDP(c, epsilon=0.5, closed=True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        x_font = (pts[:, 0] - pad) * scale + lsb
        y_font = (h_img - (pts[:, 1] - pad)) * scale
        font_pts = np.column_stack([x_font, y_font])

        is_hole = (hier[i][3] != -1)
        area = polygon_area_oriented(font_pts)

        # TrueType winding rule: outer contour must be CW (area < 0), hole CCW (area > 0)
        if is_hole:
            if area < 0:
                font_pts = font_pts[::-1]
        else:
            if area > 0:
                font_pts = font_pts[::-1]

        pen.moveTo(tuple(font_pts[0]))
        for pt in font_pts[1:]:
            pen.lineTo(tuple(pt))
        pen.closePath()

    advance_width = int(round(w_img * scale + 2 * lsb))
    return pen.glyph(), advance_width


def build_font() -> None:
    """Extract all characters and build TaiwanPlate-Regular.ttf."""
    print("Extracting glyph masks from official spec images...")
    masks: dict[str, np.ndarray] = {}

    for p_idx, r_idx, chars in GRID_LAYOUT:
        img_path = SPEC_DIR / f"page_{p_idx}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        y1 = 0 if r_idx == 0 else int(h * 0.5)
        y2 = int(h * 0.5) if r_idx == 0 else h
        for c_idx, char in enumerate(chars):
            x1, x2 = COL_BOUNDS[c_idx]
            cell = img[y1:y2, x1 : min(w, x2)]
            mask = clean_extract_cell(cell, char_hint=char, k_size=7)
            if mask is None:
                raise RuntimeError(f"Failed to extract glyph for '{char}' from page {p_idx}")
            masks[char] = mask

    print(f"Extracted {len(masks)} glyphs.")

    # TrueType Font construction
    fb = FontBuilder(unitsPerEm=1000, isTTF=True)

    glyph_names = [".notdef", "space", "hyphen"]
    glyph_names += [f"uni{ord(c):04X}" for c in sorted(masks.keys())]

    fb.setupGlyphOrder(glyph_names)

    cmap: dict[int, str] = {
        ord(" "): "space",
        ord("-"): "hyphen",
    }
    for c in masks.keys():
        cmap[ord(c)] = f"uni{ord(c):04X}"
    fb.setupCharacterMap(cmap)

    glyphs: dict[str, object] = {}
    metrics: dict[str, tuple[int, int]] = {}

    # .notdef
    pen_notdef = TTGlyphPen(None)
    pen_notdef.moveTo((50, 0))
    pen_notdef.lineTo((50, 700))
    pen_notdef.lineTo((400, 700))
    pen_notdef.lineTo((400, 0))
    pen_notdef.closePath()
    glyphs[".notdef"] = pen_notdef.glyph()
    metrics[".notdef"] = (450, 50)

    # space
    pen_space = TTGlyphPen(None)
    glyphs["space"] = pen_space.glyph()
    metrics["space"] = (250, 0)

    # hyphen '-'
    pen_hyphen = TTGlyphPen(None)
    hyph_w = 160
    hyph_h = 55
    hyph_y = 350 - hyph_h // 2
    hyph_x = 40
    # TrueType outer contour must be clockwise: (x, y) -> (x, y+h) -> (x+w, y+h) -> (x+w, y)
    pen_hyphen.moveTo((hyph_x, hyph_y))
    pen_hyphen.lineTo((hyph_x, hyph_y + hyph_h))
    pen_hyphen.lineTo((hyph_x + hyph_w, hyph_y + hyph_h))
    pen_hyphen.lineTo((hyph_x + hyph_w, hyph_y))
    pen_hyphen.closePath()
    glyphs["hyphen"] = pen_hyphen.glyph()
    metrics["hyphen"] = (hyph_w + 2 * hyph_x, hyph_x)

    # Convert all alphanumeric glyphs
    for char, mask in masks.items():
        g_name = f"uni{ord(char):04X}"
        glyph, adv = glyph_mask_to_ttglyph(mask, cap_height=700.0, lsb=40.0)
        glyphs[g_name] = glyph
        metrics[g_name] = (adv, 40)

    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)

    name_strings = {
        "familyName": "TaiwanPlate",
        "styleName": "Regular",
        "uniqueFontIdentifier": "TaiwanPlate-Regular:1.0",
        "fullName": "TaiwanPlate Regular",
        "psName": "TaiwanPlate-Regular",
        "version": "Version 1.0",
    }
    fb.setupNameTable(name_strings)
    fb.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        usWinAscent=800,
        usWinDescent=200,
    )
    fb.setupPost()

    OUTPUT_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(OUTPUT_FONT_PATH))
    print(f"Saved font to {OUTPUT_FONT_PATH}")

    # Generate verification sheet
    render_verification_sheet()


def render_verification_sheet() -> None:
    """Render verification test sheet showing all glyphs and sample plates."""
    VERIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    font_size = 64
    font = ImageFont.truetype(str(OUTPUT_FONT_PATH), font_size)

    img = Image.new("RGB", (1200, 600), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    lines = [
        "0 1 2 3 4 5 6 7 8 9",
        "A B C D E F G H I J K L M",
        "N O P Q R S T U V W X Y Z -",
        "XHU-013   HJ9-037   AQ-560   LAB-6531",
    ]

    y = 50
    for line in lines:
        draw.text((50, y), line, font=font, fill=(20, 20, 20))
        y += 120

    img.save(str(VERIFICATION_PATH))
    print(f"Saved verification sheet to {VERIFICATION_PATH}")


if __name__ == "__main__":
    build_font()
