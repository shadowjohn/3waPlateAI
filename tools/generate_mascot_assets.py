"""Generate banner, avatar, and loading assets from the mascot image using Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def remove_white_background(img: Image.Image, tolerance: int = 15) -> Image.Image:
    """Convert white background to transparent RGBA."""
    rgba = img.convert("RGBA")
    data = np.array(rgba)
    
    # White background mask (r, g, b > 255 - tolerance)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    mask = (r > (255 - tolerance)) & (g > (255 - tolerance)) & (b > (255 - tolerance))
    
    # Soft alpha feathering on the edges
    data[:, :, 3] = np.where(mask, 0, a)
    out = Image.fromarray(data)
    return out

def create_circular_avatar(mascot: Image.Image, size: int = 256) -> Image.Image:
    """Create a circular framed avatar of the mascot."""
    # Crop head and upper body
    w, h = mascot.size
    crop_box = (int(w * 0.15), int(h * 0.02), int(w * 0.85), int(h * 0.65))
    cropped = mascot.crop(crop_box).resize((size, size), Image.Resampling.LANCZOS)
    
    # Circular mask
    mask = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((0, 0, size, size), fill=255)
    
    avatar = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    avatar.paste(cropped, (0, 0), mask)
    
    # Draw gold ring
    draw = ImageDraw.Draw(avatar)
    draw.ellipse((2, 2, size - 3, size - 3), outline=(245, 166, 35, 230), width=6)
    draw.ellipse((6, 6, size - 7, size - 7), outline=(255, 255, 255, 120), width=2)
    return avatar

def create_readme_banner(mascot_rgba: Image.Image, width: int = 1200, height: int = 480) -> Image.Image:
    """Create a wide horizontal high-tech banner for README.md."""
    # Dark blue gradient background
    base = Image.new("RGB", (width, height), (17, 36, 58))
    draw = ImageDraw.Draw(base)
    
    # Draw vertical subtle gradient and sci-fi grid
    for y in range(height):
        ratio = y / height
        r = int(17 * (1 - ratio) + 26 * ratio)
        g = int(36 * (1 - ratio) + 50 * ratio)
        b = int(58 * (1 - ratio) + 80 * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # Cyber grid lines
    grid_color = (40, 70, 105)
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        
    # Glowing decorative elements
    draw.rectangle([(0, 0), (width, 8)], fill=(46, 109, 164))
    draw.rectangle([(0, height - 8), (width, height)], fill=(245, 166, 35))
    
    # Paste mascot on the right side
    # Resize mascot keeping aspect ratio
    mh = int(height * 0.95)
    mw = int(mascot_rgba.width * (mh / mascot_rgba.height))
    mascot_resized = mascot_rgba.resize((mw, mh), Image.Resampling.LANCZOS)
    
    base_rgba = base.convert("RGBA")
    
    # Add a soft cyan glow behind mascot
    glow = Image.new("RGBA", (mw + 100, mh + 100), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((20, 20, mw + 80, mh + 80), fill=(46, 138, 230, 90))
    glow = glow.filter(ImageFilter.GaussianBlur(30))
    base_rgba.paste(glow, (width - mw - 80, int((height - mh) / 2) - 30), glow)
    
    # Paste mascot
    mascot_x = width - mw - 30
    mascot_y = int((height - mh) / 2) + 10
    base_rgba.paste(mascot_resized, (mascot_x, mascot_y), mascot_resized)
    
    # Text area
    text_draw = ImageDraw.Draw(base_rgba)
    
    # Title badge
    text_draw.rounded_rectangle([(70, 70), (280, 106)], radius=18, fill=(46, 109, 164))
    # Plate tag representation
    text_draw.rounded_rectangle([(70, 125), (420, 195)], radius=10, fill=(255, 255, 255), outline=(30, 30, 30), width=3)
    text_draw.rectangle([(230, 155), (255, 165)], fill=(30, 30, 30)) # dash
    
    # Decorative text boxes / labels
    text_draw.rounded_rectangle([(70, 220), (560, 320)], radius=8, fill=(15, 25, 40, 180), outline=(50, 90, 130), width=1)
    
    # Bottom tags
    tags = ["FastAPI + Uvicorn", "Taiwan Plate Font", "PyTorch CTC", "ONNX Runtime", "Port 1688 / 1788"]
    cur_x = 70
    for t in tags:
        tw = len(t) * 9 + 20
        text_draw.rounded_rectangle([(cur_x, 340), (cur_x + tw, 372)], radius=6, fill=(35, 60, 90), outline=(70, 120, 170))
        cur_x += tw + 12

    return base_rgba

def main():
    root = Path(__file__).resolve().parent.parent
    src_img = root / "assets" / "mascot.jpg"
    if not src_img.exists():
        print("Mascot image not found!")
        return
        
    orig = Image.open(src_img)
    mascot_trans = remove_white_background(orig)
    
    # Save transparent
    mascot_trans.save(root / "assets" / "mascot_transparent.png")
    mascot_trans.save(root / "web" / "assets" / "mascot_transparent.png")
    print("Saved transparent mascot")
    
    # Save avatar
    avatar = create_circular_avatar(mascot_trans, size=256)
    avatar.save(root / "assets" / "mascot_avatar.png")
    avatar.save(root / "web" / "assets" / "mascot_avatar.png")
    print("Saved mascot avatar")
    
    # Save README banner
    banner = create_readme_banner(mascot_trans, 1200, 480)
    banner.convert("RGB").save(root / "assets" / "readme_banner.jpg", quality=95)
    banner.convert("RGB").save(root / "web" / "assets" / "readme_banner.jpg", quality=95)
    print("Saved README banner")

if __name__ == "__main__":
    main()
