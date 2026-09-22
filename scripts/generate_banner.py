"""Recreate high-tech readme_banner.jpg with authentic typography and mascot."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH, HEIGHT = 1200, 480

# 1. Background gradient: deep cyber blue to dark navy
bg = Image.new("RGBA", (WIDTH, HEIGHT), (10, 15, 30, 255))
draw = ImageDraw.Draw(bg)

# Vertical gradient
for y in range(HEIGHT):
    factor = y / HEIGHT
    r = int(10 * (1 - factor) + 14 * factor)
    g = int(22 * (1 - factor) + 28 * factor)
    b = int(48 * (1 - factor) + 64 * factor)
    draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

# Tech Grid Overlay
grid_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
grid_draw = ImageDraw.Draw(grid_layer)
grid_step = 30
for x in range(0, WIDTH, grid_step):
    grid_draw.line([(x, 0), (x, HEIGHT)], fill=(0, 210, 255, 12), width=1)
for y in range(0, HEIGHT, grid_step):
    grid_draw.line([(0, y), (WIDTH, y)], fill=(0, 210, 255, 12), width=1)

# Subtle diagonal speed lines
for i in range(-HEIGHT, WIDTH, 120):
    grid_draw.line([(i, 0), (i + HEIGHT, HEIGHT)], fill=(0, 242, 254, 8), width=2)

bg = Image.alpha_composite(bg, grid_layer)

# Ambient glow behind title and mascot
glow_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
glow_draw = ImageDraw.Draw(glow_layer)
glow_draw.ellipse([80, 60, 480, 360], fill=(0, 242, 254, 25))
glow_draw.ellipse([800, 40, 1180, 440], fill=(255, 180, 0, 20))
glow_draw.ellipse([850, 100, 1150, 420], fill=(0, 160, 255, 30))
glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(50))
bg = Image.alpha_composite(bg, glow_layer)

# 2. Place Mascot on the right
mascot_path = Path("assets/mascot_transparent.png")
if mascot_path.exists():
    mascot = Image.open(mascot_path).convert("RGBA")
    m_target_h = 460
    m_target_w = int(mascot.width * (m_target_h / mascot.height))
    mascot_resized = mascot.resize((m_target_w, m_target_h), Image.Resampling.LANCZOS)
    
    pos_x = WIDTH - m_target_w + 30
    pos_y = HEIGHT - m_target_h + 10
    
    shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse([pos_x + 60, HEIGHT - 45, pos_x + m_target_w - 60, HEIGHT - 5], fill=(0, 0, 0, 160))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    bg = Image.alpha_composite(bg, shadow)
    
    bg.paste(mascot_resized, (pos_x, pos_y), mascot_resized)

# 3. Fonts
font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 52)
font_badge = ImageFont.truetype(r"C:\Windows\Fonts\msjhbd.ttc", 13)
font_tag = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 13)
font_sub = ImageFont.truetype(r"C:\Windows\Fonts\msjhbd.ttc", 18)
font_card_title = ImageFont.truetype(r"C:\Windows\Fonts\msjhbd.ttc", 15)
font_card_desc = ImageFont.truetype(r"C:\Windows\Fonts\msjhbd.ttc", 12)
plate_font = ImageFont.truetype("assets/fonts/TaiwanPlate-Regular.ttf", 36)
chip_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 11)

# Overlay for text and UI elements
ui_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
d = ImageDraw.Draw(ui_layer)

# Top badge
badge_text = "3wa.org · OPEN SOURCE AI STUDIO"
badge_w = 290
badge_h = 26
d.rounded_rectangle([60, 36, 60 + badge_w, 36 + badge_h], radius=13, fill=(15, 45, 80, 220), outline=(0, 242, 254, 180), width=1)
d.ellipse([72, 45, 80, 53], fill=(0, 255, 170, 255))
d.text((88, 41), badge_text, font=font_badge, fill=(0, 242, 254, 255))

# Main Title: 3waPlateAI
title_x, title_y = 60, 74
d.text((title_x + 2, title_y + 2), "3waPlateAI", font=font_title, fill=(0, 50, 100, 220))
d.text((title_x, title_y), "3waPlateAI", font=font_title, fill=(255, 255, 255, 255))

# Title Accent Badge: v1.0 Production
tag_x = title_x + 355
d.rounded_rectangle([tag_x, title_y + 14, tag_x + 156, title_y + 42], radius=6, fill=(255, 180, 0, 230))
d.text((tag_x + 12, title_y + 20), "CTC v1.0 RELEASE", font=font_tag, fill=(15, 20, 25, 255))

# Subtitle
sub_text = "台灣車牌辨識與合成訓練全能工作站 · 高精度 End-to-End CTC"
d.text((60, 138), sub_text, font=font_sub, fill=(185, 215, 245, 255))

# Mini Taiwan Plate Showcase Card
plate_card_x, plate_card_y = 60, 176
plate_card_w, plate_card_h = 680, 96
d.rounded_rectangle([plate_card_x, plate_card_y, plate_card_x + plate_card_w, plate_card_y + plate_card_h], 
                    radius=12, fill=(16, 32, 58, 220), outline=(40, 80, 130, 200), width=1)

# Inside Plate Card: realistic Taiwan License Plate
p_x, p_y = plate_card_x + 16, plate_card_y + 12
p_w, p_h = 175, 71
# Plate background (white with black border)
d.rounded_rectangle([p_x, p_y, p_x + p_w, p_y + p_h], radius=6, fill=(248, 249, 250, 255), outline=(30, 30, 30, 255), width=2)
# Plate inner border line
d.rounded_rectangle([p_x + 4, p_y + 4, p_x + p_w - 4, p_y + p_h - 4], radius=4, outline=(70, 70, 70, 200), width=1)
# Plate bolt holes
d.ellipse([p_x + 14, p_y + 10, p_x + 20, p_y + 16], fill=(160, 160, 160, 255), outline=(50, 50, 50, 255))
d.ellipse([p_x + p_w - 20, p_y + 10, p_x + p_w - 14, p_y + 16], fill=(160, 160, 160, 255), outline=(50, 50, 50, 255))
# Plate text
d.text((p_x + 24, p_y + 14), "3WA-8888", font=plate_font, fill=(15, 15, 15, 255))

# Green detection box effect around plate
d.rounded_rectangle([p_x - 3, p_y - 3, p_x + p_w + 3, p_y + p_h + 3], radius=8, outline=(34, 197, 94, 255), width=2)
d.rounded_rectangle([p_x + p_w - 56, p_y - 12, p_x + p_w + 3, p_y + 4], radius=4, fill=(34, 197, 94, 255))
d.text((p_x + p_w - 50, p_y - 10), "99.8%", font=chip_font, fill=(0, 0, 0, 255))

# Info next to plate
d.text((p_x + p_w + 24, plate_card_y + 16), "即時多尺度特徵偵測 · 雙向 LSTM 解碼", font=font_sub, fill=(255, 255, 255, 255))
d.text((p_x + p_w + 24, plate_card_y + 44), "規範約束: LLL-DDDD (白牌自用車)  |  延遲: 4.2 ms", font=font_card_title, fill=(140, 180, 220, 255))
d.text((p_x + p_w + 24, plate_card_y + 68), "支援格式: 新舊式汽機車、營業車、電動車規範 (34 類)", font=font_card_desc, fill=(0, 242, 254, 255))

# 6 Feature Badges in 2 rows of 3 with all-English tech chip tags
badges = [
    ("RULE", (0, 210, 255), "台灣標準規範", "34 類專屬字集規則"),
    ("TRAIN", (255, 150, 0), "PyTorch CTC", "自建 10,000+ 合成訓練"),
    ("ONNX", (34, 197, 94), "極速推論 < 5ms", "跨平台 CPU/GPU 推論"),
    ("VISION", (140, 120, 255), "智慧多尺度定位", "高靈敏邊緣與色域偵測"),
    ("VRAM", (255, 90, 140), "顯存動態監控", "即時 GPU / 記憶體分析"),
    ("DEPLOY", (255, 210, 0), "雙工服務架構", "Web 工作站 + REST API"),
]

card_w = 216
card_h = 76
start_bx = 60
start_by = 290
gap_x = 16
gap_y = 12

for i, (chip_text, chip_color, b_title, b_desc) in enumerate(badges):
    row = i // 3
    col = i % 3
    bx = start_bx + col * (card_w + gap_x)
    by = start_by + row * (card_h + gap_y)
    
    # Card background
    d.rounded_rectangle([bx, by, bx + card_w, by + card_h], radius=10, 
                        fill=(18, 36, 64, 220), outline=(0, 180, 255, 60), width=1)
    
    # Left accent vertical bar
    d.rounded_rectangle([bx, by + 10, bx + 3, by + card_h - 10], radius=2, fill=chip_color)
    
    # Tag chip
    chip_w = 52 if len(chip_text) <= 4 else 60
    d.rounded_rectangle([bx + 12, by + 13, bx + 12 + chip_w, by + 29], radius=4, fill=(*chip_color[:3], 40), outline=chip_color, width=1)
    # Center chip text
    d.text((bx + 18, by + 14), chip_text, font=chip_font, fill=chip_color)
    
    # Title
    d.text((bx + 12 + chip_w + 8, by + 13), b_title, font=font_card_title, fill=(255, 255, 255, 255))
    
    # Desc
    d.text((bx + 12, by + 45), b_desc, font=font_card_desc, fill=(140, 180, 220, 255))

# Composite layers
final_img = Image.alpha_composite(bg, ui_layer).convert("RGB")
output_path = Path("assets/readme_banner.jpg")
final_img.save(output_path, "JPEG", quality=95)
print(f"Generated {output_path} successfully ({WIDTH}x{HEIGHT})")
