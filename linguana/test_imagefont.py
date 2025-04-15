from PIL import Image, ImageDraw, ImageFont, ImageOps

def apply_gradient_to_text(text, font, gradient_colors, size):
    mask = Image.new("L", size, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.text((0, 0), text, font=font, fill=255)

    gradient = Image.new("RGB", size, color=0)
    for y in range(size[1]):
        ratio = y / size[1]
        r = int(gradient_colors[0][0] * (1 - ratio) + gradient_colors[1][0] * ratio)
        g = int(gradient_colors[0][1] * (1 - ratio) + gradient_colors[1][1] * ratio)
        b = int(gradient_colors[0][2] * (1 - ratio) + gradient_colors[1][2] * ratio)
        ImageDraw.Draw(gradient).line([(0, y), (size[0], y)], fill=(r, g, b))

    result = Image.new("RGB", size)
    result.paste(gradient, mask=mask)
    return result

# Example usage
font = ImageFont.truetype("/data/projects/OCR-SAM/fonts/NotoSans-Regular.ttf", 100)
size = (600, 150)
text_img = apply_gradient_to_text("FAST", font, [(255, 255, 100), (255, 140, 0)], size)
text_img.show()
