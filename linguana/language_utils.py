import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo
import cairo
import math
import numpy as np
from PIL import Image

class MultilinguaTextRenderer:
    """
    A class for rendering multilingual text on images using Pango/Cairo
    which handles complex font selection and text layout automatically.
    """
    
    def __init__(self, font_config=None):
        """
        Initialize the text renderer with optional font configurations
        
        Args:
            font_config: Dictionary mapping language/script to font families
        """
        self.font_config = font_config or {}
        # Default font configuration if none provided
        if not self.font_config:
            self.font_config = {
                'default': 'Noto Sans',
                'ja': 'Noto Sans JP',
                'zh': 'Noto Sans SC',
                'zh-Hant': 'Noto Sans TC',
                'ko': 'Noto Sans KR',
                'th': 'Noto Sans Thai',
                'ar': 'Noto Sans Arabic',
                'he': 'Noto Sans Hebrew',
                'ru': 'Noto Sans',  # Cyrillic is covered by Noto Sans
            }
    
    def _create_pango_layout(self, context, text, font_size, language=None):
        """Create a Pango layout with appropriate font configuration"""
        layout = PangoCairo.create_layout(context)
        
        # Create font description
        font_family = self.font_config.get(language, self.font_config['default'])
        font_desc = Pango.FontDescription()
        font_desc.set_family(font_family)
        font_desc.set_size(font_size * Pango.SCALE)
        
        # Set the font
        layout.set_font_description(font_desc)
        
        # Set text with proper language tag if specified
        if language:
            # Create a marked-up text with language information
            marked_text = f'<span lang="{language}">{text}</span>'
            layout.set_markup(marked_text, -1)
        else:
            layout.set_text(text, -1)
        
        return layout
    
    def overlay_rotated_text(
        self,
        image_path,
        text,
        polygon,
        font_size=24,
        language=None,
        font_color=(0, 0, 0),
        outline_color=None,
        outline_width=0,
        highlight_color=None,
        output_path=None
    ):
        """
        Overlay text on an image within a rotated polygon with custom styling
        
        Args:
            image_path: Path to the input image
            text: Text to overlay
            polygon: List of (x,y) points defining the rotated bounding box
            font_size: Size of the font in points
            language: ISO language code for font selection (e.g., 'ja', 'zh')
            font_color: RGB tuple for the text color
            outline_color: RGB tuple for the outline color (None for no outline)
            outline_width: Width of the outline in pixels
            highlight_color: RGB tuple for background highlight (None for transparent)
            output_path: Path to save the result (if None, returns the image)
            
        Returns:
            PIL Image with the overlaid text
        """
        # Load the image
        pil_img = Image.open(image_path).convert("RGBA")
        img_width, img_height = pil_img.size
        
        # Calculate rotation angle from the polygon
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        
        # Calculate polygon dimensions
        width = math.sqrt((polygon[1][0] - polygon[0][0])**2 + (polygon[1][1] - polygon[0][1])**2)
        height = math.sqrt((polygon[3][0] - polygon[0][0])**2 + (polygon[3][1] - polygon[0][1])**2)
        
        # Calculate the centroid of the polygon
        centroid_x = sum(p[0] for p in polygon) / len(polygon)
        centroid_y = sum(p[1] for p in polygon) / len(polygon)
        
        # Create a Cairo surface and context for the overlay
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_width, img_height)
        context = cairo.Context(surface)
        
        # Draw highlight if specified
        if highlight_color is not None:
            # Save the context state
            context.save()
            
            # Set the color with alpha
            r, g, b = highlight_color
            context.set_source_rgba(r/255, g/255, b/255, 0.8)
            
            # Draw the polygon
            context.move_to(polygon[0][0], polygon[0][1])
            for point in polygon[1:]:
                context.line_to(point[0], point[1])
            context.close_path()
            context.fill()
            
            # Restore the context state
            context.restore()
        
        # Create a new context for the text
        text_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width*1.5), int(height*1.5))
        text_context = cairo.Context(text_surface)
        
        # Create Pango layout
        layout = self._create_pango_layout(text_context, text, font_size, language)
        
        # Get text dimensions
        text_width, text_height = layout.get_pixel_size()
        
        # Position the text in the center of the surface
        text_context.translate(
            (text_surface.get_width() - text_width) / 2,
            (text_surface.get_height() - text_height) / 2
        )
        
        # Draw text outline if specified
        if outline_color is not None and outline_width > 0:
            r, g, b = outline_color
            text_context.set_source_rgb(r/255, g/255, b/255)
            
            # Draw the text at several offsets for the outline effect
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the center (will be drawn with text color)
                    
                    text_context.save()
                    text_context.translate(dx, dy)
                    PangoCairo.show_layout(text_context, layout)
                    text_context.restore()
        
        # Draw the main text
        r, g, b = font_color
        text_context.set_source_rgb(r/255, g/255, b/255)
        PangoCairo.show_layout(text_context, layout)
        
        # Rotate the text surface
        rotated_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_width, img_height)
        rotated_context = cairo.Context(rotated_surface)
        
        # Position at the centroid
        rotated_context.translate(centroid_x, centroid_y)
        rotated_context.rotate(angle_radians)
        rotated_context.translate(
            -text_surface.get_width() / 2,
            -text_surface.get_height() / 2
        )
        
        # Draw the text surface onto the rotated surface
        rotated_context.set_source_surface(text_surface, 0, 0)
        rotated_context.paint()
        
        # Convert the Cairo surface to a PIL Image
        buf = surface.get_data()
        overlay = Image.frombuffer('RGBA', (img_width, img_height), buf, 'raw', 'BGRA', 0, 1)
        
        # Convert the rotated Cairo surface to a PIL Image
        buf = rotated_surface.get_data()
        rotated_overlay = Image.frombuffer('RGBA', (img_width, img_height), buf, 'raw', 'BGRA', 0, 1)
        
        # Composite the images
        if highlight_color is not None:
            result = Image.alpha_composite(pil_img, overlay)
        else:
            result = pil_img.copy()
        
        result = Image.alpha_composite(result, rotated_overlay)
        
        # Save or return the result
        if output_path:
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                result = result.convert('RGB')
            result.save(output_path)
            return output_path
        else:
            return result

    def detect_language(self, text):
        """
        Attempt to detect the language/script of the text
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO language code or None if detection fails
        """
        try:
            # Try using langdetect for language detection
            from langdetect import detect
            return detect(text)
        except:
            # Fallback to basic Unicode range detection
            unicode_ranges = {
                'ja': [(0x3040, 0x309F), (0x30A0, 0x30FF)],  # Hiragana and Katakana
                'zh': [(0x4E00, 0x9FFF)],  # CJK Unified Ideographs
                'ko': [(0xAC00, 0xD7AF)],  # Hangul Syllables
                'th': [(0x0E00, 0x0E7F)],  # Thai
                'ar': [(0x0600, 0x06FF)],  # Arabic
                'he': [(0x0590, 0x05FF)],  # Hebrew
                'ru': [(0x0400, 0x04FF)],  # Cyrillic
            }
            
            # Count characters in each range
            counts = {lang: 0 for lang in unicode_ranges}
            
            for char in text:
                char_code = ord(char)
                for lang, ranges in unicode_ranges.items():
                    for start, end in ranges:
                        if start <= char_code <= end:
                            counts[lang] += 1
                            break
            
            # Find the dominant language
            dominant_lang = max(counts.items(), key=lambda x: x[1])[0] if counts else None
            
            # If no clear detection or too few characters matched
            if dominant_lang is None or counts[dominant_lang] < len(text) * 0.3:
                return None
                
            return dominant_lang


def extract_script_from_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    script = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Try to interpret the line as a subtitle number
        try:
            subtitle_num = int(line)
            # If this is a subtitle number, the text should be 2 lines ahead
            # (after the timestamp line)
            if i + 2 < len(lines):
                text_line = lines[i + 2].strip()
                if text_line:  # Only add non-empty lines
                    script.append(text_line)
            i += 4  # Skip to the next subtitle block (usually 4 lines per subtitle)
        except ValueError:
            # If it's not a subtitle number, just move to next line
            i += 1
    
    return script


# Example usage
if __name__ == "__main__":
    renderer = MultilinguaTextRenderer()
    
    # Example for Japanese text
    japanese_text = "こんにちは世界"
    polygon = [(100, 100), (300, 150), (250, 250), (50, 200)]
    
    # Automatically detect language and render
    language = renderer.detect_language(japanese_text)
    result = renderer.overlay_rotated_text(
        # "input.jpg",
        "/data/projects/OCR-SAM/imgs/ex12.jpg",
        japanese_text,
        polygon,
        font_size=36,
        language=language,
        font_color=(0, 0, 255),
        outline_color=(255, 255, 255),
        outline_width=3,
        highlight_color=(255, 255, 0),
        output_path="output_japanese.jpg"
    )
    
    # Example for mixed language text with explicit language tagging
    mixed_text = "Hello, こんにちは, 你好, 안녕하세요"
    result = renderer.overlay_rotated_text(
        "input.jpg",
        mixed_text,
        polygon,
        font_size=36,
        # No language specified - Pango will handle mixed text
        font_color=(255, 0, 0),
        outline_color=(255, 255, 255),
        outline_width=2,
        output_path="output_mixed.jpg"
    )
    
    
    # Extract script from srt file
    # Usage
    file_path = "dubbing_30073257582_full_subtitles_pre_no_ads.srt"
    script = extract_script_from_srt(file_path)

    # Print or save the script
    for line in script:
        print(line)

    # If you want to save to a file
    with open("extracted_script.txt", "w", encoding="utf-8") as output_file:
        for line in script:
            output_file.write(line + "\n")