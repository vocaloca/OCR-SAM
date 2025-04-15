from PIL import Image, ImageDraw, ImageFont, ImageColor
import math
import numpy as np
import os
from fontTools.ttLib import TTFont
from io import BytesIO
import unicodedata
from pathlib import Path

FONTS_DIR = Path(__file__).parent.parent.joinpath("fonts")


class MultilingualTextRenderer:
    """
    Text renderer with support for multiple languages/scripts using Pillow
    """
    
    def __init__(self, fonts_dir=FONTS_DIR):
        """Initialize with a directory of font files"""
        self.fonts_dir = fonts_dir
        
        # Manual mapping of scripts to font files
        self.font_map = {
            'default': FONTS_DIR / "NotoSans-Regular.ttf",
            'latin': FONTS_DIR / "NotoSans-Regular.ttf",
            'cyrillic': FONTS_DIR / "NotoSans-Regular.ttf",
            'japanese': FONTS_DIR / "NotoSansCJKjp-Regular.otf",
            'chinese': FONTS_DIR / "NotoSansCJKsc-Regular.otf",
            'korean': FONTS_DIR / "NotoSansCJKkr-Regular.otf",
            'thai': FONTS_DIR / "NotoSansThai-Regular.ttf",
            'arabic': FONTS_DIR / "NotoSansArabic-Regular.ttf"
        }
        
        # Verify font files exist
        for script, font_path in self.font_map.items():
            if not font_path.exists():
                print(f"Warning: Font for {script} not found at {font_path}")
        
    def _build_font_map(self):
        """Build a map of script to available fonts"""
        font_map = {
            'default': None,  # Will be set to first found font
        }
        
        # Check if fonts directory exists
        if not os.path.exists(self.fonts_dir):
            os.makedirs(self.fonts_dir, exist_ok=True)
            print(f"Created fonts directory at {self.fonts_dir}")
            return font_map
            
        # Scan available fonts
        for font_file in os.listdir(self.fonts_dir):
            if font_file.lower().endswith(('.ttf', '.otf')):
                font_path = os.path.join(self.fonts_dir, font_file)
                
                try:
                    # Use fontTools to analyze font capabilities
                    font = TTFont(font_path)
                    
                    # Get font name
                    font_name = None
                    for record in font['name'].names:
                        if record.nameID == 4:  # Full font name
                            if record.isUnicode():
                                font_name = record.string.decode('utf-16-be')
                                break
                            else:
                                font_name = record.string.decode('latin1')
                                break
                    
                    if font_name is None:
                        font_name = os.path.splitext(font_file)[0]
                    
                    # Check which scripts this font supports
                    cmap = font.getBestCmap()
                    supported_scripts = self._analyze_font_support(cmap)
                    
                    # Add font to map for each supported script
                    for script in supported_scripts:
                        if script not in font_map or font_map[script] is None:
                            font_map[script] = font_path
                    
                    # Set default font if not set
                    if font_map['default'] is None:
                        font_map['default'] = font_path
                
                except Exception as e:
                    print(f"Error analyzing font {font_file}: {e}")
        
        return font_map
    
    def _analyze_font_support(self, cmap):
        """Analyze which scripts a font supports based on its character map"""
        # Define representative characters for different scripts
        script_samples = {
            'latin': range(0x0041, 0x007A),  # Basic Latin
            'cyrillic': range(0x0410, 0x044F),  # Cyrillic
            'greek': range(0x0391, 0x03C9),  # Greek
            'arabic': range(0x0627, 0x064A),  # Arabic
            'hebrew': range(0x05D0, 0x05EA),  # Hebrew
            'devanagari': range(0x0915, 0x0939),  # Devanagari (Hindi)
            'thai': range(0x0E01, 0x0E30),  # Thai
            'japanese': [0x3042, 0x3044, 0x3046, 0x3048, 0x304A],  # Hiragana sample
            'chinese': [0x4E00, 0x4E01, 0x4E03, 0x4E07, 0x4E08],  # Common Hanzi
            'korean': range(0xAC00, 0xAC19),  # Hangul sample
        }
        
        supported_scripts = []
        
        for script, chars in script_samples.items():
            # Check if most characters in the sample are supported
            supported_count = sum(1 for char in chars if char in cmap)
            threshold = len(list(chars)) * 0.7  # 70% threshold
            
            if supported_count >= threshold:
                supported_scripts.append(script)
        
        return supported_scripts
    
    def detect_script(self, text):
        """Detect the dominant script in the text"""
        if not text:
            return 'default'
            
        # Count characters by script
        script_counts = {}
        
        for char in text:
            # Get script name for this character
            try:
                script = unicodedata.name(char).split()[0].lower()
                
                # Map script names to our categories
                if script in ('latin', 'ascii'):
                    script = 'latin'
                elif script in ('cyrillic',):
                    script = 'cyrillic'
                elif script in ('greek',):
                    script = 'greek'
                elif script in ('arabic',):
                    script = 'arabic'
                elif script in ('hebrew',):
                    script = 'hebrew'
                elif script in ('devanagari',):
                    script = 'devanagari'
                elif script in ('thai',):
                    script = 'thai'
                elif script in ('hiragana', 'katakana', 'cjk'):
                    if 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF:
                        script = 'japanese'
                    elif 0xAC00 <= ord(char) <= 0xD7A3:
                        script = 'korean'
                    elif 0x4E00 <= ord(char) <= 0x9FFF:
                        script = 'chinese'
                # Direct check for Korean Hangul
                elif 0xAC00 <= ord(char) <= 0xD7A3:
                    script = 'korean'
                else:
                    script = 'default'
                    
                script_counts[script] = script_counts.get(script, 0) + 1
            except Exception as e:
                print(f"Error processing character '{char}': {e}")
                continue
        
        # Find dominant script
        if not script_counts:
            return 'default'
            
        dominant_script = max(script_counts.items(), key=lambda x: x[1])[0]
        return dominant_script
    
    def get_font_for_text(self, text, size=12):
        """Get appropriate font for the given text"""
        # Detect script of the text
        script = self.detect_script(text)
        
        # Special handling for CJK scripts and mixed text
        if script == 'default' or self.has_multiple_scripts(text):
            print("Text contains multiple scripts, using a universal font")
            
            # Check for CJK characters
            has_japanese = any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF for c in text if not c.isspace())
            has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in text if not c.isspace())
            has_chinese = any(0x4E00 <= ord(c) <= 0x9FFF for c in text if not c.isspace())
            
            # If contains CJK, prioritize appropriate CJK font
            if has_japanese and self.font_map['japanese'].exists():
                print("Using Japanese font for mixed text with Japanese characters")
                return ImageFont.truetype(str(self.font_map['japanese']), size)
            elif has_korean and self.font_map['korean'].exists():
                print("Using Korean font for mixed text with Korean characters")
                return ImageFont.truetype(str(self.font_map['korean']), size)
            elif has_chinese and self.font_map['chinese'].exists():
                print("Using Chinese font for mixed text with Chinese characters")
                return ImageFont.truetype(str(self.font_map['chinese']), size)
            
            # Fallback to universal font
            universal_font = self.get_universal_font()
            if universal_font:
                try:
                    return ImageFont.truetype(universal_font, size)
                except Exception as e:
                    print(f"Error loading universal font: {e}")
                    # Fall through to standard font selection
        
        # Direct font selection for specific scripts
        if script in ['japanese', 'korean', 'chinese']:
            font_path = self.font_map.get(script)
            if font_path and font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except Exception as e:
                    print(f"Error loading CJK font for {script}: {e}")
        
        # Try to get font from our map
        font_path = self.font_map.get(script, self.font_map['default'])
        
        # If we don't have a specific font for this script, try system unicode font
        if font_path is None or not os.path.exists(font_path):
            print(f"Font for script '{script}' not found at {font_path}")
            print("Trying system fonts...")
            system_font = self.get_system_unicode_font()
            if system_font:
                font_path = system_font
        
        # Load the font
        try:
            font = ImageFont.truetype(str(font_path), size)
            return font
        except Exception as e:
            print(f"Error loading font for {script} script: {e}")
            
            # If our font failed, try system unicode font as last resort
            if font_path != self.get_system_unicode_font():
                system_font = self.get_system_unicode_font()
                if system_font:
                    try:
                        return ImageFont.truetype(system_font, size)
                    except Exception as e:
                        print(f"Error loading system font: {e}")
            
            # Fall back to default PIL font if all else fails
            print(f"Falling back to default PIL font for {script} script")
            return ImageFont.load_default()
    
    def get_system_unicode_font(self):
        """Get a system font with good Unicode coverage"""
        # Common system fonts with good Unicode coverage
        system_fonts = [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS Korean
            "/Library/Fonts/Arial Unicode.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/noto/NotoSansArabic-Regular.ttf"  # Linux Arabic
        ]
        
        for font_path in system_fonts:
            if os.path.exists(font_path):
                print(f"Found system font: {font_path}")
                try:
                    return font_path
                except Exception as e:
                    print(f"Error checking system font {font_path}: {e}")
                    continue
        
        print("No suitable system fonts found")
        return None
    
    def has_multiple_scripts(self, text):
        """Check if text contains characters from multiple scripts"""
        scripts = set()
        for char in text:
            if char.isspace():
                continue
                
            try:
                # Get the script block for this character
                char_script = None
                
                # Direct script checks based on Unicode ranges
                if 0x0041 <= ord(char) <= 0x007A:  # Latin
                    char_script = 'latin'
                elif 0x0410 <= ord(char) <= 0x044F:  # Cyrillic
                    char_script = 'cyrillic'
                elif 0x0391 <= ord(char) <= 0x03C9:  # Greek
                    char_script = 'greek'
                elif 0x0600 <= ord(char) <= 0x06FF:  # Arabic
                    char_script = 'arabic'
                elif 0x0900 <= ord(char) <= 0x097F:  # Devanagari
                    char_script = 'devanagari'
                elif 0x0E00 <= ord(char) <= 0x0E7F:  # Thai
                    char_script = 'thai'
                elif 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF:  # Japanese kana
                    char_script = 'japanese'
                elif 0xAC00 <= ord(char) <= 0xD7A3:  # Korean Hangul
                    char_script = 'korean'
                elif 0x4E00 <= ord(char) <= 0x9FFF:  # CJK Unified Ideographs
                    char_script = 'chinese'
                    
                if char_script:
                    scripts.add(char_script)
                    if len(scripts) > 1:
                        return True
            except:
                continue
                
        return False
    
    def get_universal_font(self):
        """Get a font with good multi-script coverage"""
        universal_fonts = [
            # Common universal fonts
            FONTS_DIR / "NotoSans-Regular-Latest.ttf",  # Try latest version first
            FONTS_DIR / "NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            # Add more universal fonts here
        ]
        
        for font_path in [str(path) for path in universal_fonts]:
            if os.path.exists(font_path):
                print(f"Using universal font: {font_path}")
                return font_path
                
        # If no universal font found, try to get best system font
        return self.get_system_unicode_font()
    
    def overlay_rotated_text(
        self,
        image_path,
        text,
        polygon,
        font_size=24,
        font_color=(0, 0, 0),
        outline_color=None,
        outline_width=0,
        highlight_color=None,
        output_path=None,
        custom_font=None
    ):
        """
        Overlay text on an image within a rotated polygon with custom styling
        
        Args:
            image_path: Path to the input image
            text: Text to overlay
            polygon: List of (x,y) points defining the rotated bounding box
            font_size: Size of the font in points
            font_color: RGB tuple for the text color
            outline_color: RGB tuple for the outline color (None for no outline)
            outline_width: Width of the outline in pixels
            highlight_color: RGB tuple for background highlight (None for transparent)
            output_path: Path to save the result (if None, returns the image)
            custom_font: Optional PIL ImageFont object to use instead of auto-selecting
            
        Returns:
            PIL Image with the overlaid text
        """
        # Load the image
        img = Image.open(image_path).convert("RGBA")
        img_width, img_height = img.size
        
        # Calculate rotation angle from the polygon
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        angle_degrees = math.degrees(math.atan2(dy, dx))
        
        # Calculate polygon dimensions
        width = math.sqrt((polygon[1][0] - polygon[0][0])**2 + 
                         (polygon[1][1] - polygon[0][1])**2)
        height = math.sqrt((polygon[3][0] - polygon[0][0])**2 + 
                          (polygon[3][1] - polygon[0][1])**2)
        
        # Calculate the centroid of the polygon
        centroid_x = sum(p[0] for p in polygon) / len(polygon)
        centroid_y = sum(p[1] for p in polygon) / len(polygon)
        
        # Draw highlight if needed
        if highlight_color is not None:
            highlight = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(highlight)
            
            # Convert polygon to flat list for PIL
            flat_polygon = [coord for point in polygon for coord in point]
            
            # Draw with alpha
            highlight_with_alpha = highlight_color + (180,)  # Add alpha
            draw.polygon(flat_polygon, fill=highlight_with_alpha)
            
            # Composite highlight onto the image
            img = Image.alpha_composite(img, highlight)
        
        # Get appropriate font
        font = custom_font if custom_font else self.get_font_for_text(text, font_size)
        
        # Create a transparent image for the text
        text_img = Image.new('RGBA', (int(width*1.5), int(height*1.5)), 
                           (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center position
        text_x = (text_img.width - text_width) // 2
        text_y = (text_img.height - text_height) // 2
        
        # Draw outline if specified
        if outline_color is not None and outline_width > 0:
            # Draw text multiple times with offsets
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip center (will be drawn in main color)
                    draw.text((text_x + dx, text_y + dy), text, font=font, 
                             fill=outline_color)
        
        # Draw main text
        draw.text((text_x, text_y), text, font=font, fill=font_color)
        
        # Rotate the text
        rotated_text = text_img.rotate(-angle_degrees, resample=Image.BICUBIC, 
                                     expand=True)
        
        # Calculate paste position
        paste_x = int(centroid_x - rotated_text.width / 2)
        paste_y = int(centroid_y - rotated_text.height / 2)
        
        # Paste onto the image
        img.paste(rotated_text, (paste_x, paste_y), rotated_text)
        
        # Convert back to RGB if needed for JPEG
        if output_path and (output_path.lower().endswith('.jpg') or 
                           output_path.lower().endswith('.jpeg')):
            img = img.convert('RGB')
        
        # Save or return
        if output_path:
            img.save(output_path)
            return output_path
        else:
            return img

# Example usage
if __name__ == "__main__":
    # Initialize renderer
    renderer = MultilingualTextRenderer()
    
    # Example polygon (rotated bounding box)
    polygon = [(100, 100), (300, 150), (250, 250), (50, 200)]
    
    # Test with different scripts
    examples = {
        "English": "Hello World!",
        "Japanese": "こんにちは世界",
        "Chinese": "你好，世界",
        "Korean": "안녕하세요 세계",
        "Russian": "Привет, мир",
        "Thai": "สวัสดีชาวโลก",
        "Arabic": "مرحبا بالعالم",
        "Mixed": "Hello こんにちは 你好 안녕하세요"
    }
    
    # Print available fonts
    print("\nAvailable fonts:")
    for script, font_path in renderer.font_map.items():
        if font_path.exists():
            print(f"  {script}: {font_path} (exists)")
        else:
            print(f"  {script}: {font_path} (missing)")
    
    # Render each example
    for name, text in examples.items():
        print(f"\nProcessing {name} text...")
        print(f"Detected script: {renderer.detect_script(text)}")
        
        # Direct font selection for CJK scripts
        custom_font = None
        if name in ["Japanese", "Korean", "Chinese"]:
            script = name.lower()
            font_path = renderer.font_map.get(script)
            if font_path and font_path.exists():
                try:
                    print(f"Directly using {script} font: {font_path}")
                    custom_font = ImageFont.truetype(str(font_path), 36)
                except Exception as e:
                    print(f"Error loading {script} font: {e}")
        elif name == "Mixed":
            # Use Japanese font for mixed text since it has good Unicode coverage
            font_path = renderer.font_map.get('japanese')
            if font_path and font_path.exists():
                try:
                    print(f"Using Japanese font for mixed text: {font_path}")
                    custom_font = ImageFont.truetype(str(font_path), 36)
                except Exception as e:
                    print(f"Error loading Japanese font for mixed text: {e}")
        
        output = renderer.overlay_rotated_text(
            "/data/projects/OCR-SAM/imgs/ex12.jpg",
            text,
            polygon,
            font_size=36,
            font_color=(0, 0, 255),
            outline_color=(255, 255, 255),
            outline_width=2,
            highlight_color=(255, 255, 0),
            output_path=f"output_{name.lower()}.jpg",
            custom_font=custom_font
        )
        print(f"Rendered {name} text")