from typing import Union
import os
import numpy as np
import math
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import requests
import unicodedata
import re
from fontTools.ttLib import TTFont

# Import the clip_polygon_to_image_bounds function from image_ocr_utils
from linguana.image_ocr_utils import clip_polygon_to_image_bounds

FONTS_DIR = Path(__file__).parent.parent.joinpath("fonts")


class MultilingualTextRenderer:
    """
    Text renderer with support for multiple languages/scripts using Pillow
    """
    
    def __init__(self, fonts_dir=FONTS_DIR):
        """Initialize with a directory of font files"""
        self.fonts_dir = fonts_dir
        
        # Ensure fonts directory exists
        if not os.path.exists(self.fonts_dir):
            os.makedirs(self.fonts_dir, exist_ok=True)
            print(f"Created fonts directory at {self.fonts_dir}")
        
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
        
        # Load or download required fonts for critical languages
        self.load_critical_fonts(['japanese', 'chinese', 'korean'])
        
        # Verify font files exist
        for script, font_path in self.font_map.items():
            if not font_path.exists():
                print(f"Warning: Font for {script} not found at {font_path}")
        
    def load_critical_fonts(self, languages):
        """Ensure critical language fonts are available by downloading if needed"""
        jp_url = 'https://github.com/googlefonts/noto-cjk/raw/main/'
        jp_url += 'Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf'
        
        cn_url = 'https://github.com/googlefonts/noto-cjk/raw/main/'
        cn_url += 'Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'
        
        kr_url = 'https://github.com/googlefonts/noto-cjk/raw/main/'
        kr_url += 'Sans/OTF/Korean/NotoSansCJKkr-Regular.otf'
        
        latin_url = 'https://github.com/googlefonts/noto-sans/raw/main/'
        latin_url += 'fonts/NotoSans-Regular.ttf'
        
        font_urls = {
            'japanese': jp_url,
            'chinese': cn_url,
            'korean': kr_url,
            'latin': latin_url
        }
        
        for lang in languages:
            font_path = self.font_map.get(lang)
            if font_path and not font_path.exists():
                if lang in font_urls:
                    try:
                        print(f"Font for {lang} not found. Attempting to download...")
                        self._download_font(font_urls[lang], font_path)
                        print(f"Downloaded {lang} font to {font_path}")
                    except Exception as e:
                        print(f"Failed to download {lang} font: {e}")
                        # Create an in-memory font as fallback
                        self._create_fallback_font(lang)
    
    def _download_font(self, url, save_path):
        """Download a font file from URL"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Make sure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the font file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def _create_fallback_font(self, language):
        """Create an in-memory fallback font when download fails"""
        # Map language to font object that will be stored in memory
        self._memory_fonts = getattr(self, '_memory_fonts', {})
        
        # Try to use a system font as fallback
        system_font = None
        if language == 'japanese':
            system_fonts = [
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
                "/System/Library/Fonts/AppleGothic.ttf",  # macOS
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Linux
            ]
        elif language == 'chinese':
            system_fonts = [
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Linux
            ]
        elif language == 'korean':
            system_fonts = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Linux
            ]
        else:
            system_fonts = []
            
        # Try to load one of the system fonts
        for font_path in system_fonts:
            if os.path.exists(font_path):
                system_font = font_path
                print(f"Using system font for {language}: {font_path}")
                break
        
        if system_font:
            try:
                # Test that we can load it
                ImageFont.truetype(system_font, 12)  # Just test loading
                # Store the path for later use
                self._memory_fonts[language] = system_font
            except Exception as e:
                print(f"Error loading system font for {language}: {e}")
                # Fall back to default
                self._memory_fonts[language] = None
        else:
            self._memory_fonts[language] = None
            print(f"No fallback font available for {language}")
                
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
        
        # Check memory fonts first (for critical languages)
        if hasattr(self, '_memory_fonts') and script in self._memory_fonts and self._memory_fonts[script]:
            try:
                return ImageFont.truetype(self._memory_fonts[script], size)
            except Exception:
                pass  # Continue to other methods if this fails
        
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
            FONTS_DIR / "ArchivoNarrow/ArchivoNarrow-BoldItalic.ttf",
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
        image: Union[str, Path, Image.Image, np.core.ndarray],
        text,
        polygon,
        font_size=24,
        font_color=(0, 0, 0),
        outline_color=None,
        outline_width=0,
        highlight_color=None,
        output_path=None
    ):
        """
        Overlay text on an image within a rotated polygon with custom styling
        
        Args:
            image: Path to input image, PIL Image, or numpy array
            text: Text to overlay
            polygon: Flat list of 8 elements [x1,y1,x2,y2,x3,y3,x4,y4] defining the rotated bounding box
            font_size: Size of the font in points
            font_color: RGB tuple for the text color
            outline_color: RGB tuple for the outline color (None for no outline)
            outline_width: Width of the outline in pixels
            highlight_color: RGB tuple for background highlight (None for transparent)
            output_path: Path to save the result (if None, returns the image)
            
        Returns:
            PIL Image with the overlaid text
        """
        # Load the image
        if isinstance(image, np.core.ndarray):
            img = Image.fromarray(image).convert("RGBA")
        elif isinstance(image, str) or isinstance(image, Path):
            img = Image.open(image).convert("RGBA")
        elif isinstance(image, Image.Image):
            img = image.convert("RGBA")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        img_width, img_height = img.size
        
        # Convert flat polygon array to list of points
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        points_np = np.array(points)
        
        # Calculate center point (centroid) of the polygon
        centroid_x = np.mean(points_np[:, 0])
        centroid_y = np.mean(points_np[:, 1])
        center = (centroid_x, centroid_y)
        
        # Get the minimum area rectangle using OpenCV
        rect = cv2.minAreaRect(points_np.astype(np.int32))
        _, (width, height), angle = rect
        
        # We'll use OpenCV's minAreaRect values directly to avoid 
        # relying on the order of points in the polygon
        
        # Handle rotation to keep text right-side up
        # OpenCV's minAreaRect returns angle in range [-90, 0)
        if width < height:
            angle += 90
            width, height = height, width
        
        # Normalize angle to prevent upside-down text
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
            
        # # Center coordinates for text placement
        # centroid_x, centroid_y = center
        
        # Draw highlight if needed
        if highlight_color is not None:
            highlight = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(highlight)
            
            # Draw with alpha
            # Convert highlight_color to tuple if it's a list and add alpha
            if isinstance(highlight_color, list):
                highlight_with_alpha = tuple(highlight_color) + (255,)
            else:
                highlight_with_alpha = highlight_color + (255,)  # Add alpha
            draw.polygon(points, fill=highlight_with_alpha)
            
            # Composite highlight onto the image
            img = Image.alpha_composite(img, highlight)
        
        # Get appropriate font
        font = self.get_font_for_text(text, font_size)
        
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
                              fill=tuple(outline_color))
        
        # Draw main text
        draw.text((text_x, text_y), text, font=font, fill=tuple(font_color))
        
        # Rotate the text
        rotated_text = text_img.rotate(-angle, resample=Image.BICUBIC, 
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

    def calculate_optimal_font_size(
        self, 
        text, 
        width, 
        height, 
        img_width, 
        img_height,
        angle,
        centroid_x,
        centroid_y,
        min_font_size=8,
        max_font_size=72,
        vertical_margin=0.1,
        horizontal_margin=0.1
    ):
        """
        Calculate the optimal font size to fit text in a rotated rectangle
        
        Args:
            text: Text to render
            width: Width of the bounding rectangle
            height: Height of the bounding rectangle
            img_width: Width of the destination image
            img_height: Height of the destination image
            angle: Rotation angle in degrees
            centroid_x: X coordinate of polygon centroid
            centroid_y: Y coordinate of polygon centroid
            min_font_size: Minimum acceptable font size
            max_font_size: Maximum font size to try
            vertical_margin: Vertical margin as fraction of height
            horizontal_margin: Horizontal margin as fraction of width
            
        Returns:
            optimal_font_size: The largest font size that fits within constraints
        """
        # Calculate available space with margins
        available_width = width * (1 - horizontal_margin)
        available_height = height * (1 - vertical_margin)
        
        # Binary search to find the optimal font size
        low = min_font_size
        high = max_font_size
        optimal_font_size = min_font_size
        
        while low <= high:
            mid = (low + high) // 2
            font = self.get_font_for_text(text, mid)
            
            # Get text size
            img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(img)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Check if text fits within available space
            if (text_height <= available_height and text_width <= available_width):
                # This size works, try a larger one
                optimal_font_size = mid
                low = mid + 1
            else:
                # Too big, try a smaller one
                high = mid - 1
        
        # Ensure the text doesn't exceed image dimensions
        # Check if the rotated text would fit within image bounds
        size_ok = False
        while not size_ok and optimal_font_size > min_font_size:
            # Create test text image
            test_font = self.get_font_for_text(text, optimal_font_size)
            
            # Get text dimensions
            test_img = Image.new('RGB', (1, 1))
            test_draw = ImageDraw.Draw(test_img)
            test_bbox = test_draw.textbbox((0, 0), text, font=test_font)
            test_width = test_bbox[2] - test_bbox[0]
            test_height = test_bbox[3] - test_bbox[1]
            
            # Create image with padding for rotation
            test_img_with_text = Image.new('RGBA', 
                (test_width + 20, test_height + 20), (0, 0, 0, 0))
            test_draw_with_text = ImageDraw.Draw(test_img_with_text)
            
            # Position text in center
            text_x = 10
            text_y = 10
            test_draw_with_text.text((text_x, text_y), text, 
                                    font=test_font, fill=(0, 0, 0))
            
            # Rotate to check dimensions
            rotated_test = test_img_with_text.rotate(-angle, 
                                                    resample=Image.BICUBIC, 
                                                    expand=True)
            
            # Calculate paste position
            paste_x = int(centroid_x - rotated_test.width / 2)
            paste_y = int(centroid_y - rotated_test.height / 2)
            
            # Check if it would fit within image bounds
            if (paste_x < 0 or paste_y < 0 or 
                paste_x + rotated_test.width > img_width or 
                paste_y + rotated_test.height > img_height):
                # Reduce size and try again
                optimal_font_size -= 1
            else:
                # This size works
                size_ok = True
        
        return optimal_font_size
    
    def overlay_fitted_text(
        self,
        image: Union[str, Path, Image.Image, np.core.ndarray],
        text,
        polygon,
        max_font_size=72,
        min_font_size=8,
        vertical_margin=0.1,  # fraction of height
        horizontal_margin=0.1,  # fraction of width
        max_extension_factor=1.5,  # max horizontal extension factor
        font_color=(0, 0, 0),
        outline_color=None,
        outline_width=0,
        highlight_color=None,
        output_path=None,
        other_polygons=None,  # List of other polygons to check for overlap
        max_overlap_threshold=0.1,  # Maximum allowed overlap (fraction of area)
    ):
        """
        Overlay text on an image with automatic font size adjustment to fit the polygon
        
        Args:
            image: Path to input image, PIL Image, or numpy array
            text: Text to overlay
            polygon: Flat list of 8 elements [x1,y1,x2,y2,x3,y3,x4,y4] defining the rotated bounding box
            max_font_size: Maximum font size to try
            min_font_size: Minimum font size to accept
            vertical_margin: Fraction of polygon height to leave as margin (0.1 = 10%)
            horizontal_margin: Fraction of polygon width to leave as margin (0.1 = 10%)
            max_extension_factor: Maximum extension factor for width
            font_color: RGB tuple for the text color
            outline_color: RGB tuple for the outline color (None for no outline)
            outline_width: Width of the outline in pixels
            highlight_color: RGB tuple for background highlight (None for transparent)
            output_path: Path to save the result (if None, returns the image)
            other_polygons: List of other polygons to check for overlap
            max_overlap_threshold: Maximum allowed overlap (fraction of area)
            
        Returns:
            PIL Image with the overlaid text, and new polygon if it was extended or split
        """
        # Load the image
        if isinstance(image, np.core.ndarray):
            img = Image.fromarray(image).convert("RGBA")
        elif isinstance(image, str) or isinstance(image, Path):
            img = Image.open(image).convert("RGBA")
        elif isinstance(image, Image.Image):
            img = image.convert("RGBA")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        img_width, img_height = img.size
        
        # Convert flat polygon array to list of points
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        points_np = np.array(points)
        
        # Calculate center point (centroid) of the polygon
        centroid_x = np.mean(points_np[:, 0])
        centroid_y = np.mean(points_np[:, 1])
        
        # Get the minimum area rectangle using OpenCV
        rect = cv2.minAreaRect(points_np.astype(np.int32))
        _, (width, height), angle = rect
        original_width, original_height = width, height
        
        # Handle rotation to keep text right-side up
        # OpenCV's minAreaRect returns angle in range [-90, 0)
        if width < height:
            angle += 90
            width, height = height, width
            original_width, original_height = original_height, original_width
        
        # Normalize angle to prevent upside-down text
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        
        # Calculate font size based on vertical height (prioritizing height fit)
        available_height = height * (1 - vertical_margin * 2)  # Allow margin on both top and bottom
        
        # Start with a font size proportional to the height
        vertical_font_size = int(available_height * 0.95)  # Heuristic: fonts typically have some internal padding
        vertical_font_size = min(max_font_size, max(min_font_size, vertical_font_size))
        
        # Create a test font to see if the text fits within the original width
        font = self.get_font_for_text(text, vertical_font_size)
        
        # Measure text dimensions
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Check if we need to extend the polygon
        needs_extension = text_width > width * (1 - horizontal_margin * 2)
        needs_splitting = False
        new_polygons = []
        
        if needs_extension:
            # Calculate how much we need to extend
            required_width = text_width / (1 - horizontal_margin * 2)
            extension_factor = required_width / width
            
            # Limit extension factor based on max_extension_factor and available image width
            max_width_based_factor = (img_width * 0.85) / width  # Limit to 85% of image width
            max_allowed_factor = min(max_extension_factor, max_width_based_factor)
            
            if extension_factor > max_allowed_factor:
                if text and ' ' in text:
                    # Text is too long and needs splitting
                    needs_splitting = True
                    split_point = len(text) // 2
                    # Find nearest space to split at
                    while split_point > 0 and split_point < len(text) - 1:
                        if text[split_point] == ' ':
                            break
                        split_point -= 1
                    
                    if split_point == 0:  # No suitable split found, try from the middle to the end
                        split_point = len(text) // 2
                        while split_point < len(text) - 1:
                            if text[split_point] == ' ':
                                break
                            split_point += 1
                    
                    if split_point > 0 and split_point < len(text) - 1:
                        part1 = text[:split_point].strip()
                        part2 = text[split_point:].strip()
                        
                        # Original polygon points and center
                        points_np = np.array(points)
                        original_centroid_x = centroid_x
                        original_centroid_y = centroid_y
                        
                        # Create two new polygons - one moved up and one moved down
                        # Calculate vertical offset - half the height of the polygon
                        offset_distance = height / 2.0
                        
                        # Create offsets for the two polygons
                        up_offset = -offset_distance * 0.75  # Move up by 3/4 of half height
                        down_offset = offset_distance * 0.75  # Move down by 3/4 of half height
                        
                        # Create candidate polygons at these positions
                        polygon1_points = points_np.copy()
                        polygon1_points[:, 1] += up_offset
                        
                        polygon2_points = points_np.copy()
                        polygon2_points[:, 1] += down_offset
                        
                        # Convert to flat arrays
                        polygon1 = polygon1_points.flatten().tolist()
                        polygon2 = polygon2_points.flatten().tolist()
                        
                        # Check if either polygon is outside image boundaries
                        polygon1_outside = False
                        polygon2_outside = False
                        
                        # Check polygon1 (upper polygon)
                        for i in range(0, len(polygon1), 2):
                            if polygon1[i] < 0 or polygon1[i] >= img_width or polygon1[i+1] < 0 or polygon1[i+1] >= img_height:
                                polygon1_outside = True
                                break
                                
                        # Check polygon2 (lower polygon)
                        for i in range(0, len(polygon2), 2):
                            if polygon2[i] < 0 or polygon2[i] >= img_width or polygon2[i+1] < 0 or polygon2[i+1] >= img_height:
                                polygon2_outside = True
                                break
                        
                        # If polygon1 is outside and polygon2 is not, just use polygon2 at the original position
                        if polygon1_outside and not polygon2_outside:
                            polygon1 = polygon.copy()  # Use original position for polygon1
                            polygon2_points = points_np.copy()
                            polygon2_points[:, 1] += height * 0.9  # Move down by almost the full height
                            polygon2 = polygon2_points.flatten().tolist()
                        
                        # If polygon2 is outside and polygon1 is not, just use polygon1 at the original position
                        elif polygon2_outside and not polygon1_outside:
                            polygon2 = polygon.copy()  # Use original position for polygon2
                            polygon1_points = points_np.copy()
                            polygon1_points[:, 1] -= height * 0.9  # Move up by almost the full height
                            polygon1 = polygon1_points.flatten().tolist()
                        
                        # If both are outside, try to fit both within the image
                        elif polygon1_outside and polygon2_outside:
                            # Try to move polygons to fit within image
                            # Find min/max Y values for the original polygon
                            min_y = min(points_np[:, 1])
                            max_y = max(points_np[:, 1])
                            
                            # Calculate available space above and below
                            space_above = min_y
                            space_below = img_height - max_y
                            
                            # If more space above, prioritize moving polygon1 up and polygon2 to original
                            if space_above > space_below:
                                # If enough space above for polygon1
                                if space_above > height:
                                    max_up_offset = min(space_above - 10, height * 0.9)  # Leave 10px margin
                                    polygon1_points = points_np.copy()
                                    polygon1_points[:, 1] -= max_up_offset
                                    polygon1 = polygon1_points.flatten().tolist()
                                    polygon2 = polygon.copy()  # Keep polygon2 at original position
                                else:
                                    # Not enough space, try to squeeze both where they fit
                                    polygon1 = polygon.copy()
                                    polygon2_points = points_np.copy()
                                    max_down_offset = min(space_below - 10, height * 0.5)
                                    if max_down_offset > 20:  # If we can move down by at least 20px
                                        polygon2_points[:, 1] += max_down_offset
                                        polygon2 = polygon2_points.flatten().tolist()
                                    else:
                                        # Almost no space to move, try splitting horizontally instead
                                        # Create side-by-side polygons (experimental)
                                        polygon1 = polygon.copy()
                                        
                                        # Create a horizontally shifted polygon (if there's width to spare)
                                        if width > 100:  # Only try if there's enough width
                                            polygon2_points = points_np.copy()
                                            # Shift by width and adjust position to avoid going off-screen
                                            shift_amount = min(width * 0.8, img_width - max(points_np[:, 0]) - 20)
                                            if shift_amount > 50:  # Only if we can shift by a meaningful amount
                                                polygon2_points[:, 0] += shift_amount
                                                polygon2 = polygon2_points.flatten().tolist()
                                            else:
                                                # No good options, use original polygon for both
                                                polygon2 = polygon.copy()
                                        else:
                                            # No good options, use original polygon for both
                                            polygon2 = polygon.copy()
                            else:
                                # More space below, prioritize moving polygon2 down and polygon1 to original
                                if space_below > height:
                                    max_down_offset = min(space_below - 10, height * 0.9)  # Leave 10px margin
                                    polygon2_points = points_np.copy()
                                    polygon2_points[:, 1] += max_down_offset
                                    polygon2 = polygon2_points.flatten().tolist()
                                    polygon1 = polygon.copy()  # Keep polygon1 at original position
                                else:
                                    # Similar fallback as above
                                    polygon2 = polygon.copy()
                                    polygon1_points = points_np.copy()
                                    max_up_offset = min(space_above - 10, height * 0.5)
                                    if max_up_offset > 20:
                                        polygon1_points[:, 1] -= max_up_offset
                                        polygon1 = polygon1_points.flatten().tolist()
                                    else:
                                        # Almost no space to move, horizontal splitting as above
                                        polygon1 = polygon.copy()
                                        
                                        if width > 100:
                                            polygon2_points = points_np.copy()
                                            shift_amount = min(width * 0.8, img_width - max(points_np[:, 0]) - 20)
                                            if shift_amount > 50:
                                                polygon2_points[:, 0] += shift_amount
                                                polygon2 = polygon2_points.flatten().tolist()
                                            else:
                                                polygon2 = polygon.copy()
                                        else:
                                            polygon2 = polygon.copy()
                        
                        # Check for overlaps with other polygons
                        if other_polygons:
                            # Calculate overlap scores for different polygon placements
                            def calculate_overlap(poly, other_polys):
                                """Calculate total overlap ratio for a polygon against others"""
                                if not other_polys:
                                    return 0.0
                                
                                poly_points = np.array([
                                    (poly[i], poly[i+1]) for i in range(0, len(poly), 2)
                                ]).astype(np.int32)
                                
                                poly_area = cv2.contourArea(poly_points)
                                if poly_area == 0:
                                    return 0.0  # Avoid division by zero
                                
                                # Create mask for the polygon
                                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                                cv2.fillPoly(mask, [poly_points], 1)
                                
                                # Create a single combined mask for all other polygons
                                other_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                                for other_poly in other_polys:
                                    if np.allclose(other_poly, polygon, rtol=1e-5, atol=1e-8):  # Skip original polygon
                                        continue
                                    
                                    other_points = np.array([
                                        (other_poly[i], other_poly[i+1]) 
                                        for i in range(0, len(other_poly), 2)
                                    ]).astype(np.int32)
                                    
                                    # Fill the combined mask
                                    cv2.fillPoly(other_mask, [other_points], 1)
                                
                                # Calculate overlap only once using the combined mask
                                overlap_area = np.sum(np.logical_and(mask, other_mask))
                                overlap_ratio = overlap_area / poly_area
                                
                                return overlap_ratio
                            
                            # Visualize the polygons for debugging
                            def debug_visualize_polygon_overlap():
                                # Prepare polygon data
                                all_polygons = [polygon1] + other_polygons
                                labels = ["polygon1"] + [f"other_poly{i}" for i in range(len(other_polygons))]
                                colors = [(255, 0, 0)] + [(0, 0, 255) for _ in range(len(other_polygons))]
                                
                                # Determine image dimensions based on polygons
                                max_x = max(max(poly[i] for i in range(0, len(poly), 2)) for poly in all_polygons)
                                max_y = max(max(poly[i+1] for i in range(0, len(poly), 2)) for poly in all_polygons)
                                img_width = max(int(max_x * 1.2), 800)
                                img_height = max(int(max_y * 1.2), 600)
                                
                                # Visualize the polygons
                                visualize_polygons(
                                    all_polygons, 
                                    labels=labels,
                                    colors=colors,
                                    img_width=img_width,
                                    img_height=img_height,
                                    save_path='tmp_polygons_overlap.png',
                                    title=f"Polygon1 Overlap: {polygon1_overlap:.2f}"
                                )
                            
                            # Calculate overlaps for our two polygons
                            original_overlap = calculate_overlap(polygon, other_polygons)
                            polygon1_overlap = calculate_overlap(polygon1, other_polygons)
                            polygon2_overlap = calculate_overlap(polygon2, other_polygons)
                            debug_visualize_polygon_overlap()
                            
                            # If both have high overlap, try adjusting positions
                            if polygon1_overlap > max_overlap_threshold or polygon2_overlap > max_overlap_threshold:
                                # Try different positions with different offsets
                                best_poly1 = polygon1
                                best_poly2 = polygon2
                                best_total_overlap = polygon1_overlap + polygon2_overlap
                                
                                # Try various offsets to find the position with least overlap
                                for p1_offset in [-height, -height*0.75, -height*0.5, -height*0.25, 0, height*0.25]:
                                    for p2_offset in [0, height*0.25, height*0.5, height*0.75, height, height*1.25]:
                                        # Skip if p1 and p2 are too close
                                        if abs(p1_offset - p2_offset) < height * 0.4:
                                            continue
                                            
                                        # Create test polygons
                                        test_poly1_points = points_np.copy()
                                        test_poly1_points[:, 1] += p1_offset
                                        test_poly1 = test_poly1_points.flatten().tolist()
                                        
                                        test_poly2_points = points_np.copy()
                                        test_poly2_points[:, 1] += p2_offset
                                        test_poly2 = test_poly2_points.flatten().tolist()
                                        
                                        # Skip if either polygon is outside image boundaries
                                        outside = False
                                        for poly in [test_poly1, test_poly2]:
                                            for i in range(0, len(poly), 2):
                                                if (poly[i] < 0 or poly[i] >= img_width or 
                                                    poly[i+1] < 0 or poly[i+1] >= img_height):
                                                    outside = True
                                                    break
                                            if outside:
                                                break
                                        
                                        if outside:
                                            continue
                                        
                                        # Calculate total overlap
                                        other_polys_excluding_test = [p for p in other_polygons 
                                                                      if not np.allclose(p, polygon)]
                                        overlap1 = calculate_overlap(test_poly1, other_polys_excluding_test + [test_poly2])
                                        overlap2 = calculate_overlap(test_poly2, other_polys_excluding_test + [test_poly1])
                                        total_overlap = overlap1 + overlap2
                                        
                                        if total_overlap < best_total_overlap:
                                            best_total_overlap = total_overlap
                                            best_poly1 = test_poly1
                                            best_poly2 = test_poly2
                                
                                # Use the best positions found
                                polygon1 = best_poly1
                                polygon2 = best_poly2
                                
                                debug_visualize_polygon_overlap()

                        # Instead of recursively applying texts one after another, apply them to separate images first
                        # Create a copy of the original image for the part1 rendering
                        img_part1 = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
                        
                        # Render part1 on the first image copy
                        part1_result = self.overlay_rotated_text(
                            img_part1,
                            part1,
                            polygon1,
                            font_size=vertical_font_size,
                            font_color=font_color,
                            outline_color=outline_color,
                            outline_width=outline_width,
                            highlight_color=highlight_color
                        )
                        
                        # Create another copy of the original image for the part2 rendering
                        img_part2 = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
                        
                        # Render part2 on the second image copy
                        part2_result = self.overlay_rotated_text(
                            img_part2,
                            part2,
                            polygon2,
                            font_size=vertical_font_size,
                            font_color=font_color,
                            outline_width=outline_width,
                            outline_color=outline_color,
                            highlight_color=highlight_color
                        )
                        
                        # Now composite the two text renderings onto the original image
                        # Extract just the text parts with alpha from part1_result and part2_result
                        # by comparing with original image
                        original_array = np.array(img)
                        part1_array = np.array(part1_result)
                        part2_array = np.array(part2_result)
                        
                        # Create a new result image starting with the original
                        final_result = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
                        final_array = np.array(final_result)
                        
                        # For each rendering, find the changed pixels and apply them to the final image
                        # This is a simple approach that works if highlight_color is None
                        # Otherwise, a more sophisticated alpha compositing would be needed
                        if highlight_color is None:
                            # Apply where part1 is different from original
                            mask1 = np.any(part1_array != original_array, axis=2)
                            final_array[mask1] = part1_array[mask1]
                            
                            # Apply where part2 is different from original
                            mask2 = np.any(part2_array != original_array, axis=2)
                            final_array[mask2] = part2_array[mask2]
                            
                            # Convert back to PIL Image
                            final_result = Image.fromarray(final_array)
                        else:
                            # When highlight_color is used, we need to use alpha compositing
                            # Convert to RGBA if not already
                            if final_result.mode != 'RGBA':
                                final_result = final_result.convert('RGBA')
                            
                            # Composite part1 result onto final result
                            if part1_result.mode != 'RGBA':
                                part1_result = part1_result.convert('RGBA')
                            final_result = Image.alpha_composite(final_result, part1_result)
                            
                            # Composite part2 result onto final result
                            if part2_result.mode != 'RGBA':
                                part2_result = part2_result.convert('RGBA')
                            final_result = Image.alpha_composite(final_result, part2_result)
                        
                        # Save or return the result with both polygons
                        new_polygons = [polygon1, polygon2]
                        if output_path:
                            final_result.save(output_path)
                            return output_path, new_polygons
                        else:
                            return final_result, new_polygons
                
                # If we can't split or the extension is too large, reduce font size
                adjusted_font_size = self.calculate_optimal_font_size(
                    text, 
                    width * max_extension_factor, 
                    height, 
                    img_width, 
                    img_height,
                    angle,
                    centroid_x,
                    centroid_y,
                    min_font_size,
                    vertical_font_size,  # Cap at the vertical font size
                    vertical_margin,
                    horizontal_margin
                )
                
                # Use the adjusted font size
                return self.overlay_rotated_text(
                    img,
                    text,
                    polygon,
                    font_size=adjusted_font_size,
                    font_color=font_color,
                    outline_color=outline_color,
                    outline_width=outline_width,
                    highlight_color=highlight_color,
                    output_path=output_path
                )
            
            else:
                # We can extend the polygon
                extended_width = width * extension_factor
                
                # Create an extended polygon by scaling horizontally
                # Get the angle in radians
                angle_rad = math.radians(angle)
                
                # Calculate the direction vectors for width and height
                width_dir_x = math.cos(angle_rad)
                width_dir_y = math.sin(angle_rad)
                height_dir_x = -math.sin(angle_rad)
                height_dir_y = math.cos(angle_rad)
                
                # Scale factors
                width_scale = extension_factor
                height_scale = 1.0  # Keep height the same
                
                # Create new corners based on the scaled width and height
                half_width = extended_width / 2
                half_height = height / 2
                
                # Generate new corners
                corners = []
                corners.append((
                    centroid_x - width_dir_x * half_width - height_dir_x * half_height,
                    centroid_y - width_dir_y * half_width - height_dir_y * half_height
                ))
                corners.append((
                    centroid_x + width_dir_x * half_width - height_dir_x * half_height,
                    centroid_y + width_dir_y * half_width - height_dir_y * half_height
                ))
                corners.append((
                    centroid_x + width_dir_x * half_width + height_dir_x * half_height,
                    centroid_y + width_dir_y * half_width + height_dir_y * half_height
                ))
                corners.append((
                    centroid_x - width_dir_x * half_width + height_dir_x * half_height,
                    centroid_y - width_dir_y * half_width + height_dir_y * half_height
                ))
                
                # Convert corners to flat array
                extended_polygon = [coord for corner in corners for coord in corner]
                
                # Ensure the extended polygon stays within image boundaries
                extended_polygon = clip_polygon_to_image_bounds(extended_polygon, img_width, img_height)
                
                # Check if extension causes overlap with other polygons
                if other_polygons:
                    # Calculate extended polygon area
                    extended_points_np = np.array([
                        (extended_polygon[i], extended_polygon[i+1]) 
                        for i in range(0, len(extended_polygon), 2)
                    ])
                    extended_area = cv2.contourArea(extended_points_np.astype(np.int32))
                    
                    excessive_overlap = False
                    for other_poly in other_polygons:
                        if np.allclose(other_poly, polygon, rtol=1e-5, atol=1e-8):  # Skip self
                            continue
                        
                        # Convert other polygon to points array
                        other_points = [(other_poly[i], other_poly[i+1]) for i in range(0, len(other_poly), 2)]
                        other_points_np = np.array(other_points).astype(np.int32)
                        
                        # Calculate intersection area
                        intersection_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(intersection_mask, [extended_points_np.astype(np.int32)], 1)
                        
                        other_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(other_mask, [other_points_np], 1)
                        
                        intersection_area = np.sum(np.logical_and(intersection_mask, other_mask))
                        overlap_ratio = intersection_area / extended_area
                        
                        if overlap_ratio > max_overlap_threshold:
                            excessive_overlap = True
                            break
                    
                    if excessive_overlap:
                        # Fallback to the optimal font size without extension
                        adjusted_font_size = self.calculate_optimal_font_size(
                            text, 
                            width, 
                            height, 
                            img_width, 
                            img_height,
                            angle,
                            centroid_x,
                            centroid_y,
                            min_font_size,
                            vertical_font_size,  # Cap at the vertical font size
                            vertical_margin,
                            horizontal_margin
                        )
                        
                        # Use the original polygon with adjusted font size
                        return self.overlay_rotated_text(
                            img,
                            text,
                            polygon,
                            font_size=adjusted_font_size,
                            font_color=font_color,
                            outline_color=outline_color,
                            outline_width=outline_width,
                            highlight_color=highlight_color,
                            output_path=output_path
                        )
                
                # No overlap or no other polygons to check - proceed with extension
                # Use the vertical font size with the extended polygon
                img_result = self.overlay_rotated_text(
                    img,
                    text,
                    extended_polygon,
                    font_size=vertical_font_size,
                    font_color=font_color,
                    outline_color=outline_color,
                    outline_width=outline_width,
                    highlight_color=highlight_color,
                    output_path=output_path
                )
                
                # Return the extended polygon along with the result
                if output_path:
                    return output_path, [extended_polygon]
                else:
                    return img_result, [extended_polygon]
        
        # If we don't need extension, use the vertical font size
        result = self.overlay_rotated_text(
            img,
            text,
            polygon,
            font_size=vertical_font_size,
            font_color=font_color,
            outline_color=outline_color,
            outline_width=outline_width,
            highlight_color=highlight_color,
            output_path=output_path
        )
        
        # No changes to the polygon
        if output_path:
            return output_path, [polygon]
        else:
            return result, [polygon]

def visualize_polygons(polygons, labels=None, img_width=800, img_height=600, save_path='tmp__polygons_overlap.png',
                      colors=None, background_color=(255, 255, 255), draw_labels=True, title=None):
    """
    Visualize multiple polygons on a canvas and save the image.
    
    Args:
        polygons: List of polygons, each as a flat list [x1,y1,x2,y2,...]
        labels: Optional list of labels for each polygon
        img_width: Width of the output image
        img_height: Height of the output image
        save_path: Path to save the output image
        colors: List of RGB colors for each polygon, or None for automatic colors
        background_color: RGB color for the background
        draw_labels: Whether to draw labels
        title: Optional title for the image
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    import os
    
    # Create blank image
    img = Image.new('RGB', (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Generate colors if not provided
    if colors is None:
        colors = []
        for _ in range(len(polygons)):
            r = random.randint(0, 200)  # Keep under 200 to ensure contrast
            g = random.randint(0, 200)
            b = random.randint(0, 200)
            colors.append((r, g, b))
    
    # Ensure labels list is the same length as polygons
    if labels is None:
        labels = [f"Polygon {i+1}" for i in range(len(polygons))]
    elif len(labels) < len(polygons):
        labels.extend([f"Polygon {i+1}" for i in range(len(labels), len(polygons))])
    
    # Draw each polygon
    for i, poly in enumerate(polygons):
        # Convert flat array to list of points
        points = [(poly[j], poly[j+1]) for j in range(0, len(poly), 2)]
        
        # Draw polygon
        draw.polygon(points, outline=(0, 0, 0), fill=colors[i] + (128,))  # Add alpha
        
        # Draw points as circles
        for point in points:
            draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill=(255, 0, 0))
        
        # Draw label
        if draw_labels:
            # Calculate centroid for label placement
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Draw label with contrasting color
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except:
                    font = ImageFont.load_default()
            
            # Draw white background for text for better readability
            text_bbox = draw.textbbox((centroid_x, centroid_y), labels[i], font=font)
            padding = 2
            draw.rectangle(
                (text_bbox[0]-padding, text_bbox[1]-padding, 
                 text_bbox[2]+padding, text_bbox[3]+padding), 
                fill=(255, 255, 255)
            )
            
            draw.text((centroid_x, centroid_y), labels[i], fill=(0, 0, 0), font=font, anchor="mm")
    
    # Draw title
    if title:
        try:
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                title_font = ImageFont.load_default()
        
        draw.text((img_width//2, 20), title, fill=(0, 0, 0), font=title_font, anchor="mm")
    
    # Save image
    img.save(save_path)
    print(f"Polygons visualization saved to {os.path.abspath(save_path)}")
    return save_path

# Example usage
if __name__ == "__main__":
    # Initialize renderer
    renderer = MultilingualTextRenderer()
    
    # Example polygon (rotated bounding box)
    # polygon = [(100, 100), (300, 150), (250, 250), (50, 200)]
    polygon = [100, 100, 300, 150, 250, 250, 50, 200]
    
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
        
        # Let the renderer handle font selection automatically
        output = renderer.overlay_fitted_text(
            "/data/projects/OCR-SAM/imgs/ex12.jpg",
            text,
            polygon,
            max_font_size=72,
            min_font_size=8,
            vertical_margin=0.1,
            horizontal_margin=0.1,
            font_color=(0, 0, 255),
            outline_color=(255, 255, 255),
            outline_width=2,
            highlight_color=(255, 255, 0),
            output_path=f"output_{name.lower()}.jpg"
        )
        print(f"Rendered {name} text")