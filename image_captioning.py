from functools import partial
import base64
import os
from io import BytesIO
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from datetime import datetime
import matplotlib.pyplot as plt
# Add these imports at the top with other imports
import matplotlib.font_manager as fm
from pydantic import BaseModel

from linguana.gcp import delete_from_gcs, generate_signed_url, get_secret

OPENAI_API_KEY_PROJECT_ID = "357470755328"  # supertale-main
OPENAI_API_KEY_SECRET_ID =  "image-eraser-openai"

os.environ["OPENAI_API_KEY"] = get_secret(OPENAI_API_KEY_PROJECT_ID, OPENAI_API_KEY_SECRET_ID)

logging.basicConfig(level=logging.INFO)

OCR_USER_PROMPT = """
Extract and recognize any text in the image. 
If no text is found, return 'None'. 
Do not add any explanations or additional text.
"""
OCR_SYSTEM_PROMPT = """
You are an expert in text recognition and OCR. 
Your task is to extract and recognize text from images, 
including non-Latin scripts like Arabic, Japanese, Korean, etc. 
Return only the recognized text exactly as it appears, or 'None' if no text is found.
"""

def format_lines(num_words):
    lines = []
    for i in range(num_words):
        lines.append(f'Line {i+1}: ...')
    return '\n'.join(lines)

OCR_USE_PROMPT_SINGLE_SHOT = lambda num_lines: f"""
Transcribe the text in the image.
your output should be in the following format:
{format_lines(num_lines)}
Each line must correspond to a line in the original text in the image
Return only the transcription, without additional words or explanations.
Return 'None' if unable to transcribe or if no text found in the image.
"""

OCR_USE_PROMPT_SINGLE_LINE = """
Transcribe the text in the image.
Return only the transcription, without additional words or explanations.
Return 'None' if unable to transcribe or if no text found in the image.
"""

OCR_COUNT_LINES_USER_PROMPT = """
How many text lines appear in the image?
Return a single numeric output. 
Return 0 if no text was detected.
"""

LANGUAGE_DETECTION_USER_PROMPT = """
Detect the language of the text in the image.
Return the language code in ISO 639-1 format, or 'None' if no text is found.
"""
LANGUAGE_DETECTION_SYSTEM_PROMPT = """
You are an expert in language detection.
Your task is to detect the language of the text in the image.
Return the language code in ISO 639-1 format, or 'None' if no text is found.
"""

TEXT_LANGUAGE_DETECTION_USER_PROMPT = lambda text: f"""
Detect the language of the following text: {text}
Return the language code in ISO 639-1 format, or 'None' if no text is found.
"""
TEXT_LANGUAGE_DETECTION_SYSTEM_PROMPT = """
You are an expert in language detection.
Your task is to detect the language of a given text.
Return the language code in ISO 639-1 format, or 'None' if no text is found.
"""

WORDS_ORDER_USER_PROMPT = lambda words: f"""
Given the image and the following transcribed words: {words}, determine their natural visual reading order based solely on the spatial layout of the image.

⚠️ Do not assume any default reading direction (e.g., LTR, RTL, TTB). Instead, use only visual context—such as positioning, alignment, grouping, and size—to infer how a human would naturally read these words.

Return the output strictly in the following format:
Line 1: word1, word2, ...
Line 2: word3, word4, ...
Include only this structured result. Do not include explanations, comments, or extra formatting.
"""
WORDS_ORDER_SYSTEM_PROMPT = """
You are a vision-language assistant that specializes in interpreting text layouts from images, especially those containing multilingual or mixed-script content. Your task is to determine the natural visual reading order of a set of transcribed words based on their spatial arrangement in the image, not on linguistic assumptions.

Always prioritize visual cues such as alignment, positioning, grouping, orientation, and relative proximity. Do not rely on default reading directions (e.g., left-to-right) unless clearly supported by the visual context.

Your output should be the sequence of transcribed words arranged according to their inferred natural visual reading order.
"""

FONT_ANALYSIS_USER_PROMPT = """
what is the font color, outline, style and highlight colors of the text in the image?
If there is a color gradient in the font color, select the average RGB color.
Your answer should be in the following json format:
"""
FONT_ANALYSIS_SYSTEM_PROMPT = """
You are an expert in font analysis.
Your task is to analyze the font color, outline, style and highlight color of the text in the image.
Return the result in the following json format:
Do not add any explanations or additional text or other characters such as ```, escape characters, json, etc.
"""


class FontAnalysisFormat(BaseModel):
    font_color: str
    font_color_rgb: List[int]
    outline_color: str
    outline_color_rgb: List[int]
    outline_width: int
    highlight_color_exist: bool
    highlight_color: str
    highlight_color_rgb: List[int]
    font_style_italic: bool
    font_style_bold: bool
    font_style_underline: bool
    font_style_strikethrough: bool
    
    class Config:
        json_schema_extra = {
            "properties": {
                "font_color": {"type": "string"},
                "font_color_rgb": {"type": "array", "items": {"type": "integer"}},
                "outline_color": {"type": "string"},
                "outline_color_rgb": {"type": "array", "items": {"type": "integer"}},
                "outline_width": {"type": "integer"},
                "highlight_color_exist": {"type": "boolean"},
                "highlight_color": {"type": "string"},
                "highlight_color_rgb": {"type": "array", "items": {"type": "integer"}},
                "font_style_italic": {"type": "boolean"},
                "font_style_bold": {"type": "boolean"},
                "font_style_underline": {"type": "boolean"},
                "font_style_strikethrough": {"type": "boolean"},
            }
        }

TRANSLATE_USER_PROMPT = lambda src_text_lines, target_language: f"""
you are an AI text translator.
Translate the following text to {target_language} in {len(src_text_lines)} lines.
Do not add additional text, explanations, glossary or anything else other than the translated text.
You'll need to follow the following steps:
1. Concatenate the texts.
2, Translate the entire text.
3. Split the translated text to exactly {len(src_text_lines)} lines, separarated by newlines: {format_lines(len(src_text_lines))}

{src_text_lines}
"""

TRANSLATE_SYSTEM_PROMPT = lambda target_language: f"""
You are an AI text translator.
Translate the following text to {target_language}.
Do not add additional text, explanations, glossary or anything else other than the translated text.
"""
TRANSLATE_USER_PROMPT_WITH_CONTEXT = lambda src_text_lines, target_language, context: f"""
{TRANSLATE_USER_PROMPT(src_text_lines, target_language)}

Use the following context to help you translate the text:
{context}
"""

def translate_text(src_text_lines: List[str], target_language: str = "English", llm_model: str = "gpt-4o", pass_through_english: bool = True, context: Optional[str] = None):  # -> List[str]:
    """
    Translate the text to English.
    """
    client = OpenAI()
    
    # if the language of any of the src_text_lines is English, translate it to English first, and then to the target language
    translate_to_english = False
    if pass_through_english:
        for line in src_text_lines:
            
            response = client.chat.completions.create(messages=[
                {"role": "system", "content": TEXT_LANGUAGE_DETECTION_USER_PROMPT(line)},
                {"role": "user", "content": TEXT_LANGUAGE_DETECTION_SYSTEM_PROMPT}
            ],
            model=llm_model,
            )
            if response.choices[0].message.content.strip().lower() != "english":
                translate_to_english = True
                break
    
    if translate_to_english and target_language != "English":
    
        response_en_text = client.chat.completions.create(messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT("English")},
            {"role": "user", "content": TRANSLATE_USER_PROMPT(src_text_lines, "English") if context is None else TRANSLATE_USER_PROMPT_WITH_CONTEXT(src_text_lines, "English", context)}
        ],
        model=llm_model,
        ).choices[0].message.content
        response_en_text = list(map(lambda x: x.strip(), response_en_text.split('\n')))
        
        
        context_en = client.chat.completions.create(messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT("English")},
            {"role": "user", "content": TRANSLATE_USER_PROMPT(context.split('\n'), "English") if context is not None else TRANSLATE_USER_PROMPT(context.split('\n'), "English")}
        ],
        model=llm_model,
        ).choices[0].message.content
        # context_en = list(map(lambda x: x.strip(), context_en.split('\n')))
        
        response_target_text = client.chat.completions.create(messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT(target_language)},
            {"role": "user", "content": TRANSLATE_USER_PROMPT(response_en_text, target_language) if context is None else TRANSLATE_USER_PROMPT_WITH_CONTEXT(response_en_text, target_language, context_en)}
        ],
        model=llm_model,
        ).choices[0].message.content
    else:
        response_target_text = client.chat.completions.create(messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT(target_language)},
            {"role": "user", "content": TRANSLATE_USER_PROMPT(src_text_lines, target_language) if context is None else TRANSLATE_USER_PROMPT_WITH_CONTEXT(src_text_lines, target_language, context)}
        ],
        model=llm_model,
        ).choices[0].message.content
        
    return response_target_text


def image_captioning(
    client: OpenAI, 
    image: Union[str, Path, np.core.ndarray], 
    prompt: str, system_prompt: str = "You are an AI assistant.", 
    model: str = "gpt-4o", 
    response_format: Optional[BaseModel] = None, 
    ) -> str:
    """
    Generate a caption for an image using OpenAI's GPT-4 model.
    """
    image = Path(image) if isinstance(image, str) else image

    if str(image).startswith("http"):  # URL
        image = {"url": str(image)}
    elif str(image).startswith("gs://"):  # GCS file
        signed_url = generate_signed_url(bucket_name="linguana-models", blob=str(image).split("/", 3)[-1])
        image = {"url": signed_url}
    elif isinstance(image, np.core.ndarray):
        # Ensure array is C-contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        pil_image = Image.fromarray(image)
        # Convert PIL image to base64 without saving to file
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image = {"url": f"data:image/png;base64,{img_str}"}
    elif image.exists():  # local file or numpy array
        # Convert local file to base64
        with image.open("rb") as image_file:
            img_str = base64.b64encode(image_file.read()).decode()
        image = {"url": f"data:image/png;base64,{img_str}"}
    else:
        raise ValueError(f"Invalid image path: {image}")

    if response_format is not None:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image}
                    ]
                 },
                ],
            response_format=response_format,
        )
    else:
        # API request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image}
                    ]
                 },
                ],
        )
    return response.choices[0].message.content


def image_editing_prompt(prompt: str, llm_model: str = "gpt-4o", masked_image=None, target_img_caption=True, max_retries: int = 3):
    '''
    Generate image inpainting prompt based on masked image or image description
    Args:
        prompt: original image description
        llm_model: LLM model name
        masked_image: PIL Image with masked region
        target_img_caption: whether to use masked image for caption generation
    Returns:
        prompt: original image description
        image_inpainting_prompt: static description for inpainting
    '''
    if prompt is None:
        raise ValueError("prompt is None")
        
    vlm_model = OpenAI()
    image_inpainting_prompt = None
    retries = 0
    while image_inpainting_prompt is None or image_inpainting_prompt.lower().startswith("i'm sorry, i can't"):
        retries += 1
        if retries > max_retries:
            print("ERROR: Failed to generate inpainting prompt. Return original prompt.")
            return prompt, prompt
        if target_img_caption:
            if masked_image is None:
                raise ValueError("masked_image is None when target_img_caption=True")
                
            # Convert PIL image to base64
            buffered = BytesIO()
            masked_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            system_prompt = "You are an expert in image description. Based on the given masked image, please generate a concise description for target for following inpainting. the target is the unmasked region in the masked image, so the description should be focused on the unmasked region."
            
            user_prompt = """Please generate a description for the unmasked target in the given masked image. Requirements:
            1. Keep the description concise and precise
            2. Only describe unmasked visual elements
            3. Black background is not a visual element
            Only return the description, no other words."""  # noqa: E501

            # Call OpenAI vision API with image
            response = vlm_model.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ]
            )
        else:
            # Original text-only prompt processing
            system_prompt = "You are an expert in image description. Based on the given image description, please generate a concise description, focusing on the most important visual elements."
            
            user_prompt = f"""Image description: {prompt}
            Please generate a static description for the image. Requirements:
            1. Keep the description concise and precise
            2. Only describe key visual elements
            3. Any text is considered as foreground. Refer to the background to be filled.
            Only return the description, no other words."""
            
            response = vlm_model.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

        image_inpainting_prompt = response.choices[0].message.content
    return prompt, image_inpainting_prompt




# Add this configuration after the logger setup but before any plotting code (around line 40-50)
# Configure matplotlib to use fonts that support multiple languages
def configure_multilingual_fonts():
    import os
    from pathlib import Path
    import sys
    
    # Print available fonts for debugging (comment out in production)
    # available_fonts = set(fm.get_font_names())
    # print("Available fonts:", sorted([f for f in available_fonts]))
    
    # Common system font locations for finding Asian fonts
    font_locations = []
    
    if sys.platform == 'darwin':  # macOS
        font_locations.extend([
            '/System/Library/Fonts',
            '/Library/Fonts',
            str(Path.home() / 'Library/Fonts')
        ])
    elif sys.platform == 'linux':  # Linux
        font_locations.extend([
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            str(Path.home() / '.fonts')
        ])
    elif sys.platform == 'win32':  # Windows
        font_locations.extend([
            'C:/Windows/Fonts'
        ])
    
    # Scan for font files and add them
    for location in font_locations:
        if Path(location).exists():
            for font_ext in ['.ttf', '.otf', '.ttc']:
                font_files = list(Path(location).glob(f'**/*{font_ext}'))
                for font_file in font_files:
                    if any(asian_name in str(font_file).lower() for asian_name in 
                          ['noto', 'cjk', 'chinese', 'korean', 'japanese', 'thai', 
                           'arabic', 'hindi', 'tamil', 'nanum', 'heiti', 'mingliu', 
                           'meiryo', 'msmincho', 'yugothic']):
                        try:
                            fm.fontManager.addfont(str(font_file))
                        except Exception:
                            pass
    
    # Create comprehensive fallback list with a wide range of Asian fonts
    asian_fonts = [
        # CJK (Chinese, Japanese, Korean) fonts
        'Noto Sans CJK JP', 'Noto Sans CJK KR', 'Noto Sans CJK SC', 'Noto Sans CJK TC',
        'Source Han Sans', 'Source Han Serif', 
        'NanumGothic', 'Malgun Gothic', 'Batang', 'Gulim',           # Korean
        'MS Gothic', 'Meiryo', 'Yu Gothic', 'Hiragino Sans',          # Japanese
        'SimHei', 'SimSun', 'Microsoft YaHei', 'WenQuanYi Zen Hei',   # Chinese
        'PMingLiU', 'MingLiU',                                        # Traditional Chinese
        
        # South/Southeast Asian scripts
        'Noto Sans Thai', 'Leelawadee UI', 'Tahoma', 
        'Noto Sans Devanagari', 'Lohit Devanagari', 'Mangal',         # Hindi/Devanagari
        'Noto Sans Tamil', 'Lohit Tamil',                             # Tamil
        
        # Middle Eastern scripts
        'Noto Sans Arabic', 'Tahoma', 'Arial Unicode MS',
        
        # Fallbacks with wide Unicode coverage
        'DroidSansFallback', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    
    # Set font family configuration
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = asian_fonts
    
    # Force matplotlib to rebuild the font cache
    try:
        fm.findfont('DejaVu Sans', rebuild_if_missing=True)
    except:
        pass
    
    # Enable Unicode minus sign
    plt.rcParams['axes.unicode_minus'] = False
    
    # add all Latin fonts under fonts directory
    for font_file in (Path(__file__).parent / "fonts/ArchivoNarrow").glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(font_file))
        except Exception:
            pass


def extract_context_from_srt(srt_path: Union[str, Path]):  # -> str:
    """
    Extract only the subtitle text from an SRT file.
    
    Args:
        srt_path: Path to the SRT file
        
    Returns:
        A string containing only the subtitle texts joined with newlines
    """
    srt_path = Path(srt_path) if isinstance(srt_path, str) else srt_path
    subtitles = []
    with srt_path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    # Use regex to match SRT entries
    import re
    # Pattern matches: number, timestamp, and subtitle text (handles various line break formats)
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\s*\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for _, _, subtitle_text in matches:
        # Clean up the subtitle text
        cleaned_text = subtitle_text.strip()
        
        # TODO: check if it helps to remove speaker tag
        # Remove speaker tag if present
        if "##speaker_" in cleaned_text:
            cleaned_text = cleaned_text.split("##speaker_")[0].strip()
        
        subtitles.append(cleaned_text)
    
    return "\n".join(subtitles)


if __name__ == "__main__":
    
    print(extract_context_from_srt("/data/projects/OCR-SAM/v1.0.0/subtitles/subtitles_30065059574.srt"))
    
    # OPENAI_API_KEY_PROJECT_ID = "357470755328"
    # OPENAI_API_KEY_SECRET_ID =  "image-eraser-openai"
    # from linguana.gcp import get_secret

    # os.environ["OPENAI_API_KEY"] = get_secret(OPENAI_API_KEY_PROJECT_ID, OPENAI_API_KEY_SECRET_ID)
    
    # BUCKET_NAME = "linguana-models"
    
    # # Create timestamped directory for results
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_dir = Path("/data/projects/OCR-SAM/captioning_" + timestamp)
    # results_dir.mkdir(exist_ok=True)
    
    # for img_path in Path("/data/projects/OCR-SAM/det_polygon_imgs").glob("*.png"):
    #     image_caption = image_captioning(
    #         client=OpenAI(), 
    #         image=img_path, 
    #         prompt="Extract and recognize any text in the image. If no text is found, return 'None'. Do not add any explanations or additional text.",
    #         system_prompt="You are an expert in text recognition and OCR. Your task is to extract and recognize text from images, including non-Latin scripts like Arabic, Japanese, Korean, etc. Return only the recognized text exactly as it appears, or 'None' if no text is found.")  # noqa: E501
    #     # overlay the caption on the image and save to the results directory with original filename
    #     image = Image.open(img_path)
    #     # draw = ImageDraw.Draw(image)
    #     # # Use a font that supports multiple languages
    #     # try:
    #     #     # Try to use a font that supports multiple languages
    #     #     font = ImageFont.truetype("Arial Unicode MS", 20)  # Windows
    #     # except (OSError, IOError):
    #     #     try:
    #     #         font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 20)  # macOS
    #     #     except (OSError, IOError):
    #     #         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)  # Linux
    #     # draw.text((10, 10), image_caption, fill="red", font=font)
    #     # Save to results directory with original filename
    #     image.save(results_dir / img_path.with_stem(f"{img_path.stem}__{image_caption}").name)
    #     # print(image_caption)
        
        
    # # client = OpenAI()
    # # # local file
    # # response = image_captioning(client, "/data/projects/image-eraser/imgs/erase_2.jpg", "I want to remove the text using latent diffusion model. Describe the background to be filled in a single sentence.")
    # # print(response)
    
    # # Try image editing prompt
    # from PIL import Image
    # image_caption = image_captioning(
    #     client=OpenAI(), 
    #     image=Path("/data/projects/image-eraser-data/tmp/thumbs_30075095401_hq.png"), 
    #     prompt="Describe the image in a single or couple of sentences.")  # noqa: E501
    # masked_image = Image.open("/data/projects/image-eraser-data/tmp/thumbs_30075095401_hq__masked.png")
    
    # image_caption, image_inpainting_prompt = image_editing_prompt(
    #     prompt=image_caption, 
    #     llm_model="gpt-4o", 
    #     masked_image=masked_image, 
    #     target_img_caption=True)
    
    
    # # image url
    # response = image_captioning(
    #     client=OpenAI(), 
    #     image="https://images.unsplash.com/photo-1598966835412-6de6f92c243d?q=80&w=2283&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
    #     prompt="I want to remove the text using latent diffusion model. Describe the background to be filled in a single sentence.")
    # print(response)

    