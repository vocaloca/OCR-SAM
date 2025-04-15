import json
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple
import cv2
import gradio as gr
import numpy as np
import os
import sys
import PIL.Image as Image
from openai import OpenAI
import torch
from logging import getLogger, INFO
import image_captioning
from linguana import gcp, image_ocr_utils
from matplotlib import pyplot as plt

# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
from mmocr.utils.polygon_utils import offset_polygon

# SAM
from segment_anything import SamPredictor, sam_model_registry

from linguana.language_utils_simplified import MultilingualTextRenderer



# Add latent diffusion path
sys.path.append('latent_diffusion')
from latent_diffusion.ldm_erase_text import (
    erase_text_from_image, instantiate_from_config, OmegaConf
)

MODEL_FOLDER = Path(__file__).parent / 'checkpoints'

BoxType = Tuple[float, float, float, float, float, float, float, float]  # x1, y1, x2, y2, x3, y3, x4, y4

# Call the configuration function
image_captioning.configure_multilingual_fonts()

# Diffusion model
try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    # Fallback for case where huggingface_hub has version conflicts
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Monkey patch huggingface_hub for compatibility with diffusers
    import huggingface_hub
    if (not hasattr(huggingface_hub, 'cached_download') and 
            hasattr(huggingface_hub, 'hf_hub_download')):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    from diffusers import StableDiffusionInpaintPipeline

logger = getLogger(__name__)
logger.setLevel(INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

det_config = ('mmocr_dev/configs/textdet/dbnetpp/'
              'dbnetpp_swinv2_base_w16_in21k.py')
det_weight = f'{MODEL_FOLDER}/mmocr/db_swin_mix_pretrain.pth'
rec_config = 'mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py'
rec_weight = (f'{MODEL_FOLDER}/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth')
sam_checkpoint = f'{MODEL_FOLDER}/sam/sam_vit_h_4b8939.pth'
device = 'cuda'
sam_type = 'vit_h'

try:
    gcp.download_blob('linguana-models', 'image-eraser/', MODEL_FOLDER)
    # gcp.download_blob('linguana-models', 'image-eraser-eyal/', MODEL_FOLDER)
    logger.info("Downloaded models from GCP")
except Exception as e:
    logger.error(f"Error initializing models or pipelines: {e}")
    raise   

# BUILD MMOCR
mmocr_inferencer = MMOCRInferencer(
    det_config, det_weight, device=device)
# # Build SAM
# sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
# sam = sam.to(device)
# sam_predictor = SamPredictor(sam)


def multi_mask2one_mask(masks):
    _, _, h, w = masks.shape
    whole_mask = None
    for i, mask in enumerate(masks):
        mask_image = mask.reshape(h, w, 1)
        whole_mask = mask_image if i == 0 else whole_mask + mask_image
    whole_mask = np.where(whole_mask == 0, 0, 255)
    return whole_mask


def numpy2PIL(numpy_image):
    out = Image.fromarray(numpy_image.astype(np.uint8))
    return out


# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


def crop_image_polygons(img: np.ndarray, polygons: list):  # -> List[np.ndarray]:
    """Crop the image with the polygons
    """
    crop_imgs = []
    for polygon in polygons:
        polygon = list(map(int, polygon))
        # Make sure the coordinates are within the image boundaries
        x1 = max(0, polygon[0])
        y1 = max(0, polygon[1])
        x2 = min(img.shape[1], polygon[2])
        y2 = min(img.shape[0], polygon[3])
        print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
        
        # Ensure the crop region is valid (width and height > 0)
        if x2 > x1 and y2 > y1:
            crop_imgs.append(img[y1:y2, x1:x2])
        else:
            print(f'Invalid crop region: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
            # # Add a small empty image as placeholder if the crop is invalid
            # crop_imgs.append(np.zeros((10, 10, 3), dtype=np.uint8))
    return crop_imgs


def create_mask_rotate_crop(
    img: np.ndarray, polygons: list, box_expansion: float = 0.1
):  # -> List[dict]:
    
    """Create a mask from polygon points, rotate so the long side is 
    horizontal, and crop the image.
    
    Args:
        img (np.ndarray): Input image
        polygons (list): List of polygons, each with 4 pairs of (x,y) points
        box_expansion (float, optional): Factor to expand bounding box by. 
            Defaults to 0.1 (10%).
        
    Returns:
        List[dict]: List with cropped image, mask, and angle dictionaries
    """
    results = []
    
    for polygon in polygons:
        # Convert to numpy array and reshape to points
        points = np.array(polygon).reshape(-1, 2)
        
        # Create mask of the polygon
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
        
        # Find rotated rectangle
        rect = cv2.minAreaRect(points.astype(np.int32))
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.intp)  # Using np.intp instead of np.int0
        
        # Get center, width, height and angle from the rectangle
        center, (width, height), angle = rect
        
        # Handle rotation to keep text right-side up
        # OpenCV's minAreaRect returns angle in range [-90, 0)
        if width < height:
            angle += 90
            width, height = height, width
        
        # Normalize angle to prevent upside-down text (more than 90 degree rotation)
        # We want angle in range [-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
            
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate the original image
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Rotate the mask as well
        rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        
        # Find the bounding box of the rotated mask
        x, y, w, h = cv2.boundingRect(rotated_mask)
        
        # Apply box expansion if needed
        if box_expansion > 0:
            expansion_x = int(w * box_expansion)
            expansion_y = int(h * box_expansion)
            
            # Ensure boundaries stay within image
            x = max(0, x - expansion_x)
            y = max(0, y - expansion_y)
            new_width = w + 2 * expansion_x
            new_height = h + 2 * expansion_y
            w = min(rotated_img.shape[1] - x, new_width)
            h = min(rotated_img.shape[0] - y, new_height)
        
        # Crop the rotated image using the bounding box
        cropped_img = rotated_img[y:y+h, x:x+w]
        cropped_mask = rotated_mask[y:y+h, x:x+w]
        
        # Create result dictionary
        result = {
            'image': cropped_img,
            'mask': cropped_mask,
            'angle': angle,
            'original_points': points
        }
        
        results.append(result)
    
    return results

def get_ocr_single_shot_results(img: np.core.ndarray):
    """Get OCR results from an image using a single shot approach
    """
    try:
        num_lines = int(get_text_or_language_from_img(img, mode='ocr_count_lines'))
    except Exception as e:
        logger.error(f"Error getting number of lines: {e}")
        num_lines = 0
    if num_lines > 0:
        text = get_text_or_language_from_img(img, mode='ocr_single_shot', num_lines=num_lines)
        # parse the text
        lines = text.split('\n')
        lines = [line.split(':')[1].strip() for line in lines]
        return lines
    else:
        return []


def fix_json_none_values(json_dict):
    for key, value in json_dict.items():
        if value == 'none' or value == 'null' or value == 'None':
            json_dict[key] = None
    return json_dict

def get_valid_json_from_llm(response_text):
    # Try to extract JSON from the response if there's extra text
    json_pattern = r'({.*})'
    json_match = re.search(json_pattern, response_text, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = response_text
    
    try:
        # Parse and validate the JSON
        parsed_json = json.loads(json_string)
        parsed_json = fix_json_none_values(parsed_json)
        return parsed_json
    except json.JSONDecodeError:
        # Handle invalid JSON
        return {"error": "Invalid JSON response", "raw_response": response_text}
    
def get_text_or_language_from_img(img: np.core.ndarray, mode: Literal['ocr', 'ocr_count_lines', 'ocr_single_shot', 'ocr_single_line', 'language', 'words_order', 'font_analysis'], **kwargs):
    """Get text or language from (cropped) images
    """
    parse_json = False
    if mode == 'ocr':
        user_prompt = image_captioning.OCR_USER_PROMPT
        system_prompt = image_captioning.OCR_SYSTEM_PROMPT
    elif mode == 'ocr_count_lines':
        user_prompt = image_captioning.OCR_COUNT_LINES_USER_PROMPT
        system_prompt = image_captioning.OCR_SYSTEM_PROMPT
    elif mode == 'ocr_single_shot':
        user_prompt = image_captioning.OCR_USE_PROMPT_SINGLE_SHOT(kwargs['num_lines'])
        system_prompt = image_captioning.OCR_SYSTEM_PROMPT
    elif mode == 'ocr_single_line':
        user_prompt = image_captioning.OCR_USE_PROMPT_SINGLE_LINE
        system_prompt = image_captioning.OCR_SYSTEM_PROMPT
    elif mode == 'language':
        user_prompt = image_captioning.LANGUAGE_DETECTION_USER_PROMPT
        system_prompt = image_captioning.LANGUAGE_DETECTION_SYSTEM_PROMPT
    elif mode == 'words_order':
        words = ", ".join(kwargs['words'])
        user_prompt = image_captioning.WORDS_ORDER_USER_PROMPT(words)
        system_prompt = image_captioning.WORDS_ORDER_SYSTEM_PROMPT
    elif mode == 'font_analysis':
        user_prompt = image_captioning.FONT_ANALYSIS_USER_PROMPT
        system_prompt = image_captioning.FONT_ANALYSIS_SYSTEM_PROMPT
        parse_json = True
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    max_retries = 3
    
    while max_retries > 0:
        text = image_captioning.image_captioning(
            client=OpenAI(), 
            image=img, 
            prompt=user_prompt,
            system_prompt=system_prompt)  # noqa: E501
        if not parse_json:
            return text
        
        try:
            parsed_text = json.loads(text)
            if kwargs.get('validate_keys', None) and not set(kwargs['validate_keys']).issubset(set(parsed_text.keys())):
                logger.error(f"Invalid keys: {set(parsed_text.keys())} != {set(kwargs['validate_keys'])}")
                max_retries -= 1
                continue
            parsed_text = fix_json_none_values(parsed_text)
            return parsed_text
        except json.JSONDecodeError:
            try:
                # Apply fallback parsing if direct parsing fails
                parsed_text = get_valid_json_from_llm(text)
                if kwargs.get('validate_keys', None) and not set(kwargs['validate_keys']).issubset(set(parsed_text.keys())):
                    logger.error(f"Invalid keys: {set(parsed_text.keys())} != {set(kwargs['validate_keys'])}")
                    max_retries -= 1
                    continue
                return parsed_text
            except Exception as e:
                logger.error(f"Error parsing JSON: {e}, trying again...")
                max_retries -= 1
                continue
    
    raise Exception("Failed to parse JSON (max_retries={})".format(max_retries))


def parse_words_order(words: str):
    """Parse the words order from the string
    """
    # the format is Line 1: word1, word2, ... Line 2: word3, word4, ...
    lines = words.split('\n')
    words_order = []
    for line in lines:
        words = line.split(':')[1].strip()
        words_order.append([word.strip() for word in words.split(',')])
    return words_order

def run_text_recognition(img: np.ndarray, erased_image: np.ndarray, det_polygons: Optional[List[BoxType]] = None):
    """Run MMOCR and SAM

    Args:
        img (np.ndarray): Input image
        det_config (str): Path to the config file of the selected detection
            model.
        det_weight (str): Path to the custom checkpoint file of the selected
            detection model.
        rec_config (str): Path to the config file of the selected recognition
            model.
        rec_weight (str): Path to the custom checkpoint file of the selected
            recognition model.
        sam_checkpoint (str): Path to the custom checkpoint file of the
            selected SAM model.
        sam_type (str): Type of the selected SAM model. Defaults to 'vit_h'.
        device (str): Device used for inference. Defaults to 'cuda'.
    """
    # Build MMOCR

    det_polygons = det_polygons or mmocr_inferencer(img)['predictions'][0]['det_polygons']  # text detection only
    det_polygon_imgs = create_mask_rotate_crop(
        img, det_polygons, box_expansion=0.1
    )
    rec_texts = get_ocr_single_shot_results(img)
    # print(rec_texts)
    
    # Calculate the line groupings
    lines, line_polygons = image_ocr_utils.group_polygons_in_lines(det_polygons, rec_texts, num_lines=len(rec_texts), det_polygon_imgs=det_polygon_imgs)
    print("expected num_lines={}, got num_lines={}".format(len(rec_texts), len(lines)))
    
    # plot the polygons on top of the image, each line in a different color
    plt.figure()
    plt.imshow(img)
    # Define a list of distinct colors for each line
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'lime', 'pink']
    for i, (line, line_polygon) in enumerate(zip(lines, line_polygons)):
        # Use modulo to cycle through colors if there are more lines than colors
        line_color = colors[i % len(colors)]
        
        # Draw each individual text box in the line
        for idx in line:
            polygon = np.array(det_polygon_imgs[idx]['original_points'])
            polygon = np.concatenate([polygon, polygon[:1]], axis=0)
            plt.plot(polygon[:, 0], polygon[:, 1], '--', color=line_color, linewidth=2)
        
        # Draw the line's bounding polygon
        if len(line) > 0:
            line_polygon_points = line_polygon.reshape(-1, 2)
            line_polygon_points = np.concatenate([line_polygon_points, line_polygon_points[:1]], axis=0)
            plt.plot(line_polygon_points[:, 0], line_polygon_points[:, 1], '-', 
                     color=line_color, linewidth=3, alpha=0.7)
            
    plt.savefig('tmp_output.png')
    plt.close()
        
    # Create output directory if it doesn't exist
    output_dir = f'{ROOT_DIR}/det_polygon_imgs'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, det_polygon_result in enumerate(det_polygon_imgs):
        # save all det_polygon_imgs
        cv2.imwrite(f'{output_dir}/{i}.png', det_polygon_result['image'])
        
    det_bboxes = torch.tensor(
        np.array([poly2bbox(poly) for poly in line_polygons]),
        device='cuda')

    # class LinePolygon:
    #     def __init__(self, polygon, image):
    #         self.polygon = polygon
    #         self.image = image
    #         self.text = None
    #         self.font_analysis = None
    #     @property
    #     def text(self):
    #         return self.text
    #     @text.setter
    #     def text(self, text):
    #         self.text = text
    #     @property
    #     def polygon(self):
    #         return self.polygon
    #     @polygon.setter
    #     def polygon(self, polygon):
    #         self.polygon = polygon
    #     @property
    #     def image(self):
    #         return self.image
    #     @image.setter
    #     def image(self, image):
    #         self.image = image
    #     @property
    #     def font_analysis(self):
    #         return self.font_analysis
    #     @font_analysis.setter
    #     def font_analysis(self, font_analysis):
    #         self.font_analysis = font_analysis
                    

    # TODO: match each line_polygon to the rec_texts, using API calls to ChatGPT-4o
    line_polygon_imgs = create_mask_rotate_crop(
        img, line_polygons, box_expansion=0.1
    )
    line_polygon_rec_texts = []
    line_polygon_font_analysis = []
    for line_polygon_img in line_polygon_imgs:
        text = get_text_or_language_from_img(line_polygon_img['image'], mode='ocr_single_line')
        font_analysis = get_text_or_language_from_img(line_polygon_img['image'], mode='font_analysis', 
                                                      validate_keys=['font color (RGB)', 'outline color (RGB)', 'highlight color (RGB)', 'highlight color exist'])
        line_polygon_rec_texts.append(text)
        line_polygon_font_analysis.append(font_analysis)
    print(line_polygon_rec_texts)
    
    # Match each line_polygon to the rec_texts based on textual similarity
    text_matches = image_ocr_utils.match_text_to_lines(line_polygon_rec_texts, rec_texts)
    
    # Create a sorted/matched version of rec_texts
    matched_texts = []
    for match_idx in text_matches:
        if 0 <= match_idx < len(rec_texts):
            matched_texts.append(rec_texts[match_idx])
        else:
            # If no good match found, use the line polygon text directly
            idx = text_matches.index(match_idx)
            if idx < len(line_polygon_rec_texts):
                matched_texts.append(line_polygon_rec_texts[idx])
            else:
                matched_texts.append("")
    
    print("Original rec_texts:", rec_texts)
    print("Line polygon texts:", line_polygon_rec_texts)
    print("Matched indices:", text_matches)
    print("Final matched texts:", matched_texts)
    
    # Render a new text on an erased image
    renderer = MultilingualTextRenderer()
    
    matched_texts_translated = image_captioning.translate_text(matched_texts, "gpt-4o")
    
    image = erased_image.copy()
    for idx, (translated_text, polygon, polygon_font) in enumerate(
            zip(matched_texts_translated, line_polygons, line_polygon_font_analysis)):
        
        image = renderer.overlay_rotated_text(
            image,
            translated_text,
            polygon,
            font_size=24,  # TODO: fit to the polygon size
            font_color=polygon_font["font color (RGB)"],
            outline_color=polygon_font["outline color (RGB)"],
            outline_width=2, # TODO: should have polygon_font["outline width"],
            highlight_color=polygon_font["highlight color (RGB)"] if polygon_font["highlight color exist"] else None,
            output_path='tmp_translated_text.png',  # TODO: remove this
            )
    
    # Draw results
    plt.figure(figsize=(12, 12))
    # close axis
    plt.axis('off')
    # convert img to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    outputs = {}
    output_str = ''
    for idx, (rec_text, polygon, bbox) in enumerate(
            zip(matched_texts, line_polygons, det_bboxes)):
        polygon = np.array(polygon).reshape(-1, 2)
        # convert polygon to closed polygon
        polygon = np.concatenate([polygon, polygon[:1]], axis=0)
        plt.plot(polygon[:, 0], polygon[:, 1], '--', color='g', linewidth=2)
        # plot text on the left top corner of the polygon
        text_string = f'idx:{idx}, {rec_text}'
        bbox = bbox.cpu().numpy()
        plt.text(
            bbox[0],
            bbox[1],
            text_string,
            color='b',
            fontsize=13,
        )
        output_str += f'{idx}:{rec_text}' + '\n'
        outputs[idx] = dict(polygon=polygon.tolist())
    plt.savefig('output.png')
    # convert plt to numpy
    img = cv2.cvtColor(
        np.array(plt.gcf().canvas.renderer._renderer), cv2.COLOR_RGB2BGR)
    plt.close()
    return img, output_str, outputs


if __name__ == '__main__':

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input Image')
                erased_image = gr.Image(label='Erased Image')
                sam_results = gr.Textbox(label='Detection Results')
                mask_results = gr.Textbox(label='Mask Results', max_lines=2)
                mmocr_sam = gr.Button('Run MMOCR and SAM')
                text_index = gr.Textbox(
                    label='Select Text Index. It can be multiple indices '
                          'separated by commas.'
                )
                diffusion_type = gr.Radio(
                    choices=['Stable Diffusion', 'Latent Diffusion'],
                    label='Erasing Model')
                mask_type = gr.Radio(
                    choices=['SAM', 'MMOCR'], label='Mask Type')
                dilate_iter = gr.Slider(
                    1,
                    5,
                    value=2,
                    step=1,
                    label='The dilate iteration to dilate the SAM ouput mask',
                )
                
                # Add a new button for mask-rotate-crop functionality
                rotate_crop_btn = gr.Button('Rotate and Crop Text Areas')
                
            with gr.Column(scale=1):
                output_image = gr.Image(label='Output Image')
                # Add a gallery for displaying rotated crops
                rotated_crops = gr.Gallery(label='Rotated Crops').style(grid=4)
                
                gr.Markdown("## Image Examples")
                gr.Examples(
                    examples=[
                        'imgs/ex1.jpg', 'imgs/ex2.jpg', 'imgs/ex3.jpg',
                        'imgs/ex4.jpg', 'imgs/ex5.jpg', 'imgs/ex6.jpg',
                        'imgs/ex7.jpg', 'imgs/ex8.jpg', 'imgs/ex9.jpg',
                        'imgs/ex10.jpg', 'imgs/ex11.jpg', 'imgs/ex12.jpg',
                        'imgs/ex13.jpg', 'imgs/ex14.jpg', 'imgs/ex15.jpg'
                    ],
                    inputs=input_image,
                )
            mmocr_sam.click(
                fn=run_text_recognition,
                inputs=[input_image, erased_image],
                outputs=[output_image, sam_results, mask_results])
                
            # Add function to handle rotate and crop functionality
            def process_rotate_crop(img, mask_results):
                if not img or not mask_results:
                    return []
                    
                mask_data = eval(mask_results)
                polygons = [
                    np.array(mask_data[idx]['polygon']) for idx in mask_data
                ]
                results = create_mask_rotate_crop(
                    img, polygons, box_expansion=0.1
                )
                
                # Convert results to a format suitable for the gallery
                gallery_images = [result['image'] for result in results]
                return gallery_images
                
            rotate_crop_btn.click(
                fn=process_rotate_crop,
                inputs=[input_image, mask_results],
                outputs=[rotated_crops]
            )

    # Simple launch with minimal options to avoid pydantic schema generation issues
    demo.launch(
        debug=True,
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False  # Disable API documentation to avoid schema generation
    )
