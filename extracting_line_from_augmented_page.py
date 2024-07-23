from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import numpy as np
import cv2
from enum import Enum
import random
import math

# Font paths dictionary
FONT_PATHS = {
    "Uchen1": './fonts/Uchen/monlam_uni_ochan1.ttf',
    "Uchen2": './fonts/Uchen/Monlam Uni Ochan2.ttf',
    # Add more font types and paths as needed
}

class DistortionMode(Enum):
    """ A simple selection for the mode used for font contour distortion """
    additive = 0
    subtractive = 1

def distort_line(image: np.array, mode: DistortionMode = DistortionMode.additive, edge_tresh1: int = 100, edge_tresh2: int = 200, kernel_width: int = 2, kernel_height: int = 1, kernel_iterations=2):
    if type(image) is not np.array:
        image = np.array(image)
    edges = cv2.Canny(image, edge_tresh1, edge_tresh2)
    if edges is None:
        return image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    kernel = np.ones((kernel_width, kernel_height), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=kernel_iterations)
    edges = cv2.dilate(edges, kernel, iterations=kernel_iterations)
    indices = np.where(edges[:, :] == 255)
    cv_image_added = image.copy()
    if mode == DistortionMode.additive:
        cv_image_added[indices[0], indices[1], :] = [0]
    else:
        cv_image_added[indices[0], indices[1], :] = [255]
    return cv_image_added


class WaveDeformer:
    def __init__(self, grid: int = 20, multiplier: int = 6, offset: int = 70) -> None:
        self.multiplier = multiplier
        self.offset = offset
        self.grid = grid

    def transform(self, x, y):
        y = y + self.multiplier * math.sin(x / self.offset)
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (
            *self.transform(x0, y0),
            *self.transform(x0, y1),
            *self.transform(x1, y1),
            *self.transform(x1, y0),
        )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = self.grid
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]

def deform_image(image: np.array):
    grid_size = random.randint(1, 140)
    multiplier = random.randint(1, 3)
    offset = random.randint(1, 100)
    deformer = WaveDeformer(grid=grid_size, multiplier=multiplier, offset=offset)
    deformed_img = ImageOps.deform(Image.fromarray(image), deformer)
    deformed_img = np.array(deformed_img)
    return deformed_img

class SyntheticImageGenerator:
    def __init__(self, font_size=30, font_type="Uchen1", background_image=None) -> None:
        self.font_size = font_size
        self.font_type = font_type
        self.background_image = background_image

    def get_pages(self, vol_text):
        pages = vol_text.split('\n\n')
        return pages

    def calculate_image_size(self, text):
        dummy_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        font_file_name = FONT_PATHS.get(self.font_type, './fonts/monlam_uni_ouchan2.ttf')
        font = ImageFont.truetype(font_file_name, self.font_size, encoding='utf-16')
        lines = text.split('\n')
        max_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        total_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines) + 20  # Add padding
        return max_width + 80, total_height + 40  # Add padding

    def save_transcript(self, page_text, base_dir, page_file_name):
        transcript_dir = base_dir / 'transcriptions'
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        transcript_path = transcript_dir / f'{page_file_name}.txt'
        transcript_path.write_text(page_text, encoding='utf-8')

    def apply_background(self, img):
        """
        Applies a background to the synthetic page image if no background is already present.
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        background_dir = Path('/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/backgrounds')  # Update this path
        background_files = list(background_dir.glob('*.png'))
        if not background_files:
            return img
        # Randomly select a background image
        background_file = random.choice(background_files)
        background = Image.open(background_file).convert('RGBA')
        # Resize background to match the synthetic page image
        background = background.resize(img.size, Image.ANTIALIAS)
        # Create a new image with a white background and the synthetic image on top
        combined = Image.new('RGBA', img.size, (255, 255, 255, 255))
        combined.paste(background, (0, 0), background)
        combined.paste(img, (0, 0), img)
        return combined.convert('RGB')  # Convert back to RGB if needed

    def apply_augmentation(self, img):
        possible_augmentations = [
            ("brightness", lambda img: self.augment_brightness(img)),
            ("contrast", lambda img: self.augment_contrast(img)),
            ("sharpness", lambda img: self.augment_sharpness(img)),
            ("rotate", lambda img: self.augment_rotate(img)),
            ("distort", lambda img: self.augment_distort(img)),
            ("deform", lambda img: self.augment_deform(img))
        ]
        selected_augmentations = random.sample(possible_augmentations, random.randint(1, 3))       
        rotation_angle = 0
        for name, func in selected_augmentations:
            img, param = func(img)
            if name == "rotate":
                rotation_angle = param       
        img = self.apply_background(img)    
        return img, rotation_angle

    def augment_brightness(self, img):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        return img, factor

    def augment_contrast(self, img):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        return img, factor

    def augment_sharpness(self, img):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return img, factor

    def augment_rotate(self, img):
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        return img, angle

    def augment_distort(self, img):
        img_np = np.array(img)
        distorted_img = distort_line(img_np)
        return Image.fromarray(distorted_img), None

    def augment_deform(self, img):
        img_np = np.array(img)
        deformed_img = deform_image(img_np)
        return Image.fromarray(deformed_img), None

    def extract_lines(self, img, text, rotation_angle):
        if rotation_angle != 0:
            img = img.rotate(-rotation_angle, expand=True, fillcolor=(255, 255, 255))   
        lines = text.split('\n')
        y = 30
        font_file_name = FONT_PATHS.get(self.font_type, './fonts/monlam_uni_ouchan2.ttf')
        font = ImageFont.truetype(font_file_name, self.font_size, encoding='utf-16')
        draw = ImageDraw.Draw(img)
        line_images = []
        for line in lines:
            line_bbox = draw.textbbox((40, y), line, font=font)
            line_img = img.crop((line_bbox[0], line_bbox[1], line_bbox[2], line_bbox[3]))
            # Convert line image to 'RGB' to ensure consistency
            line_img = line_img.convert('RGB')
            line_img_np = np.array(line_img)
            # Check if the line image contains non-white pixels
            if np.any(line_img_np != [255, 255, 255]):
                line_images.append(line_img)
            y += line_bbox[3] - line_bbox[1]
        if rotation_angle != 0:
            # Correct the rotation of the line images
            for i in range(len(line_images)):
                line_images[i] = line_images[i].rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255))
        return line_images

    def generate_images(self, vol_text, output_folder):
        pages = self.get_pages(vol_text)
        for idx, page_text in enumerate(pages):
            image_size = self.calculate_image_size(page_text)
            page_img = Image.new('RGB', image_size, (255, 255, 255))
            draw = ImageDraw.Draw(page_img)
            font_file_name = FONT_PATHS.get(self.font_type, './fonts/monlam_uni_ouchan2.ttf')
            font = ImageFont.truetype(font_file_name, self.font_size, encoding='utf-16')
            y = 30
            for line in page_text.split('\n'):
                draw.text((40, y), line, font=font, fill=(0, 0, 0))
                y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
            augmented_img, rotation_angle = self.apply_augmentation(page_img)           
            page_file_name = f'page_{idx:04d}'
            if rotation_angle != 0:
                page_file_name += f'_rot{rotation_angle:.2f}'
            page_file_path = output_folder / f'{page_file_name}.png'
            augmented_img.save(page_file_path)
           # Save transcription
            self.save_transcript(page_text, output_folder, page_file_name)
            # Extract and save line images
            line_images = self.extract_lines(augmented_img, page_text, rotation_angle)
            lines_dir = output_folder / 'lines'
            lines_dir.mkdir(parents=True, exist_ok=True)           
            for line_idx, line_img in enumerate(line_images):
                line_file_name = f'{page_file_name}_line_{line_idx:02d}.png'
                line_img.save(lines_dir / line_file_name)

def main():
    # Define paths and parameters
    text_folder = Path('./texts/kangyur')  # Replace with your text folder path
    output_folder = Path('./just_output')  # Replace with your desired output folder path   
    # Initialize the SyntheticImageGenerator
    generator = SyntheticImageGenerator(
        font_size=30,
        font_type="Uchen1",  # Replace with your desired font type
        background_image=None  # Set path to a background image if needed
    )   
    # Iterate over text files in the text folder
    for text_file in text_folder.glob('*.txt'):
        print(f"Processing file: {text_file.name}")
        with open(text_file, 'r', encoding='utf-8') as f:
            vol_text = f.read()       
        # Generate synthetic images and extract lines
        generator.generate_images(vol_text, output_folder)

if __name__ == "__main__":
    main()
