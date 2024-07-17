import os
import uuid
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import albumentations as A

# Utility functions
def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_filename(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]

def resize_to_width(image: np.array, target_width: int) -> np.array:
    width_ratio = target_width / image.shape[1]
    return cv2.resize(image, (target_width, int(image.shape[0] * width_ratio)), interpolation=cv2.INTER_LINEAR)

# Augmentation functions
class DistortionMode:
    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"

def distort_image(image: np.array, mode: str = DistortionMode.ADDITIVE, edge_tresh1: int = 100, edge_tresh2: int = 200, kernel_width: int = 2, kernel_height: int = 1, kernel_iterations=2):
    edges = cv2.Canny(image, edge_tresh1, edge_tresh2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    kernel = np.ones((kernel_width, kernel_height), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=kernel_iterations)
    edges = cv2.dilate(edges, kernel, iterations=kernel_iterations)
    indices = np.where(edges == 255)
    if mode == DistortionMode.ADDITIVE:
        image[indices[0], indices[1], :] = [0]
    else:
        image[indices[0], indices[1], :] = [255]
    return image

def binarize_background(image: np.array, block_size: int = 47, threshold: int = 17) -> np.array:
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, threshold)
    image = cv2.erode(image, kernel=(6, 6), iterations=3)
    image = cv2.bitwise_not(image)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def add_background(image: np.array, background: np.array, threshold: int = 10):
    binarized_bg = binarize_background(background, threshold=threshold)
    image_height, image_width, _ = image.shape
    if binarized_bg.shape[1] < image_width:
        binarized_bg = resize_to_width(binarized_bg, image_width)
    x_anchor = random.randint(0, binarized_bg.shape[1] - image_width)
    y_anchor = random.randint(0, binarized_bg.shape[0] - image_height)
    background_slice = binarized_bg[y_anchor:y_anchor + image_height, x_anchor:x_anchor + image_width]
    stacked = np.dstack((background_slice, image))
    stacked = np.sum(stacked, axis=-1)
    norm_stack = cv2.normalize(stacked, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_stack = np.where(norm_stack == -1, 1.0, norm_stack)
    norm_stack = np.where(norm_stack < 0.7, 0.0, norm_stack)
    return cv2.cvtColor((norm_stack * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Image generation class
class SyntheticPageImageGenerator:
    def __init__(self, font_path: str, output_dir: str, background_dir: str):
        self.font_path = font_path
        self.output_dir = output_dir
        self.backgrounds = glob(os.path.join(background_dir, "*.jpg"))
        self.init_output_dirs()

    def init_output_dirs(self):
        self.page_dir = os.path.join(self.output_dir, "pages")
        self.line_dir = os.path.join(self.output_dir, "lines")
        create_dir(self.page_dir)
        create_dir(self.line_dir)

    def text_size(self, text: str, font: ImageFont) -> tuple:
        dummy_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def generate_page_image(self, text: str, font: ImageFont, background: np.array = None):
        lines = text.split("\n")
        max_width = max([self.text_size(line, font)[0] for line in lines])
        total_height = sum([self.text_size(line, font)[1] for line in lines])
        page_image = Image.new("RGB", (max_width + 20, total_height + 20), (255, 255, 255))
        draw = ImageDraw.Draw(page_image)
        y = 10
        for line in lines:
            draw.text((10, y), line, font=font, fill=(0, 0, 0))
            y += self.text_size(line, font)[1]
        page_image = np.array(page_image)
        if background is not None:
            page_image = add_background(page_image, background)
        return page_image

    def save_page_image(self, image: np.array, base_filename: str, augmentation_name: str, angle: int):
        filename = f"{base_filename}_{augmentation_name}_{angle}deg.png"
        cv2.imwrite(os.path.join(self.page_dir, filename), image)
        return filename

    def extract_lines(self, page_image: np.array, text: str, font: ImageFont, page_filename: str, angle: int):
        lines = text.split("\n")
        y = 10
        for line_idx, line in enumerate(lines):
            line_height = self.text_size(line, font)[1]
            line_image = page_image[y:y + line_height, 10:-10]
            y += line_height
            if line_image.size == 0:  # Check if the cropped image is empty
                continue
            line_filename = f"{os.path.splitext(page_filename)[0]}_line_{line_idx}_{angle}deg.png"
            cv2.imwrite(os.path.join(self.line_dir, line_filename), line_image)

    def bend_page_image(self, image: np.array, max_curvature_height_ratio: float = 0.00001, max_angle: int = 10):
        original_height, original_width = image.shape[:2]
        bent_image = np.zeros_like(image)
        for y in range(original_height):
            curvature = max_curvature_height_ratio * (y - original_height / 2) ** 2
            angle = int(np.degrees(np.arctan(curvature)))
            if abs(angle) > max_angle:
                angle = np.sign(angle) * max_angle
                curvature = np.tan(np.radians(angle))

            for x in range(original_width):
                new_x = int(x + curvature * (y - original_height / 2) ** 2)
                if 0 <= new_x < original_width:
                    bent_image[y, x] = image[y, new_x]
        self.bending_angle = angle  # Store the bending angle
        return bent_image

    def random_augmentation(self, image: np.array):
        # Define random augmentations using Albumentations
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=30),
        ])
        augmented = transform(image=image)
        return augmented['image']

    def run(self, text: str):
        fonts = [ImageFont.truetype(self.font_path, size=size) for size in range(20, 41, 5)]
        base_filename = "generated_page"
        for font in fonts:
            background = None
            if self.backgrounds:
                background = cv2.imread(random.choice(self.backgrounds))
            page_image = self.generate_page_image(text, font, background)
            bent_page_image = self.bend_page_image(page_image)  # Adjust curvature as needed
            # Apply random augmentation
            augmented_image = self.random_augmentation(bent_page_image)
            # Save the augmented image
            page_filename = self.save_page_image(augmented_image, base_filename, "bent_augmented", self.bending_angle)
            # Extract lines from the augmented image
            self.extract_lines(augmented_image, text, font, page_filename, self.bending_angle)


# Example usage
if __name__ == "__main__":
    # Example Tibetan text
    text = "བདེ་བར་སྐྱེ་མཆོག་གཏུག་དགུ་རྒྱུ་འཚོར་ལོ་གཉེར།\nབདེ་ལེགས་ལྟར་རྩིས་བརྒྱུད་བརྟན་པ་སྐྱབས་ཤིང་།"
    font_path = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/fonts/Uchen/monlam_uni_ouchan5.ttf"
    output_dir = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/line_output"
    background_dir = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/backgrounds/I1KG812760788.jpg"
    generator = SyntheticPageImageGenerator(font_path, output_dir, background_dir)
    generator.run(text)
