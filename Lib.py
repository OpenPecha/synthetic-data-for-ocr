import logging
import os
import re
import cv2
import math
import random
import numpy as np

from enum import Enum
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps


class DistortionMode(Enum):
    """
    A simple selection for the mode used for font contour distortion
    """

    additive = 0
    subtractive = 1


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_filename(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]


def get_pages(file_path: str, page_splitter: str = "\n\n"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        labels = content.split(page_splitter)
        labels = [x for x in labels if x != ""]

    return labels


def clean_line(line: str) -> str:
    return re.sub("[\[\(].*?[\]\)]", "", line)


def cleanup_lines(line: str) -> list[str]:
    sample_lines = line.split(",")
    sample_lines = [x.replace("[", "") for x in sample_lines]
    sample_lines = [x.replace("]", "") for x in sample_lines]
    sample_lines = [x.replace("\n", "") for x in sample_lines]
    sample_lines = [x.replace("\n", "") for x in sample_lines]
    sample_lines = [x.replace("{", "") for x in sample_lines]
    sample_lines = [x.replace(" '", "") for x in sample_lines]
    sample_lines = [x.replace("'", "") for x in sample_lines]
    sample_lines = [x.strip() for x in sample_lines]

    return sample_lines


def read_file(file_path: str) -> list[str]:
    """
    Reading a text file containing multiple lines of text, removes custom tags if present.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()
        content = [x.replace("<line>", "") for x in content]
        content = [x.replace("<gap>", "") for x in content]
        content = [x for x in content if x != ""]
        content = [clean_line(x) for x in content]

        return content


def resize_to_width(image: np.array, target_width: int) -> np.array:
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * width_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image


def binarize_background(
    image: np.array, block_size: int = 47, threshold: int = 17
) -> np.array:

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    height, width = image.shape
    x_offset = int(width * 0.02)
    y_offset = int(height * 0.05)
    image = image[0 + y_offset : height - y_offset, 0 + x_offset : width - x_offset]
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        threshold,
    )

    image = cv2.erode(image, kernel=(6, 6), iterations=3)
    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def get_background_anchor(
    background_image: np.array, text_width: int, text_height: int
) -> tuple[int, int]:
    if len(background_image.shape) > 2:
        background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2GRAY)

    # not using any check for the height, assuming that any background image that has been chosen is much higher than a single line
    _, bg_width = background_image.shape

    if text_width > bg_width:
        """
        just overscale the background by 10% so that there aren't any issues with 0 values later, specifically if the text_width
        returned by ImageFont.getlength() is equal to the width of the background image
        """
        background_image = resize_to_width(
            background_image, int(text_width + text_width * 0.10)
        )

    bg_x_anchor_range = background_image.shape[1] - text_width
    bg_y_anchor_range = background_image.shape[0] - text_height

    x_anchor = int(random.randint(0, bg_x_anchor_range))
    y_anchor = int(random.randint(0, bg_y_anchor_range))

    return x_anchor, y_anchor


def distort_line(
    image: np.array,
    mode: DistortionMode = DistortionMode.additive,
    edge_tresh1: int = 100,
    edge_tresh2: int = 200,
    kernel_width: int = 2,
    kernel_height: int = 1,
    kernel_iterations=2,
):
    """
    This functions performs a simple distortion of the script contours by running a simple canny edge detection, which is further distorted.
    The resulting pixels can be used in two ways: "additive" or "substractive".
    - additive will add the sampled pixels to the outer contour
    - subtractive will subtract the sampled pixels from the line, so that the visible font thins out a bit

    The chosen default parameters were briefly tested on a couple of examples, other may eventually work as well or better, this is open to experimentation.
    """

    if type(image) is not np.array:
        image = np.array(image)

    edges = cv2.Canny(image, edge_tresh1, edge_tresh2)

    if edges is None:
        logging.warning("Edge detection returned None")
        return image

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    kernel = np.ones((kernel_width, kernel_height), np.uint8)

    # TODO: add a random number choice for the kernel iterations within a sensible range
    edges = cv2.erode(edges, kernel, iterations=kernel_iterations)
    edges = cv2.dilate(edges, kernel, iterations=kernel_iterations)

    indices = np.where(edges[:, :] == 255)
    cv_image_added = image.copy()

    if mode == DistortionMode.additive:
        cv_image_added[indices[0], indices[1], :] = [0]
    else:
        """
        TODO: results for subtractive aren't yet satisfying, so stick to additive for time being
        """
        cv_image_added[indices[0], indices[1], :] = [255]

    return cv_image_added


def deform_image(image: np.array):
    grid_size = random.randint(1, 140)
    multiplier = random.randint(1, 3)
    offset = random.randint(1, 100)

    deformer = WaveDeformer(grid=grid_size, multiplier=multiplier, offset=offset)
    deformed_img = ImageOps.deform(Image.fromarray(image), deformer)
    deformed_img = np.array(deformed_img)

    # TODO: maybe add a little check here or above that makes sure the image is really negative before
    deformed_img = cv2.bitwise_not(deformed_img)

    return deformed_img


def add_background(image: np.array, background: np.array, threshold: int = 10):
    binarized_bg = binarize_background(background, threshold=threshold)

    image_height, image_width, _ = image.shape
    _, background_width, _ = binarized_bg.shape

    if background_width < image_width:
        binarized_bg = resize_to_width(binarized_bg, image_width)

    bg_x_anchor_range = binarized_bg.shape[1] - image_width
    bg_y_anchor_range = binarized_bg.shape[0] - image_height

    x_anchor = int(random.randint(0, bg_x_anchor_range))
    y_anchor = int(random.randint(0, bg_y_anchor_range))
    background_slice = binarized_bg[
        y_anchor : y_anchor + image_height, x_anchor : x_anchor + image_width, :
    ]

    stacked = np.dstack((background_slice[:, :, :], image[:, :, :]))
    stacked = np.sum(stacked, axis=-1)
    norm_stack = cv2.normalize(
        stacked, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    norm_stack = np.where(norm_stack == -1, 1.0, norm_stack)
    norm_stack = np.where(norm_stack < 0.7, 0.0, norm_stack)
    norm_stack *= 255
    norm_stack = norm_stack.astype(np.uint8)
    norm_stack = cv2.cvtColor(norm_stack, cv2.COLOR_GRAY2RGB)

    return norm_stack


class WaveDeformer:
    """
    see: https://www.pythoninformer.com/python-libraries/pillow/imageops-deforming/
    """

    def __init__(self, grid: int = 20, multiplier: int = 2, offset: int = 120) -> None:
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


class AdvancedSyntheticImageGenerator:
    def __init__(
        self,
        font_path: str,
        output_dir: str,
        add_background: bool = True,
        distort_image: bool = True,
        jiggle_line: bool = True,
    ):
        self._font = font_path
        self._add_background = add_background
        self._distort_image = distort_image
        self._jiggle_line = jiggle_line
        self._output_dir = output_dir

        self.out_img_dir, self.out_lbl_dir = self.init_output_dirs()

    def init_output_dirs(self):
        out_img_dir = os.path.join(self._output_dir, "lines")
        out_lbl_dir = os.path.join(self._output_dir, "transcriptions")

        create_dir(out_img_dir)
        create_dir(out_lbl_dir)

        return out_img_dir, out_lbl_dir

    def save_line_label(
        self, image: np.array, page_idx: int, line_idx: int, label: str, label_name: str
    ):
        img_out = f"{self.out_img_dir}/{label_name}_{page_idx}_{line_idx}.jpg"
        lbl_out = f"{self.out_lbl_dir}/{label_name}_{page_idx}_{line_idx}.txt"

        cv2.imwrite(img_out, image)

        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write(label)

    def generate_line(
        self, line: str, font: ImageFont, background: np.array, oversize: float = 0.1
    ):
        _, _, width, height = font.getbbox(
            line, direction="ltr", anchor="lt", language="xct"
        )
        tmp_img = Image.new(
            "RGBA",
            size=(width + int(width * oversize), height + int(height * oversize)),
        )

        d = ImageDraw.Draw(tmp_img)
        d.text(
            (int(width * 0.01), int(height * 0.1)),
            line,
            font=font,
            anchor="lt",
            language="xct",
            fill=(0, 0, 0),
            stroke_width=0,
        )
        cv_img = np.asarray(tmp_img.convert("RGBA"))
        x = cv_img.copy()

        if self._distort_image:
            x = distort_line(cv_img, mode=DistortionMode.additive, kernel_iterations=1)

        if self._add_background:
            x = add_background(x, background, threshold=17)

            if self._jiggle_line:
                x = deform_image(x)
                return x
            else:
                x = cv2.bitwise_not(x)
                return x

        else:
            return x

    def run_on_directory(self, labels: list[str], font: str, font_size: int = 30):
        _font = ImageFont.truetype(font, size=font_size)
        backgrounds = glob("backgrounds/*.jpg")
        if len(backgrounds) == 0:
            self._add_background = False

        logging.info(f"Generating lines based on {len(labels)} files...")
        for _, label in enumerate(labels):

            background_idx = random.randint(0, len(backgrounds) - 1)
            background = cv2.imread(backgrounds[background_idx])

            label_name = get_filename(label)

            with open(label, "r", encoding="utf-8") as f:
                content = f.read()
                pages = content.split("\n\n")

            logging.info(f"Working on: {label_name}")
            for p_idx, page in tqdm(enumerate(pages), total=len(pages)):
                lines = page.split("\n")

                for line_idx, line in enumerate(lines):
                    _, _, width, height = _font.getbbox(
                        line, direction="ltr", anchor="lt", language="xct"
                    )
                    tmp_img = Image.new(
                        "RGBA", size=(width + int(width * 0.05), height)
                    )

                    d = ImageDraw.Draw(tmp_img)
                    d.text(
                        (int(width * 0.01), int(height * 0.1)),
                        line,
                        font=_font,
                        anchor="lt",
                        language="xct",
                        fill=(0, 0, 0),
                        stroke_width=0,
                    )
                    cv_img = np.asarray(tmp_img.convert("RGBA"))
                    x = cv_img.copy()

                    if self._distort_image:
                        x = distort_line(
                            cv_img, mode=DistortionMode.additive, kernel_iterations=1
                        )

                    if self._add_background:
                        x = add_background(x, background, threshold=17)

                    if self._jiggle_line:
                        x = deform_image(x)

                    else:
                        x = cv2.bitwise_not(x)

                    self.save_line_label(
                        image=x,
                        page_idx=p_idx,
                        line_idx=line_idx,
                        label=line,
                        label_name=label_name,
                    )

    def run_on_file(
        file_path: str, backgrounds: list[str], font: str, font_size: int = 30
    ):
        pass
