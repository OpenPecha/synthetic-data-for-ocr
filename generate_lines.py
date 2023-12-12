
import os
import shutil
import logging
import argparse
import sys
from glob import glob
from PIL import features
from natsort import natsorted

from Lib import create_dir
from Lib import AdvancedSyntheticImageGenerator
DEFAULT_FONT = "fonts/monlam_uni_ouchan4.ttf"


def check_libraqm():
    return features.check("raqm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--font", type=str, required=False, default=DEFAULT_FONT)
    parser.add_argument("--size", type=int, required=False, default=30)
    parser.add_argument(
        "--add_background", choices=["yes", "no"], required=False, default="yes"
    )
    parser.add_argument(
        "--distort", choices=["yes", "no"], required=False, default="yes"
    )
    parser.add_argument(
        "--jiggle", choices=["yes", "no"], required=False, default="yes"
    )

    if not check_libraqm():
        logging.error("raqm library not found!")
        sys.exit(1)

    args = parser.parse_args()

    dataset = args.dataset
    font = args.font
    font_size = args.size
    add_background = True if args.add_background == "yes" else False
    distort = True if args.distort == "yes" else False
    jiggle = True if args.jiggle == "yes" else False

    output_dir = str(os.path.join("output", dataset))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    create_dir(output_dir)

    if not os.path.exists(output_dir):
        try:
            create_dir(output_dir)
        except OSError as e:
            logging.error(
                f"Output directory {output_dir} does not exist and could not be created."
            )
            sys.exit(1)

    label_dir = os.path.join("texts", dataset)
    label_files = natsorted(glob(f"{label_dir}/*.txt"))

    if len(label_files) == 0:
        logging.error(f"No files found in {label_dir}.")
        sys.exit(1)

    logging.info(f"Processing {len(label_files)}")

    font_generator = AdvancedSyntheticImageGenerator(font, output_dir)
    font_generator.run_on_directory(label_files, font=font, font_size=font_size)