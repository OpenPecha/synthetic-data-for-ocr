import os

# Import functions from the straight and bent/wrinkled modules
from testing1 import handle_straight_synthetic
from testing import process_bent_wrinkled_synthetic_pages

def main():
    # Define directories
    text_folder = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/texts/kangyur"
    font_path = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/fonts/Tsugthung/monlam_uni_tiktong.ttf"
    straight_dir = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/output"
    line_output_dir = "/Users/ogyenthoga/Desktop/Work/synthetic-data-for-ocr/output"
    #bent_output_dir = "path/to/bent_output_dir"

    # Ensure output directories exist
    os.makedirs(straight_dir, exist_ok=True)
    os.makedirs(line_output_dir, exist_ok=True)
    #os.makedirs(bent_output_dir, exist_ok=True)

    # Step 1: Handle straight synthetic page images
    handle_straight_synthetic(text_folder, font_path, straight_dir, line_output_dir)
    print("Straight synthetic page images processing complete.")

    # Step 2: Process bent and wrinkled synthetic pages using straight synthetic pages
    #process_bent_wrinkled_synthetic_pages(straight_dir, bent_output_dir)
    #print("Bent and wrinkled synthetic page images processing complete.")

if __name__ == '__main__':
    main()
