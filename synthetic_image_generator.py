from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMAGE_PATH = './images'
TEXT_PATH = Path('./texts/kangyur')
FONT_PATHS = {
    'Uchen': './fonts/monlam_uni_ochan1.ttf',
    'Chouk': './fonts/monlam_uni_chouk.ttf',
}

class SyntheticImageGenerator:

    def __init__(self, image_width, image_height, font_size=24, font_type="Uchen") -> None:
        self.image_width = image_width
        self.iamge_height = image_height
        self.font_size = font_size
        self.font_type = font_type

    def get_pages(self, vol_text):
        pages = vol_text.split('\n\n')
        return pages

    def save_image(self, text, img_file_name):
        font_file_name = FONT_PATHS.get(self.font_type, 'Uchen')
        img = Image.new('RGB', (self.image_width,self.iamge_height), color = (255, 255, 255))
        d = ImageDraw.Draw(img)

        # Define font and text color
        if len(text)<100:
            size = int(self.font_size*1.5)
        else:
            size = self.font_size
        font = ImageFont.truetype(font_file_name, size=size, encoding='utf-16')
        text_color = (0,0,0)

        # Write text on image
        
        d.text((40,20), text, fill=text_color,spacing=12, font=font)

        # Save the image
        img.save(img_file_name)

    def save_images(self, pages, vol_number):
        for page_number, page in enumerate(pages,1):
            img_file_name = f"{IMAGE_PATH}/{self.font_type}/v{vol_number:03}/{page_number:04}.jpg"
            self.save_image(page, img_file_name)

    def create_synthetic_data(self):
        vol_paths = list(TEXT_PATH.iterdir())
        vol_paths.sort()
        for vol_number, vol_path in enumerate(vol_paths, 1):
            Path(f"{IMAGE_PATH}/{self.font_type}/v{vol_number:03}").mkdir(parents=True, exist_ok=True)
            vol_text = vol_path.read_text(encoding='utf-8')
            pages = self.get_pages(vol_text)
            self.save_images(pages, vol_number)

if __name__ == "__main__":
    # synthetic_image_generator = SyntheticImageGenerator(
    #     image_width=1700,
    #     image_height=280,
    #     font_size=28,
    #     font_type="Uchen"
    # )

    # synthetic_image_generator = SyntheticImageGenerator(
    #     image_width=2000,
    #     image_height=350,
    #     font_size=28,
    #     font_type="Chouk"
    # )

    synthetic_image_generator.create_synthetic_data()
    

