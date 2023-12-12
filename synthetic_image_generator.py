from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMAGE_PATH = './images'
TEXT_PATH = Path('./texts/kangyur')
FONT_PATHS = {
    'Uchen1': './fonts/monlam_uni_ochan1.ttf',
    # 'Uchen4': './fonts/monlam_uni_ochan4.ttf',
    # 'Uchen5': './fonts/monlam_uni_uchen5.ttf',
    # 'Chouk': './fonts/monlam_uni_chouk.ttf',
    # 'Drutsa': './fonts/monlam_uni_dutsa1.ttf',
    # 'Paytsik': './fonts/monlam_uni_paytsik.ttf',
    # 'Tikrang': './fonts/monlam_uni_tikrang.ttf',
    # 'Tiktong': './fonts/monlam_uni_tiktong.ttf',
}

class SyntheticImageGenerator:

    def __init__(self, image_width, image_height, font_size=24, font_type="Uchen") -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.font_size = font_size
        self.font_type = font_type

    def get_pages(self, vol_text):
        pages = vol_text.split('\n\n')
        return pages
    
    def save_line(self, line, base_dir, line_file_name):
        (base_dir/ f'transcriptions/{line_file_name}.txt').write_text(line, encoding='utf-8')


    def save_image(self, text, img_file_name):
        font_file_name = FONT_PATHS.get(self.font_type, 'Uchen')
        img = Image.new('RGB', (self.image_width,self.image_height), color = (255, 255, 255))
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

    def save_images(self, lines, vol_number):
        line_number = 1
        for line in lines:
            if line and line_number<2000:
                img_file_name = Path(f"{IMAGE_PATH}/{self.font_type}/v{vol_number:03}/{self.font_type}_{line_number:06}.jpg")
                self.save_line(line, img_file_name.parent, img_file_name.stem)
                self.save_image(line, img_file_name)
                line_number += 1

    def create_synthetic_data(self):
        vol_paths = list(TEXT_PATH.iterdir())
        vol_paths.sort()
        for vol_number, vol_path in enumerate(vol_paths, 1):
            Path(f"{IMAGE_PATH}/{self.font_type}/v{vol_number:03}").mkdir(parents=True, exist_ok=True)
            vol_text = vol_path.read_text(encoding='utf-8')
            lines = vol_text.split('\n')
            self.save_images(lines, vol_number)
            break

if __name__ == "__main__":
    # synthetic_image_generator = SyntheticImageGenerator(
    #     image_width=1700,
    #     image_height=280,
    #     font_size=28,
    #     font_type="Uchen"
    # )

    for font_type in FONT_PATHS.keys():
        # try:
        synthetic_image_generator = SyntheticImageGenerator(
            image_width=2000,
            image_height=80,
            font_size=30,
            font_type=font_type
        )
        synthetic_image_generator.create_synthetic_data()
        # except:
        #     print(f"Error in {font_type}")
            
    

