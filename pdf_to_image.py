import os
import glob
from pdf2image import convert_from_path

class PdfToImage:
    
    def __init__(self, pdf_path='pdfs', pages_path='page_images'):
        self.pdf_path = pdf_path
        self.pages_path = pages_path
    
    def pdf_to_png(self):
        """Converts PDF to PNG images and saves them."""
        self._clear_folder(self.pages_path)  # Clear folder before saving new images
        pages = self._pdf_to_image(self.pdf_path)  
        self._image_to_png(pages)
    
    def _clear_folder(self, pages_path):
        """Deletes all files in the specified folder."""
        if not os.path.exists(pages_path):
            print(f"Folder {pages_path} does not exist.")
            return
        
        files = glob.glob(os.path.join(pages_path, "*"))  # Get all files in the folder
        for file in files:
            try:
                os.remove(file)  # Delete each file
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    
    def _pdf_to_image(self, pdf_path, dpi=600):
        """Converts PDF pages to images."""
        return convert_from_path(pdf_path, dpi=dpi)
          
    def _image_to_png(self, list_of_images):
        """Saves images as PNG files."""
        os.makedirs(self.pages_path, exist_ok=True)  # Ensure directory exists
        for i, image in enumerate(list_of_images):
            image.save(os.path.join(self.pages_path, f'page_{i}.png'), 'PNG')


if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    PdfToImage(pdf_path='pdfs/1-s2.0-S0308814617312839-Lentils.pdf').pdf_to_png()
