from pdf_to_image import PdfToImage
from image_processor import ImageProcessor
from ocr import Ocr
from table_transformer import TableTransformer
from table_constructor import TableConstruction
from csv_creator import DataFrameSaver
from anomaly_detector import AnomalyDetector
import numpy as np

import os

def extract_tables_from_pdf(pdf_path, page_img_dir, table_img_dir, save_dir):
    # Convert PDF to images
    PdfToImage(pdf_path).pdf_to_png()

    # Get paths of generated images
    _, _, pages = next(os.walk(page_img_dir))
    pages_with_path = [os.path.join(page_img_dir, page) for page in pages]

    # Detect and crop tables
    image_processor = ImageProcessor()
    table_transformer = TableTransformer(image_processor=image_processor)
    cropped_tables = table_transformer.detect_and_crop_tables(pages_with_path)

    # Get cropped table image paths
    _, _, images = next(os.walk(table_img_dir))
    images_with_path = [os.path.join(table_img_dir, img) for img in images]

    # Process and reconstruct tables
    saver = DataFrameSaver()
    for i, img_path in enumerate(images_with_path):
        try:
            ocr_instance = Ocr(img_path)
        except Exception:
            continue

        resize_float = 0.35
        while True:
            try:
                df = ocr_instance.apply_easyocr(expansion=False, dilation=False, resize_float=resize_float)
                final_table = TableConstruction(df).table_creator()
                break
            except Exception:
                resize_float = np.random.uniform(0.29, 0.34)

        print(final_table)
        detector = AnomalyDetector(threshold=10)
        print(f"Possible anomalies in Table {i}:\n{detector.find_large_anomalies(final_table)}\n{'='*40}")

        saver.save(final_table, f"{save_dir}/Table_{i}")

# One-liner execution
extract_tables_from_pdf(
    pdf_path='/Users/hosseindavarzanisani/gitrep_pdfTable/TableExtraction/pdfs/foods-11-00289.pdf',
    page_img_dir='/Users/hosseindavarzanisani/gitrep_pdfTable/TableExtraction/page_images',
    table_img_dir='/Users/hosseindavarzanisani/gitrep_pdfTable/TableExtraction/table_images',
    save_dir='/Users/hosseindavarzanisani/gitrep_pdfTable/TableExtraction/csv_tables'
)
