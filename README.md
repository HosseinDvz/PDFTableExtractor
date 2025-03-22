

# PDF Table Extractor
This AI-driven table extractor efficiently processes PDF files, expertly handling even the most complex and unstructured tables. Leveraging advanced algorithms, it accurately extracts data and reconstructs it into a clean, well-organized CSV format. While primarily designed for numerical tables, it can effectively process various table types.

## Features
- It detects and extracts tables from PDFs using DETR.
- Uses OCR to extract text and bounding box coordinates.
- Reconstructs tables with high accuracy.
- Outputs structured CSV files for easy data processing.

## Installation & Setup
```bash
git clone https://github.com/HosseinDvz/pdf-table-extraction.git
pip install -r requirements.txt
```

## Usage
Run the following command to extract tables from a PDF (sample.pdf is a path to the PDF file, others are folder names to save PDF page images  and table images and final CSV files). You can run either the extract_tables.py or the main.py:
```bash
extract_tables("sample.pdf", "page_images", "table_images", "csv_tables")
```

## Examples  
For examples of extracted tables, check this repository's folder (https://github.com/HosseinDvz/PDFTableExtractor/blob/main/csv_tables/).  


## Technical Details
This project leverages advanced deep learning and image processing techniques to achieve accurate table extraction:
- **DEtection TRansformer (DETR)** for detecting table regions.
- **EasyOCR** for extracting text and coordinates.
- **Custom Python functions** for table reconstruction.

## Contribution
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Commit your changes: `git commit -m "Added new feature"`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a Pull Request.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, reach out via [email](mailto:hosdvz@gmail.com) or visit my [LinkedIn Profile](https://www.linkedin.com/in/hosseindvz).

