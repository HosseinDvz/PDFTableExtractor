

# PDF Table Extractor
This AI-driven table extractor efficiently processes PDF files, expertly handling even the most complex and unstructured tables. Leveraging advanced algorithms, it accurately extracts data and reconstructs it into a clean, well-organized CSV format. While primarily designed for numerical tables, it can effectively process various table types.

## Features
- It detects and extracts tables from PDFs using DETR.
- Uses OCR to extract text and bounding box coordinates.
- Reconstructs tables with high accuracy.
- Validates reconstructed tables by Gemini
- It operates within Gemini’s free tier limits by sending minimal structured data.
- Outputs structured CSV files for easy data processing.

## Installation & Setup
```bash
git clone https://github.com/yourusername/pdf-table-extraction.git
cd pdf-table-extraction
pip install -r requirements.txt
```

## Usage
Run the main function:


## Examples  
Check this repository's folder for examples of extracted tables (https://github.com/HosseinDvz/PDFTableExtractor/blob/main/validated_tables/).  


## Technical Details
This project leverages advanced deep learning and image processing techniques to achieve accurate table extraction:
- **DEtection TRansformer (DETR)** for detecting table regions. I do not utilize the transformer part of DETR, which detects the structure of tables. 
- **EasyOCR** for extracting text and coordinates.
- **Custom Python functions** for table reconstruction from the output of OCR. These functions are the replacement of the transformer part of DETR.
- **Validation** for sending the pair of table images and constructed tables to Gemini for correcting possible OCR errors.

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

