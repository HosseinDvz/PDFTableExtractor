

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
### Create a .env file in the project root and add your [free Gemini API key](https://ai.google.dev/gemini-api/docs/api-key).:
GEMINI_API_KEY="YOUR_FREE_GEMINI_API"

## Usage
Run the main.py


## Examples  
Check this repository's folder for examples of extracted tables (https://github.com/HosseinDvz/PDFTableExtractor/blob/main/validated_tables/).  


## Technical Details  
This project combines deep learning and image processing techniques to extract tables from PDFs with high accuracy:

- **DEtection TRansformer (DETR)** is used to detect the locations of tables within PDF pages.  
  *Note: The transformer module within DETR, typically used for structure recognition, is not utilized in this project.*

- **EasyOCR** is employed to extract both text content and bounding box coordinates from the detected table regions.

- **Custom Python logic** reconstructs the tables using the OCR output. This logic effectively replaces the structural parsing functionality usually handled by DETR's transformer component. The
out put of thid step can be find in **csv_tables** folder

- **Validation with Gemini**: Each table image and its reconstructed version are sent to Google's Gemini model to correct potential OCR errors. The validation is optimized to stay within the limits of Gemini’s free tier by sending minimal, structured data.


## Contribution
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Commit your changes: `git commit -m "Added new feature"`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a Pull Request.

## 🔮 Future Updates
We are committed to making AskMyDoc more accessible and cost-effective. To achieve this, we plan to:

Explore Alternative Embedding Methods: Investigate open-source or more affordable embedding models to replace the current paid OpenAI embeddings.​

Integrate Gemini for Response Generation: Utilize Google's Gemini API for generating answers, leveraging its generous free tier and advanced capabilities.​
Google AI Studio

By implementing these changes, we aim to reduce or eliminate usage costs, making AskMyDoc freely available to a broader audience.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, reach out via [email](mailto:hosdvz@gmail.com) or visit my [LinkedIn Profile](https://www.linkedin.com/in/hosseindvz).

