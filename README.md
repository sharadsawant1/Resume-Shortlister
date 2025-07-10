# Resume Shortlister

A Streamlit-based application for uploading, processing, and shortlisting resumes in PDF format. The app extracts information (e.g., name, email, skills, experience) from resumes, creates a vector store for semantic analysis, and ranks candidates based on required skills and minimum experience using natural language processing and machine learning techniques.

## Features

- **PDF Resume Upload**: Upload multiple PDF resumes via a user-friendly Streamlit interface.
- **Information Extraction**: Automatically extracts structured data (name, email, phone, location, skills, experience, education) from resumes.
- **Skill Matching**: Shortlists candidates based on required skills using exact matches and optional LLM-based scoring.
- **Experience Filtering**: Filters candidates by minimum years of experience.
- **Vector Store**: Uses FAISS and sentence-transformer embeddings for efficient resume storage and retrieval.
- **Interactive Results**: Displays shortlisted candidates in a table with expandable details, including match scores and summaries.

## Tech Stack

- **Python**: Core programming language.
- **Streamlit**: Web interface for uploading resumes and viewing results.
- **pypdf**: PDF text extraction.
- **LangChain**: Text splitting and vector store integration.
- **FAISS**: Vector store for semantic search.
- **HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for text embeddings.
- **Ollama**: Optional LLM (`llama2`) for nuanced skill matching.
- **Pandas**: Data handling and table display.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/resume-shortlister.git
   cd resume-shortlister
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then install required packages:
   ```bash
   pip install streamlit pypdf langchain faiss-cpu sentence-transformers ollama pandas
   ```

3. **Set Up Ollama (Optional)**:
   For LLM-based skill matching, install Ollama and pull the `llama2` model:
   ```bash
   # Install Ollama (follow instructions at https://ollama.ai)
   ollama pull llama2
   ollama run llama2
   ```
   If Ollama is unavailable, the app falls back to exact skill matching.

4. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Open the provided URL (e.g., `http://localhost:8501`) in a browser.

## Usage

1. **Upload Resumes**:
   - Use the file uploader to select one or more PDF resumes.
   - Ensure PDFs contain readable text (scanned or image-based PDFs may not work).

2. **Specify Criteria**:
   - Enter required skills (comma-separated, e.g., `Python, JavaScript, React, AWS`).
   - Set the minimum experience (in years, e.g., `2.5`).
   - Choose the number of top candidates to display (e.g., `5`).

3. **Process and View Results**:
   - Click "Process Resumes" to extract information and shortlist candidates.
   - View results in a table with columns for name, email, experience, skills, match score, source file, and summary.
   - Expand each candidate’s row for detailed information.

## Example

**Input**:
- Uploaded PDFs: `sharad_resume.pdf`, `sharad_cv.pdf`
- Required Skills: `Python, JavaScript, React, Django, AWS`
- Minimum Experience: `0` years
- Top Candidates: `5`

**Output** (Sample Table):
| Name          | Email                    | Experience (Years) | Skills                                  | Match Score | Source File         | Summary                              |
|---------------|--------------------------|--------------------|----------------------------------------|-------------|---------------------|--------------------------------------|
| Sharad Sawant | sharad.s@example.com     | 7.0                | Python, JavaScript, React, PostgreSQL, Kubernetes | 8           | sharad_cv.pdf       | Full-stack developer proficient in... |
| Sharad Sawant | sharad.sawant@example.com | 5.0                | Python, Django, Flask, SQL, AWS        | 7           | sharad_resume.pdf   | Python developer with expertise in... |

## Project Structure

```
resume-shortlister/
├── app.py          # Main Streamlit application
├── README.md       # Project documentation
└── requirements.txt # Python dependencies
```

## Dependencies

See `requirements.txt` for a complete list. Key dependencies:
- `streamlit>=1.31.0`
- `pypdf>=4.0.0`
- `langchain>=0.1.0`
- `faiss-cpu>=1.7.4`
- `sentence-transformers>=2.2.2`
- `ollama>=0.1.0` (optional)
- `pandas>=2.0.0`

## Limitations

- **PDF Compatibility**: Relies on `pypdf` for text extraction, which may fail for scanned or image-based PDFs.
- **Regex Parsing**: Simple regex patterns for extracting information may miss complex resume formats.
- **Ollama Dependency**: LLM-based scoring requires a running Ollama server; otherwise, falls back to exact matching.
- **Performance**: Processing large numbers of resumes may be slow due to embedding generation and LLM calls.

## Future Improvements

- Add semantic search using FAISS for better skill matching.
- Support scanned PDFs with OCR integration (e.g., `pytesseract`).
- Enhance regex patterns or use LLMs for robust information extraction.
- Add CSV export for shortlisted candidates.
- Implement skill autocomplete or predefined skill categories.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.



## Contact

For questions or feedback, open an issue or contact [Sharad Sawant](mailto:swntshrd1@gmail.com).
