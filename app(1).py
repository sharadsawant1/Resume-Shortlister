import os
import re
from typing import List, Dict, Optional
import streamlit as st
import tempfile
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.docstore.document import Document
import pandas as pd

class ResumeProcessor:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Initialize LLM
        try:
            self.llm = Ollama(model="llama2")
        except Exception as e:
            st.error(f"Failed to initialize Ollama LLM: {e}. Using exact skill matching only.")
            self.llm = None
        # Initialize vector store
        self.vector_store = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF resume"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            return text
        except Exception as e:
            st.warning(f"Error reading PDF {pdf_path}: {e}")
            return ""

    def parse_resume_info(self, text: str) -> Dict:
        """Extract structured information from resume text"""
        info = {
            "name": self._extract_name(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "location": self._extract_location(text),
            "education": self._extract_education(text),
            "experience_years": self._extract_experience(text),
            "skills": self._extract_skills(text),
            "raw_text": text
        }
        return info

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills section"""
        skills = []
        skill_section_pattern = r"(?i)(skills|technical skills|key skills|competencies):?\s*(.*?)(?=\n\w+:|$)"
        match = re.search(skill_section_pattern, text, re.DOTALL)
        if match:
            skill_text = match.group(2)
            skills = re.split(r"[,•\n]", skill_text)
            skills = [s.strip() for s in skills if s.strip()]
        return skills

    def _extract_experience(self, text: str) -> Optional[float]:
        """Extract years of experience"""
        patterns = [
            r"(\d+)\s*years?.*experience",
            r"experience.*(\d+)\s*years?",
            r"(\d+)\+?\s*years?"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return 0.0

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address"""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        phone_pattern = r"(\+?\d[\d\s-]{7,}\d)"
        match = re.search(phone_pattern, text)
        return match.group(0) if match else ""

    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name (first line usually)"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[0] if lines else ""

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location"""
        location_pattern = r"(?i)(current location|location|address):?\s*(.*?)(?=\n\w+:|$)"
        match = re.search(location_pattern, text, re.DOTALL)
        return match.group(2).strip() if match else ""

    def _extract_education(self, text: str) -> List[str]:
        """Extract education section"""
        edu_section_pattern = r"(?i)(education|academic background|qualifications):?\s*(.*?)(?=\n\w+:|$)"
        match = re.search(edu_section_pattern, text, re.DOTALL)
        if match:
            edu_text = match.group(2)
            return [line.strip() for line in edu_text.split('\n') if line.strip()]
        return []

    def process_resumes(self, pdf_paths: List[str]):
        """Process multiple resumes and create vector store"""
        documents = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                st.warning(f"File not found: {pdf_path}")
                continue
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                st.warning(f"Empty text extracted from: {pdf_path}")
                continue
            info = self.parse_resume_info(text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "name": info["name"] or "Unknown",
                        "email": info["email"] or "",
                        "skills": ", ".join(info["skills"]) if info["skills"] else "",
                        "experience_years": info["experience_years"] or 0.0,
                        "source": os.path.basename(pdf_path)
                    }
                ))
        if documents:
            try:
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            except Exception as e:
                st.error(f"Failed to create vector store: {e}")
        else:
            st.error("No valid documents to process")

    def shortlist_resumes(self, required_skills: List[str], min_experience: float, top_k: int = 5) -> List[Dict]:
        """Shortlist resumes based on skills and experience"""
        if not self.vector_store:
            raise ValueError("No resumes processed yet. Call process_resumes() first.")
        filtered_docs = []
        for doc in self.vector_store.docstore._dict.values():
            exp = doc.metadata.get("experience_years", 0.0)
            if exp >= min_experience:
                filtered_docs.append(doc)
        docs_with_scores = []
        for doc in filtered_docs:
            candidate_skills = [s.strip().lower() for s in doc.metadata.get("skills", "").split(",")]
            required_skills_lower = [s.strip().lower() for s in required_skills]
            exact_matches = sum(1 for skill in required_skills_lower if skill in candidate_skills)
            rating = exact_matches
            if self.llm:
                prompt = f"""
                Evaluate how well this candidate's skills match the required skills.
                Required Skills: {', '.join(required_skills)}
                Candidate Skills: {doc.metadata.get('skills', '')}
                Provide a score from 1-10 where:
                - 1-3 = Poor match
                - 4-6 = Some relevant skills
                - 7-8 = Good match
                - 9-10 = Excellent match
                Provide only the number.
                """
                try:
                    rating = int(self.llm(prompt).strip())
                except:
                    rating = exact_matches
            docs_with_scores.append((doc, rating))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = docs_with_scores[:top_k]
        results = []
        for doc, rating in top_candidates:
            results.append({
                "name": doc.metadata.get("name", "Unknown"),
                "email": doc.metadata.get("email", ""),
                "skills": doc.metadata.get("skills", ""),
                "experience_years": doc.metadata.get("experience_years", 0.0),
                "match_score": rating,
                "summary": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                "source": doc.metadata.get("source", "")
            })
        return results

# Streamlit UI
st.title("Resume Shortlister")
st.markdown("Upload PDF resumes, specify required skills and minimum experience, and view shortlisted candidates.")

# Initialize session state for processor
if "processor" not in st.session_state:
    st.session_state.processor = ResumeProcessor()

# File uploader
uploaded_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

# Input for required skills
default_skills = "Python, JavaScript, React, Django, AWS"
required_skills = st.text_input("Required Skills (comma-separated)", value=default_skills)
required_skills_list = [s.strip() for s in required_skills.split(",") if s.strip()]

# Input for minimum experience
min_experience = st.number_input("Minimum Experience (years)", min_value=0.0, value=0.0, step=0.5)

# Input for number of top candidates
top_k = st.number_input("Number of Top Candidates to Show", min_value=1, value=5, step=1)

# Process button
if st.button("Process Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    else:
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        pdf_paths = []
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(pdf_path)
        
        # Process resumes
        with st.spinner("Processing resumes..."):
            try:
                st.session_state.processor.process_resumes(pdf_paths)
                candidates = st.session_state.processor.shortlist_resumes(
                    required_skills=required_skills_list,
                    min_experience=min_experience,
                    top_k=top_k
                )
                
                # Display results
                if candidates:
                    st.success("Resumes processed successfully!")
                    st.subheader("Shortlisted Candidates")
                    # Create DataFrame for display
                    df = pd.DataFrame(candidates)
                    df = df[["name", "email", "experience_years", "skills", "match_score", "source", "summary"]]
                    df.columns = ["Name", "Email", "Experience (Years)", "Skills", "Match Score", "Source File", "Summary"]
                    st.dataframe(df, use_container_width=True)
                    
                    # Optional: Detailed view for each candidate
                    for i, candidate in enumerate(candidates, 1):
                        with st.expander(f"Candidate #{i}: {candidate['name']}"):
                            st.write(f"**Email**: {candidate['email']}")
                            st.write(f"**Experience**: {candidate['experience_years']} years")
                            st.write(f"**Skills**: {candidate['skills']}")
                            st.write(f"**Match Score**: {candidate['match_score']}/10")
                            st.write(f"**Source File**: {candidate['source']}")
                            st.write(f"**Summary**: {candidate['summary']}")
                else:
                    st.warning("No candidates matched the criteria.")
            except Exception as e:
                st.error(f"Error processing resumes: {e}")
        
        # Clean up temporary files
        for pdf_path in pdf_paths:
            try:
                os.remove(pdf_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass