import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
 
# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LangChain LLM Setup
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4
    )

# Extract text from PDF
@st.cache_data(show_spinner=False)
def extract_pdf_text(file):
    try:
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text:
            raise ValueError("No text extracted from the PDF")
        return text
    except Exception as e:
        raise ValueError(f"PDF Error: {e}")

# Create LangChain PromptTemplate
def build_prompt():
    return PromptTemplate.from_template(
        """
        Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
        - Technical fields
        - Software engineering
        - Data science
        - Data analysis
        - Big data engineering
        
        Evaluate the following resume against the job description. Consider that the job market 
        is highly competitive. Provide detailed feedback for resume improvement.
    
        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Give response in ONLY this JSON format:
        {{
            "JD Match": "percentage between 0-100",
            "MissingKeywords": ["keyword1", "keyword2", ...],
            "Profile Summary": "detailed analysis of the match and specific improvement suggestions"
        }}
        """
    )

# Streamlit App
st.set_page_config(page_title="Smart ATS Resume Analyzer", layout="centered")
st.title("📄 Smart ATS Resume Analyzer")
st.subheader("Optimize Your Resume for ATS")

with st.sidebar:
    st.title("🎯 Resume Analyzer")
    st.markdown("""
    This smart ATS helps you:
    - Match your resume to job description
    - Find missing keywords
    - Get expert improvement suggestions
    """)

job_description = st.text_area("Job Description", placeholder="Paste the job description here...")
resume_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")

if st.button("🔍 Analyze Resume"):
    if not GOOGLE_API_KEY:
        st.error("Please set GOOGLE_API_KEY in your .env file.")
    elif not job_description:
        st.warning("Please enter the job description.")
    elif not resume_file:
        st.warning("Please upload a resume PDF file.")
    else:
        with st.spinner("Analyzing resume with Gemini..."):
            try:
                resume_text = extract_pdf_text(resume_file)
                prompt = build_prompt()
                chain = prompt | get_llm()
                full_input = {"resume_text": resume_text, "job_description": job_description}
                response = chain.invoke(full_input)
                
                # Extract JSON safely
                try:
                    response_json = json.loads(response.content)
                except:
                    import re
                    match = re.search(r'{.*}', response.content, re.DOTALL)
                    response_json = json.loads(match.group() if match else '{}')

                st.success("✅ Analysis Complete!")

                st.metric("Match Score", response_json.get("JD Match", "N/A"))
                
                st.subheader("🔑 Missing Keywords")
                keywords = response_json.get("MissingKeywords", [])
                st.write(", ".join(keywords) if keywords else "No major keywords missing!")

                st.subheader("📝 Profile Summary")
                st.write(response_json.get("Profile Summary", "No summary provided."))

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


