import streamlit as st
import nltk
import io
import requests
import json
import re
import PyPDF2
import os
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env if it exists
load_dotenv()
# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HR-Tek Systems ATS Checker",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 4rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem;
    color: #2563eb;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 1.8rem !important;
    color: #4b5563;
    margin-bottom: 2rem;
    font-weight: 500;
}

/* Headers (Resume, Intern Role, Job Description) */
h2 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #1e3a8a !important;
    padding-bottom: 0.5rem;
}

/* Widget Labels */
.stRadio p, .stSelectbox label, .stFileUploader label, .stTextInput label {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}

/* Radio button options */
.stRadio div[role='radiogroup'] label {
    font-size: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NLTK ----------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

download_nltk_data()

# ---------------- JOB DESCRIPTIONS ----------------
PREDEFINED_JOB_DESCRIPTIONS = {
    "Cloud / DevOps Intern (AWS Focused)": """
About the Role:
Help HR-Tek optimize cloud usage and deploy products on AWS.

Responsibilities:
- Deploy environments on AWS
- Configure autoscaling & monitoring
- Work on CI/CD pipelines
- Optimize AWS credits

Skills: AWS, EC2, S3, RDS, IAM, Terraform, Linux
""",

    "GenAI Intern": """
About the Role:
Build AI-driven HR advisory tools.

Responsibilities:
- Build HR chatbots
- Work with LLMs
- Create roadmap & recommendation engines

Skills: Python, LangChain, OpenAI APIs, NLP, REST APIs
""",

    "UI/UX Design": """
Responsibilities:
- Create wireframes & prototypes
- Conduct usability testing
- Improve dashboards

Skills: Figma, UX Research
""",

    "Full-Stack Development Intern": """
Responsibilities:
- React frontend development
- Backend APIs
- Debug & scale product

Skills: React, Node.js, Python, REST APIs
""",

    "Custom Job Description": ""
}

# ---------------- GOOGLE DRIVE HELPERS ----------------
def extract_file_id_from_gdrive_url(url):
    patterns = [
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
        r"/d/([a-zA-Z0-9-_]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_file_from_gdrive(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)

        # Handle 'virus scan' warning by looking for cookies
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(url, params=params, stream=True)
                break

        if response.status_code == 200:
            if response.content.startswith(b"%PDF"):
                return response.content
            elif b"<!DOCTYPE html>" in response.content or b"<html" in response.content:
                st.error("⚠️ Google Drive link is not public. Please change permissions to 'Anyone with the link can view'.")
                return None
            else:
                st.error("⚠️ Downloaded file is not a valid PDF.")
                return None
        else:
            st.error(f"⚠️ Google Drive download failed. Status code: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Google Drive download error: {e}")
        return None

def extract_text_from_gdrive_pdf(file_content):
    try:
        pdf = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        text = text.strip()
        if not text and OCR_AVAILABLE:
            st.info("ℹ️ Scanned PDF detected. Attempting OCR extraction...")
            text = extract_text_with_ocr(file_content)

        return text if text else None

    except Exception as e:
        st.error(f"PDF read error: {e}")
        return None

# ---------------- OCR HELPER ----------------
def extract_text_with_ocr(pdf_content):
    if not OCR_AVAILABLE:
        return None
    try:
        images = convert_from_bytes(pdf_content, dpi=300)
        extracted_text = ""
        for image in images:
            page_text = pytesseract.image_to_string(image, lang='eng')
            extracted_text += page_text + "\n"
        return extracted_text.strip()
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return None

# ---------------- PDF UPLOAD ----------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        
        text = text.strip()
        if not text and OCR_AVAILABLE:
             st.info("ℹ️ Scanned PDF detected. Attempting OCR extraction...")
             uploaded_file.seek(0)
             text = extract_text_with_ocr(uploaded_file.read())

        return text if text else None

    except PyPDF2.errors.PdfReadError:
        st.error("Corrupted or password-protected PDF.")
        return None

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ---------------- GEMINI API ----------------
def get_gemini_analysis(resume_text, jd_text):
    # Try to get API key from streamlit secrets, then environment variable
    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("❌ GEMINI_API_KEY not found. Please set it in .streamlit/secrets.toml or a .env file.")
        return None

    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={api_key}"
    )

    prompt = f"""
You are an advanced, enterprise-grade Applicant Tracking System (ATS) used by top technology companies.
Your goal is to evaluate the resume against the job description with high strictness and accuracy.

**Analysis Rules:**
1. **Strict Keyword Matching**: Check for specific technical skills (e.g., Python, AWS, React) mentioned in the Job Description. Missing critical skills should lower the score.
2. **Contextual Analysis**: Do not just look for keywords; ensure the candidate has *experience* using them (e.g., "Used Python for data analysis" is better than just listing "Python").
3. **Scoring Logic**:
   - 90-100%%: Perfect match, exceeds requirements (Topper/Ideal Candidate).
   - 75-89%%: Strong match, fits most requirements.
   - 50-74%%: Average match, gaps in critical skills.
   - <50%%: Poor match, missing major requirements.

**Output Requirement:**
Return a valid JSON object with detailed feedback:
{{
  "compatibilityScore": <integer 0-100>,
  "strengths": [
    "<Technical Skill Match: e.g., 'Strong proficiency in AWS and Terraform as required'>",
    "<Experience Match: e.g., 'Relevant internship experience in Cloud DevOps'>",
    "<Soft Skill/Formatting: e.g., 'Clear project descriptions and quantified achievements'>"
  ],
  "areasForImprovement": [
    "<Missing Skill: e.g., 'Missing explicit mention of CI/CD pipelines (Jenkins/GitLab)'>",
    "<Experience Gap: e.g., 'No prior experience with containerization (Docker/Kubernetes) which is a key requirement'>",
    "<Formatting/Detail: e.g., 'Project sections lack metrics or outcomes'>"
  ]
}}

**Data for Analysis:**
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }

    headers = {"Content-Type": "application/json"}

    max_retries = 2
    delay = 2  # seconds

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )

            # 🔴 RATE LIMIT HANDLING
            if response.status_code == 429:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    st.warning(
                        "🚦 High traffic right now. Please wait 1–2 minutes and try again."
                    )
                    return None

            response.raise_for_status()

            result = response.json()
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

            # Clean the response - remove markdown code blocks
            cleaned_text = raw_text.strip()
            
            # Remove markdown code fences
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_text:
                cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()
            
            # Try to extract JSON if it's embedded in text
            if not cleaned_text.startswith("{"):
                # Look for JSON object in the text
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(0)
            
            try:
                parsed_result = json.loads(cleaned_text)
                return parsed_result
            except json.JSONDecodeError as je:
                # Log the problematic response for debugging
                st.error(f"⚠️ AI response was malformed. Please retry once.")
                with st.expander("Debug Info (Click to expand)"):
                    st.text(f"Raw response:\n{raw_text[:500]}")
                    st.text(f"Cleaned:\n{cleaned_text[:500]}")
                    st.text(f"Error: {str(je)}")
                return None

        except json.JSONDecodeError:
            st.error("⚠️ AI response was malformed. Please retry once.")
            return None

        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg = error_details.get("error", {}).get("message", str(e))
                except:
                    error_msg = e.response.text or str(e)
                st.error(f"❌ Gemini API Error ({e.response.status_code}): {error_msg}")
            else:
                st.error(f"⚠️ Connection Error: {e}")
            return None

    return None

# ---------------- UI ----------------
col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 2, 1])
with col_logo_2:
    if os.path.exists("hr-tek-systems-logo.jpg"):
        st.image("hr-tek-systems-logo.jpg", use_column_width=True)

st.markdown('<h1 class="main-title">ATS Resume Compatibility Checker</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">As a Hybrid SAAS HR Digital Transformation</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.header("Resume")
    method = st.radio("Resume Input Method", ["Upload File", "Google Drive Link"])
    resume_text = None

    if method == "Upload File":
        file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
        if file:
            resume_text = extract_text_from_pdf(file)
    else:
        link = st.text_input("Google Drive PDF Link")
        if link and "drive.google.com" in link:
            file_id = extract_file_id_from_gdrive_url(link)
            if file_id:
                content = download_file_from_gdrive(file_id)
                if content:
                    resume_text = extract_text_from_gdrive_pdf(content)

    st.header("Intern Role")
    selected_jd = st.selectbox(
        "Select Job Description",
        list(PREDEFINED_JOB_DESCRIPTIONS.keys())
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Analyze Compatibility", type="primary", use_container_width=True)

with col2:
    st.header("Job Description")
    if selected_jd != "Custom Job Description":
        job_description = PREDEFINED_JOB_DESCRIPTIONS[selected_jd]
        st.text_area("Job Description", job_description, height=400, label_visibility="collapsed")
    else:
        job_description = st.text_area("Paste Job Description", height=300, label_visibility="collapsed")
    

# ---------------- ANALYSIS ----------------
if analyze_button:
    if not resume_text:
        st.error("Could not extract text from the uploaded PDF. Please ensure it's not a scanned image or check OCR configuration.")
    elif not job_description:
        st.warning("Please provide a job description.")
    else:
        with st.spinner("Analyzing the resume..."):
            result = get_gemini_analysis(resume_text, job_description)

            if result:
                st.markdown(f"""
                <div style="text-align: center; margin-top: 2rem;">
                    <h2>Compatibility Score</h2>
                    <h1 style="font-size: 5rem; color: #2563eb;">{result['compatibilityScore']}%</h1>
                </div>
                """, unsafe_allow_html=True)
