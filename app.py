import streamlit as st
import nltk
import os
import io
import requests
import json
import re
import PyPDF2

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

.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #2563eb;
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
        response = requests.get(url, stream=True)

        if response.status_code == 200 and response.content.startswith(b"%PDF"):
            return response.content
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
        if not text:
            st.warning("⚠️ This appears to be a scanned PDF. Please upload a text-based PDF.")

        return text if text else None

    except Exception as e:
        st.error(f"PDF read error: {e}")
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
        if not text:
            st.warning("⚠️ This appears to be a scanned PDF. Please upload a text-based PDF.")

        return text if text else None

    except PyPDF2.errors.PdfReadError:
        st.error("Corrupted or password-protected PDF.")
        return None

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ---------------- GEMINI API ----------------
def get_gemini_analysis(resume_text, jd_text):
    api_key = st.secrets["GEMINI_API_KEY"]
    prompt = f"""
You are an expert ATS system.

Analyze the resume against the job description.

Return ONLY valid JSON in this format:

{{
  "compatibilityScore": 0-100,
  "strengths": "- bullet points",
  "areasForImprovement": "- bullet points"
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }

    models = [
        "gemini-2.0-flash",
        "gemini-flash-latest",
    ]

    import time
    
    last_error_msg = "Unknown error"
    
    for model in models:
        api_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={api_key}"
        )

        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 404:
                last_error_msg = f"Model {model} not found or not supported."
                print(f"DEBUG: 404 for model {model}")
                continue
            
            if response.status_code == 400:
                try:
                    error_detail = response.json()
                    print(f"DEBUG: 400 Error - {error_detail}")
                except:
                    print(f"DEBUG: 400 Error - {response.text}")
                st.error("🚫 Unable to process the request. Please try again.")
                return None

            if response.status_code == 401:
                print(f"DEBUG: 401 Unauthorized")
                st.error("🚫 Authentication failed. Please contact support.")
                return None

            if response.status_code == 429:
                 print(f"DEBUG: 429 Rate limit for {model}")
                 last_error_msg = "Service temporarily unavailable. Please try again later."
                 continue

            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    print(f"DEBUG: {response.status_code} Error - {error_detail}")
                    last_error_msg = f"Service error ({response.status_code}). Please try again."
                except:
                    print(f"DEBUG: {response.status_code} Error - {response.text[:200]}")
                    last_error_msg = "Service error. Please try again."
                continue
            
            # If we get here, the request was successful
            result = response.json()
            
            # Check if candidates exist (handling safety blocks)
            if not result.get("candidates") or not result["candidates"][0].get("content"):
                st.error("⚠️ Unable to analyze this resume. Please try a different file.")
                return None

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
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
                if match:
                    cleaned_text = match.group(0)
            
            try:
                parsed_result = json.loads(cleaned_text)
                return parsed_result
            except json.JSONDecodeError:
                # Fallback: Try to find just the score
                score_match = re.search(r'"compatibilityScore":\s*(\d+)', raw_text)
                if score_match:
                    return {"compatibilityScore": int(score_match.group(1))}
                
                st.error("⚠️ Analysis incomplete. Please try again.")
                return None

        except requests.exceptions.Timeout:
            print(f"DEBUG: Timeout error with {model}")
            last_error_msg = f"Connection timeout. Please try again."
            continue
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Connection error with {model}: {str(e)}")
            last_error_msg = f"Connection error. Please try again."
            continue
    
    print(f"DEBUG: All models failed. Last error: {last_error_msg}")
    st.error(f"❌ Analysis failed: {last_error_msg}")
    return None

# ---------------- LOGO ----------------
if os.path.exists("hr-tek-systems-logo.jpg"):
    import base64
    with open("hr-tek-systems-logo.jpg", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 0rem;">
            <img src="data:image/jpeg;base64,{logo_base64}" width="300">
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<h1 class="main-title">ATS Resume Compatibility Checker</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray; font-size: 1.2rem;">As a Hybrid SAAS HR Digital Transformation</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.header("Resume")
    method = st.radio("Resume Input Method", ["Upload File", "Google Drive Link"])
    resume_text = None

    if method == "Upload File":
        file = st.file_uploader("Upload PDF", type=["pdf"])
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

with col2:
    if selected_jd != "Custom Job Description":
        job_description = PREDEFINED_JOB_DESCRIPTIONS[selected_jd]
        st.text_area("Job Description", job_description, height=400, disabled=True)
    else:
        job_description = st.text_area("Paste Job Description", height=300)

# ---------------- ANALYSIS ----------------
if st.button("Analyze Compatibility", type="primary", use_container_width=True):
    if not resume_text:
        st.error("Could not extract text from the uploaded PDF. Please ensure it's a text-based PDF (not a scanned image).")
    elif not job_description:
        st.warning("Please provide a job description.")
    else:
        with st.spinner("Analyzing Resume..."):
            result = get_gemini_analysis(resume_text, job_description)

            if result:
                # Only Display Score as requested
                st.markdown(f"""
                <div style="text-align: center; margin-top: 2rem;">
                    <h2>Compatibility Score</h2>
                    <h1 style="font-size: 5rem; color: #2563eb;">{result.get('compatibilityScore', 0)}%</h1>
                </div>
                """, unsafe_allow_html=True)
