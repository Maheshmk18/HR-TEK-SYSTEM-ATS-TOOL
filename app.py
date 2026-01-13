import streamlit as st
import nltk
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

        return text.strip() if text else None

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

        return text.strip() if text else None

    except PyPDF2.errors.PdfReadError:
        st.error("Corrupted or password-protected PDF.")
        return None

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ---------------- GEMINI API ----------------
def get_gemini_analysis(resume_text, jd_text):
    api_key = st.secrets["GEMINI_API_KEY"]
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={api_key}"
    )

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
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
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

            cleaned_text = (
                raw_text.replace("```json", "")
                .replace("```", "")
                .strip()
            )

            return json.loads(cleaned_text)

        except json.JSONDecodeError:
            st.error("⚠️ AI response was malformed. Please retry once.")
            return None

        except requests.exceptions.RequestException:
            st.warning(
                "⚠️ Temporary AI service issue. Please retry in a moment."
            )
            return None

    return None

# ---------------- UI ----------------
st.markdown('<h1 class="main-title">ATS Resume Compatibility Checker</h1>', unsafe_allow_html=True)

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
        st.warning("Please upload a valid resume.")
    elif not job_description:
        st.warning("Please provide a job description.")
    else:
        with st.spinner("Analyzing resume with Gemini..."):
            result = get_gemini_analysis(resume_text, job_description)

            if result:
                st.metric(
                    "Compatibility Score",
                    f"{result['compatibilityScore']}%"
                )
                st.subheader("✅ Strengths")
                st.markdown(result["strengths"])
                st.subheader("⚠️ Areas for Improvement")
                st.markdown(result["areasForImprovement"])
