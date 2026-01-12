import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
import io
import requests
import json
import time
import re
from urllib.parse import urlparse, parse_qs


st.set_page_config(page_title="HR-Tek Systems ATS Checker", page_icon="📄", layout="wide")

st.markdown("""
<style>
    /* Hide Streamlit branding */
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

@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data packs."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data()

PREDEFINED_JOB_DESCRIPTIONS = {
    "Cloud / DevOps Intern (AWS Focused)": """Cloud / DevOps Intern (AWS Focused):-

About the Role:
-> Help HR-Tek optimize cloud usageand deploy the product on AWS with scalability, monitoring, and automation in mind.

Responsibilities:
-> Deploy staging and production environments on AWS.
-> Set up auto-scaling groups, load balancers, and monitoring (CloudWatch).
-> Work on CI/CD pipelines using AWS CodePipeline / Jenkins.
-> Optimize AWS credits for cost efficiency.

Preferred Skills: AWS, EC2, S3, RDS, VPC, IAM, CloudFormation/Terraform, Linux basics.
    """,

#     "Data Science & Analytics": """Data Science & Analytics:-
    
# About the Role:
# -> Build smart HR analytics that help companies understand attrition, engagement, and HR maturity through dashboards.

# Responsibilities:
# -> Analyze HR datasets and create predictive models.
# -> Develop dashboards for attrition, engagement, and trends.
# -> Support HR Maturity Assessment scoring logic.
# -> Work with Power BI / Python for data visualization.

# Preferred Skills: Python (Pandas, NumPy, Scikit-learn), SQL, Power BI / Tableau, statistics.
#     """,

    "GenAI Intern": """GenAI Intern (Generative AI for HR Advisory):-
    
About the Role:
-> Support HR-Tek in building integration-ready modules with enterprise HR systems like SAP, Workday, and Salesforce.
-> Build AI-driven features like HR chatbots, auto-generated digital roadmaps, and vendor-fit recommendations for clients.

Responsibilities:
-> Research integration frameworks of major HRIS (SAP, Workday, Darwinbox).
-> Build mock APIs for data exchange between HR-Tek and external tools.
-> Document integration workflows for enterprise readiness.
-> Work on data mapping & testing.
-> Train and fine-tune LLMs for HR queries.
-> Build a chatbot for HR digital advisory.
-> Prototype AI-based roadmap & recommendation engines.
-> Experiment with OpenAI / Hugging Face models.

Preferred Skills: REST APIs, JSON, Postman, Salesforce basics, SAP/Workday APIs (preferred), Python, LangChain, OpenAI APIs, NLP basics, prompt engineering.
    """,

#     "Product Testing & QA": """Product Testing & QA:-
    
# About the Role:
# -> Help HR-Tek ensure reliability and smooth scaling by testing features across browsers, devices, and automating regression testing.

# Responsibilities:
# -> Write Selenium scripts for automated regression testing.
# -> Conduct manual testing for new features.
# -> Ensure cross-browser & cross-device compatibility.
# -> Report bugs & work closely with developers for fixes.

# Preferred Skills: Selenium WebDriver, TestNG, basic Python/Java, manual testing methods.
#     """,

    "UI/UX Design": """UI/UX Design:-
    
About the Role:
-> Design clean, modern, and user-friendly dashboards for HR leaders, ensuring the product feels intuitive and enterprise-grade.
    
Responsibilities:
-> Create wireframes, mockups, and Figma prototypes.
-> Conduct usability testing with mock users.
-> Improve dashboard designs for clarity and adoption.
-> Work with developers to implement designs.
    
Preferred Skills: Figma, Adobe XD, UX research, design systems.
    """,

    "Full-Stack Development Intern": """Full-Stack Development Intern (React + Node/Python):-
    
About The Role:
-> Work on building and enhancing HR-Tek’s web-based application, contributing to core features, bug fixes, and scaling
modules.

Responsibilities:
-> Develop front-end features using React.js.
-> Build backend services using Node.js / Python.
-> Debug, test, and fix issues across the product.
-> Assist in integrating APIs with external HR systems.

Preferred Skills: React, Node.js/Python, Firebase, REST APIs, Git.
    """,

    "Custom Job Description": ""
}

def extract_file_id_from_gdrive_url(url):
    """Extract file ID from Google Drive URL"""
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/d/([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_file_from_gdrive(file_id):
    """Download file from Google Drive using file ID"""
    try:
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(download_url, headers=headers, stream=True, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                
                html_content = response.text
                confirm_match = re.search(r'confirm=([a-zA-Z0-9-_]+)', html_content)
                
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = requests.get(confirm_url, headers=headers, stream=True, allow_redirects=True)
                    
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type:
                        return None
            
            if 'application/pdf' in content_type or response.content.startswith(b'%PDF'):
                return response.content
            else:
                return None
        else:
            return None
            
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {e}")
        return None

def extract_text_from_gdrive_pdf(file_content):
    """Extracts text from PDF content downloaded from Google Drive."""
    try:

        if not file_content.startswith(b'%PDF'):
            st.error("Downloaded content is not a valid PDF file. Please check the Google Drive link and sharing permissions.")
            return None
            
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        if text.strip():
            st.success("✅ PDF text extracted successfully!")
            return text.strip()
        else:
            st.warning("⚠️ PDF file is empty or contains only images. Please ensure your resume has selectable text.")
            return None
            
    except Exception as e:
        st.error(f"Error reading PDF from Google Drive: {e}")
        return None


def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text.strip()
    except PyPDF2.errors.PdfReadError:
        st.error("This PDF file appears to be corrupted or password-protected. Please upload a different file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the PDF: {e}")
        return None

def get_gemini_analysis(resume_text, jd_text):
    """
    Calls the Gemini API to get an ATS analysis of the resume against the job description.
    Returns a structured JSON response with score, strengths, and improvements.
    """
    apiKey = st.secrets["GEMINI_API_KEY"]
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/129
    :generateContent?key={apiKey}"

    system_prompt = """
    You are an expert Applicant Tracking System (ATS) and a professional career coach. Your task is to analyze a resume against a job description.
    Provide a compatibility score as a percentage (as a number, not a string), a concise bulleted list of the candidate's strengths, and a concise bulleted list of actionable areas for improvement.
    The response MUST be in the specified JSON format.
    """

    user_prompt = f"""
    Analyze the following resume and job description and provide a compatibility analysis.

    **Resume Text:**
    ---
    {resume_text}
    ---

    **Job Description Text:**
    ---
    {jd_text}
    ---
    """

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "compatibilityScore": {
                "type": "NUMBER",
                "description": "A percentage score from 0 to 100 representing the match."
            },
            "strengths": {
                "type": "STRING",
                "description": "A concise, bulleted summary of how the resume aligns with the key requirements. Start with a brief introductory sentence."
            },
            "areasForImprovement": {
                "type": "STRING",
                "description": "A concise, bulleted list of actionable suggestions for how to improve the resume for this specific role. Start with a brief introductory sentence."
            }
        },
        "required": ["compatibilityScore", "strengths", "areasForImprovement"]
    }

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "temperature": 0.3
        }
    }

    max_retries = 3
    base_delay = 1  # in seconds
    for attempt in range(max_retries):
        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' not in result or not result['candidates']:
                 st.error("API Error: The response did not contain any candidates. Retrying...")
                 raise ValueError("Empty candidates list")

            json_text = result['candidates'][0]['content']['parts'][0]['text']
            analysis = json.loads(json_text)
            return analysis

        except requests.exceptions.HTTPError as http_err:
             st.error(f"HTTP error occurred: {http_err}. Please check the API key and endpoint.")
             return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"API request failed: {e}. Retrying in {base_delay}s...")
                time.sleep(base_delay)
                base_delay *= 2
            else:
                st.error(f"API request failed after {max_retries} attempts. Please check your connection.")
                return None
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse the API response. The format might be incorrect. Error: {e}")
            st.code(response.text if 'response' in locals() else "No response received.")
            return None
    return None


# MAIN APP

logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
with logo_col2:
    st.image("hr-tek-systems-logo.jpg", width=830)

# Centered title
st.markdown('<h1 class="main-title">ATS Resume Compatibility Checker</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">As a Hybrid SAAS HR Digital Transformation</p>', unsafe_allow_html=True)

# --- Layout with columns ---
col1, col2 = st.columns(2)

with col1:
    st.header("Resume")
        
    # Choice between manual upload and Google Drive link
    upload_method = st.radio(
        "Choose how to provide your resume:",
        ["Upload File", "Google Drive Link"],
        help="Select whether to upload a file directly or provide a Google Drive link"
    )
        
    resume_text = None
        
    if upload_method == "Upload File":
        uploaded_resume = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
        if uploaded_resume is not None:
            resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        gdrive_url = st.text_input(
            "Google Drive Link:",
            placeholder="https://drive.google.com/file/d/1ABC.../view?usp=sharing",
            help="Paste the Google Drive shareable link to your resume PDF"
        )
            
        if gdrive_url:
            if "drive.google.com" in gdrive_url:
                file_id = extract_file_id_from_gdrive_url(gdrive_url)
                if file_id:
                    with st.spinner('Downloading file from Google Drive...'):
                        file_content = download_file_from_gdrive(file_id)
                        if file_content:
                            resume_text = extract_text_from_gdrive_pdf(file_content)
                        else:
                            st.error("❌ Failed to download file from Google Drive. Please check the link and sharing permissions.")
                else:
                    st.error("❌ Could not extract file ID from the Google Drive URL. Please check the link format.")
            else:
                st.error("❌ Please provide a valid Google Drive URL")


    st.header("Intern Role")
        
    selected_jd = st.selectbox(
        "Choose a predefined job description:",
        options=list(PREDEFINED_JOB_DESCRIPTIONS.keys()),
        help="Select from predefined job descriptions or choose 'Custom Job Description' to paste your own"
    )

with col2:        
    if selected_jd != "Custom Job Description":
        st.header("Job Description:")
        # Use a unique key that changes with the selection to force re-render
        st.text_area(
            "", 
            value=PREDEFINED_JOB_DESCRIPTIONS[selected_jd], 
            height=400, 
            disabled=True, 
            key=f"preview_jd_{selected_jd}"  # Dynamic key based on selection
        )
        job_description = PREDEFINED_JOB_DESCRIPTIONS[selected_jd]
    else:
        st.header("Custom Job Description:")
        job_description = st.text_area("Paste your custom job description here", height=200, key="custom_jd")

if st.button("Analyze Compatibility", type="primary", use_container_width=True):
    if resume_text and job_description:
        
        if not resume_text:
            st.error("Could not extract any text from the uploaded PDF. Please ensure the file is not corrupted, password-protected, or an image-only PDF.")
        else:
            with st.spinner('Gemini is analyzing your documents... This may take a moment.'):
                analysis_result = get_gemini_analysis(resume_text, job_description)

                if analysis_result:
                    st.header("Analysis Results")
                    
                    st.info(f"Analyzed against: **{selected_jd}**")
                    
                    score = analysis_result.get("compatibilityScore", 0)
                    
                    if score > 75:
                        delta_text = "Excellent Match!"
                    elif score > 50:
                        delta_text = "Good Match"
                    else:
                        delta_text = "Needs Improvement"
                        
                    st.metric(label="Compatibility Score", value=f"{score}%", delta=delta_text)

                    # st.subheader("✅ Strengths")
                    # st.markdown(analysis_result.get("strengths", "No strengths identified."))

                    # st.subheader("Areas for Improvement")
                    # st.markdown(analysis_result.get("areasForImprovement", "No specific areas for improvement identified."))
                else:
                    st.error("Could not extract text from the uploaded PDF. Please ensure it's not a scanned image.")

    else:
        st.warning("Please provide your resume and select/paste a job description to proceed.")
