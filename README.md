MahnoorKhan111
AI-Resume-Analyzer
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Security and quality
Insights
Settings
MahnoorKhan111
AI-Resume-Analyzer
Private
Go to file
t
T
MahnoorKhan111
MahnoorKhan111
Update requirements.txt
249baee
 · 
21 minutes ago
Name		
.devcontainer
Added Dev Container Folder
47 minutes ago
.gitignore
Initial project upload
2 days ago
README.md
Add project documentation README
2 days ago
app.py
Update app.py
32 minutes ago
best_resume_model.joblib
Initial project upload
2 days ago
calibrated_resume_model updated.joblib
Updated app.py and calibrated SVM model
1 hour ago
requirements.txt
Update requirements.txt
21 minutes ago
resume_model.joblib
Initial project upload
2 days ago
skills_dictionary.joblib
Initial project upload
2 days ago
test_gemini.py
Initial project upload
2 days ago
Repository files navigation
README
📄 AI Resume Analyzer
An AI-powered Resume Analyzer built using Python, Streamlit, and Google's Gemini AI. The application analyzes uploaded resumes, identifies professional fields, evaluates skills and experience, and generates personalized career recommendations.

🎯 Project Overview
The AI Resume Analyzer helps users understand their professional profiles through automated resume analysis.

Users can upload their resumes in PDF, DOCX, or TXT format and receive an AI-generated professional assessment.

✨ Key Features
Multiple Resume Formats: Upload PDF, DOCX, and TXT files.
Resume Text Extraction: Automatically extract text from uploaded resumes.
Resume Statistics: Calculate total words and characters.
AI-Powered Analysis: Use Google Gemini AI to assess professional profiles.
Professional Classification: Identify the primary professional field and category.
Skills Assessment: Identify technical skills and explicitly stated soft skills.
Education and Experience Analysis: Extract educational qualifications and professional experience.
Career Recommendations: Suggest suitable job roles and career improvement strategies.
Skill Gap Analysis: Identify missing skills and areas for professional development.
Professional Dashboard: Present assessment results in an organized dashboard.
PDF Report Generation: Download a formatted professional assessment report.
🛠️ Technologies Used
Technology	Purpose
Python	Application programming
Streamlit	Web application interface
Google Gemini AI	Intelligent resume assessment
PyPDF	PDF text extraction
Python-docx	Word document processing
ReportLab	PDF report generation
Python-dotenv	Environment variable management
📂 Project Structure
AI-Resume-Analyzer/
│
├── app.py
├── app1.py
├── app_backup_working.py
├── requirements.txt
├── best_resume_model.joblib
├── calibrated_resume_model.joblib
├── resume_model.joblib
├── skills_dictionary.joblib
├── test_gemini.py
├── .gitignore
└── README.md
⚙️ Installation and Setup
1. Clone the repository
git clone https://github.com/MahnoorKhan111/AI-Resume-Analyzer.git
2. Open the project directory
cd AI-Resume-Analyzer
3. Install dependencies
pip install -r requirements.txt
4. Configure Gemini API Key
Create a .env file in the project directory:

GEMINI_API_KEY=your_actual_gemini_api_key
Replace the placeholder with your own API key.

Important: Never upload your API key or .env file to GitHub.

5. Run the application
streamlit run app.py
The application will open in your default web browser.

📊 Application Workflow
Upload a resume in PDF, DOCX, or TXT format.
Extract and display resume content.
View resume statistics.
Generate an AI-powered professional assessment.
View the professional dashboard.
Review career recommendations and identified skill gaps.
Download the complete assessment as a PDF report.
🔐 Privacy and Security
Gemini API credentials should be stored securely.
API keys must never be committed to a public repository.
Users should avoid uploading confidential or sensitive resume information.
AI-generated assessments should be reviewed and are not guaranteed hiring decisions.
⚠️ Limitations
Scanned PDFs may require OCR support.
Complex resume layouts may affect text extraction.
AI-generated recommendations may occasionally be inaccurate.
Assessment quality depends on the information provided in the resume.
🚀 Future Enhancements
OCR support for scanned resumes.
Resume improvement and optimization suggestions.
ATS compatibility analysis.
Advanced skill matching.
Job description and resume matching.
Enhanced professional dashboards.
👩‍💻 Developer
Developed by: Mahnoor Khan

Project: AI Resume Analyzer

Category: Artificial Intelligence / Machine Learning

Framework: Python and Streamlit

AI Resume Analyzer — Making professional resume assessment smarter and easier.

About

No description, website, or topics provided.
Resources
Readme
Activity
Stars
0 stars
Watchers
0 watching
Forks
0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Contributors
1
 (1)
@MahnoorKhan111
MahnoorKhan111
Languages
Python
100%
