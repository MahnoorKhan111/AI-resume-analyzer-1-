import io
import json
import os
import re
import html
import hashlib
from pathlib import Path

import joblib
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload a PDF, DOCX, or TXT resume. The SVM model predicts a category "
    "from the training dataset, while Gemini independently analyzes the "
    "professional field and provides a detailed assessment."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "calibrated_resume_model updated.joblib"
GEMINI_MODEL = "gemini-2.5-flash"

# These are the exact 24 labels stored in the uploaded trained model.
DATASET_CATEGORIES = [
    "ACCOUNTANT",
    "ADVOCATE",
    "AGRICULTURE",
    "APPAREL",
    "ARTS",
    "AUTOMOBILE",
    "AVIATION",
    "BANKING",
    "BPO",
    "BUSINESS-DEVELOPMENT",
    "CHEF",
    "CONSTRUCTION",
    "CONSULTANT",
    "DESIGNER",
    "DIGITAL-MEDIA",
    "ENGINEERING",
    "FINANCE",
    "FITNESS",
    "HEALTHCARE",
    "HR",
    "INFORMATION-TECHNOLOGY",
    "PUBLIC-RELATIONS",
    "SALES",
    "TEACHER",
]

# These pairs are described as related, not identical.
RELATED_CATEGORY_PAIRS = {
    frozenset(("ACCOUNTANT", "FINANCE")),
    frozenset(("BANKING", "FINANCE")),
}


# ============================================================
# 2. LOAD THE TRAINED SVM MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep the model file in the same GitHub folder as app.py."
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved model does not provide predict() and predict_proba()."
        )

    model_labels = {str(label) for label in model.classes_}
    expected_labels = set(DATASET_CATEGORIES)

    if model_labels != expected_labels:
        raise ValueError(
            "The model's category labels do not match the categories "
            "configured in app.py. Please use the supplied model file."
        )

    return model


# ============================================================
# 3. CLEANING AND RESUME TEXT EXTRACTION
# ============================================================

def clean_resume(text):
    """Apply the same basic text normalization used before model prediction."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_resume_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    if filename.endswith(".docx"):
        document = Document(io.BytesIO(uploaded_file.getvalue()))
        parts = [paragraph.text for paragraph in document.paragraphs]

        for table in document.tables:
            for row in table.rows:
                parts.append(" ".join(cell.text for cell in row.cells))

        return "\n".join(parts)

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    raise ValueError("Please upload a PDF, DOCX, or TXT file.")


# ============================================================
# 4. SVM MACHINE-LEARNING PREDICTION
# ============================================================

def predict_resume_category(resume_text):
    model = load_model()
    model_input = clean_resume(resume_text)

    if not model_input:
        raise ValueError("The resume does not contain readable text.")

    # The saved artifact includes TF-IDF and the classifier.
    # Pass raw cleaned text; do not vectorize it again here.
    probabilities = model.predict_proba([model_input])[0]
    classes = [str(label) for label in model.classes_]

    ranked_indices = probabilities.argsort()[::-1]
    top_results = [
        {
            "category": classes[index],
            "score": float(probabilities[index]) * 100
        }
        for index in ranked_indices[:3]
    ]

    prediction = classes[int(probabilities.argmax())]
    return prediction, top_results


# ============================================================
# 5. GEMINI API
# ============================================================

def get_gemini_client():
    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Community "
            "Cloud > App settings > Secrets. Do not put the key in app.py."
        )

    return genai.Client(api_key=api_key)


def analyze_with_gemini(resume_text, svm_prediction, top_results):
    """
    Gemini independently classifies the resume using the same dataset label
    vocabulary, or OUT_OF_SCOPE when no label is a reasonable fit.
    It also returns the detailed professional assessment.
    """
    client = get_gemini_client()

    category_enum = DATASET_CATEGORIES + ["OUT_OF_SCOPE"]

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "primary_professional_field": {"type": "STRING"},
            "professional_category": {"type": "STRING"},
            "dataset_category": {
                "type": "STRING",
                "enum": category_enum
            },
            "dataset_category_explanation": {"type": "STRING"},
            "recommended_job_title": {"type": "STRING"},
            "classification_explanation": {"type": "STRING"},
            "professional_summary": {"type": "STRING"},
            "technical_skills": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "soft_skills": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "education": {"type": "STRING"},
            "professional_experience": {"type": "STRING"},
            "career_recommendations": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "identified_skill_gaps": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "certifications_and_training": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "overall_evaluation": {"type": "STRING"}
        },
        "required": [
            "primary_professional_field",
            "professional_category",
            "dataset_category",
            "dataset_category_explanation",
            "recommended_job_title",
            "classification_explanation",
            "professional_summary",
            "technical_skills",
            "soft_skills",
            "education",
            "professional_experience",
            "career_recommendations",
            "identified_skill_gaps",
            "certifications_and_training",
            "overall_evaluation"
        ]
    }

    svm_top_text = ", ".join(
        f"{item['category']} ({item['score']:.1f}%)"
        for item in top_results
    )

    prompt = f"""
You are an objective professional resume analyst.

Analyze only the information actually present in the resume.
Never invent education, employment, years of experience, skills,
certifications, achievements, or personal details.

The machine-learning model in this app was trained on exactly these
24 dataset categories:
{", ".join(DATASET_CATEGORIES)}

Your dataset_category field MUST be exactly one of those 24 labels,
or OUT_OF_SCOPE if none is a reasonable match. Do not create a new
category. Do not choose an unrelated label just to agree with SVM.

The SVM model's top prediction is: {svm_prediction}
Its top three estimates are: {svm_top_text}

Make your own independent assessment. Do not copy the SVM prediction
unless the resume evidence supports it. If the resume describes a field
not represented in the list (for example, a specialized research field
with no suitable dataset category), choose OUT_OF_SCOPE.

Explain your choice briefly. Do not describe model probability as a
guarantee of correctness.

Return concise but useful information for every required JSON field:
- primary_professional_field: actual field described by the resume
- professional_category: a readable category name
- dataset_category: exact dataset label or OUT_OF_SCOPE
- dataset_category_explanation: why that label fits, or why no label fits
- recommended_job_title: one suitable role based on the resume
- classification_explanation: compare your independent category with SVM
- professional_summary: concise summary
- technical_skills: list only skills supported by the resume
- soft_skills: list only skills explicitly stated or clearly evidenced
- education: summarize education; say Not stated if absent
- professional_experience: summarize roles and responsibilities; say Not stated if absent
- career_recommendations: practical recommendations
- identified_skill_gaps: relevant gaps; do not invent requirements
- certifications_and_training: list those present; say Not stated if absent
- overall_evaluation: balanced overall assessment

RESUME TEXT:
{resume_text[:30000]}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=5000
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini returned a response that could not be read as JSON. "
            "Please try generating the assessment again."
        ) from error

    dataset_category = str(result.get("dataset_category", "")).strip()
    if dataset_category not in category_enum:
        raise ValueError(
            "Gemini returned an invalid dataset category. Please try again."
        )

    return result


# ============================================================
# 6. COMPARE THE TWO CLASSIFICATIONS
# ============================================================

def compare_classifications(svm_category, gemini_category):
    if gemini_category == "OUT_OF_SCOPE":
        return (
            "OUT OF SCOPE",
            "Gemini found that the resume's professional field does not "
            "reasonably fit one of the 24 categories in the training dataset."
        )

    if svm_category == gemini_category:
        return (
            "EXACT AGREEMENT",
            "Both systems selected the same dataset category."
        )

    pair = frozenset((svm_category, gemini_category))
    if pair in RELATED_CATEGORY_PAIRS:
        return (
            "RELATED CATEGORIES",
            "The systems selected different but related categories. "
            "They are not an exact match."
        )

    return (
        "DIFFERENT PREDICTIONS",
        "The systems selected different categories. Review the resume "
        "evidence and the two explanations; neither result should be "
        "treated as guaranteed correct."
    )


# ============================================================
# 7. DISPLAY HELPERS
# ============================================================

def show_list(title, items):
    st.markdown(f"**{title}**")
    if not items:
        st.write("Not identified.")
        return

    for item in items:
        st.markdown(f"- {item}")


def show_gemini_result(result):
    st.markdown("### Gemini Professional Assessment")
    st.write("**Primary Professional Field:**", result["primary_professional_field"])
    st.write("**Professional Category:**", result["professional_category"])
    st.write("**Gemini Dataset Category:**", result["dataset_category"])
    st.write("**Dataset Category Explanation:**", result["dataset_category_explanation"])
    st.write("**Recommended Job Title:**", result["recommended_job_title"])

    st.markdown("**Classification Explanation**")
    st.write(result["classification_explanation"])

    st.markdown("**Professional Summary**")
    st.write(result["professional_summary"])

    show_list("Technical Skills", result["technical_skills"])
    show_list("Soft Skills", result["soft_skills"])

    st.markdown("**Education**")
    st.write(result["education"])

    st.markdown("**Professional Experience**")
    st.write(result["professional_experience"])

    show_list("Career Recommendations", result["career_recommendations"])
    show_list("Identified Skill Gaps", result["identified_skill_gaps"])
    show_list("Certifications and Training", result["certifications_and_training"])

    st.markdown("**Overall Evaluation**")
    st.write(result["overall_evaluation"])


# ============================================================
# 8. PDF REPORT
# ============================================================

def generate_pdf(resume_name, svm_prediction, top_results, gemini_result,
                 comparison_status, comparison_explanation):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=7
    )

    story = [
        Paragraph("AI RESUME ANALYSIS REPORT", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"<b>Resume file:</b> {html.escape(resume_name)}", body_style),
        Paragraph("1. MACHINE-LEARNING PREDICTION (SVM)", styles["Heading2"]),
        Paragraph(
            f"<b>Predicted dataset category:</b> {html.escape(svm_prediction)}",
            body_style
        ),
        Paragraph("<b>Top three model estimates:</b>", body_style)
    ]

    for item in top_results:
        story.append(
            Paragraph(
                f"{html.escape(item['category'])}: {item['score']:.2f}%",
                body_style
            )
        )

    story.extend([
        Spacer(1, 8),
        Paragraph("2. SVM AND GEMINI COMPARISON", styles["Heading2"]),
        Paragraph(f"<b>Status:</b> {html.escape(comparison_status)}", body_style),
        Paragraph(html.escape(comparison_explanation), body_style),
        Paragraph("3. GEMINI PROFESSIONAL ASSESSMENT", styles["Heading2"])
    ])

    fields = [
        ("Primary Professional Field", "primary_professional_field"),
        ("Professional Category", "professional_category"),
        ("Gemini Dataset Category", "dataset_category"),
        ("Dataset Category Explanation", "dataset_category_explanation"),
        ("Recommended Job Title", "recommended_job_title"),
        ("Classification Explanation", "classification_explanation"),
        ("Professional Summary", "professional_summary"),
        ("Education", "education"),
        ("Professional Experience", "professional_experience"),
        ("Overall Evaluation", "overall_evaluation"),
    ]

    for label, key in fields:
        story.append(Paragraph(f"<b>{label}:</b>", body_style))
        story.append(Paragraph(html.escape(str(gemini_result.get(key, ""))), body_style))

    list_fields = [
        ("Technical Skills", "technical_skills"),
        ("Soft Skills", "soft_skills"),
        ("Career Recommendations", "career_recommendations"),
        ("Identified Skill Gaps", "identified_skill_gaps"),
        ("Certifications and Training", "certifications_and_training"),
    ]

    for label, key in list_fields:
        story.append(Paragraph(f"<b>{label}:</b>", body_style))
        values = gemini_result.get(key, [])
        if values:
            bullet_items = [
                ListItem(Paragraph(html.escape(str(value)), body_style))
                for value in values
            ]
            story.append(ListFlowable(bullet_items, bulletType="bullet"))
        else:
            story.append(Paragraph("Not identified.", body_style))

    story.extend([
        Spacer(1, 12),
        Paragraph(
            "Note: SVM estimates are model outputs, not guarantees. "
            "Gemini provides an independent AI assessment. Different "
            "predictions should be reviewed rather than hidden.",
            styles["Italic"]
        )
    ])

    document.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# 9. MAIN APP INTERFACE
# ============================================================

st.subheader("Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, and TXT."
)

if uploaded_file is not None:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    # Prevent results from a previously uploaded resume being displayed
    # after the user selects a different file.
    if st.session_state.get("current_file_hash") != file_hash:
        st.session_state["current_file_hash"] = file_hash
        for key in [
            "svm_prediction", "top_results", "gemini_result",
            "comparison_status", "comparison_explanation",
            "resume_text", "resume_name"
        ]:
            st.session_state.pop(key, None)

    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        resume_text = extract_resume_text(uploaded_file)

        if not resume_text.strip():
            st.error(
                "No readable text was found. If this is a scanned PDF, "
                "it may need OCR before analysis."
            )
            st.stop()

        word_count = len(resume_text.split())
        char_count = len(resume_text)

        st.subheader("📊 Resume Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Total Words", word_count)
        col2.metric("Total Characters", char_count)

        with st.expander("📄 View Extracted Resume Text"):
            st.text_area(
                "Extracted content",
                resume_text,
                height=250,
                disabled=True
            )

        st.divider()
        st.subheader("🤖 Machine Learning (SVM) Prediction")

        if "svm_prediction" not in st.session_state:
            with st.spinner("SVM is predicting the resume category..."):
                prediction, top_results = predict_resume_category(resume_text)

            st.session_state["svm_prediction"] = prediction
            st.session_state["top_results"] = top_results
            st.session_state["resume_text"] = resume_text
            st.session_state["resume_name"] = uploaded_file.name

        svm_prediction = st.session_state["svm_prediction"]
        top_results = st.session_state["top_results"]

        st.metric("SVM Predicted Category", svm_prediction)
        st.caption(
            "The SVM model is a closed-set classifier. It always chooses "
            "among its 24 training categories; its top estimate is not "
            "proof that the category is correct."
        )

        for item in top_results:
            st.write(f"**{item['category']} — {item['score']:.2f}%**")
            st.progress(min(max(item["score"] / 100, 0.0), 1.0))

        st.divider()
        st.subheader("✨ Gemini AI Professional Assessment")
        st.write(
            "Gemini independently analyzes the resume and selects a "
            "dataset category only when a reasonable match exists."
        )

        if st.button("Generate AI Resume Assessment", type="primary"):
            try:
                with st.spinner("Gemini is analyzing the resume..."):
                    gemini_result = analyze_with_gemini(
                        resume_text,
                        svm_prediction,
                        top_results
                    )

                status, explanation = compare_classifications(
                    svm_prediction,
                    gemini_result["dataset_category"]
                )

                st.session_state["gemini_result"] = gemini_result
                st.session_state["comparison_status"] = status
                st.session_state["comparison_explanation"] = explanation

            except Exception as gemini_error:
                st.error(f"Gemini analysis failed: {gemini_error}")
                st.info(
                    "Check that GEMINI_API_KEY is correctly added to "
                    "Streamlit Secrets and that the API key has access "
                    "to the selected Gemini model."
                )

        if "gemini_result" in st.session_state:
            st.divider()
            st.subheader("🔎 SVM + Gemini Classification Comparison")

            status = st.session_state["comparison_status"]
            explanation = st.session_state["comparison_explanation"]
            gemini_result = st.session_state["gemini_result"]

            if status == "EXACT AGREEMENT":
                st.success(status)
            elif status == "RELATED CATEGORIES":
                st.info(status)
            else:
                st.warning(status)

            st.write(explanation)

            comparison_col1, comparison_col2 = st.columns(2)
            comparison_col1.metric("SVM Category", svm_prediction)
            comparison_col2.metric(
                "Gemini Dataset Category",
                gemini_result["dataset_category"]
            )

            show_gemini_result(gemini_result)

            st.divider()
            st.subheader("📥 Download Professional PDF Report")

            pdf_buffer = generate_pdf(
                st.session_state["resume_name"],
                svm_prediction,
                top_results,
                gemini_result,
                status,
                explanation
            )

            st.download_button(
                label="Download Resume Analysis PDF",
                data=pdf_buffer,
                file_name="Resume_Analysis_Report.pdf",
                mime="application/pdf"
            )

            st.success("Your PDF report is ready to download!")

    except Exception as error:
        st.error(f"An error occurred: {error}")
        st.exception(error)
