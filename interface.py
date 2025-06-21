import streamlit as st
import PyPDF2
from mains import (
    predict_roles,
    ats_score_10_point_without_jd,
    analyze_job_role_skills,
    analyze_jd_skills,
    role_skills_map
)


st.set_page_config(page_title="Resume Insights Hub", layout="wide")

st.title("📄 Smart Resume Intelligence System")
st.markdown("Upload your resume to get ATS insights, job role prediction, and skill match analysis.")

uploaded_file = st.file_uploader("🔼 Upload Resume (.txt or .pdf)", type=["txt", "pdf"])

resume_text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
    else:
        resume_text = uploaded_file.read().decode("utf-8")


if resume_text:
    st.success("✅ Resume Uploaded Successfully!")


    tabs = st.tabs(["📊 ATS Score", "🤖 Job Role Prediction", "📌 Skill Gap Analysis", "📄 JD Skill Match"])

    with tabs[0]:
        st.subheader("📊 ATS Score (Out of 10)")
        score_details = ats_score_10_point_without_jd(resume_text)
        for key, value in score_details.items():
            st.write(f"**{key}**: {value}")

  
    with tabs[1]:
        st.subheader("🔍 Predicted Job Roles")
        roles = predict_roles(resume_text)
        for role, score in roles:
            st.write(f"**{role.title()}**")


    with tabs[2]:
        st.subheader("📌 Select Desired Job Role")
        job_role = st.selectbox("Choose a Role", list(role_skills_map.keys()))
        if st.button("Analyze Skill Gap"):
            result = analyze_job_role_skills(resume_text, job_role)
            st.write(f"### 🔧 Skill Gap for `{job_role.title()}`")
            st.write(f"**Matched Skills:** {', '.join(result['matched_skills'])}")
            st.write(f"**Missing Skills:** {', '.join(result['missing_skills'])}")
            st.write(f"**Skill Match Score:** {result['score']}%")

            st.subheader("📚 Learning Resources")
            for skill, links in result["resources"].items():
                st.markdown(f"**{skill.title()}**:")
                for platform, url in links.items():
                    st.markdown(f"- [{platform}]({url})")

    with tabs[3]:
        st.subheader("📄 Paste Job Description")
        jd_input = st.text_area("Paste the JD below to compare with your resume:", height=200)

        submit_col = st.columns([6, 1, 6])[1]  
        with submit_col:
            submit = st.button("🔍 Submit")

        if submit and jd_input:
            jd_analysis = analyze_jd_skills(resume_text, jd_input)
            st.write(f"**JD Match Score:** {jd_analysis['score']}%")
            st.write(f"**Matched Skills:** {', '.join(jd_analysis['matched_skills'])}")
            st.write(f"**Missing Skills for JD:** {', '.join(jd_analysis['missing_skills'])}")

            st.subheader("📚 Resources for Missing JD Skills")
            for skill, links in jd_analysis["resources"].items():
                st.markdown(f"**{skill.title()}**:")
                for platform, url in links.items():
                    st.markdown(f"- [{platform}]({url})")
