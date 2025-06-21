
import re
import spacy
import numpy as np
import joblib


nlp = spacy.load("en_core_web_sm")

model = joblib.load("job_role_predictor.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")


def preprocess(text):
    text = text.lower()
    doc = nlp(text)
    clean_words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(clean_words)

def predict_roles(text,n=5):
    cleaned = preprocess(text)
    X_input = tfidf.transform([cleaned]).toarray()
    probs = model.predict_proba(X_input)[0]
    
    results = []
    for i, prob in enumerate(probs):
        if prob > 0.25:
            role = le.inverse_transform([i])[0]
            results.append((role, prob))
    
    return results

role_skills_map = {
    "data scientist": {"python", "sql", "tableau", "machine learning", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"},
    "data engineer": {"python", "sql", "etl", "hadoop", "spark", "aws", "azure", "airflow", "bigquery", "data warehousing"},
    "software engineer": {"python", "java", "c++", "data structures", "algorithms", "git", "sql", "api", "docker"},
    "data analyst": {"excel", "sql", "tableau", "powerbi", "data visualization", "python", "statistics", "matplotlib", "seaborn"},
    "ml engineer": {"python", "tensorflow", "keras", "scikit-learn", "numpy", "pandas", "mlops", "docker", "git", "opencv"}
}


master_skills = set().union(*role_skills_map.values())

def extract_skills(text, skills_vocab):
    doc = nlp(text.lower())
    tokens = set(token.text for token in doc if token.is_alpha)
    return tokens & skills_vocab

def generate_course_links(skill):
    query = skill.replace(" ", "+")
    return {
        "Coursera": f"https://www.coursera.org/search?query={query}",
        "Udemy": f"https://www.udemy.com/courses/search/?q={query}",
        "YouTube": f"https://www.youtube.com/results?search_query={query}+tutorial"
    }

def get_learning_resources_for_missing_skills(missing_skills):
    return {skill: generate_course_links(skill) for skill in missing_skills}

def analyze_job_role_skills(resume_text, job_role):
    required_skills = role_skills_map.get(job_role.lower(), set())
    resume_skills = extract_skills(resume_text, master_skills)
    missing_skills = required_skills - resume_skills
    matched_skills = required_skills & resume_skills
    score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
    resources = get_learning_resources_for_missing_skills(missing_skills)
    return {
        "required_skills": list(required_skills),
        "resume_skills": list(resume_skills),
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills),
        "score": round(score, 2),
        "resources": resources
    }

def analyze_jd_skills(resume_text, jd_text):
    resume_skills = extract_skills(resume_text, master_skills)
    jd_skills = extract_skills(jd_text, master_skills)
    missing_skills = jd_skills - resume_skills
    matched_skills = jd_skills & resume_skills
    score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
    resources = get_learning_resources_for_missing_skills(missing_skills)
    return {
        "jd_skills": list(jd_skills),
        "resume_skills": list(resume_skills),
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills),
        "score": round(score, 2),
        "resources": resources
    }

def ats_score_10_point_without_jd(resume_text):
    score = 0
    details = {}

    lower_resume = resume_text.lower()
    has_email = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
    has_phone = re.search(r'\b\d{10}\b|\+\d{2,3}[-\s]?\d{10}', resume_text)

    if not re.search(r'<table|<img', resume_text):
        score += 1
        details['Simple Formatting'] = 1
    else:
        details['Simple Formatting'] = 0

    if has_email or has_phone:
        score += 1
        details['Contact Info'] = 1
    else:
        details['Contact Info'] = 0

    predicted_role = predict_roles(resume_text, n=1)[0][0]
    expected_skills = role_skills_map.get(predicted_role.lower(), set())
    resume_skills = extract_skills(resume_text, master_skills)
    matched_keywords = expected_skills & resume_skills
    if matched_keywords:
        score += 2
        details['Job-Relevant Keywords'] = 2
    else:
        details['Job-Relevant Keywords'] = 0

    if 'skills' in lower_resume and resume_skills:
        score += 2
        details['Technical Skills Listed'] = 2
    else:
        details['Technical Skills Listed'] = 0

    if 'education' in lower_resume:
        score += 1
        details['Education Section'] = 1
    else:
        details['Education Section'] = 0

    if 'project' in lower_resume or 'experience' in lower_resume:
        score += 2
        details['Projects/Experience'] = 2
    else:
        details['Projects/Experience'] = 0

    standard_sections = ['summary', 'skills', 'education', 'projects', 'experience']
    found_sections = [sec for sec in standard_sections if sec in lower_resume]
    if len(found_sections) >= 3:
        score += 1
        details['Standard Sections Used'] = 1
    else:
        details['Standard Sections Used'] = 0

    
    details['Total ATS Score (out of 10)'] = score
    return details
