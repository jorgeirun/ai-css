from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from nltk.corpus import stopwords
from nltk import download
from PyPDF2 import PdfReader
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download stopwords if not already done
download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.lower()  # Convert to lowercase
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return text

# Define a function for rule-based scoring
def calculate_rule_based_score(skills, job_keywords):
    skills = skills.lower() if isinstance(skills, str) else ''
    matches = sum(1 for keyword in job_keywords if keyword in skills)
    return matches * 10  # Each match is worth 10 points

# Extract dynamic keywords from job description
def extract_keywords(description, top_n=10):
    words = clean_text(description).split()
    word_counts = Counter(words)
    common_keywords = [word for word, _ in word_counts.most_common(top_n) if word not in ENGLISH_STOP_WORDS]
    return common_keywords

# Load pre-trained model and vectorizer
try:
    regressor = joblib.load("candidate_scoring_regressor.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    data = pd.read_excel("ZIPDEV - Candidate Database Code challenge (1).xlsx")

    # Prepare initial candidate data
    data['resume_combined'] = data[['Summary', 'Experiences', 'Skills']].fillna('').apply(' '.join, axis=1)
    data['cleaned_resume'] = data['resume_combined'].apply(clean_text)
except Exception as e:
    raise Exception(f"Failed to load models or data: {str(e)}")

# POST endpoint to process the uploaded PDF
@app.post("/process-job-description/")
async def process_job_description(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Step 1: Process Job Description
        pdf_reader = PdfReader(file.file)
        job_description = "".join(page.extract_text() for page in pdf_reader.pages)
        cleaned_description = clean_text(job_description)

        # Extract dynamic keywords
        job_keywords = extract_keywords(job_description)

        # Compute similarity scores
        tfidf_features = tfidf_vectorizer.transform(data['cleaned_resume'])
        similarity_scores = cosine_similarity(tfidf_vectorizer.transform([cleaned_description]), tfidf_features).flatten()

        # Update features with similarity scores and dynamic rule-based scores
        X = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        X['skills_binary'] = data['Skills'].apply(lambda x: 1 if isinstance(x, str) and len(x) > 0 else 0)
        X['similarity_score'] = similarity_scores
        data['rule_based_score'] = data['Skills'].apply(lambda skills: calculate_rule_based_score(skills, job_keywords))
        X['rule_based_score'] = data['rule_based_score']

        # Predict scores for candidates
        data['predicted_score'] = regressor.predict(X)

        # Rank candidates by predicted score
        ranked_candidates = data.sort_values(by='predicted_score', ascending=False)

        # Select the top candidates
        top_candidates = ranked_candidates[['Name', 'Job title', 'predicted_score']].head(30).to_dict(orient="records")

        return JSONResponse(content={"top_candidates": top_candidates})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")