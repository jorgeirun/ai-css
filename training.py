# pip install fastapi uvicorn pandas numpy scikit-learn nltk PyPDF2 joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from nltk.corpus import stopwords
from nltk import download
from PyPDF2 import PdfReader
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

# Step 1: Data Preparation
original_file = "ZIPDEV - Candidate Database Code challenge (1).xlsx"
data = pd.read_excel(original_file)

# Combine relevant columns into 'resume_combined'
data['resume_combined'] = data[['Summary', 'Experiences', 'Skills']].fillna('').apply(' '.join, axis=1)

# Clean the 'resume_combined' column
data['cleaned_resume'] = data['resume_combined'].apply(clean_text)

# Step 2: Feature Engineering
tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_resume'])

# Convert TF-IDF to DataFrame
X = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Add binary feature for skills
X['skills_binary'] = data['Skills'].apply(lambda x: 1 if isinstance(x, str) and len(x) > 0 else 0)

# Step 3: Process Job Description for Training
pdf_path = "JD - Engineering Role.pdf"
pdf_reader = PdfReader(pdf_path)
job_description = "".join(page.extract_text() for page in pdf_reader.pages)
cleaned_description = clean_text(job_description)

# Extract dynamic keywords
job_keywords = extract_keywords(job_description)

# Compute similarity scores
similarity_scores = cosine_similarity(tfidf_vectorizer.transform([cleaned_description]), tfidf_features).flatten()

# Add dynamic features to the dataset
X['similarity_score'] = similarity_scores
data['rule_based_score'] = data['Skills'].apply(lambda skills: calculate_rule_based_score(skills, job_keywords))
X['rule_based_score'] = data['rule_based_score']

# Generate target variable
data['score'] = np.random.randint(50, 101, size=len(data))  # Random scores for regression

# Step 4: Model Training
y_regression = data['score']  # Target for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Save models and vectorizer
joblib.dump(regressor, "candidate_scoring_regressor.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Evaluate Regression Model
y_pred_reg = regressor.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
# print("\nModel RMSE:", rmse)

# Step 5: Model Application
pdf_path = "JD - Engineering Role.pdf"
# pdf_path = "JD - GoLang Dev.pdf"
pdf_reader = PdfReader(pdf_path)
job_description = "".join(page.extract_text() for page in pdf_reader.pages)
cleaned_description = clean_text(job_description)

# Recompute similarity scores for the new job description
similarity_scores = cosine_similarity(tfidf_vectorizer.transform([cleaned_description]), tfidf_features).flatten()

# Update features with similarity scores and dynamic rule-based scores
X['similarity_score'] = similarity_scores
data['rule_based_score'] = data['Skills'].apply(lambda skills: calculate_rule_based_score(skills, extract_keywords(job_description)))
X['rule_based_score'] = data['rule_based_score']

# Predict scores for candidates
data['predicted_score'] = regressor.predict(X)

# Rank candidates by predicted score
ranked_candidates = data.sort_values(by='predicted_score', ascending=False)

# Display the top candidates
print("\nTop Candidates by Regression:")
print(ranked_candidates[['Name', 'Job title', 'predicted_score']].head(30))