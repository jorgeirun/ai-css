#  AI Candidare Score System

## Project Overview

This project implements a FastAPI backend to process job descriptions and rank candidates based on their predicted scores. It includes a simple HTML/JavaScript frontend to upload job description files and view the top-ranked candidates.

## Libraries needed

Requisites
```
Python 3.7+
pip install fastapi uvicorn pandas numpy scikit-learn nltk PyPDF2 joblib
```

Download NLTK Stopwords
```
python -m nltk.downloader stopwords
```

## Run project

#### Start the FastAPI Backend

	1.	Ensure the required files are in the project directory:
	•	main.py (backend script).
	•	candidate_scoring_regressor.pkl (pre-trained model).
	•	tfidf_vectorizer.pkl (TF-IDF vectorizer).
	•	ZIPDEV - Candidate Database Code challenge (1).xlsx (candidate data).

	2.	Run the backend server:
    uvicorn main:app --reload


#### Frontend
Open the ```index.hml``` file in an browser.


