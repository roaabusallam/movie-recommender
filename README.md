# Movie Recommendation System

A simple movie recommendation application built with Python.

The application suggests similar movies based on:

- Movie overview
- Genres
- Keywords
- Cast
- Director

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## How It Works

1. Load movie and credits data
2. Merge both files into one table
3. Clean missing and duplicate values
4. Combine movie information into one text column
5. Convert text to numbers using TF-IDF
6. Calculate similarity between movies
7. Show five similar movie recommendations

## Run the Project

Install the libraries:

pip install -r requirements.txt

Run the application:

streamlit run app.py