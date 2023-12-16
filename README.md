This Movie Recommender is a Flask web application that suggests movies based on user input. It utilizes natural language processing and cosine similarity to recommend movies with similar features. The backend, built with Python, employs pandas for data manipulation and scikit-learn for text vectorization.

## Features

- **Cosine Similarity:** Recommends movies by calculating the cosine similarity between their features.
- **Responsive Design:** Ensures optimal user experience on various devices.
- **Web Scraping:** Fetches movie data dynamically from the dataset.

## Libraries Used

- **Flask:** Web framework for Python.
- **pandas:** Data manipulation and analysis library.
- **scikit-learn:** Machine learning library for cosine similarity calculation.

## Usage

1. Clone the repository.

   ```bash
   git clone https://github.com/rahulsbeelur/Movie-Recommendation-using-Scikit-learn.git
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app

   ```bash
   python app.py
   ```

## Proof of Concept

This project demonstrates the application of cosine similarity in recommending movies based on user input. The web interface is user-friendly, allowing users to receive movie recommendations effortlessly.
