from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Step 1: Reading CSV File
df = pd.read_csv("movie_dataset.csv")

# Step 2: Selecting features
features = ['keywords', 'cast', 'genres', 'director']

# Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna(' ')

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        print("Error")

df["combined_features"] = df.apply(combine_features, axis=1)

# Create count matrix and compute cosine similarity outside the route
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# Functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def preprocess_title(title):
    return title.replace(" ", "").lower()

def get_index_from_title(title):
    title = preprocess_title(title)
    matching_indices = df[df['title'].str.replace(" ", "").str.lower().str.contains(title)].index
    if not matching_indices.empty and len(matching_indices) > 0:
        return matching_indices[0]
    return None

# Your existing routes and Flask application code...

@app.route('/')
def index():
    return render_template('index.html')

# app.py

# ... (your existing Flask app code)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_user_likes = request.form['movie_name']

    # Step 6: Getting index of this movie from its title
    movie_index = get_index_from_title(movie_user_likes)

    if movie_index is not None:
        similar_movies = list(enumerate(cosine_sim[movie_index]))

        # Step 7: Getting a list of similar movies in descending order of similarity score
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        # Step 8: Extracting titles and normalized similarity scores of first 10 movies
        recommended_movies = [(get_title_from_index(movie[0]), round(movie[1] * 100, 2)) for movie in sorted_similar_movies[:10]]

        # Format recommendations as HTML table rows
        recommendations_html = ""
        for movie in recommended_movies:
            recommendations_html += f"<tr><td>{movie[0]}</td><td>{movie[1]}%</td></tr>"

        return recommendations_html
    else:
        return "<p>No similar movies in the dataset.</p>"


if __name__ == '__main__':
    app.run(debug=True)
