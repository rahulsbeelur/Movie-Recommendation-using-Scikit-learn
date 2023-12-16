from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))

csv_file_path = os.path.join(current_directory, 'movie_dataset.csv')

df = pd.read_csv(csv_file_path)

features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
    df[feature] = df[feature].fillna(' ')

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        print("Error")

df["combined_features"] = df.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_user_likes = request.form['movie_name']
    # Get the selected number of recommendations
    num_recommendations = int(request.form['num_recommendations'])

    movie_index = get_index_from_title(movie_user_likes)

    if movie_index is not None:
        similar_movies = list(enumerate(cosine_sim[movie_index]))

        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        recommended_movies = [(get_title_from_index(movie[0]), round(movie[1] * 100, 2)) for movie in sorted_similar_movies[:num_recommendations]]

        recommendations_html = ""
        for movie in recommended_movies:
            recommendations_html += f"<tr><td>{movie[0]}</td><td>{movie[1]}%</td></tr>"

        return recommendations_html
    else:
        return "<p>No similar movies in the dataset.</p>"

if __name__ == '__main__':
    app.run(debug=True)
