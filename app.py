from flask import Flask, render_template, request
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
movies = pd.read_csv('static/movies.csv')
credits = pd.read_csv('static/credits.csv')

# Preprocessing functions
def convert(obj):
    if isinstance(obj, str):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except (ValueError, SyntaxError):
            return []
    return obj


def convert3(obj):
    if isinstance(obj, str):
        try:
            return [i['name'] for i in ast.literal_eval(obj)[:3]]
        except (ValueError, SyntaxError):
            return []
    return obj[:3] if isinstance(obj, list) else []


def fetch_director(obj):
    if isinstance(obj, str):
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return [i['name']]
        except (ValueError, SyntaxError):
            return []
    return []

# print(movies.columns)
# print(credits.columns)

# Step 2: Merge the DataFrames using the correct column
movies = movies.merge(credits, on='title')  # Update 'movie_id' or use the correct column name

# Step 3: After merging, print columns to identify the correct ones
# print(movies.columns)

# Step 4: Preprocess and clean the data
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
# movies['overview'] = movies['overview'].apply(lambda x: x.split())
# Apply a safe split to the 'overview' column
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

movies['director'] = movies['crew']
movies['actors'] = movies['cast']
movies['genres_str'] = movies['genres']

# Remove spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create 'tags' and new DataFrame
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags', 'director', 'actors', 'genres_str']]  # Make sure 'title' exists
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# print(movies['director'].values)
# print(movies['actors'].values)
# print(movies['genres_str'].values)
# print(movies['actors'])
# print(new_df.columns)


# Count vectorizer and similarity matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(new_df['tags'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# Recommendation function
def get_recommendations(title, cosine_sim = cosine_sim):
    indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = new_df.iloc[movie_indices]
    print(recommended_movies)

    return recommended_movies[['title', 'director', 'actors', 'genres_str']].to_dict(orient='records')


@app.route('/', methods=['GET', 'POST'])
def home():
    selected_movie = None
    recommended_movies = []

    if request.method == 'POST':
        selected_movie = request.form['movie']
        recommended_movies = get_recommendations(selected_movie)
        print(recommended_movies)


    movie_list = sorted(new_df['title'].unique().tolist())
    return render_template(
        'home.html',
        movie_list=movie_list,
        selected_movie=selected_movie,
        recommended_movies=recommended_movies
    )


if __name__ == '__main__':
    app.run(debug=True)
