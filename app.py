from flask import Flask, render_template, request
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from query import recommend_movies

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
            return [i['name'] for i in ast.literal_eval(obj)[:5]]
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
# print(movies.columns)
# print(movies['director'].values)
# print(movies['actors'].values)
# print(movies['genres_str'].values)
# print(movies['actors'])
# print(new_df.columns)
# print(movies['tags'][0])


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
    # print(recommended_movies)
    return recommended_movies[['title', 'director', 'actors', 'genres_str']].to_dict(orient='records')

def recommend_by_actor(actor_name):
    # Filter movies by actor and recommend similar ones
    movies_by_actor = new_df[new_df['actors'] == actor_name]['title'].tolist()
    return movies_by_actor

def recommend_by_director(director_name):
    # Filter movies by director and recommend similar ones
    movies_by_director = new_df[new_df['director'] == director_name]['title'].tolist()
    return movies_by_director

def recommend_by_genre(genre_name):
    # Filter movies by genre and recommend similar ones
    movies_by_genre = new_df[new_df['genres_str'] == genre_name]['title'].tolist()
    return movies_by_genre

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_movie = None
    selected_actor = None
    selected_director = None
    selected_genre = None
    recommended_movies = []
    no_results_message = None

    actors_list = sorted(set([actor for sublist in new_df['actors'].dropna() for actor in
                              (sublist if isinstance(sublist, list) else [sublist])]))
    directors_list = sorted(set([director for sublist in new_df['director'].dropna() for director in
                                 (sublist if isinstance(sublist, list) else [sublist])]))
    genres_list = sorted(set([genre for sublist in new_df['genres_str'].dropna() for genre in
                              (sublist if isinstance(sublist, list) else [sublist])]))

    if request.method == 'POST':
        search_type = request.form.get('search_type')

        if search_type == 'by_query':
            user_query = request.form.get('query','').strip()
            if user_query:
                query_result = recommend_movies(user_query, movies)
                recommended_movies = query_result[['title', 'director', 'actors', 'genres_str']].to_dict(
                    orient='records')
            else:
                recommended_movies = []

        elif search_type == 'by_movie':
            selected_movie = request.form.get('movie', None)
            if selected_movie:
                filtered_movies = get_recommendations(selected_movie)
                recommended_movies = filtered_movies

        elif search_type == 'by_filters':
            selected_actor = request.form.get('actor', None)
            selected_director = request.form.get('director', None)
            selected_genre = request.form.get('genre', None)

            filtered_movies = new_df

            # filtered_movies = [movie for movie in filtered_movies if movie['title'] == selected_movie]
            if selected_actor:
                filtered_movies = filtered_movies[filtered_movies['actors'].apply(lambda actors: selected_actor in actors)]
            if selected_director:
                filtered_movies = filtered_movies[filtered_movies['director'].apply(lambda directors: selected_director in directors)]
            if selected_genre:
                filtered_movies = filtered_movies[filtered_movies['genres_str'].apply(lambda genres: selected_genre in genres)]

            if not filtered_movies.empty:
                recommended_movies = filtered_movies[['title', 'director', 'actors', 'genres_str']].to_dict(
                orient='records')
            else:
                recommended_movies = []
        # recommended_movies = filtered_movies['title'].tolist()
        # recommended_movies = filtered_movies
        # recommended_movies = get_recommendations(selected_movie)
        print(recommended_movies)

        if not recommended_movies:
            no_results_message = "No movies found matching your criteria."


    movie_list = sorted(new_df['title'].unique().tolist())


    return render_template(
        'home.html',
        movie_list=movie_list,
        actors_list=actors_list,
        directors_list=directors_list,
        genres_list=genres_list,
        selected_movie=selected_movie,
        selected_actor=selected_actor,
        selected_director=selected_director,
        selected_genre=selected_genre,
        recommended_movies=recommended_movies,
        no_results_message=no_results_message,
    )


if __name__ == '__main__':
    app.run(debug=True)
