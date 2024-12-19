import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('static/movies.csv')
credits = pd.read_csv('static/credits.csv')

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
newmovies = movies.copy()
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
def get_recommendations(title):
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

# def recommend_by_actor(actor_name):
#     # Filter movies by actor and recommend similar ones
#     movies_by_actor = new_df[new_df['actors'] == actor_name]['title'].tolist()
#     return movies_by_actor
#
# def recommend_by_director(director_name):
#     # Filter movies by director and recommend similar ones
#     movies_by_director = new_df[new_df['director'] == director_name]['title'].tolist()
#     return movies_by_director
#
# def recommend_by_genre(genre_name):
#     # Filter movies by genre and recommend similar ones
#     movies_by_genre = new_df[new_df['genres_str'] == genre_name]['title'].tolist()
#     return movies_by_genre
