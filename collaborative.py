import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from content import newmovies as movies
data = pd.read_csv('static/ratings.csv')
# movies = pd.read_csv('static/movies.csv')

ratings = data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

cosine_sim = cosine_similarity(ratings)
similarity = pd.DataFrame(cosine_sim, index=ratings.index, columns=ratings.index)


def user_recommend(user_id):
    try:
        # Get the similarity scores for the specified user
        user_similarities = similarity[user_id]

        # Sort the users by similarity (excluding the user itself)
        similar_users = user_similarities.drop(user_id).sort_values(ascending=False)

        # List of recommended movies (we will recommend movies that the similar users liked)
        recommended_movies = set()

        # Iterate through similar users
        for similar_user in similar_users.index:
            # Get movies rated by the similar user
            movies_rated_by_similar_user = ratings.loc[similar_user][ratings.loc[similar_user] > 0]

            # Filter out movies that the target user has already rated
            already_rated = ratings.loc[user_id][ratings.loc[user_id] > 0].index
            movies_to_recommend = movies_rated_by_similar_user.index.difference(already_rated)

            # Add these movies to the recommended list
            recommended_movies.update(movies_to_recommend)

            # Stop once we've got enough recommendations
            if len(recommended_movies) >= 10:
                break

        movie_ids = list(recommended_movies)[:10]
        new_df = pd.DataFrame(columns=['title', 'director', 'actors', 'genres_str'])

        for recommended_id in movie_ids:
            movie_info = movies[movies['id'] == recommended_id]
            if not movie_info.empty:
                new_df = pd.concat([new_df, movie_info[['title', 'director', 'actors', 'genres_str']]])

        # Reset the index for cleanliness
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    except KeyError as e:
        print(f"User ID {user_id} or movie ID not found: {e}")
        return pd.DataFrame(columns=['title', 'director', 'actors', 'genres_str'])

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame(columns=['title', 'director', 'actors', 'genres_str'])
