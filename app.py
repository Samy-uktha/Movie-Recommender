from flask import Flask, render_template, request
# import pandas as pd
# import ast
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from content import get_recommendations, new_df, movies
from query import recommend_movies
from collaborative import user_recommend, similarity

app = Flask(__name__)

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
        # user_id = request.form.get('userid')
        search_type = request.form.get('search_type')
        # print("hello user",user_id)

        if search_type == 'by_query':
            user_query = request.form.get('query','').strip()
            if user_query:
                query_result = recommend_movies(user_query, movies)
                recommended_movies = query_result[['title', 'director', 'actors', 'genres_str']].to_dict(orient='records')
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
                recommended_movies = filtered_movies[['title', 'director', 'actors', 'genres_str']].to_dict(orient='records')
            else:
                recommended_movies = []

        elif search_type == 'by_user':
            user_id = int(request.form.get('userid'))
            print("got user id",user_id)
            if user_id in similarity:
                usermovies = user_recommend(user_id)
                recommended_movies = usermovies[['title', 'director', 'actors', 'genres_str']].to_dict(orient='records')
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
