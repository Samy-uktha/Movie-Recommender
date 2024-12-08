import pickle
import requests
from flask import Flask, render_template, request

app = Flask(__name__, static_folder='static')

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
# print("movie values", list(movies['title'].values()))
# movie_list = [movies['title'] for movie in movies.values()]
movie_list = list(movies['title'].values())
# print(movie_list)

def recommend(movie):
    # if movie not in movie_list:
    #     return ["Movie not found"]
    # index = movie_list.index(str(movie))
    # print("index is", index )
    # distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    # recommended_movie_names = []
    # for i in distances[1:6]:
    #     # movie_id = movies.iloc[i[0]].movie_id
    #     recommended_movie_names.append(movies[i[0]]['title'])

    movie = movie.strip().lower()  # Strip spaces and convert to lowercase
    movie_titles = [title.strip().lower() for title in movies['title'].values()]
    # if movie not in movie_titles:
    #     return ["Movie not found"]
    index = movie_titles.index(movie)
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )
    # print(distances)
    recommended_movie_names = []
    for i,_ in distances[1:6]:
        if 0 <= i < len(movies):
            movie_data = list(movies.values())[i]
            # movie_data = movies[movie_id]
            recommended_movie_names.append(movie_data['title'])
        else:
            print("no movies to recommend")
    # movie_list = sorted(
    #     list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    # recommended_movie_names = [movies['title'][i[0]] for i in movie_list if i[0] in movies['title']]

    return recommended_movie_names


@app.route("/", methods=["GET", "POST"])
def home():
    # movie_list = movies['title'].tolist()
    recommended_movie_names = []
    selected_movie = ""

    if request.method == "POST":
        selected_movie = request.form.get('movie')
        recommended_movie_names = recommend(selected_movie)

    return render_template('home.html', movie_list=movie_list, recommended_movie_names=recommended_movie_names, selected_movie=selected_movie)


if __name__ == "__main__":
    app.run(debug=True)
