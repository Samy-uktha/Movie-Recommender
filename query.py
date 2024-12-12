import nltk
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_tags(tags):
    filtered_words = [re.sub(r'\W+', '', word.lower()) for word in tags if word.lower() not in stop_words]
    return [word for word in filtered_words if word]

def recommend_movies(query, movies):
    # temp_df = movies.copy()
    movies['tags2'] = movies['tags']
    movies['tags2'] = movies['tags2'].apply(lambda x: ' '.join(x))

    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    vector_matrix = vectorizer.fit_transform(movies['tags2'])
    cosine_sim = cosine_similarity(vector_matrix)
    query = ' '.join(query.lower().split())
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vector_matrix)
    sorted_indices = scores[0].argsort()[::-1]
    recommendations = [(movies.iloc[i]['title'],movies.iloc[i]['director'],movies.iloc[i]['actors'],movies.iloc[i]['genres_str'])
                       for i in sorted_indices if scores[0][i] > 0]
    recommendations = recommendations[:10]
    new_df = pd.DataFrame()
    new_df['title'] = [recommendations[i][0] for i in range(len(recommendations))]
    new_df['director'] = [recommendations[i][1] for i in range(len(recommendations))]
    new_df['actors'] = [recommendations[i][2] for i in range(len(recommendations))]
    new_df['genres_str'] = [recommendations[i][3] for i in range(len(recommendations))]
    return new_df

