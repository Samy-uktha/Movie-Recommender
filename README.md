Movie Recommender Website

Developed a Flask-based movie recommender system that uses both content based filtering and collaborative filtering and provides personalized movie recommendations through multiple search functionalities, including natural language queries, direct movie selection, and filters like genre, actor, and director.
Key featues:

1) Content based filtering -
  Implemented filtering based on movie metadata such as genres, cast, and directors using Pandas. Applied Scikit-learn CountVectorizer to convert tags into numerical representations and calculated cosine similarity scores.

2) Collaborative filtering -
   Integrated collaborative filtering to suggest movies based on user behavior by tracking user ratings and using similarity metrics such as Cosine Similarity

3) Filter-based search -
   Enabled conditional filtering by specific movie features like actors, directors to narrow search results based on user selection.

4) Natural language query support -
   Enabled query parsing to extract relevant filters from user input, enhancing search accuracy through keyword matching.
