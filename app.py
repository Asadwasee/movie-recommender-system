import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# ---------- Step 1: Load or generate movie_dict.pkl ----------
try:
    movies_dict = pickle.load(open('movie_dict.pkl','rb'))
    movies = pd.DataFrame(movies_dict)
except:
    # CSV se generate kare
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    # Helper functions
    def convert(obj):
        l = []
        for i in ast.literal_eval(obj):
            l.append(i['name'])
        return l

    def convert3(obj):
        l = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                l.append(i['name'])
                counter += 1
            else:
                break
        return l

    def fetch_director(obj):
        l = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                l.append(i['name'])
                break
        return l

    # Apply conversions
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['id','title','tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    # Save pickle
    pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
    movies = pd.DataFrame(new_df.to_dict())

# ---------- Step 2: Load or generate similarity.pkl ----------
try:
    similarity = pickle.load(open('similarity.pkl','rb'))
except:
    cv = CountVectorizer(max_features=5000,stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    pickle.dump(similarity, open('similarity.pkl','wb'))

# ---------- Step 3: Streamlit UI ----------
st.title("Movie Recommender System")

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommend_movies = []
    recommended_movie_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]]['id']
        recommend_movies.append(movies.iloc[i[0]]['title'])
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommend_movies, recommended_movie_posters

selected_movie_name = st.selectbox('Which movie would you like to watch?', movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
