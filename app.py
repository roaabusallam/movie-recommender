import streamlit as st
import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Movie Recommender")

st.title("Movie Recommendation System")

st.write("Choose a movie to get similar recommendations.")


@st.cache_data

def load_movie_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")

    credits = pd.read_csv("tmdb_5000_credits_small.csv")

    df = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id"
    )

    df = df[[
        "title_x",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]]

    df = df.rename(
        columns={"title_x": "title"}
    )

    df = df.dropna()

    df = df.drop_duplicates(
        subset="title" #حذف المتكرر بالنسبة للاسماء
    )

    df = df.reset_index(drop=True) #يعيد ترتيب ارقام الصفوف

    return df

#دالج تستقبل الكلمات المفتاحية
def get_names(text):

    data = ast.literal_eval(text) #تحول النص إلى قائمة بايثون

    names = []

    for item in data:
        names.append(item["name"])

    return " ".join(names)

#دالة تستقبل أسماء الممثلين
def get_cast(text):

    data = ast.literal_eval(text)

    names = []

    for item in data[:3]:#أول ثلاثة ممثلين فقط
        names.append(item["name"])

    return " ".join(names)
#تجميع البيانات من قائمة الى نص كامل

def get_director(text):

    data = ast.literal_eval(text)

    for item in data:
        if item["job"] == "Director":
            return item["name"]

    return ""


df = load_movie_data()

df["genres"] = df["genres"].apply(get_names)

df["keywords"] = df["keywords"].apply(get_names)

df["cast"] = df["cast"].apply(get_cast)

df["crew"] = df["crew"].apply(get_director)


df["movie_info"] = (
    df["overview"]
    + " "
    + df["genres"]
    + " "
    + df["keywords"]
    + " "
    + df["cast"]
    + " "
    + df["crew"]
)

vectorizer = TfidfVectorizer(
    stop_words="english" #تجاهل الكلمات الإنجليزية الشائعة جدًا
)

movie_vectors = vectorizer.fit_transform(
    df["movie_info"]
)

#يحسب درجة تشابه كل فيلم مع كل الأفلام الأخرى
similarity = cosine_similarity(
    movie_vectors
)


def recommend_movies(movie_title):

    movie_index = df[
        df["title"] == movie_title
    ].index[0] #تبحث عن الفيلم في الجدول وتأخذ رقم صفه

    scores = similarity[movie_index]

    movies_with_scores = list(
        enumerate(scores)
    )#تربط كل درجة برقم الفيلم الخاص بها

    movies_with_scores = sorted(
        movies_with_scores,
        key=lambda item: item[1], # درجة التشابه
        reverse=True #ترتيب تنازلي، من الأكبر للأصغر
    )
    #ترتب الأفلام من الأعلى تشابهًا إلى الأقل
    recommendations = []

    for index, score in movies_with_scores[1:6]:#نأخذ من الفيلم الثاني إلى السادس تجاوزنا أول فيلم لأن أول نتيجة ستكون الفيلم نفسه، وليس اقتراحًا جديدًا

        movie_name = df.iloc[index]["title"]

        recommendations.append(movie_name)

    return recommendations

#تأخذ جميع أسماء الأفلام وترتبها أبجديًا
movie_names = sorted(
    df["title"].values
)
#قائمة منسدلة
selected_movie = st.selectbox(
    "Select a movie",
    movie_names
)

#الزر
if st.button("Recommend"):

    recommended_movies = recommend_movies(
        selected_movie
    )

    st.subheader("Recommended Movies")

    for movie in recommended_movies:
        st.write("- " + movie)
