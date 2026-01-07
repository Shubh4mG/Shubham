import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess genres
movies['combined_features'] = movies['genres'].str.replace('|', ' ')

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Calculate average rating and count of ratings per movie
movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']

# Merge with movies
movies = pd.merge(movies, movie_stats, on='movieId', how='left')
movies.fillna({'avg_rating': 0, 'rating_count': 0}, inplace=True)

# Recommend movies based on multiple selected titles
# Inside your recommend_movies() function, we'll return similarity too
def recommend_movies(selected_titles, movies, cosine_sim, n_recommendations=5):
    indices = [movies[movies['title'] == title].index[0] for title in selected_titles if title in movies['title'].values]
    sim_scores = sum(cosine_sim[idx] for idx in indices) / len(indices)

    sim_scores = list(enumerate(sim_scores))
    selected_indices = [movies[movies['title'] == title].index[0] for title in selected_titles]
    sim_scores = [s for s in sim_scores if s[0] not in selected_indices]

    # Sort by combined score
    sim_scores = sorted(sim_scores, key=lambda x: (
        x[1], 
        movies.iloc[x[0]]['rating_count'], 
        movies.iloc[x[0]]['avg_rating']
    ), reverse=True)

    recommendations = []
    for i, score in sim_scores[:n_recommendations]:
        row = movies.iloc[i]
        explanation = f"Similar to: {', '.join(selected_titles)} | Genre match score: {score:.2f} | " \
                      f"⭐ Avg Rating: {row['avg_rating']:.2f} | 👥 {int(row['rating_count'])} ratings"
        recommendations.append({
            "title": row['title'],
            "avg_rating": row['avg_rating'],
            "rating_count": int(row['rating_count']),
            "explanation": explanation
        })
    
    return recommendations


# Streamlit UI
def create_ui():
    st.title("🍿 Smart Movie Recommender")
    st.markdown("Select a few movies you like. We'll suggest popular, highly rated similar ones!")

    movie_options = sorted(movies['title'].unique())
    selected_movies = st.multiselect("Pick 1 to 3 movies you like:", movie_options, default=["Toy Story (1995)"])

    if st.button("Recommend"):
        if not selected_movies:
            st.warning("Please select at least one movie.")
        else:
            recommendations = recommend_movies(selected_movies, movies, cosine_sim)

            st.subheader("🎥 Recommended for You:")
            for rec in recommendations:
                st.markdown(f"**{rec['title']}**")
                st.markdown(f"`{rec['explanation']}`")
                st.markdown("---")

if __name__ == '__main__':
    create_ui()
