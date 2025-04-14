import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz
from autocorrect import Speller

# Load data and pre-process
books_df = pd.read_csv('books_dataset_unique.csv')
songs_df = pd.read_csv('songs_dataset_unique.csv')
merged_df = pd.merge(books_df, songs_df, left_on='Book_ID', right_on='Song_ID')
df = pd.DataFrame(merged_df)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Initialize autocorrect spell checker
spell = Speller()

# Recommendation function
def get_song_recommendations(book_title, n=3, cosine_sim=cosine_sim):
    book_title = book_title.lower()
    book_title_corrected = spell(book_title)
    try:
        idx = df.index[df['Title'].str.lower() == book_title_corrected].tolist()[0]
    except IndexError:
        return None
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    song_indices = [i[0] for i in sim_scores]
    return df['Song Name'].iloc[song_indices].tolist()

# Streamlit UI
st.title("Book to Song Recommender")

# Search for book title with suggestions
book_title = st.text_input("Enter a book title", placeholder="Type to search books...")
books = books_df['Title'].tolist()

# Display suggestions
if book_title:
    suggestions = [book for book in books if book_title.lower() in book.lower()][:5]
    if suggestions:
        st.write("Suggestions:")
        for suggestion in suggestions:
            if st.button(suggestion, key=suggestion):
                book_title = suggestion
                st.session_state['book_title'] = suggestion  # Store selected suggestion

# Retrieve book_title from session state if set
if 'book_title' in st.session_state:
    book_title = st.session_state['book_title']

# Number of recommendations
num_recommendations = st.slider("Number of song recommendations", 1, 10, 3)

# Get recommendations
if st.button("Get Recommendations"):
    if book_title:
        recommendations = get_song_recommendations(book_title, num_recommendations)
        if recommendations is None:
            st.error(f"No songs found for '{book_title}'. Try another title.")
        else:
            st.success(f"Song recommendations for '{book_title}':")
            for i, song in enumerate(recommendations, 1):
                st.write(f"{i}. {song}")
    else:
        st.warning("Please enter a book title.")
