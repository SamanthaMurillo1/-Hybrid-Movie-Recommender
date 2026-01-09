import streamlit as st
import pandas as pd
import requests
import pickle
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# This is the Python version of "require('dotenv').config()"
load_dotenv() 

# This is the Python version of "const apiKey = process.env.API_KEY"
TMDB_API_KEY = os.getenv("API_KEY")

# Initialize App
st.set_page_config(page_title="Hybrid Movie AI", layout="wide")

# -------------------------
# Session State Helpers
# -------------------------
def init_watchlist():
    """Initialize the watchlist in session state if it doesn't exist"""
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

init_watchlist()

def add_to_watchlist(title, movie_id):
    """Add a movie to the list if it's not already there"""
    movie_entry = {"title": title, "id": movie_id}
    if movie_entry not in st.session_state.watchlist:
        st.session_state.watchlist.append(movie_entry)
        st.toast(f"Added {title} to your queue!")
    else:
        st.warning(f"{title} is already in your watchlist.")
# -------------------------
# Load BOTH Models
# -------------------------
@st.cache_resource
def load_all_models():
    # 1. Load Original Movie Model (Title-based)
    with open("movie_data.pkl", "rb") as f:
        m1_df, cosine_sim = pickle.load(f)
    
    # 2. Load Smart Model (Semantic/Vibe-based)
    with open("smart_movie_models.pkl", "rb") as f:
        m2_df, embeddings = pickle.load(f)
    
    # 3. Load Transformer Brain
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return m1_df, cosine_sim, m2_df, embeddings, st_model

m1_df, cosine_sim, m2_df, embeddings, st_model = load_all_models()

# -------------------------
# Logic Functions
# -------------------------

def get_title_recommendations(title: str, top_n=10):
    """Logic for exact movie title matches"""
    idx = m1_df[m1_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : 1 + top_n]
    movie_indices = [i[0] for i in sim_scores]
    return m1_df[['title', 'movie_id']].iloc[movie_indices].reset_index(drop=True)

def get_vibe_recommendations(query: str, top_n=10):
    """Logic for 'Smart Search' vibe queries"""
    query_vec = st_model.encode([query])
    similarity = cosine_similarity(query_vec, embeddings).flatten()
    
    res_df = m2_df.copy()
    res_df['similarity'] = similarity
    
    # 1. Kill the 'Noise' (Threshold)
    res_df = res_df[res_df['similarity'] > 0.30] 
    
    # 2. Add Tone Bias
    if "animated" in query.lower() or "kids" in query.lower():
        # Animation usually has high tone_score (happy/positive)
        res_df['similarity'] *= (1 + res_df['tone_score'])
        
    # 3. Final Calculation (Favor Text Match heavily)
    res_df['pop_boost'] = res_df['popularity'] / res_df['popularity'].max()
    res_df['final_score'] = (res_df['similarity'] * 0.9) + (res_df['pop_boost'] * 0.1)
    
    return res_df.sort_values(by='final_score', ascending=False).head(top_n)
    
def fetch_poster(movie_id: int):
    # Use the variable loaded from your .env file
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/500x750?text=Missing+API+Key"
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        return f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"


# -------------------------
# UI Implementation
# -------------------------
# 1. Initialize session state keys
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# SIDEBAR: Watchlist Display
with st.sidebar:
    st.header("⏳ Watchlist")
    if st.session_state.watchlist:
        for i, item in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {item['title']}")
            with col2:
                # Use a unique key for the remove button
                if st.button("❌", key=f"remove_sidebar_{item['id']}"):
                    st.session_state.watchlist.pop(i)
                    st.rerun()
        
        if st.button("Clear All"):
            st.session_state.watchlist = []
            st.rerun()
    else:
        st.info("Your list is empty.")



#MAIN AREA
st.title("🎬 Hybrid Movie Recommender")

user_input = st.text_input("Enter a movie title OR describe a vibe:", 
                          placeholder="e.g. 'Batman' OR 'happy animated movies'")

# When Search is clicked, we update the session_state
if st.button("Search"):
    if user_input:
        # Create a lowercase version of the input for comparison
        user_input_lower = user_input.strip().lower()
        
        # Create a lowercase mapping of titles to their original versions
        # This ensures 'avatar' matches 'Avatar'
        title_map = {t.lower(): t for t in m1_df['title'].values}
        if user_input_lower in title_map:
            # Get the correctly capitalized title for the recommendation function
            original_title = title_map[user_input_lower]
            
            st.session_state.recommendations = get_title_recommendations(original_title)
            st.success(f"Found exact match for '{original_title}'. Using Title-Based Recommender.")
        else:
            # If no match, treat it as a description/vibe
            st.session_state.recommendations = get_vibe_recommendations(user_input)
            st.info(f"No exact title match. Using Smart AI to find: '{user_input}'")
    else:
        st.warning("Please enter a title or a description.")
  


    # -------------------------
    # Display Results Grid
    # -------------------------
if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    
    if not recommendations.empty:
        st.divider()
        cols = st.columns(5)
        for i, row in enumerate(recommendations.itertuples()):
            with cols[i % 5]:
                poster = fetch_poster(row.movie_id)
                st.image(poster)
                st.markdown(f"**[{row.title}](https://www.themoviedb.org/movie/{row.movie_id})**")
                
                # Logic for adding to watchlist
                if st.button("➕ Queue", key=f"btn_{row.movie_id}"):
                    add_to_watchlist(row.title, row.movie_id)
                    # trigger rerun so the sidebar updates immediately 
                    # while the session_state.recommendations keeps the movies on screen
                    st.rerun()

    