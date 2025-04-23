# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and dataset
knn = joblib.load('knn_model.pkl')
data_url = "https://github.com/ArinB/MSBA-CA-Data/raw/main/CA05/movies_recommendation_data.csv"
df = pd.read_csv(data_url)

# Remove label column if present
if 'Label' in df.columns:
    df.drop(columns=['Label'], inplace=True)

st.title("Movie Recommendation Engine")

# --- INPUT FORM ---
st.subheader("Enter Movie Features")
with st.form("movie_form"):
    imdb_rating = st.number_input("IMDB Rating", min_value=0.0, max_value=10.0, value=7.0)
    biography = st.checkbox("Biography")
    drama = st.checkbox("Drama")
    thriller = st.checkbox("Thriller")
    comedy = st.checkbox("Comedy")
    crime = st.checkbox("Crime")
    mystery = st.checkbox("Mystery")
    history = st.checkbox("History")
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    # Prepare input vector for model
    input_vector = np.array([
        imdb_rating,
        int(biography),
        int(drama),
        int(thriller),
        int(comedy),
        int(crime),
        int(mystery),
        int(history)
    ]).reshape(1, -1)

    # Get nearest neighbors
    distances, indices = knn.kneighbors(input_vector)

    # Display recommended movies
    st.subheader("Recommended Movies")
    recommended_titles = df.iloc[indices[0], 1].values  # Movie Titles
    st.write(pd.DataFrame({"Recommended Movie Titles": recommended_titles}))
