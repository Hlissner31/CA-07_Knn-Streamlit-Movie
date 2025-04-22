# CA05 – kNN based Movie Recommender Engine

## 1. The Application
This project implements a k-Nearest Neighbors (kNN) based movie recommender engine. At scale, similar techniques are used by Amazon for product recommendations, Netflix for movie suggestions, and YouTube for video recommendations. While large-scale systems use more sophisticated approaches, this project demonstrates a fundamental implementation of a content-based recommender system.

### Question Addressed
Given a movie dataset, the system determines the 5 most similar movies to a queried movie based on genre and IMDB rating.

## 2. Data Source and Contents
The dataset is a small subset extracted from the UCI IMDB dataset.

**Data File:** `movies_recommendation_data.csv`  
**File Location:**(https://github.com/ArinB/MSBA-CA-Data/raw/main/CA05/movies_recommendation_data.csv)

### Dataset Features
- **IMDB Rating** (Numerical)
- **Genres** (Binary encoded for Biography, Drama, Thriller, Comedy, Crime, Mystery, History)
- **Labels Column** (Can be ignored as it's not used for classification or regression)

### Limitations
- The dataset lacks information on actors, directors, or themes, which could improve recommendations.
- Similarity is solely determined by genre and IMDB rating using the kNN algorithm.

## 3. Building the Recommender System
This project simulates a recommendation website where a user can explore similar movies based on a selected title. For example, if a user views "The Post" and wants similar recommendations, the system retrieves 5 movies with the closest features.

### Features of "The Post"
- **IMDB Rating:** 7.2
- **Genres:** Biography (Yes), Drama (Yes), Thriller (No), Comedy (No), Crime (No), Mystery (No), History (Yes)

### Implementation Steps
1. **Load the Dataset**: Read and preprocess the movie dataset.
2. **Feature Extraction**: Represent movies as feature vectors (IMDB rating and genre).
3. **Apply kNN Algorithm**:
   - Fit a kNN model to find similar movies using Euclidean distance.
   - Retrieve the top 5 most similar movies.
4. **Display Recommendations**: The system outputs similar movies directly within the Jupyter Notebook.

## 4. Installation
Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scikit-learn
```

## 5. Example Output
For the movie `"The Post"`, the system might output:

```bash
Top 5 movies similar to 'The Post':
1. Movie A
2. Movie B
3. Movie C
4. Movie D
5. Movie E
```