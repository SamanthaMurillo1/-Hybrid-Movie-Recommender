# 🎬 Hybrid Movie AI: Smart Recommender

This project is an advanced **Hybrid Movie Recommendation System**. It goes beyond traditional title-matching by combining a **Content-Based Filtering model** with a **Custom NLP "Vibe" Search** powered by Sentence Transformers.

## 🌟 Demo video link
* **Youtube Link:** https://youtu.be/j19Rvhgn-OY

## 🌟 Key Features

* **Dual-Mode Search:** The system automatically detects if you entered a specific movie title or a "vibe" (e.g., "sad space thriller" or "happy animated movies").
* **Smart AI "Vibe" Search:** Uses a custom NLP model to analyze the sentiment and "soul" of a movie.
* **TMDB Integration:** Fetches high-quality posters and provides direct links to TMDB pages for full descriptions and ratings.
* **Watchlist Queue:** Add movies to a session-based queue to keep track of what you want to watch next.
* **Interactive UI:** Built with Streamlit for a sleek, responsive user experience.

---

## 🧠 Theory & Logic

### 1. Content-Based Filtering (The Classic Model)

This model uses **Cosine Similarity** to find movies with similar metadata (genres, keywords, cast, and crew). If you search for "The Dark Knight," it finds movies that are structurally similar based on the TMDB 5000 dataset.

### 2. Smart Vibe Search (The NLP Model)

While traditional systems look at "who directed this," the **Smart Vibe Search** looks at "what does this movie *feel* like?"

#### Sentence Embeddings

I used the `all-MiniLM-L6-v2` **Sentence Transformer**. This model converts movie overviews into **dense vectors** (384-dimensional numerical representations). Unlike a keyword search, this understands that "galactic journey" is semantically similar to "space travel," even if the words don't match.

#### Hybrid Scoring & Tone Bias

To make the recommendations feel more "human," I added a final layer of logic:

* **Tone Sentiment:** I assigned a `tone_score` to movies. If you search for "happy" movies, the system boosts positive sentiment movies.
* **Weighted Ranking:** The final rank is a weighted average to ensure quality:


---

## 🚀 Installation


1. **Clone the repository:**
```bash
git clone https://github.com/SamanthaMurillo1/-Hybrid-Movie-Recommender.git
cd -Hybrid-Movie-Recommender

```


2. **Activate your environment:**
```bash
conda activate movie_ai

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up your Secrets:**
Create a `.env` file in the root directory and add your TMDB API Key:
```text
API_KEY=your_tmdb_api_key_here

```



---

## 🏗️ How to Generate the Models

Since `.pkl` files are often too large for GitHub, you may need to generate them locally before running the app:

1. **Download the Data:** Download the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata).
2. **Run the Preprocessing Notebook:** Open `Movie_Recommender.ipynb` and run all cells. This generates `movie_data.pkl`.
3. **Run the NLP Notebook:** Open `nlp_preprocess.ipynb` and run all cells. This uses the Sentence Transformer to generate `smart_movie_models.pkl`.
4. Once both `.pkl` files are in your project folder, the Streamlit app is ready to go.

---

## 🛠️ Usage

1. **Run the App:**
```bash
streamlit run app2.py

```
1.5. **Open Local Host:**
Local URL: http://localhost:8501


2. **Search:** * Type a movie name like **"Batman"** for exact recommendations.
* Type a vibe like **"dark space thriller"** to let the AI find a match.


3. **Explore:** Click on any movie title; it is a direct link to the **TMDB movie page** where you can see full descriptions and ratings.
4. **WatchList:** Use the **➕ Add** button to save movies to your sidebar watchlist.

---

## 📊 Dataset

The dataset used for this project is the [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata), which includes metadata for nearly 5,000 movies.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Give credit. 

credit to https://github.com/dutta-sujoy/Movie-Recommendation-System/tree/main?tab=readme-ov-file I built off of this project

