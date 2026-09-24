# 🎬 CineMatch - Content-Based Movie Recommendation System

CineMatch is an interactive **content-based movie recommendation system** that recommends movies based on similarities in their content.

The system analyzes movie information such as **overview, genres, keywords, cast, and director**, transforms this information using **TF-IDF**, and calculates **cosine similarity** to identify movies with similar content profiles.

The project includes an interactive **Streamlit web application** with movie posters and additional movie information retrieved using the **TMDB API**.

---

## 🌐 Live Demo

🚀 **Try CineMatch here:**  
https://cinematch-tashini.streamlit.app/

💻 **GitHub Repository:**  
https://github.com/tashini-savindya/cinematch-movie-recommendation-system

---

## ✨ Features

- Search and select a movie from the dataset
- Generate content-based movie recommendations
- Choose between **5 and 10 recommendations**
- Display movie posters
- Display movie release year
- Display TMDB user rating
- Display movie genres
- Display runtime
- Display movie overview
- Show a **Content Similarity** score for each recommendation
- Interactive and user-friendly Streamlit interface
- Movie metadata retrieved dynamically using the TMDB API

---

## 🧠 How CineMatch Works

CineMatch uses a **content-based recommendation approach**.

Instead of recommending movies based on ratings from other users, the system examines the characteristics of each movie and finds movies with similar content.

The recommendation process is:

```text
TMDB Movie Dataset
        ↓
Data Cleaning
        ↓
Feature Extraction
        ↓
Text Preprocessing
        ↓
Stemming
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Top Similar Movies
        ↓
TMDB API Metadata
        ↓
Streamlit Web Application
```

---

## 📊 Dataset

The project uses the **TMDB 5000 Movie Dataset**.

The main dataset files used are:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

The movies and credits datasets are combined so that information about each movie can be used to construct its content profile.

Important features used by the recommendation system include:

- Movie title
- Overview
- Genres
- Keywords
- Cast
- Director

---

## 🧹 Data Preprocessing

Several preprocessing steps are performed before building the recommendation system.

### 1. Dataset Merging

The movies dataset and credits dataset are merged using the movie title.

### 2. Feature Selection

The following important features are selected:

```text
movie_id
title
overview
genres
keywords
cast
crew
```

### 3. Feature Extraction

The JSON-like columns are processed to extract useful information.

For example:

- Genre names are extracted from `genres`
- Keyword names are extracted from `keywords`
- The top cast members are extracted from `cast`
- The director is extracted from `crew`

### 4. Feature Combination

The selected information is combined into a single text representation called **tags**.

Conceptually:

```text
Overview + Genres + Keywords + Cast + Director
```

This creates a content profile for every movie.

---

## 🔤 Text Preprocessing

The combined movie tags are normalized before vectorization.

The preprocessing includes:

- Converting text to lowercase
- Combining important movie features
- Applying word stemming

### Stemming

The **Porter Stemmer** from NLTK is used to reduce related words to a common stem.

This helps the recommendation system recognize related forms of words as similar features.

---

## 🔢 TF-IDF Vectorization

CineMatch uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert movie text into numerical vectors.

The project uses:

```python
TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
```

TF-IDF gives greater importance to words that are useful for distinguishing one movie from another while reducing the importance of very common words.

Each movie is therefore represented as a numerical **TF-IDF vector**.

---

## 📐 Cosine Similarity

After creating TF-IDF vectors, CineMatch uses **cosine similarity** to measure how similar two movies are.

Cosine similarity compares the direction of two vectors.

A higher cosine similarity value means the two movies have more similar content profiles.

When a user selects a movie, CineMatch:

1. Finds the selected movie's TF-IDF vector
2. Compares it with the TF-IDF vectors of the other movies
3. Calculates cosine similarity scores
4. Sorts movies by similarity
5. Returns the most similar movies

The **Content Similarity** value displayed by CineMatch represents cosine similarity between movie TF-IDF vectors.

> **Important:** Content Similarity is not a probability, prediction confidence, or model accuracy.

---

## ⚡ Storage Optimization

An earlier version of the project calculated and stored the complete cosine similarity matrix.

The matrix had approximately:

```text
4806 × 4806
```

entries and produced a `similarity.pkl` file of approximately:

```text
176 MB
```

This was inefficient for GitHub storage and cloud deployment.

The final version instead stores the sparse TF-IDF matrix:

```text
tfidf_matrix.pkl
```

which is approximately:

```text
1.68 MB
```

Cosine similarity is then calculated **on demand** only for the movie selected by the user.

Conceptually:

```python
scores = cosine_similarity(
    tfidf_matrix[position],
    tfidf_matrix
).flatten()
```

This significantly reduces storage requirements while maintaining the same recommendation approach.

---

## 🎯 Example Recommendation

If the user selects:

```text
Interstellar
```

CineMatch identifies movies with similar content profiles.

Example recommendations include:

```text
Apollo 13
The Right Stuff
The Martian
Space Pirate Captain Harlock
Space Cowboys
```

The ranking is generated using cosine similarity between the TF-IDF representations of the movies.

---

## 🌐 TMDB API Integration

The application uses the **TMDB API** to retrieve additional movie information dynamically.

This includes:

- Movie posters
- Release year
- TMDB rating
- Genres
- Runtime
- Movie overview

The recommendation algorithm itself is based on the processed TMDB 5000 dataset, while the API improves the application's presentation by providing additional movie metadata.

---

## 🖥️ Streamlit Web Application

The recommendation system is deployed as an interactive web application using **Streamlit**.

Users can:

1. Select a movie
2. Choose the number of recommendations
3. Generate recommendations
4. View recommended movie posters
5. View content similarity
6. Explore additional information about each movie

The deployed application is available here:

https://cinematch-tashini.streamlit.app/

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| NLTK | Text stemming |
| Streamlit | Interactive web application |
| Requests | TMDB API requests |
| Pickle | Saving processed movie data and TF-IDF matrix |
| Jupyter Notebook | Data exploration and model development |
| Git | Version control |
| GitHub | Source code hosting |
| TMDB API | Movie posters and metadata |

---

## 📁 Project Structure

```text
cinematch-movie-recommendation-system/
│
├── app.py
│
├── movie_recommendation.ipynb
│
├── movies.pkl
│
├── tfidf_matrix.pkl
│
├── tmdb_5000_movies.csv
│
├── tmdb_5000_credits.csv
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

The following files are intentionally excluded from GitHub:

```text
.streamlit/secrets.toml
similarity.pkl
archive.zip
archive/
```

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/tashini-savindya/cinematch-movie-recommendation-system.git
```

### 2. Navigate to the project

```bash
cd cinematch-movie-recommendation-system
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Configure the TMDB API token

Create the following directory and file:

```text
.streamlit/secrets.toml
```

Add your own TMDB Read Access Token:

```toml
TMDB_TOKEN = "YOUR_TMDB_READ_ACCESS_TOKEN"
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will then open in your browser.

---

## 🔐 Security

API credentials are **not stored in the GitHub repository**.

The local TMDB credential file:

```text
.streamlit/secrets.toml
```

is excluded using `.gitignore`.

For the deployed application, the TMDB token is stored securely using **Streamlit Secrets**.

Users who clone this repository must provide their own TMDB API credentials.

---

## 📦 Model Files

### `movies.pkl`

Contains the processed movie information required by the Streamlit application.

### `tfidf_matrix.pkl`

Contains the sparse TF-IDF representation of the movie content.

### Why `similarity.pkl` is not included

The original complete similarity matrix was approximately **176 MB**.

Instead of storing this large file, CineMatch calculates the required cosine similarity scores dynamically using the much smaller sparse TF-IDF matrix.

---

## 📈 Future Improvements

Possible future improvements include:

- Collaborative filtering
- Hybrid recommendation systems
- User accounts and personalized recommendation history
- User ratings and preferences
- Advanced NLP embeddings
- Sentence Transformer embeddings
- Genre-based filtering
- Release-year filtering
- Recommendation explanations
- Improved search functionality
- Additional evaluation methods
- Larger and more recent movie datasets

---

## 🎓 Project Purpose

CineMatch was developed as a **Data Science portfolio project** to demonstrate practical knowledge of:

- Data preprocessing
- Natural Language Processing
- Feature engineering
- TF-IDF vectorization
- Similarity-based recommendation systems
- Python development
- API integration
- Interactive application development
- Model/data serialization
- Git and GitHub
- Cloud deployment

---

## 🔗 Project Links

**Live Application:**  
https://cinematch-tashini.streamlit.app/

**GitHub Repository:**  
https://github.com/tashini-savindya/cinematch-movie-recommendation-system

---

## 🙋‍♀️ Author

**Tashini Kariyawasam**  
Data Science Undergraduate  
SLIIT Faculty of Computing

---

## 🙏 Acknowledgements

This project uses movie data and metadata associated with **The Movie Database (TMDB)**.

Movie posters and additional metadata displayed in the web application are provided through the TMDB API.

**This product uses the TMDB API but is not endorsed or certified by TMDB.**