# 🎬 CineMatch - Content-Based Movie Recommendation System

CineMatch is a content-based movie recommendation system developed as a Data Science portfolio project. It recommends movies based on similarities in their content, including movie overviews, genres, keywords, cast members, and directors.

The project applies Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity to identify movies with similar content profiles. The recommendation model is integrated into an interactive Streamlit web application, while the TMDB API is used to retrieve movie posters and additional movie information.

---

## 🚀 Features

- Search and select movies from the TMDB 5000 Movie Dataset
- Generate between 5 and 10 movie recommendations
- Content-based movie recommendation
- TF-IDF text vectorization
- Porter stemming for text preprocessing
- Cosine similarity for movie comparison
- Movie poster integration using the TMDB API
- Display movie release year
- Display TMDB rating
- Display movie runtime
- Display movie genres
- Display movie overview
- Interactive movie detail sections
- Responsive Streamlit interface
- On-demand similarity calculation to reduce model storage requirements

---

## 🧠 How CineMatch Works

CineMatch uses a **content-based filtering approach**.

Instead of using ratings from other users, the system analyzes the characteristics of each movie and recommends movies with similar content.

### 1. Data Preparation

Movie information is obtained from the **TMDB 5000 Movie Dataset**.

The main features used by the recommendation system are:

- Movie overview
- Genres
- Keywords
- Top cast members
- Director

These features are cleaned and combined into a single text representation for each movie.

### 2. Natural Language Processing

Text preprocessing is performed before creating the recommendation model.

Porter stemming is used to reduce related words to a common stem.

For example:

```text
connected
connecting
connection
```

can be reduced to related root forms, helping the model treat similar words more consistently.

### 3. TF-IDF Vectorization

The processed movie text is converted into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF gives greater importance to words that are useful for distinguishing one movie from another.

### 4. Cosine Similarity

When a user selects a movie, CineMatch compares its TF-IDF vector with the vectors of the other movies using **cosine similarity**.

Movies with higher cosine similarity scores have more similar content profiles.

The movies with the highest scores are returned as recommendations.

> The displayed content similarity value is a cosine similarity score. It should not be interpreted as a probability or model accuracy.

---

## ⚙️ Recommendation Pipeline

```text
TMDB 5000 Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Text Preprocessing
        ↓
Porter Stemming
        ↓
TF-IDF Vectorization
        ↓
Selected Movie
        ↓
Cosine Similarity
        ↓
Top-N Similar Movies
        ↓
Streamlit Application
```

---

## 💡 Storage Optimization

An earlier implementation stored the complete pairwise cosine similarity matrix.

For approximately 4,800 movies, this produced a file of roughly **176 MB**.

The final application instead stores the sparse TF-IDF feature matrix and calculates cosine similarity only when a user requests recommendations.

The TF-IDF matrix is approximately **1.68 MB**, significantly reducing the storage required by the application while preserving the same recommendation approach.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| NLTK | Porter stemming |
| Streamlit | Interactive web application |
| TMDB API | Movie posters and metadata |
| Requests | TMDB API communication |
| Pickle | Saving processed model data |
| Jupyter Notebook | Model development and experimentation |

---

## 📂 Project Structure

```text
CineMatch/
│
├── app.py
├── movies.pkl
├── tfidf_matrix.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── notebook.ipynb
│
└── .streamlit/
    └── secrets.toml
```

> `.streamlit/secrets.toml` contains the TMDB API credential and is excluded from GitHub using `.gitignore`.

The old `similarity.pkl` file is also excluded because the final application calculates cosine similarity dynamically.

---

## 💻 Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY_NAME
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the TMDB API

Create a folder named:

```text
.streamlit
```

Inside it, create:

```text
secrets.toml
```

Add your TMDB API Read Access Token:

```toml
TMDB_TOKEN = "YOUR_TMDB_READ_ACCESS_TOKEN"
```

Do not commit this file to GitHub.

### 4. Run CineMatch

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 Example Recommendation

For a movie such as **Interstellar**, CineMatch identifies movies with similar content profiles based on the features used by the model.

The recommendation results include the movie poster, title, release year, TMDB rating, content similarity score, genres, runtime, and overview.

---

## 🔐 Security

API credentials are not stored directly in the Python source code.

The TMDB API token is stored using Streamlit secrets:

```text
.streamlit/secrets.toml
```

This file is excluded from version control using `.gitignore`.

---

## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset**.

The dataset contains movie information such as titles, overviews, genres, keywords, cast information, crew information, and other metadata.

---

## 🔮 Possible Future Improvements

Possible improvements to CineMatch include:

- Hybrid recommendation combining content and collaborative filtering
- User accounts and personalized recommendation history
- User ratings and feedback
- Recommendation evaluation using user interaction data
- Advanced semantic embeddings
- Search functionality for partial movie titles
- Genre and year filters
- Improved mobile interface
- Additional recommendation explanations

---

## 🎓 Project Purpose

CineMatch was developed as a Data Science portfolio project to demonstrate practical knowledge of:

- Data preprocessing
- Feature engineering
- Natural Language Processing
- TF-IDF
- Similarity-based recommendation systems
- Python application development
- API integration
- Model optimization
- Streamlit deployment

---

## 🙏 Acknowledgements

Movie metadata and poster images displayed in the application are provided by TMDB.

This product uses the TMDB API but is not endorsed or certified by TMDB.

---

## 👩‍💻 Author

**Tashini Kariyawasam**

Data Science Undergraduate  
SLIIT Faculty of Computing

---

⭐ If you find this project useful, feel free to explore the repository.