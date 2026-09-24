import streamlit as st
import pickle
import requests
import html

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #888888;
    margin-top: 5px;
    margin-bottom: 30px;
}

.recommend-title {
    font-size: 2rem;
    font-weight: 750;
    margin-top: 10px;
}

.movie-title {
    font-size: 1.15rem;
    font-weight: 700;
    min-height: 55px;
    margin-top: 10px;
}

.movie-meta {
    font-size: 0.95rem;
    color: #888888;
    margin-bottom: 5px;
}

.rank {
    font-size: 1.5rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.similarity-label {
    font-size: 0.85rem;
    color: #888888;
    margin-top: 8px;
}

.similarity-value {
    font-size: 1.5rem;
    font-weight: 750;
}

.tech {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    background-color: rgba(128,128,128,0.12);
    font-size: 0.9rem;
}

.footer {
    text-align: center;
    color: #888888;
    font-size: 0.85rem;
    padding-top: 20px;
}

[data-testid="stImage"] img {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MOVIES + TF-IDF MATRIX
# ============================================================

@st.cache_resource
def load_model():

    # Load movie information
    with open("movies.pkl", "rb") as file:
        movie_data = pickle.load(file)

    # Load TF-IDF matrix
    with open("tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix_data = pickle.load(file)

    return movie_data, tfidf_matrix_data


try:

    movies, tfidf_matrix = load_model()

except FileNotFoundError:

    st.error(
        "❌ movies.pkl or tfidf_matrix.pkl could not be found. "
        "Make sure both files are in the same folder as app.py."
    )

    st.stop()


# ============================================================
# 4. CHECK MODEL FILES
# ============================================================

if len(movies) != tfidf_matrix.shape[0]:

    st.error(
        "❌ The movie dataset and TF-IDF matrix do not match."
    )

    st.stop()


# ============================================================
# 5. TMDB API CONFIGURATION
# ============================================================

try:

    TMDB_TOKEN = st.secrets["TMDB_TOKEN"]

except Exception:

    st.error(
        "❌ TMDB API token was not found. "
        "Check .streamlit/secrets.toml."
    )

    st.stop()


TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json"
}


TMDB_IMAGE_BASE = (
    "https://image.tmdb.org/t/p/w500"
)


# ============================================================
# 6. FETCH MOVIE DETAILS FROM TMDB
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):

    url = (
        f"https://api.themoviedb.org/3/movie/"
        f"{movie_id}"
    )

    try:

        response = requests.get(
            url,
            headers=TMDB_HEADERS,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()


        # Poster
        poster_path = data.get("poster_path")

        if poster_path:

            poster = (
                TMDB_IMAGE_BASE
                + poster_path
            )

        else:

            poster = None


        # Release year
        release_date = data.get(
            "release_date",
            ""
        )

        if release_date:

            year = release_date[:4]

        else:

            year = "N/A"


        # Rating
        rating = data.get(
            "vote_average",
            0
        )


        # Overview
        overview = data.get(
            "overview"
        )

        if not overview:

            overview = (
                "No overview is available "
                "for this movie."
            )


        # Genres
        genres = [
            genre["name"]
            for genre in data.get(
                "genres",
                []
            )
        ]


        # Runtime
        runtime = data.get(
            "runtime"
        )


        return {
            "poster": poster,
            "year": year,
            "rating": rating,
            "overview": overview,
            "genres": genres,
            "runtime": runtime
        }


    except requests.RequestException:

        return {
            "poster": None,
            "year": "N/A",
            "rating": 0,
            "overview":
                "Movie information is currently unavailable.",
            "genres": [],
            "runtime": None
        }


# ============================================================
# 7. RECOMMENDATION FUNCTION
# ============================================================

def recommend(movie_title, number=5):

    # Find selected movie
    matches = movies[
        movies["title"].str.lower()
        == movie_title.lower()
    ]


    if matches.empty:

        return []


    # Get DataFrame index
    dataframe_index = matches.index[0]


    # Convert index to row position
    position = movies.index.get_loc(
        dataframe_index
    )


    # ========================================================
    # CALCULATE COSINE SIMILARITY ON DEMAND
    # ========================================================

    scores = cosine_similarity(
        tfidf_matrix[position],
        tfidf_matrix
    ).flatten()


    # Sort highest similarity first
    similar_movies = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True
    )


    # Remove the selected movie itself
    similar_movies = [
        item
        for item in similar_movies
        if item[0] != position
    ]


    # Keep requested number
    similar_movies = similar_movies[
        :number
    ]


    results = []


    for movie_position, score in similar_movies:

        row = movies.iloc[
            movie_position
        ]


        movie_id = int(
            row["movie_id"]
        )


        details = fetch_movie_details(
            movie_id
        )


        results.append(
            {
                "movie_id":
                    movie_id,

                "title":
                    row["title"],

                "similarity":
                    float(score),

                "poster":
                    details["poster"],

                "year":
                    details["year"],

                "rating":
                    details["rating"],

                "overview":
                    details["overview"],

                "genres":
                    details["genres"],

                "runtime":
                    details["runtime"]
            }
        )


    return results


# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🎬 CineMatch
    </div>

    <div class="subtitle">
        Discover movies through content-based recommendations
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("---")


# ============================================================
# 9. MOVIE SELECTION
# ============================================================

st.subheader(
    "🍿 Find Your Next Movie"
)


st.write(
    "Choose a movie you enjoyed and CineMatch "
    "will find movies with similar content."
)


# Movie titles
movie_titles = sorted(
    movies["title"]
    .dropna()
    .unique()
    .tolist()
)


# Default movie
default_movie = "Interstellar"


if default_movie in movie_titles:

    default_index = (
        movie_titles.index(
            default_movie
        )
    )

else:

    default_index = 0


selected_movie = st.selectbox(
    "Select a movie",
    movie_titles,
    index=default_index
)


# ============================================================
# NUMBER OF RECOMMENDATIONS
# ============================================================

number_of_movies = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=10,
    value=5,
    step=1
)


# ============================================================
# SEARCH BUTTON
# ============================================================

find_movies = st.button(
    "🔍 Find Similar Movies",
    type="primary",
    use_container_width=True
)


# ============================================================
# 10. RESULTS
# ============================================================

if find_movies:


    # ========================================================
    # SELECTED MOVIE
    # ========================================================

    selected_row = movies[
        movies["title"]
        == selected_movie
    ]


    if not selected_row.empty:

        selected_id = int(
            selected_row.iloc[0][
                "movie_id"
            ]
        )


        selected_details = (
            fetch_movie_details(
                selected_id
            )
        )


        st.markdown("---")


        st.subheader(
            "🎥 Your Selected Movie"
        )


        poster_column, info_column = (
            st.columns(
                [1, 3],
                gap="large"
            )
        )


        # ====================================================
        # SELECTED MOVIE POSTER
        # ====================================================

        with poster_column:

            if selected_details[
                "poster"
            ]:

                st.image(
                    selected_details[
                        "poster"
                    ],
                    use_container_width=True
                )

            else:

                st.info(
                    "Poster unavailable"
                )


        # ====================================================
        # SELECTED MOVIE INFORMATION
        # ====================================================

        with info_column:

            st.title(
                selected_movie
            )


            info1, info2, info3 = (
                st.columns(3)
            )


            # Release year
            with info1:

                st.metric(
                    "Release Year",
                    selected_details[
                        "year"
                    ]
                )


            # Rating
            with info2:

                st.metric(
                    "TMDB Rating",
                    f"{selected_details['rating']:.1f}/10"
                )


            # Runtime
            with info3:

                runtime = (
                    selected_details[
                        "runtime"
                    ]
                )


                if runtime:

                    runtime_text = (
                        f"{runtime} min"
                    )

                else:

                    runtime_text = "N/A"


                st.metric(
                    "Runtime",
                    runtime_text
                )


            # Genres
            if selected_details[
                "genres"
            ]:

                st.write(
                    "**Genres:** "
                    + " • ".join(
                        selected_details[
                            "genres"
                        ]
                    )
                )


            # Overview
            st.markdown(
                "### Overview"
            )


            st.write(
                selected_details[
                    "overview"
                ]
            )


    # ========================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================

    with st.spinner(
        "Finding movies you might enjoy..."
    ):

        recommendations = recommend(
            selected_movie,
            number_of_movies
        )


    st.markdown("---")


    # ========================================================
    # SHOW RECOMMENDATIONS
    # ========================================================

    if not recommendations:

        st.warning(
            "No recommendations were found."
        )


    else:

        st.markdown(
            """
            <div class="recommend-title">
                ✨ Recommended Movies
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write(
            f"Because you selected "
            f"**{selected_movie}**, "
            f"these movies have the most "
            f"similar content profiles."
        )


        st.caption(
            "Content Similarity represents cosine "
            "similarity between movie TF-IDF vectors. "
            "It is not a probability or model accuracy."
        )


        # ====================================================
        # DISPLAY RECOMMENDATION CARDS
        # ====================================================

        for start in range(
            0,
            len(recommendations),
            5
        ):


            row_movies = recommendations[
                start:start + 5
            ]


            # IMPORTANT:
            # Always create five columns.
            #
            # This means that if the user selects:
            #
            # 6 recommendations:
            # #6 stays the same width.
            #
            # 8 recommendations:
            # #6 #7 #8 stay the same width.
            #
            # 10 recommendations:
            # #6 #7 #8 #9 #10 fill the second row.
            #
            # No giant poster problem.
            columns = st.columns(
                5,
                gap="medium"
            )


            for rank, (
                column,
                movie
            ) in enumerate(
                zip(
                    columns,
                    row_movies
                ),
                start=start + 1
            ):


                with column:


                    # ========================================
                    # RANK
                    # ========================================

                    st.markdown(
                        f"""
                        <div class="rank">
                            #{rank}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ========================================
                    # POSTER
                    # ========================================

                    if movie[
                        "poster"
                    ]:

                        st.image(
                            movie[
                                "poster"
                            ],
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "Poster unavailable"
                        )


                    # ========================================
                    # MOVIE TITLE
                    # ========================================

                    safe_title = (
                        html.escape(
                            str(
                                movie[
                                    "title"
                                ]
                            )
                        )
                    )


                    st.markdown(
                        f"""
                        <div class="movie-title">
                            {safe_title}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ========================================
                    # RELEASE YEAR
                    # ========================================

                    st.markdown(
                        f"""
                        <div class="movie-meta">
                            📅 {movie["year"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ========================================
                    # TMDB RATING
                    # ========================================

                    st.markdown(
                        f"""
                        <div class="movie-meta">
                            ⭐ {movie["rating"]:.1f}/10
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ========================================
                    # CONTENT SIMILARITY
                    # ========================================

                    st.markdown(
                        f"""
                        <div class="similarity-label">
                            Content Similarity
                        </div>

                        <div class="similarity-value">
                            {movie["similarity"]:.1%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ========================================
                    # MOVIE DETAILS
                    # ========================================

                    with st.expander(
                        "Movie details"
                    ):


                        # Genres
                        if movie[
                            "genres"
                        ]:

                            st.write(
                                "**Genres:** "
                                + ", ".join(
                                    movie[
                                        "genres"
                                    ]
                                )
                            )


                        # Runtime
                        if movie[
                            "runtime"
                        ]:

                            st.write(
                                f"**Runtime:** "
                                f"{movie['runtime']} min"
                            )


                        # Overview
                        st.write(
                            movie[
                                "overview"
                            ]
                        )


# ============================================================
# 11. HOW CINEMATCH WORKS
# ============================================================

st.markdown("---")


st.subheader(
    "🧠 How CineMatch Works"
)


st.write(
    """
    CineMatch is a **content-based movie recommendation
    system**.

    Instead of relying on user ratings, CineMatch compares
    information describing each movie to identify movies
    with similar content.
    """
)


step1, step2, step3, step4 = (
    st.columns(4)
)


# ============================================================
# STEP 1
# ============================================================

with step1:

    st.markdown(
        "### 1️⃣ Movie Features"
    )

    st.write(
        """
        Movie overview, genres, keywords,
        cast members and director information
        are combined.
        """
    )


# ============================================================
# STEP 2
# ============================================================

with step2:

    st.markdown(
        "### 2️⃣ NLP"
    )

    st.write(
        """
        Text preprocessing and Porter stemming
        normalize related word forms.
        """
    )


# ============================================================
# STEP 3
# ============================================================

with step3:

    st.markdown(
        "### 3️⃣ TF-IDF"
    )

    st.write(
        """
        TF-IDF converts the processed movie text
        into numerical feature vectors.
        """
    )


# ============================================================
# STEP 4
# ============================================================

with step4:

    st.markdown(
        "### 4️⃣ Similarity"
    )

    st.write(
        """
        Cosine similarity compares the selected
        movie vector with all other movie vectors.
        """
    )


# ============================================================
# 12. RECOMMENDATION PIPELINE
# ============================================================

st.markdown("---")


st.subheader(
    "⚙️ Recommendation Pipeline"
)


st.info(
    "TMDB 5000 Dataset  →  "
    "Data Cleaning  →  "
    "Feature Engineering  →  "
    "NLP & Stemming  →  "
    "TF-IDF Vectorization  →  "
    "Cosine Similarity  →  "
    "Top-N Recommendations"
)


# ============================================================
# 13. PROJECT INFORMATION
# ============================================================

st.markdown("---")


st.subheader(
    "📊 Project Information"
)


project1, project2, project3 = (
    st.columns(3)
)


with project1:

    st.markdown(
        "#### Recommendation Method"
    )

    st.write(
        "Content-Based Filtering"
    )


with project2:

    st.markdown(
        "#### Text Representation"
    )

    st.write(
        "TF-IDF Vectorization"
    )


with project3:

    st.markdown(
        "#### Similarity Measure"
    )

    st.write(
        "Cosine Similarity"
    )


# ============================================================
# 14. TECHNOLOGIES
# ============================================================

st.markdown("---")


st.subheader(
    "🛠️ Technologies Used"
)


st.markdown(
    """
    <span class="tech">Python</span>
    <span class="tech">Pandas</span>
    <span class="tech">Scikit-learn</span>
    <span class="tech">NLTK</span>
    <span class="tech">TF-IDF</span>
    <span class="tech">Cosine Similarity</span>
    <span class="tech">Streamlit</span>
    <span class="tech">TMDB API</span>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 15. TMDB ATTRIBUTION
# ============================================================

st.markdown("---")


st.caption(
    "Movie metadata and poster images are provided by TMDB."
)


st.caption(
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎬 CineMatch • Data Science Portfolio Project
    </div>
    """,
    unsafe_allow_html=True
)