import ast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_names(value):
    """Extrai nomes de gêneros ou palavras-chave do CSV."""

    if pd.isna(value) or not value:
        return ""

    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ""

    if not isinstance(items, list):
        return ""

    return " ".join(
        item["name"]
        for item in items
        if isinstance(item, dict) and "name" in item
    )


def load_movies(csv_path):
    """Carrega e prepara os dados dos filmes."""

    movies = pd.read_csv(csv_path, low_memory=False)

    required_columns = {
        "title",
        "overview",
        "genres",
        "keywords",
    }

    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Colunas ausentes no CSV: {missing_columns}"
        )

    movies = movies[
        ["title", "overview", "genres", "keywords"]
    ].copy()

    movies = movies.dropna(subset=["title"])
    movies = movies.reset_index(drop=True)

    movies["overview"] = movies["overview"].fillna("")

    movies["genres"] = movies["genres"].apply(extract_names)
    movies["keywords"] = movies["keywords"].apply(extract_names)

    return movies


def train_content_model(movies):
    """Treina o modelo usando sinopse, gêneros e palavras-chave."""

    content_columns = [
        column
        for column in ["overview", "genres", "keywords"]
        if column in movies.columns
    ]

    if not content_columns:
        raise ValueError(
            "Os dados precisam conter pelo menos uma coluna de conteúdo."
        )

    content = (
        movies[content_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    movie_vectors = vectorizer.fit_transform(content)

    return vectorizer, movie_vectors


def recommend_movies_by_content(
    movie_title,
    movies,
    movie_vectors,
    n_recommendations=5,
):
    """Recomenda filmes com características semelhantes."""

    if n_recommendations < 1:
        raise ValueError(
            "n_recommendations deve ser maior que zero."
        )

    matches = movies.index[
        movies["title"].str.casefold()
        == movie_title.casefold()
    ].tolist()

    if not matches:
        raise ValueError(
            f"O filme '{movie_title}' não foi encontrado."
        )

    movie_index = matches[0]

    similarities = cosine_similarity(
        movie_vectors[movie_index],
        movie_vectors,
    ).flatten()

    similarities[movie_index] = -1

    similar_indices = similarities.argsort()[::-1]

    selected_indices = similar_indices[:n_recommendations]

    recommendations = movies.iloc[selected_indices].copy()

    recommendations["similarity"] = similarities[selected_indices]

    return recommendations.reset_index(drop=True)