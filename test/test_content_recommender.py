import pandas as pd
import pytest

from src.content_recommender import (
    train_content_model,
    recommend_movies_by_content,
)

@pytest.fixture
def movie_data():
    return pd.DataFrame(
        {
            "title": [
                "Filme Espacial",
                "Aventura Espacial",
                "Comédia Romântica",
            ],
            "overview": [
                "Uma missão espacial para salvar a Terra",
                "Astronautas exploram o espaço em uma missão",
                "Um casal se apaixona em uma comédia romântica",
            ],
        }
    )


@pytest.fixture
def trained_model(movie_data):
    # Treina o modelo de recomendação com dados de teste (MOCK DATA)

    _, movie_vectors = train_content_model(movie_data)
    # O _ é uma convenção Python que significa: “recebo este valor, mas não pretendo utilizá-lo”.
    return movie_vectors


def test_recommends_requested_number_of_movies(
        movie_data, trained_model
):
    recommendations = recommend_movies_by_content(
        "Filme Espacial",
        movie_data,
        trained_model,
        n_recommendations=2
    )

    assert len(recommendations) == 2


def test_does_not_recommend_the_same_movie(
        movie_data, trained_model
):
    recommendations = recommend_movies_by_content(
        "Filme Espacial",
        movie_data,
        trained_model,
        n_recommendations=2
    )

    assert "Filme Espacial" not in recommendations["title"].values


def test_uknown_movie_raises_value_error(
        movie_data, trained_model
):
    with pytest.raises(ValueError, match="não foi encontrado"):
        recommend_movies_by_content(
            "Filme Desconhecido",
            movie_data,
            trained_model,
        )


def test_zero_recommendations_raises_value_error(
        movie_data, trained_model
):
    with pytest.raises(ValueError, match="maior que zero"):
        recommend_movies_by_content(
            "Filme Espacial",
            movie_data,
            trained_model,
            n_recommendations=0
        )