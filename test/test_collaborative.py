import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.collaborative_recommender import (
    train_collaborative_model,
    recommend_movies
)

# MOCK DATA

def create_test_data():
    # Cria uma matriz pequena de avaliações para usar nos testes

    movie_pivot = pd.DataFrame(
        {
            1: [5,4,0,0],
            2: [4,5,1,0],
            3: [0,1,5,4],
            4: [0,0,4,5],
        },
        index=[
            "Filme A",
            "Filme B",
            "Filme C",
            "Filme D"
        ]
    )

    return movie_pivot

"""
TESTS: 
test1 - model training, 
test2 - recommendations quantity, 
test3 - movie not recommended to itself, 
test4 - invalid movie
"""

def test_train_collaborative_model():
    # verifica se o modelo colaborativo é treinado corretamente

    movie_pivot = create_test_data()
    movie_sparse = csr_matrix(movie_pivot)

    model = train_collaborative_model(movie_sparse)

    assert model is not None


def test_recommend_movies():
    # verifica se o sistema retorna recomendações de filmes corretamente

    movie_pivot = create_test_data()
    movie_sparse = csr_matrix(movie_pivot)

    model = train_collaborative_model(movie_sparse)

    recomendations = recommend_movies(
        "Filme A",
        model,
        movie_pivot,
        n_recommendations=2
    )

    assert len(recomendations) == 2


def test_movie_is_not_recommended_to_itself():
    # verifica se o filme pesquisado não aparece entre as recomendações

    movie_pivot = create_test_data()
    movie_sparse = csr_matrix(movie_pivot)

    model = train_collaborative_model(movie_sparse)


    recommendations = recommend_movies(
        "Filme A",
        model,
        movie_pivot,
        n_recommendations=2
    )

    assert "Filme A" not in recommendations

def test_invalid_movie():
    # verifica se o sistema lança um erro quando o filme pesquisado não existe

    movie_pivot = create_test_data()
    movie_sparse = csr_matrix(movie_pivot)

    model = train_collaborative_model(movie_sparse)

    with pytest.raises(ValueError):
        recommend_movies(
            "Filme Inexistente",
            model,
            movie_pivot
        )