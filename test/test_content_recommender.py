import pandas as pd
import pytest

from src.content_recommender import (
    extract_names,
    load_movies,
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

###############################################################################################
######################## TESTE DE PREPARAÇÃO DOS NOVOS DADOS ##################################
###############################################################################################

def test_extract_names():
    value = "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"

    result = extract_names(value)

    assert result == "Action Adventure"


def test_extract_names_with_empty_value():
    assert extract_names(None) == ""
    assert extract_names("") == ""



def test_load_movies_(tmp_path):
    # Cria um arquivo CSV temporário com dados de teste
    csv_path = tmp_path / "movies.csv"

    data = pd.DataFrame({
        "title": ["Filme A", "Filme B"],
        "overview": ["Uma aventura espacial", None],
        "genres": [
            "[{'id': 12, 'name': 'Adventure'}]",
            "[{'id': 35, 'name': 'Comedy'}]",
        ],
        "keywords": [
            "[{'id': 1, 'name': 'space'}]",
            "[{'id': 2, 'name': 'friendship'}]",
        ],
    })

    data.to_csv(csv_path, index=False)

    movies = load_movies(csv_path)

    assert len(movies) == 2
    assert movies.loc[0, "genres"] == "Adventure"
    assert movies.loc[1, "keywords"] == "friendship"
    assert movies.loc[1, "overview"] == ""  #Verifica se o valor None foi substituído por uma string vazia


'''
O que você está aprendendo aqui?

tmp_path: fixture do pytest que cria uma pasta temporária para o teste.

to_csv(): permite gerar um CSV pequeno de exemplo, sem depender do dataset real.

assert: verifica resultados concretos das funções.

Testes de valores vazios: ajudam a evitar erros quando os dados reais possuem campos ausentes.
'''