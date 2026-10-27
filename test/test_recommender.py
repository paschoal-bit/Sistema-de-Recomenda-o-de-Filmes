import pandas as pd
import pytest

from src.hybrid_recommender import recommend_hybrid_movies

def test_hybrid_combines_recommendations(monkeypatch):
#Verifica se o híbrido combina resultados dos dois modelos.


    movie_pivot = pd.DataFrame(
        [[5, 4], [4, 5], [1, 2]],
        index=["Filme A", "Filme B", "Filme C"],
        columns=["Usuario 1", "Usuario 2"],
    )

movies = pd.DataFrame({
    "title": ["Filme A", "Filme B", "Filme C"],
    "overview": ["Aventura", "Ação", "Comédia"],
})

movie_vectors = None
collaborative_model = object()

def fake_collaborative(*args, **kwargs):
    return ["Filme B", "Filme C"]

def fake_content(*args, **kwargs):
    return pd.DataFrame({
        "title": ["Filme C", "Filme B"],
        "similarity": [0.9, 0.8],
    })

    monkeypatch.setattr(
        "src.hybrid_recommender.recommend_movies",
        fake_collaborative,
    )

    monkeypatch.setattr(
        "src.hybrid_recommender.recommend_movies_by_content",
        fake_content,
    )

    result = recommend_hybrid_movies(
        movie_title="Filme A",
        collaborative_model=collaborative_model,
        movie_pivot=movie_pivot,
        movies=movies,
        movie_vectors=movie_vectors,
        n_recommendations=2,
    )

    assert len(result) == 2
    assert set(result["title"]) == {"Filme B", "Filme C"}
    assert "hybrid_score" in result.columns


def test_hybrid_rejects_invalid_recommendation_count():
# Verifica se quantidades inválidas são rejeitadas.


    with pytest.raises(ValueError, match="maior que zero"):
        recommend_hybrid_movies(
        movie_title="Filme A",
        collaborative_model=None,
        movie_pivot=pd.DataFrame(),
        movies=pd.DataFrame(),
        movie_vectors=None,
        n_recommendations=0,
    )

