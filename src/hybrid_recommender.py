
import pandas as pd

from src.collaborative_recommender import recommend_movies
from src.content_recommender import recommend_movies_by_content


def recommend_hybrid_movies(
    movie_title,
    collaborative_model,
    movie_pivot,
    movies,
    movie_vectors,
    n_recommendations=5,
):
    """
        Recommenda filmes usando uma abordagem híbrida, combinando recomendações colaborativas e baseadas em conteúdo.
    
        Args:
            movie_title (str): O título do filme para o qual as recomendações serão feitas.
            collaborative_model: O modelo de recomendação colaborativa treinado.
            movie_pivot (pd.DataFrame): A matriz de utilidade (pivot table) usada para a recomendação colaborativa.
            movies (pd.DataFrame): O DataFrame contendo informações sobre os filmes.
            movie_vectors: Os vetores de características dos filmes, usados para a recomendação baseada em conteúdo.
            n_recommendations (int): O número de recomendações a serem retornadas.
    
        Returns:
            pd.DataFrame: Um DataFrame contendo os filmes recomendados, ordenados por relevância.
        """
        # Combina as recomendações colaborativas e baseadas em conteúdo
        # Os resultados são combinados por posição no ranking, sem somar diretamente pontuações incompatíveis.

    if n_recommendations < 1:
        raise ValueError(
            "n_recommendations deve ser maior que zero."
        )

    collaborative_results = []
    content_results = []

    # Tenta obter recomendações colaborativas.
    collaborative_title = next(
        (
            title
            for title in movie_pivot.index
            if str(title).casefold() == movie_title.casefold()
        ),
        None,
    )

    if collaborative_title is not None:
        collaborative_results = recommend_movies(
            collaborative_title,
            collaborative_model,
            movie_pivot,
            n_recommendations=n_recommendations,
        )

    # Tenta obter recomendações baseadas em conteúdo.
    content_match = movies["title"].astype(str).str.casefold().eq(
        movie_title.casefold()
    ).any()

    if content_match:
        content_recommendations = recommend_movies_by_content(
            movie_title,
            movies,
            movie_vectors,
            n_recommendations=n_recommendations,
        )

        content_results = content_recommendations["title"].tolist()

    if not collaborative_results and not content_results:
        raise ValueError(
            f"O filme '{movie_title}' não foi encontrado "
            "em nenhum dos modelos."
        )

    # Soma contribuições baseadas na posição de cada filme.
    scores = {}
    collaborative_ranks = {}
    content_ranks = {}

    for rank, title in enumerate(collaborative_results, start=1):
        scores[title] = scores.get(title, 0) + 1 / rank
        collaborative_ranks[title] = rank

    for rank, title in enumerate(content_results, start=1):
        scores[title] = scores.get(title, 0) + 1 / rank
        content_ranks[title] = rank

    # Remove o próprio filme de referência, se aparecer.
    scores.pop(movie_title, None)

    results = pd.DataFrame([
        {
            "title": title,
            "hybrid_score": score,
            "collaborative_rank": collaborative_ranks.get(title),
            "content_rank": content_ranks.get(title),
        }
        for title, score in scores.items()
    ])

    results = results.sort_values(
        "hybrid_score",
        ascending=False,
    )

    return results.head(n_recommendations).reset_index(drop=True)

    