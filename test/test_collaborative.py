import pandas as pd
from scipy.sparse import csr_matrix

from src.collaborative_recommender import (
    train_collaborative_model,
    recommend_movies
)


# Dados pequenos apenas para testar
movie_pivot = pd.DataFrame(
    {
        1: [5, 4, 0, 0],
        2: [4, 5, 1, 0],
        3: [0, 1, 5, 4],
        4: [0, 0, 4, 5],
    },
    index=[
        "Filme A",
        "Filme B",
        "Filme C",
        "Filme D"
    ]
)

movie_sparse = csr_matrix(movie_pivot)

model = train_collaborative_model(movie_sparse)

recommendations = recommend_movies(
    "Filme A",
    model,
    movie_pivot,
    n_recommendations=2
)

print("Recomendações:")
print(recommendations)