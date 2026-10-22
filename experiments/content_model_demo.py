from pathlib import Path

from src.content_recommender import (
    load_movies,
    train_content_model,
    recommend_movies_by_content
)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "tmdb_5000_movies.csv"
)

movies = load_movies(DATA_PATH)
vectorizer, movie_vectors = train_content_model(movies)

recommendations = recommend_movies_by_content(
    "The Dark Knight",
    movies,
    movie_vectors,
    n_recommendations=5
)

'''
    DEVE SER RODADO COMO MODULO A PARTIR DA PASTA RAIZ DO PROJETO PARA VERIFICAR A SAÍDA

    em (.venv) coloque o script:
    python -m experiments.test_content_model - diretamente no terminal PowerShell ou CMD

    SE EXECUTADO DIRETAMENTE PELO CAMINHO DO ARQUIVO,
    O CAMINHO PARA O CSV NÃO SERÁ ENCONTRADO E RESULTARÁ EM
    UM ERRO DE ARQUIVO NÃO ENCONTRADO

    3 LINE ERROR: FileNotFoundError
'''
print(recommendations[["title", "similarity"]].to_string(index=False))