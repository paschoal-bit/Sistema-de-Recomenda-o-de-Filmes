from sklearn.neighbors import NearestNeighbors

# treinar modelo usando KNN
# movie_matrix : scipy.sparse.csr_matrix
# isso é Matriz esparsa contendo as avaliações dos filmes.
# NearestNeighbors : Modelo KNN treinado.
def train_collaborative_model(movie_matrix):
   
    model = NearestNeighbors(
        algorithm="brute",
        metric="cosine"
    )

    model.fit(movie_matrix)

    return model


# No notebook é preciso definir o código e a função manualmente para testes
# Como: distances, suggestions = modelo.kneighbors(...)
# agora, para facilitar, criamos a função recommend_movies que encapsula esse processo.
def recommend_movies(movie_title, model, movie_pivot, n_recommendations=5):
    """
    Encontra filmes semelhantes ao filme informado.

    Parameters
    ----------
    movie_title : str
        Nome do filme utilizado como referência.

    model : NearestNeighbors
        Modelo KNN treinado.

    movie_pivot : pandas.DataFrame
        Matriz de filmes x usuários.

    n_recommendations : int
        Quantidade de recomendações desejadas.

    Returns
    -------
    list
        Lista contendo os filmes recomendados.
    """

    if movie_title not in movie_pivot.index:
        raise ValueError(
            f"O filme '{movie_title}' não foi encontrado."
        )

    movie_vector = movie_pivot.loc[[movie_title]].values

    distances, indices = model.kneighbors(
        movie_vector,
        n_neighbors=n_recommendations + 1
    )

    recommendations = []

    for index in indices[0]:
        recommended_movie = movie_pivot.index[index]

        if recommended_movie != movie_title:
            recommendations.append(recommended_movie)

    return recommendations[:n_recommendations]