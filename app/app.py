import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Encontre filmes que você pode gostar "
    "com base em suas preferências."
)

st.subheader("Escolha um filme")

movie = st.selectbox(
    "Digite ou selecione um filme:",
    [
        "Interestelar",
        "A Origem",
        "O Poderoso Chefão",
        "Matrix",
        "O Senhor dos Anéis"
    ]
)

if st.button("Recomendar filmes"):
    st.success(f"Buscando recomendações para: {movie}")

    st.write("Recomendação 1")
    st.write("Recomendação 2")
    st.write("Recomendação 3")