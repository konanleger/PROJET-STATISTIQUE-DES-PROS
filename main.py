import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Chargement des données
def load_data():
    # On charge les données
    data = pd.read_csv("propositions_clusters.csv")
    return data

data = load_data()

# Titre de l'application
st.title("Exploration des Clusters de Propositions")

# Section 1 : Affichage du nuage de points
st.header("Nuage des points par clusters après réduction de dimension (ACP)")
fig = px.scatter(
    data,
    x='PC1', 
    y='PC2', 
    color='cluster', 
    hover_name='propositions', 
    title="Clusters de propositions",
    symbol= 'vote',
    labels={'PC1': 'Composante principale 1', 'PC2': 'Composante principale 2', 'vote': 'Avis des citoyens'}
)
st.plotly_chart(fig, use_container_width=True)

# import plotly.express as px

# fig_3d = px.scatter_3d(
#     data, x='PC1', y='PC2', z='PC3',
#     color='cluster',
#     hover_name='proposition',
#     title="Visualisation des clusters en 3D"
# )
# st.plotly_chart(fig_3d)

# st.write("Analyse des votes par cluster")
# for cluster_id in data['cluster'].unique():
#     cluster_votes = data[data['cluster'] == cluster_id][['pour', 'contre', 'neutre']].mean()
#     st.write(f"Votes moyens pour Cluster {cluster_id}:")
#     st.write(cluster_votes)


# Section 2 : Exploration des clusters
st.header("Explorer les propositions par cluster")
cluster_id = st.selectbox("Choisissez un cluster :", sorted(data['cluster'].unique()))
cluster_data = data[data['cluster'] == cluster_id]

st.write(f"Propositions dans le cluster {cluster_id} :")
st.write(cluster_data[['propositions']])

# Filtrage des propositions par mots clé
keyword = st.text_input("Rechercher un mot-clé dans les propositions :")
if keyword:
    filtered_data = cluster_data [cluster_data ['propositions'].str.contains(keyword, case=False, na=False)]
    st.write(f"Propositions contenant '{keyword}':")
    st.write(filtered_data[['propositions']])

# Les mots les plus frequents par cluster
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Mots les plus fréquents
# all_words = " ".join(cluster_data['propositions'])
# word_freq = Counter(all_words.split())

# st.write("Mots les plus fréquents dans ce cluster :")
# st.write(word_freq.most_common(10))

# from scipy.spatial.distance import cdist

# # Centroides des clusters
# centroids = kmeans.cluster_centers_

# Matrice de distances entre centroides
# distance_matrix = cdist(centroids, centroids, metric='cosine')
# st.write("Matrice de similarité entre clusters :")
# st.dataframe(distance_matrix)



# Section 3 : Analyse des clusters
st.header("Analyse des clusters")
st.write("### Taille des clusters")
cluster_sizes = data['cluster'].value_counts()
st.bar_chart(cluster_sizes)

# Section 4 : Variance expliquée (optionnel)
st.header("Variance expliquée par les axes principaux")
variance = [90.10,0.54]  # Adaptez selon vos calculs d'ACP
st.write(pd.DataFrame({
    'Axe': [1, 2],
    'Variance expliquée (%)': variance
}))

# method = st.selectbox("Choisissez une méthode de clustering :", ["KMeans", "DBSCAN", "Agglomerative Clustering"])


# # Appliquer la normalisation
# scaler = StandardScaler()
# vectors_normalized = scaler.fit_transform(pro_vectors)


# # Normalisation des vecteurs
# scaler = StandardScaler()
# vectors_normalized = scaler.fit_transform(pro_vectors)

# # Réduction des vecteurs à 2 dimensions (ACP)
# pca = PCA(n_components=2)
# word_vectors_2d = pca.fit_transform(pro_vectors)

# if method == "KMeans":
#     n_clusters = st.slider("Nombre de clusters :", 0, 6, 1)
#     kmeans = KMeans(n_clusters=n_clusters)
#     data['cluster'] = kmeans.fit_predict(word_vectors_2d )
# elif method == "DBSCAN":
#     from sklearn.cluster import DBSCAN
#     eps = st.slider("Échelle de densité (epsilon) :", 0.1, 2.0, 0.5)
#     dbscan = DBSCAN(eps=eps)
#     data['cluster'] = dbscan.fit_predict(word_vectors_2d )

# Section 5 : Télécharger les données (optionnel)
st.header("Télécharger les données")
st.download_button(
    label="Télécharger les clusters en CSV",
    data=data.to_csv(index=False),
    file_name='propositions_clusters.csv',
    mime='text/csv'
)

feedback = st.text_area("Vos retours sur ce cluster :")
if st.button("Soumettre vos retours"):
    with open("feedback.txt", "a") as f:
        f.write(f"Cluster {cluster_id} feedback: {feedback}\n")
    st.success("Merci pour vos retours !")

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Chargement des données
def load_data():
    # On charge les données
    data = pd.read_csv("propositions_clusters.csv")
    return data

data = load_data()

# Titre de l'application
st.title("Exploration des Clusters de Propositions")

# Section 1 : Affichage du nuage de points
st.header("Nuage des points par clusters après réduction de dimension (ACP)")
fig = px.scatter(
    data,
    x='PC1', 
    y='PC2', 
    color='cluster', 
    hover_name='propositions', 
    title="Clusters de propositions",
    symbol= 'vote',
    labels={'PC1': 'Composante principale 1', 'PC2': 'Composante principale 2', 'vote': 'Avis des citoyens'}
)
st.plotly_chart(fig, use_container_width=True)

# import plotly.express as px

# fig_3d = px.scatter_3d(
#     data, x='PC1', y='PC2', z='PC3',
#     color='cluster',
#     hover_name='proposition',
#     title="Visualisation des clusters en 3D"
# )
# st.plotly_chart(fig_3d)

# st.write("Analyse des votes par cluster")
# for cluster_id in data['cluster'].unique():
#     cluster_votes = data[data['cluster'] == cluster_id][['pour', 'contre', 'neutre']].mean()
#     st.write(f"Votes moyens pour Cluster {cluster_id}:")
#     st.write(cluster_votes)


# Section 2 : Exploration des clusters
st.header("Explorer les propositions par cluster")
cluster_id = st.selectbox("Choisissez un cluster :", sorted(data['cluster'].unique()))
cluster_data = data[data['cluster'] == cluster_id]

st.write(f"Propositions dans le cluster {cluster_id} :")
st.write(cluster_data[['propositions']])

# Filtrage des propositions par mots clé
keyword = st.text_input("Rechercher un mot-clé dans les propositions :")
if keyword:
    filtered_data = cluster_data [cluster_data ['propositions'].str.contains(keyword, case=False, na=False)]
    st.write(f"Propositions contenant '{keyword}':")
    st.write(filtered_data[['propositions']])

# Les mots les plus frequents par cluster
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Mots les plus fréquents
# all_words = " ".join(cluster_data['propositions'])
# word_freq = Counter(all_words.split())

# st.write("Mots les plus fréquents dans ce cluster :")
# st.write(word_freq.most_common(10))

# from scipy.spatial.distance import cdist

# # Centroides des clusters
# centroids = kmeans.cluster_centers_

# Matrice de distances entre centroides
# distance_matrix = cdist(centroids, centroids, metric='cosine')
# st.write("Matrice de similarité entre clusters :")
# st.dataframe(distance_matrix)



# Section 3 : Analyse des clusters
st.header("Analyse des clusters")
st.write("### Taille des clusters")
cluster_sizes = data['cluster'].value_counts()
st.bar_chart(cluster_sizes)

# Section 4 : Variance expliquée (optionnel)
st.header("Variance expliquée par les axes principaux")
variance = [90.10,0.54]  # Adaptez selon vos calculs d'ACP
st.write(pd.DataFrame({
    'Axe': [1, 2],
    'Variance expliquée (%)': variance
}))

# method = st.selectbox("Choisissez une méthode de clustering :", ["KMeans", "DBSCAN", "Agglomerative Clustering"])


# # Appliquer la normalisation
# scaler = StandardScaler()
# vectors_normalized = scaler.fit_transform(pro_vectors)


# # Normalisation des vecteurs
# scaler = StandardScaler()
# vectors_normalized = scaler.fit_transform(pro_vectors)

# # Réduction des vecteurs à 2 dimensions (ACP)
# pca = PCA(n_components=2)
# word_vectors_2d = pca.fit_transform(pro_vectors)

# if method == "KMeans":
#     n_clusters = st.slider("Nombre de clusters :", 0, 6, 1)
#     kmeans = KMeans(n_clusters=n_clusters)
#     data['cluster'] = kmeans.fit_predict(word_vectors_2d )
# elif method == "DBSCAN":
#     from sklearn.cluster import DBSCAN
#     eps = st.slider("Échelle de densité (epsilon) :", 0.1, 2.0, 0.5)
#     dbscan = DBSCAN(eps=eps)
#     data['cluster'] = dbscan.fit_predict(word_vectors_2d )

# Section 5 : Télécharger les données (optionnel)
st.header("Télécharger les données")
st.download_button(
    label="Télécharger les clusters en CSV",
    data=data.to_csv(index=False),
    file_name='propositions_clusters.csv',
    mime='text/csv'
)

feedback = st.text_area("Vos retours sur ce cluster :")
if st.button("Soumettre vos retours"):
    with open("feedback.txt", "a") as f:
        f.write(f"Cluster {cluster_id} feedback: {feedback}\n")
    st.success("Merci pour vos retours !")
